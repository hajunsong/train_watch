"""USB(ADB)로 연결된 안드로이드 폰의 화면을 보고 마우스로 조작하는 도구.

두 가지 화면 수신 방식을 지원한다.

- scrcpy: scrcpy 서버의 H.264 스트림을 받는다. 화면이 바뀔 때 즉시 갱신되고 지연이 낮다.
- screencap: `adb screencap`을 반복 호출한다. 추가 패키지가 필요 없지만 초당 1~2장 수준이다.

실행:
    python android_control.py              # GUI 실행 (scrcpy 우선, 실패 시 screencap)
    python android_control.py --list       # 연결된 기기 목록
    python android_control.py --selftest   # GUI 없이 백엔드 동작만 확인
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# 콘솔 창이 깜빡이지 않도록 자식 프로세스를 숨긴다.
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0

# screencap 원시 출력의 포맷 코드 -> (픽셀당 바이트, PIL 모드, PIL raw 모드)
_SCREENCAP_FORMATS = {
    1: (4, "RGBA", "RGBA"),
    2: (4, "RGBA", "RGBX"),
    3: (3, "RGB", "RGB"),
    4: (2, "RGB", "BGR;16"),
}

KEYCODES = {
    "HOME": 3,
    "BACK": 4,
    "VOLUME_UP": 24,
    "VOLUME_DOWN": 25,
    "POWER": 26,
    "TAB": 61,
    "SPACE": 62,
    "ENTER": 66,
    "DEL": 67,
    "ESCAPE": 111,
    "MOVE_HOME": 122,
    "MOVE_END": 123,
    "APP_SWITCH": 187,
    "SLEEP": 223,
    "WAKEUP": 224,
    "DPAD_UP": 19,
    "DPAD_DOWN": 20,
    "DPAD_LEFT": 21,
    "DPAD_RIGHT": 22,
}


class AdbError(RuntimeError):
    pass


def find_adb() -> str:
    """PATH와 흔한 설치 위치에서 adb 실행 파일을 찾는다."""
    found = shutil.which("adb")
    if found:
        return found

    candidates: list[Path] = []
    for env_name in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        root = os.environ.get(env_name)
        if root:
            candidates.append(Path(root) / "platform-tools" / "adb.exe")

    local = os.environ.get("LOCALAPPDATA")
    if local:
        pattern = str(
            Path(local)
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "Google.PlatformTools*"
            / "platform-tools"
            / "adb.exe"
        )
        candidates.extend(Path(p) for p in glob.glob(pattern))
        candidates.append(Path(local) / "Android" / "Sdk" / "platform-tools" / "adb.exe")

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    raise AdbError(
        "adb를 찾을 수 없습니다. 'winget install Google.PlatformTools'로 설치한 뒤 "
        "터미널을 새로 열어 다시 실행하세요."
    )


def decode_screencap(buf: bytes) -> Image.Image:
    """`adb exec-out screencap`의 원시 프레임을 PIL 이미지로 변환한다."""
    if len(buf) < 12:
        raise AdbError("화면 데이터가 너무 짧습니다. 기기 연결을 확인하세요.")

    width, height, fmt = struct.unpack_from("<III", buf, 0)
    if fmt not in _SCREENCAP_FORMATS or not (0 < width <= 8192 and 0 < height <= 8192):
        raise AdbError(f"알 수 없는 화면 포맷입니다 (w={width}, h={height}, fmt={fmt}).")

    bytes_per_pixel, mode, raw_mode = _SCREENCAP_FORMATS[fmt]
    body_size = width * height * bytes_per_pixel
    # 안드로이드 버전에 따라 헤더 뒤에 colorspace 필드가 더 붙는다.
    header_size = len(buf) - body_size
    if header_size < 12:
        raise AdbError("화면 데이터가 잘렸습니다. 다시 시도하세요.")

    image = Image.frombuffer(
        mode, (width, height), buf[header_size : header_size + body_size], "raw", raw_mode, 0, 1
    )
    return image.convert("RGB")


@dataclass(frozen=True)
class DeviceInfo:
    serial: str
    state: str
    model: str


class AndroidDevice:
    """adb를 통한 화면 캡처와 입력 전송을 담당한다."""

    def __init__(self, adb_path: str, serial: str | None = None):
        self.adb_path = adb_path
        self.serial = serial
        self._shell: subprocess.Popen | None = None
        self._shell_lock = threading.Lock()

    # ------------------------------------------------------------------ adb
    def _argv(self, *args: str) -> list[str]:
        argv = [self.adb_path]
        if self.serial:
            argv += ["-s", self.serial]
        return argv + list(args)

    def run(self, *args: str, timeout: float = 20.0) -> bytes:
        proc = subprocess.run(
            self._argv(*args),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            creationflags=_NO_WINDOW,
        )
        if proc.returncode != 0:
            message = proc.stderr.decode("utf-8", "replace").strip() or "알 수 없는 adb 오류"
            raise AdbError(message)
        return proc.stdout

    @staticmethod
    def list_devices(adb_path: str) -> list[DeviceInfo]:
        proc = subprocess.run(
            [adb_path, "devices", "-l"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            creationflags=_NO_WINDOW,
        )
        devices: list[DeviceInfo] = []
        for line in proc.stdout.decode("utf-8", "replace").splitlines()[1:]:
            parts = line.split()
            if len(parts) < 2:
                continue
            model = next(
                (p.split(":", 1)[1] for p in parts[2:] if p.startswith("model:")), "unknown"
            )
            devices.append(DeviceInfo(serial=parts[0], state=parts[1], model=model))
        return devices

    def describe(self) -> str:
        try:
            model = self.run("shell", "getprop", "ro.product.model").decode().strip()
            release = self.run("shell", "getprop", "ro.build.version.release").decode().strip()
            return f"{model} (Android {release})"
        except (AdbError, subprocess.TimeoutExpired):
            return self.serial or "device"

    # -------------------------------------------------------------- 화면 캡처
    def capture(self, png: bool = True) -> Image.Image:
        """PNG 방식은 기기에서 압축해 보내므로 원시 전송보다 2배 이상 빠르다."""
        if png:
            import io

            image = Image.open(io.BytesIO(self.run("exec-out", "screencap", "-p", timeout=20)))
            image.load()
            return image.convert("RGB")
        return decode_screencap(self.run("exec-out", "screencap", timeout=20))

    # ------------------------------------------------------------------ 입력
    def _write_shell(self, line: str) -> None:
        """`adb shell`을 계속 띄워두고 명령만 흘려보낸다 (매번 실행하면 느리다)."""
        with self._shell_lock:
            for attempt in range(2):
                if self._shell is None or self._shell.poll() is not None:
                    self._shell = subprocess.Popen(
                        self._argv("shell"),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=_NO_WINDOW,
                    )
                try:
                    assert self._shell.stdin is not None
                    self._shell.stdin.write(f"{line}\n".encode("utf-8"))
                    self._shell.stdin.flush()
                    return
                except (BrokenPipeError, OSError):
                    self._shell = None
                    if attempt == 1:
                        raise AdbError("기기와의 연결이 끊어졌습니다.")

    def tap(self, x: int, y: int) -> None:
        self._write_shell(f"input tap {int(x)} {int(y)}")

    def long_press(self, x: int, y: int, duration_ms: int = 600) -> None:
        self._write_shell(f"input swipe {int(x)} {int(y)} {int(x)} {int(y)} {int(duration_ms)}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        self._write_shell(
            f"input swipe {int(x1)} {int(y1)} {int(x2)} {int(y2)} {int(duration_ms)}"
        )

    def keyevent(self, key: int | str) -> None:
        code = KEYCODES[key] if isinstance(key, str) else int(key)
        self._write_shell(f"input keyevent {code}")

    def input_text(self, text: str) -> None:
        if not text:
            return
        quoted = text.replace("'", "'\\''")
        self._write_shell(f"input text '{quoted}'")

    def close(self) -> None:
        with self._shell_lock:
            if self._shell and self._shell.poll() is None:
                try:
                    self._shell.stdin.close()  # type: ignore[union-attr]
                except OSError:
                    pass
                self._shell.terminate()
            self._shell = None


class ScreencapBackend:
    """`adb screencap`을 백그라운드에서 반복 호출한다. 추가 패키지가 필요 없다."""

    label = "screencap"
    supports_unicode_text = False

    def __init__(self, device: AndroidDevice, target_fps: float = 4.0, png: bool = True):
        self.device = device
        self.png = png
        self.interval = 1.0 / max(target_fps, 0.5)
        self._lock = threading.Lock()
        self._frame: Image.Image | None = None
        self._counter = 0
        self._error: str | None = None
        self._stamps: deque[float] = deque(maxlen=12)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="screencap")

    def start(self) -> None:
        self._frame = self.device.capture(self.png)  # 연결 실패를 즉시 알 수 있게 한 장 먼저
        self._counter = 1
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.device.close()

    def poll(self) -> tuple[Image.Image | None, int, str | None, float]:
        with self._lock:
            return self._frame, self._counter, self._error, _rate(self._stamps)

    def _loop(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                frame = self.device.capture(self.png)
            except Exception as exc:  # 연결 끊김 등은 표시만 하고 계속 재시도한다.
                with self._lock:
                    self._error = str(exc)
                self._stop.wait(0.7)
                continue
            with self._lock:
                self._frame = frame
                self._counter += 1
                self._error = None
                self._stamps.append(time.perf_counter())
            self._stop.wait(max(0.0, self.interval - (time.perf_counter() - started)))

    # ------------------------------------------------------------------ 입력
    def tap(self, x: int, y: int) -> None:
        self.device.tap(x, y)

    def long_press(self, x: int, y: int, duration_ms: int) -> None:
        self.device.long_press(x, y, duration_ms)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        self.device.swipe(x1, y1, x2, y2, duration_ms)

    def key(self, name: str) -> None:
        self.device.keyevent(name)

    def send_text(self, text: str) -> None:
        self.device.input_text(text)


class ScrcpyBackend:
    """scrcpy 서버가 보내는 H.264 스트림을 받는다. 화면 변화에 바로 반응한다."""

    label = "scrcpy"
    supports_unicode_text = True

    def __init__(
        self, adb_path: str, serial: str | None, max_size: int = 1080, max_fps: int = 30
    ):
        from py_scrcpy_sdk import ScrcpyClient, ScrcpyConfig

        self._client = ScrcpyClient(
            ScrcpyConfig(serial=serial, adb_path=adb_path, max_size=max_size, max_fps=max_fps)
        )
        self._lock = threading.Lock()
        self._array = None
        self._counter = 0
        self._error: str | None = None
        self._stamps: deque[float] = deque(maxlen=30)
        self._cached: tuple[int, Image.Image] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read_loop, daemon=True, name="scrcpy-reader")

    def start(self) -> None:
        self._client.start()
        self._store(self._client.wait_until_ready(timeout=25))
        self._thread.start()

    def stop(self) -> None:
        # 먼저 표시해 두어야 소켓이 닫힐 때 나는 예외를 오류로 오해하지 않는다.
        self._stop.set()
        self._client.stop()

    def _read_loop(self) -> None:
        try:
            for array in self._client.frames():
                if self._stop.is_set():
                    return
                self._store(array)
        except Exception as exc:
            if not self._stop.is_set():
                with self._lock:
                    self._error = str(exc)

    def _store(self, array) -> None:
        with self._lock:
            self._array = array
            self._counter += 1
            self._error = None
            self._stamps.append(time.perf_counter())

    def poll(self) -> tuple[Image.Image | None, int, str | None, float]:
        import numpy as np

        with self._lock:
            array, counter = self._array, self._counter
            error, fps = self._error, _rate(self._stamps)

        if array is None:
            return None, counter, error, fps
        if self._cached is not None and self._cached[0] == counter:
            return self._cached[1], counter, error, fps

        # PyAV 디코더는 BGR 순서로 넘겨준다 (screencap 결과와 비교해 확인).
        image = Image.fromarray(np.ascontiguousarray(array[:, :, ::-1]))
        self._cached = (counter, image)
        return image, counter, error, fps

    # ------------------------------------------------------------------ 입력
    def tap(self, x: int, y: int) -> None:
        self._client.tap(x, y)

    def long_press(self, x: int, y: int, duration_ms: int) -> None:
        self._client.long_press(x, y, duration=duration_ms / 1000)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        self._client.swipe(x1, y1, x2, y2, duration_ms=duration_ms)

    def key(self, name: str) -> None:
        self._client.press_key(KEYCODES[name])

    def send_text(self, text: str) -> None:
        if text.isascii():
            self._client.input_text(text)
        else:
            self._client.paste_text(text)  # 한글은 클립보드 붙여넣기로 전달된다


def _rate(stamps: deque[float]) -> float:
    if len(stamps) < 2:
        return 0.0
    span = stamps[-1] - stamps[0]
    return (len(stamps) - 1) / span if span > 0 else 0.0


class MirrorApp:
    """폰 화면을 표시하고 마우스/키보드 입력을 기기로 보내는 Tkinter 창."""

    TAP_TOLERANCE_PX = 12
    LONG_PRESS_MS = 450
    WHEEL_STEP_RATIO = 0.22  # 휠 한 칸에 화면 높이의 몇 배를 스와이프할지

    def __init__(self, backend, title: str, window_height: int):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.backend = backend
        self.window_height = window_height

        self._photo = None
        self._shown_counter = -1
        self._sized = False
        self._view: tuple[float, float, int, int, int, int] | None = None
        self._press: tuple[int, int, float] | None = None
        self._drag: tuple[int, int] | None = None

        self.root = tk.Tk()
        self.root.title(f"Android Control - {title}")
        self.root.minsize(320, 480)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        toolbar = ttk.Frame(self.root, padding=(6, 6, 6, 0))
        toolbar.pack(side=tk.TOP, fill=tk.X)
        for label, key in [
            ("◀ 뒤로", "BACK"),
            ("● 홈", "HOME"),
            ("■ 최근", "APP_SWITCH"),
            ("전원", "POWER"),
            ("깨우기", "WAKEUP"),
            ("음량+", "VOLUME_UP"),
            ("음량-", "VOLUME_DOWN"),
        ]:
            ttk.Button(
                toolbar, text=label, width=7, command=self._guard(lambda k=key: self.backend.key(k))
            ).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="저장", width=7, command=self._guard(self._save_screenshot)).pack(
            side=tk.LEFT, padx=2
        )

        self.canvas = tk.Canvas(self.root, background="#111318", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)

        bottom = ttk.Frame(self.root, padding=(6, 0, 6, 6))
        bottom.pack(side=tk.BOTTOM, fill=tk.X)
        self.entry = ttk.Entry(bottom)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda _event: self._send_entry())
        ttk.Button(bottom, text="전송", width=6, command=self._guard(self._send_entry)).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        self.status = ttk.Label(self.root, anchor="w", padding=(8, 0, 8, 4))
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Button-3>", self._guard(lambda _e=None: self.backend.key("BACK")))
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.root.bind("<Key>", self._on_key)

    # ---------------------------------------------------------------- 실행
    def run(self) -> None:
        self._tick()
        self.root.mainloop()

    def _on_close(self) -> None:
        self.backend.stop()
        self.root.destroy()

    def _guard(self, func):
        """콜백에서 발생한 오류를 창을 닫지 않고 상태줄에 표시한다."""

        def wrapped(*_args, **_kwargs):
            try:
                return func()
            except Exception as exc:
                self.status.configure(text=f"오류: {exc}")

        return wrapped

    # ------------------------------------------------------------- 화면 갱신
    def _tick(self) -> None:
        frame, counter, error, fps = self.backend.poll()
        if frame is not None and counter != self._shown_counter:
            self._shown_counter = counter
            self._draw(frame)

        if error:
            self.status.configure(text=f"연결 오류: {error} (재시도 중)")
        elif frame is not None:
            width, height = frame.size
            self.status.configure(
                text=(
                    f"{self.backend.label}  |  {width}x{height}  |  {fps:.1f} fps  |  "
                    "좌클릭=탭, 드래그=스와이프, 길게=롱프레스, 우클릭=뒤로, 휠=스크롤"
                )
            )
        else:
            self.status.configure(text="화면을 가져오는 중...")

        self.root.after(25, self._tick)

    def _draw(self, frame: Image.Image) -> None:
        from PIL import ImageTk

        dev_w, dev_h = frame.size
        if not self._sized:
            self._sized = True
            canvas_h = max(self.window_height - 110, 200)
            self.root.geometry(f"{int(canvas_h * dev_w / dev_h) + 24}x{self.window_height}")
            self.root.update_idletasks()

        view_w = max(self.canvas.winfo_width(), 1)
        view_h = max(self.canvas.winfo_height(), 1)
        scale = min(view_w / dev_w, view_h / dev_h)
        disp_w = max(int(dev_w * scale), 1)
        disp_h = max(int(dev_h * scale), 1)
        offset_x = (view_w - disp_w) / 2
        offset_y = (view_h - disp_h) / 2

        self._photo = ImageTk.PhotoImage(frame.resize((disp_w, disp_h), Image.BILINEAR))
        self.canvas.delete("frame")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self._photo, tags="frame")
        self._view = (offset_x, offset_y, disp_w, disp_h, dev_w, dev_h)

    # ------------------------------------------------------------ 좌표 변환
    def _to_device(self, event_x: float, event_y: float) -> tuple[int, int] | None:
        if self._view is None:
            return None
        offset_x, offset_y, disp_w, disp_h, dev_w, dev_h = self._view
        local_x = event_x - offset_x
        local_y = event_y - offset_y
        if not (0 <= local_x < disp_w and 0 <= local_y < disp_h):
            return None
        x = int(local_x / disp_w * dev_w)
        y = int(local_y / disp_h * dev_h)
        return min(max(x, 0), dev_w - 1), min(max(y, 0), dev_h - 1)

    # -------------------------------------------------------------- 마우스
    def _on_press(self, event) -> None:
        point = self._to_device(event.x, event.y)
        if point is None:
            return
        self._press = (point[0], point[1], time.perf_counter())
        self._drag = point

    def _on_motion(self, event) -> None:
        point = self._to_device(event.x, event.y)
        if point is not None:
            self._drag = point

    def _on_release(self, event) -> None:
        if self._press is None:
            return
        start_x, start_y, started = self._press
        self._press = None

        end_x, end_y = self._to_device(event.x, event.y) or self._drag or (start_x, start_y)
        held_ms = (time.perf_counter() - started) * 1000
        moved = max(abs(end_x - start_x), abs(end_y - start_y))

        try:
            if moved > self.TAP_TOLERANCE_PX:
                self.backend.swipe(
                    start_x, start_y, end_x, end_y, int(min(max(held_ms, 60), 1500))
                )
            elif held_ms >= self.LONG_PRESS_MS:
                self.backend.long_press(start_x, start_y, int(min(held_ms, 3000)))
            else:
                self.backend.tap(start_x, start_y)
        except Exception as exc:
            self.status.configure(text=f"입력 실패: {exc}")

    def _on_wheel(self, event) -> None:
        point = self._to_device(event.x, event.y)
        if point is None or self._view is None:
            return
        dev_h = self._view[5]
        notches = event.delta / 120 if event.delta else 0
        # 휠을 올리면 손가락을 아래로 끌어 화면 내용이 위로 올라가게 한다.
        target_y = min(max(point[1] + int(notches * dev_h * self.WHEEL_STEP_RATIO), 0), dev_h - 1)
        if target_y == point[1]:
            return
        try:
            self.backend.swipe(point[0], point[1], point[0], target_y, 120)
        except Exception as exc:
            self.status.configure(text=f"스크롤 실패: {exc}")

    # ------------------------------------------------------------ 키보드
    _KEYSYM_MAP = {
        "BackSpace": "DEL",
        "Return": "ENTER",
        "KP_Enter": "ENTER",
        "Escape": "BACK",
        "Tab": "TAB",
        "space": "SPACE",
        "Up": "DPAD_UP",
        "Down": "DPAD_DOWN",
        "Left": "DPAD_LEFT",
        "Right": "DPAD_RIGHT",
        "Home": "MOVE_HOME",
        "End": "MOVE_END",
    }

    def _on_key(self, event) -> None:
        if self.root.focus_get() is self.entry:
            return
        try:
            mapped = self._KEYSYM_MAP.get(event.keysym)
            if mapped:
                self.backend.key(mapped)
            elif event.char and event.char.isprintable():
                self.backend.send_text(event.char)
        except Exception as exc:
            self.status.configure(text=f"키 입력 실패: {exc}")

    def _send_entry(self) -> None:
        text = self.entry.get()
        if not text:
            return
        if not text.isascii() and not self.backend.supports_unicode_text:
            self.status.configure(
                text="screencap 방식에서는 한글을 보낼 수 없습니다. scrcpy 방식으로 실행하세요."
            )
            return
        self.backend.send_text(text)
        self.entry.delete(0, self.tk.END)

    def _save_screenshot(self) -> None:
        frame, _counter, _error, _fps = self.backend.poll()
        if frame is None:
            self.status.configure(text="저장할 화면이 아직 없습니다.")
            return
        out_dir = Path(__file__).parent / "screenshots"
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.png"
        frame.save(path)
        self.status.configure(text=f"저장: {path}")


def pick_serial(adb_path: str, requested: str | None) -> str:
    devices = AndroidDevice.list_devices(adb_path)
    ready = [d for d in devices if d.state == "device"]

    if requested:
        if any(d.serial == requested for d in ready):
            return requested
        raise AdbError(f"'{requested}' 기기를 사용할 수 없습니다.")

    if not ready:
        if devices:
            detail = ", ".join(f"{d.serial}({d.state})" for d in devices)
            raise AdbError(
                f"사용 가능한 기기가 없습니다: {detail}. "
                "폰에서 'USB 디버깅 허용'을 눌렀는지 확인하세요."
            )
        raise AdbError("연결된 기기가 없습니다. USB 연결과 USB 디버깅 설정을 확인하세요.")

    if len(ready) > 1:
        detail = ", ".join(f"{d.serial}({d.model})" for d in ready)
        raise AdbError(f"기기가 여러 대입니다. --serial 로 선택하세요: {detail}")

    return ready[0].serial


def create_backend(args, adb_path: str, serial: str):
    """요청한 방식대로 백엔드를 만든다. auto는 scrcpy를 먼저 시도한다."""
    if args.backend in ("auto", "scrcpy"):
        try:
            backend = ScrcpyBackend(adb_path, serial, args.max_size, int(args.fps))
            backend.start()
            return backend
        except Exception as exc:
            if args.backend == "scrcpy":
                raise AdbError(f"scrcpy 방식으로 연결하지 못했습니다: {exc}") from exc
            print(f"[알림] scrcpy 방식 실패({exc}). screencap 방식으로 진행합니다.")

    # screencap은 매번 화면 전체를 받아오므로 더 높은 FPS를 요구해도 USB 대역폭만 낭비한다.
    backend = ScreencapBackend(AndroidDevice(adb_path, serial), target_fps=min(args.fps, 5.0))
    backend.start()
    return backend


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="USB로 연결된 안드로이드 폰 화면 미러링 및 조작")
    parser.add_argument("--serial", help="대상 기기 시리얼 (adb devices 참고)")
    parser.add_argument(
        "--backend",
        choices=["auto", "scrcpy", "screencap"],
        default="auto",
        help="화면 수신 방식 (기본 auto: scrcpy 우선)",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="목표 FPS (기본 30)")
    parser.add_argument(
        "--max-size", type=int, default=1080, help="scrcpy 방식의 긴 변 최대 픽셀 (기본 1080)"
    )
    parser.add_argument("--height", type=int, default=880, help="창 초기 높이 (기본 880)")
    parser.add_argument("--list", action="store_true", help="연결된 기기 목록만 출력")
    parser.add_argument("--selftest", action="store_true", help="GUI 없이 백엔드 동작만 확인")
    args = parser.parse_args(argv)

    try:
        adb_path = find_adb()
    except AdbError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 2

    if args.list:
        devices = AndroidDevice.list_devices(adb_path)
        if not devices:
            print("연결된 기기가 없습니다.")
        for device in devices:
            print(f"{device.serial}\t{device.state}\t{device.model}")
        return 0

    backend = None
    try:
        serial = pick_serial(adb_path, args.serial)
        title = AndroidDevice(adb_path, serial).describe()
        backend = create_backend(args, adb_path, serial)

        if args.selftest:
            time.sleep(1.0)
            frame, counter, error, fps = backend.poll()
            if frame is None:
                print(f"[오류] 화면을 받지 못했습니다: {error}", file=sys.stderr)
                return 1
            out_dir = Path(__file__).parent / "screenshots"
            out_dir.mkdir(exist_ok=True)
            path = out_dir / "selftest.png"
            frame.save(path)
            print(f"기기: {title} ({serial})")
            print(f"백엔드: {backend.label}")
            print(f"해상도: {frame.size[0]}x{frame.size[1]}  (프레임 {counter}개, {fps:.1f} fps)")
            print(f"저장: {path}")
            return 0

        MirrorApp(backend, title, args.height).run()
        return 0
    except AdbError as exc:
        print(f"[오류] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if backend is not None:
            backend.stop()


if __name__ == "__main__":
    sys.exit(main())
