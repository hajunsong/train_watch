"""KTX 333(13:55) 매진이 풀릴 때까지 새로고침한다.

자리가 열리면 바로 예매 → 장바구니 담기 → 등록 카드 결제 화면.
예약 대기가 열리면 예약 대기 신청을 누른다.
결제 비밀번호는 사람이 입력한다.

    python watch_ktx333.py --check   # 현재 화면에서 위치만 확인
    python watch_ktx333.py           # 감시 시작
    python watch_ktx333.py --dry-run # 감지까지만, 클릭 안 함
"""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from android_control import AdbError, AndroidDevice, find_adb, pick_serial, _NO_WINDOW

BOUNDS_RE = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")

# 코레일 열차 카드에서 파란 KTX 뱃지 대비 매진 글씨 상대 위치 (실측값)
STANDARD_OFFSET = (79, 118)  # 일반실 매진
SPECIAL_OFFSET = (158, 197)  # 특실 매진
SOLD_OUT_X = (961, 1030)
SOLD_OUT_RED_MIN = 400  # 매진 글씨 1개는 약 1100 픽셀
WAITLIST_GREEN_MIN = 200  # '대기' 초록 글씨는 약 800 픽셀

REFRESH_FALLBACK = (900, 149)
BOOK_FALLBACK = (790, 2208)  # 하단 시트가 열렸을 때의 바로 예매
WAITLIST_FALLBACK = (540, 2160)  # 하단 전체 너비의 예약 대기 신청
CART_LABELS = ("장바구니 담기", "장바구니담기")
WAITLIST_LABELS = ("예약 대기 신청", "예약대기신청")
BOOK_LABELS = ("바로 예매",)
PAY_LABELS = ("결제하기", "바로결제", "바로 결제")
CARD_LABELS = ("등록카드", "등록된 카드", "등록 카드", "신용/체크카드", "신용체크카드")
PAY_AGREE_LABELS = ("전체 동의", "모두 동의", "약관 전체동의", "결제대행 서비스 약관")
PIN_LABELS = ("결제 비밀번호", "비밀번호 입력", "보안키패드", "보안 키패드", "PIN")
SANCHON_NOTICE_LABELS = (
    "이용 안내",
    "KTX-산천 2개를 연결",
    "연결해 운행하는 열차",
    "열차 번호와 호차",
)
SANCHON_CONFIRM_FALLBACK = (540, 1353)
AD_SKIP_PATH = Path(__file__).parent / "ad_skip.json"
AD_CLOSE_NEEDLES = (
    "닫기",
    "광고닫기",
    "광고 닫기",
    "건너뛰기",
    "skipad",
    "skip ad",
    "skip_ad",
    "btnclose",
    "btn_close",
    "iv_close",
    "ad_close",
    "오늘하루보지않기",
    "다시보지않기",
)


@dataclass
class Bounds:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2


@dataclass
class SeatStatus:
    name: str
    kind: str  # sold_out | waitlist | available
    red_pixels: int
    green_pixels: int
    tap: tuple[int, int]
    box: Bounds

    @property
    def sold_out(self) -> bool:
        return self.kind == "sold_out"


@dataclass
class TrainCard:
    badge: Bounds
    standard: SeatStatus
    special: SeatStatus


def parse_bounds(raw: str) -> Bounds:
    m = BOUNDS_RE.fullmatch(raw)
    if not m:
        raise ValueError(raw)
    return Bounds(*map(int, m.groups()))


def dump_ui(device: AndroidDevice, timeout: float = 20.0) -> ET.Element:
    raw = device.run("exec-out", "uiautomator", "dump", "/dev/tty", timeout=timeout)
    text = raw.decode("utf-8", "replace")
    start = text.find("<?xml")
    end = text.rfind("</hierarchy>")
    if start < 0 or end < 0:
        raise RuntimeError("화면 구조를 읽지 못했습니다.")
    return ET.fromstring(text[start : end + len("</hierarchy>")])


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value or "")


def find_node(
    root: ET.Element,
    *,
    desc: str | None = None,
    text: str | None = None,
    contains: str | None = None,
) -> tuple[ET.Element, Bounds] | None:
    needle = _norm(contains) if contains else None
    for node in root.iter("node"):
        node_text = node.attrib.get("text") or ""
        node_desc = node.attrib.get("content-desc") or ""
        if desc is not None and node_desc != desc:
            continue
        if text is not None and node_text != text:
            continue
        if needle is not None and needle not in _norm(node_text) and needle not in _norm(node_desc):
            continue
        bounds = parse_bounds(node.attrib["bounds"])
        if bounds.x2 - bounds.x1 < 4 or bounds.y2 - bounds.y1 < 4:
            continue
        return node, bounds
    return None


def clickable_bounds(root: ET.Element, inner: ET.Element) -> Bounds:
    """아이콘 노드 대신, 그걸 감싼 클릭 가능한 부모의 영역을 쓴다."""
    parent_map = {child: parent for parent in root.iter("node") for child in parent}
    node: ET.Element | None = inner
    while node is not None:
        if node.attrib.get("clickable") == "true":
            return parse_bounds(node.attrib["bounds"])
        node = parent_map.get(node)
    return parse_bounds(inner.attrib["bounds"])


def red_mask(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (r > 160) & (g < 90) & (b < 90) & ((r - g) > 80)


def waitlist_mask(rgb: np.ndarray) -> np.ndarray:
    """코레일 '대기' 글씨. 실측 RGB 약 (34, 197, 94)."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (g > 150) & (r < 80) & (b < 140) & ((g - r) > 80)


def blue_badge_bands(rgb: np.ndarray) -> list[Bounds]:
    """왼쪽의 파란 KTX / KTX-산천 뱃지를 세로 구간으로 찾는다."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    blue = (b > 140) & (r < 100) & (g < 160)
    # 헤더/하단 네비는 제외하고 열차 목록만 본다.
    rows = np.where(blue[700:2200].any(axis=1))[0] + 700
    bands: list[tuple[int, int]] = []
    start = prev = None
    for y in rows:
        if start is None:
            start = prev = int(y)
        elif y <= prev + 2:
            prev = int(y)
        else:
            bands.append((start, prev))
            start = prev = int(y)
    if start is not None and prev is not None:
        bands.append((start, prev))

    badges: list[Bounds] = []
    for y1, y2 in bands:
        xs = np.where(blue[y1 : y2 + 1].any(axis=0))[0]
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        if x1 > 80 or (y2 - y1) < 30:
            continue
        badges.append(Bounds(x1, y1, x2, y2))
    return badges


def seat_status(rgb: np.ndarray, badge: Bounds, name: str, y_off: tuple[int, int]) -> SeatStatus:
    box = Bounds(SOLD_OUT_X[0], badge.y1 + y_off[0], SOLD_OUT_X[1], badge.y1 + y_off[1])
    h, w = rgb.shape[:2]
    y1, y2 = max(box.y1, 0), min(box.y2, h)
    x1, x2 = max(box.x1, 0), min(box.x2, w)
    if y2 <= y1 or x2 <= x1:
        red = green = 0
    else:
        crop = rgb[y1:y2, x1:x2]
        red = int(red_mask(crop).sum())
        green = int(waitlist_mask(crop).sum())
    if red >= SOLD_OUT_RED_MIN:
        kind = "sold_out"
    elif green >= WAITLIST_GREEN_MIN:
        kind = "waitlist"
    elif red < 80 and green < 80:
        kind = "unknown"  # 로딩·광고·가림. 빈 칸을 좌석으로 보지 않음
    else:
        kind = "available"
    return SeatStatus(
        name=name,
        kind=kind,
        red_pixels=red,
        green_pixels=green,
        tap=((x1 + x2) // 2, (y1 + y2) // 2),
        box=Bounds(x1, y1, x2, y2),
    )


def find_ktx333(rgb: np.ndarray) -> TrainCard | None:
    """KTX-산천이 아닌 짧은 파란 'KTX' 뱃지가 333호다."""
    plains = [b for b in blue_badge_bands(rgb) if (b.x2 - b.x1) < 120]
    if not plains:
        return None
    badge = plains[0]
    return TrainCard(
        badge=badge,
        standard=seat_status(rgb, badge, "일반실", STANDARD_OFFSET),
        special=seat_status(rgb, badge, "특실", SPECIAL_OFFSET),
    )


def save_overlay(image: Image.Image, card: TrainCard, path: Path) -> None:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    draw.rectangle((card.badge.x1, card.badge.y1, card.badge.x2, card.badge.y2), outline="#3b82f6", width=4)
    colors = {"sold_out": "#ef4444", "waitlist": "#22c55e", "available": "#f59e0b", "unknown": "#94a3b8"}
    for seat in (card.standard, card.special):
        draw.rectangle((seat.box.x1, seat.box.y1, seat.box.x2, seat.box.y2), outline=colors[seat.kind], width=4)
    vis.save(path)


def seat_label(seat: SeatStatus) -> str:
    names = {"sold_out": "매진", "waitlist": "예약대기", "available": "가능", "unknown": "미확인"}
    extra = seat.green_pixels if seat.kind == "waitlist" else seat.red_pixels
    return f"{names[seat.kind]}({extra})"


def pick_target(card: TrainCard) -> SeatStatus | None:
    """실제 좌석이 있으면 그걸 먼저, 없으면 예약 대기."""
    for seat in (card.standard, card.special):
        if seat.kind == "available":
            return seat
    for seat in (card.standard, card.special):
        if seat.kind == "waitlist":
            return seat
    return None


def beep_pc() -> None:
    try:
        import winsound

        winsound.Beep(1200, 400)
        winsound.Beep(1600, 600)
        winsound.Beep(1200, 400)
    except Exception:
        print("\a", end="", flush=True)


def _shell_try(device: AndroidDevice, *args: str) -> None:
    try:
        device.run("shell", *args, timeout=8)
    except (AdbError, subprocess.TimeoutExpired, OSError):
        pass


def _phone_sound_on(device: AndroidDevice) -> None:
    """무음/진동이면 소리 모드로 바꾸고 벨·알림 볼륨을 켠다."""
    _shell_try(device, "settings", "put", "global", "zen_mode", "0")
    _shell_try(device, "settings", "put", "system", "all_sound_off", "0")
    _shell_try(device, "cmd", "audio", "set-ringer-mode", "NORMAL")
    for stream in ("RING", "NOTIFICATION", "MUSIC", "ALARM", "SYSTEM"):
        _shell_try(device, "cmd", "audio", "adj-unmute", stream)
    _shell_try(device, "cmd", "audio", "set-volume", "RING", "11")
    _shell_try(device, "cmd", "audio", "set-volume", "NOTIFICATION", "11")
    _shell_try(device, "cmd", "audio", "set-volume", "ALARM", "11")
    _shell_try(device, "cmd", "audio", "set-volume", "MUSIC", "8")


def _phone_alert(device: AndroidDevice, kind: str) -> None:
    _phone_sound_on(device)
    title = "KTX 333 예매 가능" if kind == "available" else "KTX 333 예약 대기"
    _shell_try(device, "cmd", "notification", "post", "-t", title, "ktx333", "지금 화면을 확인하세요")
    _shell_try(device, "cmd", "vibrator_manager", "synced", "oneshot", "700")
    for _ in range(6):
        _shell_try(device, "cmd", "audio", "adj-volume", "RING", "RAISE")
        time.sleep(0.12)
        _shell_try(device, "cmd", "audio", "adj-volume", "RING", "LOWER")
        time.sleep(0.12)


def alert_found(device: AndroidDevice, kind: str) -> None:
    """예매/예약대기 발견 시 PC와 폰에서 동시에 알린다. 클릭은 막지 않는다."""
    threading.Thread(target=beep_pc, daemon=True).start()
    threading.Thread(target=_phone_alert, args=(device, kind), daemon=True).start()


def wait_for_card(device: AndroidDevice, timeout: float = 4.0) -> tuple[Image.Image, np.ndarray, TrainCard | None]:
    deadline = time.perf_counter() + timeout
    image = device.capture(png=True)
    rgb = np.array(image)
    card = find_ktx333(rgb)
    while card is None and time.perf_counter() < deadline:
        time.sleep(0.25)
        image = device.capture(png=True)
        rgb = np.array(image)
        card = find_ktx333(rgb)
    return image, rgb, card


_INPUT_EVENT_64 = struct.Struct("QQHHi")
_DEV_RE = re.compile(r"add device \d+:\s*(/dev/input/event\d+)")
_ABS_RE = re.compile(
    r"([0-9a-fA-F]+)\s*:\s*value\s+-?\d+,\s*min\s+(-?\d+),\s*max\s+(-?\d+)",
    re.I,
)


def _screen_size(device: AndroidDevice) -> tuple[int, int]:
    raw = device.run("shell", "wm", "size").decode("utf-8", "replace")
    override = re.search(r"Override size:\s*(\d+)x(\d+)", raw)
    if override:
        return int(override.group(1)), int(override.group(2))
    physical = re.search(r"Physical size:\s*(\d+)x(\d+)", raw)
    if physical:
        return int(physical.group(1)), int(physical.group(2))
    return 1080, 2340


def _touch_devices(info: str) -> list[tuple[str, str, int, int]]:
    found: list[tuple[str, str, int, int]] = []
    path = ""
    name = ""
    axes: dict[int, tuple[int, int]] = {}

    def flush() -> None:
        if path and 0x35 in axes and 0x36 in axes:
            found.append((path, name, axes[0x35][1], axes[0x36][1]))

    for line in info.splitlines():
        header = _DEV_RE.search(line)
        if header:
            flush()
            path, name, axes = header.group(1), "", {}
            continue
        if "name:" in line:
            name = line.split("name:", 1)[-1].strip().strip('"')
            continue
        abs_match = _ABS_RE.search(line)
        if abs_match:
            axes[int(abs_match.group(1), 16)] = (
                int(abs_match.group(2)),
                int(abs_match.group(3)),
            )
    flush()
    found.sort(key=lambda item: (0 if "touchscreen" in item[1].lower() else 1, item[0]))
    return found


def _scale_axis(raw: int, axis_max: int, screen: int) -> int:
    if axis_max <= 0 or screen <= 1:
        return raw
    return min(screen - 1, max(0, int(round(raw * (screen - 1) / axis_max))))


class TouchMonitor:
    """폰 터치스크린의 실제 터치를 읽어 광고 닫기 좌표를 배운다."""

    def __init__(self, device: AndroidDevice):
        self.device = device
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._taps: deque[tuple[int, int]] = deque(maxlen=40)
        self._scale = (1, 1, 1080, 2340)

    def start(self) -> bool:
        try:
            info = self.device.run("shell", "getevent", "-p", timeout=8).decode(
                "utf-8", "replace"
            )
            devices = _touch_devices(info)
            width, height = _screen_size(self.device)
        except (AdbError, subprocess.TimeoutExpired) as exc:
            print(f"[알림] 터치 기록을 시작하지 못했습니다: {exc}")
            return False
        if not devices:
            print("[알림] 터치스크린 입력을 찾지 못했습니다. --learn-ad 로 좌표를 찍으세요.")
            return False
        path, name, max_x, max_y = devices[0]
        self._scale = (max_x, max_y, width, height)
        self._proc = subprocess.Popen(
            self.device._argv("exec-out", "cat", path),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=_NO_WINDOW,
        )
        self._thread = threading.Thread(target=self._loop, daemon=True, name="touch-monitor")
        self._thread.start()
        print(f"터치 기록: {name} {path} (축 {max_x}x{max_y} → {width}x{height})")
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        self._proc = None

    def drain(self) -> list[tuple[int, int]]:
        with self._lock:
            taps = list(self._taps)
            self._taps.clear()
        return taps

    def _loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        max_x, max_y, width, height = self._scale
        x = y = 0
        pending = False
        buf = b""
        while not self._stop.is_set():
            chunk = self._proc.stdout.read(24)
            if not chunk:
                break
            buf += chunk
            while len(buf) >= 24:
                _sec, _usec, ev_type, code, value = _INPUT_EVENT_64.unpack_from(buf, 0)
                buf = buf[24:]
                if ev_type != 0x03:
                    continue
                if code == 0x35:
                    x = _scale_axis(value, max_x, width)
                    pending = True
                elif code == 0x36:
                    y = _scale_axis(value, max_y, height)
                    pending = True
                elif code == 0x39 and value == -1 and pending:
                    with self._lock:
                        self._taps.append((x, y))
                    pending = False


class AdSkipStore:
    def __init__(self, sequences: list[list[tuple[int, int]]] | None = None):
        self.sequences = sequences or []

    @classmethod
    def load(cls) -> "AdSkipStore":
        if not AD_SKIP_PATH.is_file():
            return cls()
        try:
            data = json.loads(AD_SKIP_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls()
        sequences: list[list[tuple[int, int]]] = []
        for item in data.get("sequences") or []:
            taps = item.get("taps") if isinstance(item, dict) else item
            parsed = [
                (int(tap[0]), int(tap[1]))
                for tap in (taps or [])
                if isinstance(tap, (list, tuple)) and len(tap) == 2
            ]
            if parsed:
                sequences.append(parsed)
        return cls(sequences)

    def save(self) -> None:
        payload = {
            "sequences": [
                {"taps": taps, "saved_at": datetime.now().isoformat(timespec="seconds")}
                for taps in self.sequences
            ]
        }
        AD_SKIP_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_sequence(self, taps: list[tuple[int, int]], *, source: str) -> None:
        cleaned: list[tuple[int, int]] = []
        for tap in taps:
            if cleaned and max(abs(tap[0] - cleaned[-1][0]), abs(tap[1] - cleaned[-1][1])) < 18:
                cleaned[-1] = tap
            else:
                cleaned.append(tap)
        if not cleaned:
            return
        self.sequences = [seq for seq in self.sequences if seq != cleaned]
        self.sequences.insert(0, cleaned[:8])
        self.sequences = self.sequences[:6]
        self.save()
        print(f"광고 닫기 좌표 저장({source}): {cleaned}")


def _looks_like_ad_close(node: ET.Element) -> bool:
    text = node.attrib.get("text") or ""
    desc = node.attrib.get("content-desc") or ""
    rid = (node.attrib.get("resource-id") or "").rsplit("/", 1)[-1]
    blob = _norm(f"{text} {desc} {rid}").lower()
    if "새로고침" in text or "새로고침" in desc:
        return False
    return any(_norm(needle).lower() in blob for needle in AD_CLOSE_NEEDLES)


def find_ad_close(root: ET.Element) -> Bounds | None:
    hits: list[Bounds] = []
    for node in root.iter("node"):
        if not _looks_like_ad_close(node):
            continue
        try:
            bounds = clickable_bounds(root, node)
        except (KeyError, ValueError):
            continue
        if bounds.x2 - bounds.x1 < 8 or bounds.y2 - bounds.y1 < 8:
            continue
        hits.append(bounds)
    if not hits:
        return None
    return min(hits, key=lambda box: (box.y1, -box.x1))


def try_skip_ad(device: AndroidDevice, store: AdSkipStore) -> str | None:
    if store.sequences:
        taps = store.sequences[0]
        for x, y in taps:
            device.tap(x, y)
            time.sleep(0.22)
        return f"광고 닫기(기억한 좌표) {taps}"
    try:
        root = dump_ui(device, timeout=8.0)
        close = find_ad_close(root)
        if close is not None:
            device.tap(*close.center)
            return f"광고 닫기(화면글자) {close.center}"
    except Exception:
        pass
    return None


def dismiss_sancheon_notice(device: AndroidDevice) -> str | None:
    """KTX-산천 연결 운행 안내가 있으면 확인 또는 X를 누른다."""
    try:
        root = dump_ui(device, timeout=6.0)
    except Exception:
        return None
    notice = locate_label(root, SANCHON_NOTICE_LABELS)
    if notice is None:
        return None
    notice_box = notice[2]

    confirms: list[tuple[ET.Element, Bounds]] = []
    closes: list[tuple[ET.Element, Bounds]] = []
    for node in root.iter("node"):
        text = node.attrib.get("text") or ""
        desc = node.attrib.get("content-desc") or ""
        try:
            bounds = clickable_bounds(root, node)
        except (KeyError, ValueError):
            continue
        if text == "확인" or desc == "확인":
            if bounds.y1 >= notice_box.y1 - 80:
                confirms.append((node, bounds))
        elif text in ("닫기", "×", "✕") or desc in ("닫기", "close", "Close"):
            if abs(bounds.y1 - notice_box.y1) < 220 and bounds.x1 > 640:
                closes.append((node, bounds))

    if confirms:
        _node, bounds = max(confirms, key=lambda item: item[1].y1)
        device.tap(*bounds.center)
        time.sleep(0.35)
        return f"산천 안내 확인 {bounds.center}"
    if closes:
        _node, bounds = closes[0]
        device.tap(*bounds.center)
        time.sleep(0.35)
        return f"산천 안내 닫기 {bounds.center}"

    device.tap(*SANCHON_CONFIRM_FALLBACK)
    time.sleep(0.35)
    return f"산천 안내 확인 기본좌표 {SANCHON_CONFIRM_FALLBACK}"


def wait_and_learn_skip(
    device: AndroidDevice, monitor: TouchMonitor | None, store: AdSkipStore, timeout: float
) -> list[tuple[int, int]] | None:
    if monitor is None:
        return None
    monitor.drain()
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        time.sleep(0.3)
        _image, _rgb, card = wait_for_card(device, timeout=0.2)
        if card is None:
            continue
        taps = monitor.drain()
        if taps:
            store.add_sequence(taps, source="watch")
            return taps
        return None
    return None


def _pick_taps_on_screenshot(device: AndroidDevice, image: Image.Image) -> list[tuple[int, int]]:
    import tkinter as tk
    from PIL import ImageTk

    taps: list[tuple[int, int]] = []
    current = image
    root = tk.Tk()
    root.title("광고 닫기 위치 - 닫기 버튼을 클릭하세요")
    canvas = tk.Canvas(root, background="#111318", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)
    status = tk.Label(root, anchor="w", padx=8, pady=4)
    status.pack(fill=tk.X)
    buttons = tk.Frame(root)
    buttons.pack(fill=tk.X, pady=4)

    photo = None
    view: tuple[float, float, int, int, int, int] | None = None

    def redraw(frame: Image.Image) -> None:
        nonlocal photo, view, current
        current = frame
        dev_w, dev_h = frame.size
        max_h = 820
        scale = min(max_h / dev_h, 420 / dev_w)
        disp_w, disp_h = max(int(dev_w * scale), 1), max(int(dev_h * scale), 1)
        photo = ImageTk.PhotoImage(frame.resize((disp_w, disp_h), Image.BILINEAR))
        canvas.config(width=disp_w, height=disp_h)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        view = (0.0, 0.0, disp_w, disp_h, dev_w, dev_h)
        status.config(text=f"클릭 {len(taps)}회  |  광고 닫기 버튼을 누르세요. 끝나면 저장.")

    def to_device(event_x: float, event_y: float) -> tuple[int, int] | None:
        if view is None:
            return None
        _ox, _oy, disp_w, disp_h, dev_w, dev_h = view
        if not (0 <= event_x < disp_w and 0 <= event_y < disp_h):
            return None
        return int(event_x / disp_w * dev_w), int(event_y / disp_h * dev_h)

    def on_click(event) -> None:
        point = to_device(event.x, event.y)
        if point is None:
            return
        taps.append(point)
        device.tap(*point)
        status.config(text=f"탭 {point} 전송. 화면 갱신 중...")
        root.after(450, refresh)

    def refresh() -> None:
        frame = device.capture(png=True)
        rgb = np.array(frame)
        redraw(frame)
        if find_ktx333(rgb) is not None:
            status.config(text="열차 목록이 다시 보였습니다. 저장합니다.")
            root.after(400, root.destroy)

    def save() -> None:
        root.destroy()

    def undo() -> None:
        if taps:
            taps.pop()
            status.config(text=f"마지막 클릭을 취소했습니다. 남은 {len(taps)}회")

    tk.Button(buttons, text="저장", command=save).pack(side=tk.LEFT, padx=6)
    tk.Button(buttons, text="마지막 취소", command=undo).pack(side=tk.LEFT, padx=6)
    tk.Button(buttons, text="취소(저장 안 함)", command=lambda: (taps.clear(), root.destroy())).pack(
        side=tk.LEFT, padx=6
    )
    canvas.bind("<Button-1>", on_click)
    redraw(image)
    root.mainloop()
    return taps


def run_learn_ad(device: AndroidDevice) -> int:
    print("광고가 보일 때까지 기다립니다. 이미 광고면 바로 창이 열립니다.")
    print("창에서 닫기 버튼(X)을 클릭하세요. 여러 번 눌러야 하면 순서대로 클릭하면 됩니다.")
    while True:
        image = device.capture(png=True)
        if find_ktx333(np.array(image)) is None:
            break
        time.sleep(0.4)
    taps = _pick_taps_on_screenshot(device, image)
    if not taps:
        print("저장한 좌표가 없습니다.")
        return 1
    store = AdSkipStore.load()
    store.add_sequence(taps, source="learn-ad")
    return 0


def locate_label(
    root: ET.Element,
    labels: tuple[str, ...],
    *,
    exact: bool = False,
    pick: str = "first",
) -> tuple[str, ET.Element, Bounds] | None:
    found: list[tuple[str, ET.Element, Bounds]] = []
    for node in root.iter("node"):
        node_text = node.attrib.get("text") or ""
        node_desc = node.attrib.get("content-desc") or ""
        for label in labels:
            if exact:
                matched = node_text == label or node_desc == label
            else:
                matched = (
                    node_text == label
                    or node_desc == label
                    or _norm(label) in _norm(node_text)
                    or _norm(label) in _norm(node_desc)
                )
            if not matched:
                continue
            try:
                bounds = parse_bounds(node.attrib["bounds"])
            except ValueError:
                break
            if bounds.x2 - bounds.x1 < 4 or bounds.y2 - bounds.y1 < 4:
                break
            found.append((label, node, clickable_bounds(root, node)))
            break
    if not found:
        return None
    if pick == "bottom":
        return max(found, key=lambda item: item[2].y1)
    return found[0]


def is_checked(root: ET.Element, node: ET.Element) -> bool:
    parent_map = {child: parent for parent in root.iter("node") for child in parent}
    cur: ET.Element | None = node
    while cur is not None:
        if cur.attrib.get("checked") == "true":
            return True
        cur = parent_map.get(cur)
    return False


def wait_dump(device: AndroidDevice, timeout: float, predicate):
    deadline = time.perf_counter() + timeout
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            root = dump_ui(device)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.25)
            continue
        hit = predicate(root)
        if hit:
            return root, hit
        time.sleep(0.25)
    raise RuntimeError(last_error or "화면 요소를 기다리다 시간 초과")


def type_digits(device: AndroidDevice, text: str) -> None:
    device.run("shell", "input", "text", text, timeout=8)


def phone_fields(root: ET.Element) -> list[Bounds]:
    boxes: list[Bounds] = []
    for node in root.iter("node"):
        cls = node.attrib.get("class", "")
        hint = node.attrib.get("hint") or ""
        text = node.attrib.get("text") or ""
        focusable = node.attrib.get("focusable") == "true"
        if "EditText" in cls or "숫자" in hint or (focusable and re.fullmatch(r"\d{3,4}", text or "")):
            bounds = parse_bounds(node.attrib["bounds"])
            if 80 < (bounds.x2 - bounds.x1) < 500 and 50 < (bounds.y2 - bounds.y1) < 220:
                boxes.append(bounds)
    uniq: list[Bounds] = []
    for box in sorted(boxes, key=lambda b: (b.y1, b.x1)):
        if any(abs(box.x1 - u.x1) < 10 and abs(box.y1 - u.y1) < 10 for u in uniq):
            continue
        uniq.append(box)
    return uniq[:3]


def fill_waitlist_form(
    device: AndroidDevice, phone: tuple[str, str, str], shot_dir: Path
) -> None:
    """특실 포함 → 전화번호 → 개인정보 동의 → 신청."""
    root, hit = wait_dump(
        device, 8.0, lambda r: locate_label(r, ("특실 좌석 포함",))
    )
    print(tap_located(device, hit[1], hit[2], "특실 좌석 포함") if not is_checked(root, hit[1]) else "특실 좌석 포함: 이미 선택됨")
    time.sleep(0.35)

    root, hit = wait_dump(
        device, 5.0, lambda r: locate_label(r, ("좌석 배정 시 안내받을 휴대폰 번호 입력", "휴대폰 번호"))
    )
    if not is_checked(root, hit[1]):
        print(tap_located(device, hit[1], hit[2], "휴대폰 번호 안내"))
        time.sleep(0.5)
    else:
        print("휴대폰 번호 안내: 이미 선택됨")

    fields: list[Bounds] = []
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        root = dump_ui(device)
        fields = phone_fields(root)
        if len(fields) >= 3:
            break
        time.sleep(0.25)
    if len(fields) < 3:
        fields = [
            Bounds(90, 820, 350, 980),
            Bounds(400, 820, 680, 980),
            Bounds(730, 820, 990, 980),
        ]
        print("전화번호 입력칸을 못 찾아 기본 좌표를 사용합니다.")

    for part, box in zip(phone, fields):
        current = ""
        for node in dump_ui(device).iter("node"):
            b = parse_bounds(node.attrib["bounds"])
            if b.center == box.center or (
                abs(b.x1 - box.x1) < 8 and abs(b.y1 - box.y1) < 8
            ):
                current = node.attrib.get("text") or ""
                break
        if current.replace("-", "") == part:
            print(f"번호 {part}: 이미 입력됨")
            continue
        device.tap(*box.center)
        time.sleep(0.2)
        type_digits(device, part)
        print(f"번호 입력 {part} @ {box.center}")
        time.sleep(0.2)

    form = device.capture(png=True)
    form.save(shot_dir / f"waitlist-form-{time.strftime('%H%M%S')}.png")

    print(tap_label(device, ("개인정보 수집 및 이용 동의",), timeout=6.0))
    time.sleep(0.4)
    root, hit = wait_dump(
        device,
        6.0,
        lambda r: locate_label(r, ("동의",), exact=True, pick="bottom"),
    )
    print(tap_located(device, hit[1], hit[2], "동의"))
    time.sleep(0.5)
    consent = device.capture(png=True)
    consent.save(shot_dir / f"waitlist-consent-{time.strftime('%H%M%S')}.png")

    def enabled_apply(root: ET.Element):
        found = locate_label(root, ("신청",), exact=True, pick="bottom")
        if found is None:
            return None
        _label, node, bounds = found
        if node.attrib.get("enabled") != "true":
            return None
        return found

    root, hit = wait_dump(device, 8.0, enabled_apply)
    print(tap_located(device, hit[1], hit[2], "신청"))
    time.sleep(0.45)
    done = device.capture(png=True)
    done.save(shot_dir / f"waitlist-apply-{time.strftime('%H%M%S')}.png")


def tap_located(device: AndroidDevice, node: ET.Element, bounds: Bounds, label: str) -> str:
    device.tap(*bounds.center)
    return f"{label} 탭 {bounds.center} enabled={node.attrib.get('enabled')}"


def has_pin_prompt(root: ET.Element) -> bool:
    return locate_label(root, PIN_LABELS) is not None


def pay_until_password(device: AndroidDevice, shot_dir: Path) -> str:
    """등록 카드 선택·결제하기까지 진행하고, 비밀번호 칸에서 멈춘다."""
    deadline = time.perf_counter() + 28.0
    tapped_card = False
    pay_taps = 0
    while time.perf_counter() < deadline:
        try:
            root = dump_ui(device)
        except Exception:
            time.sleep(0.25)
            continue

        if has_pin_prompt(root):
            shot = device.capture(png=True)
            shot.save(shot_dir / f"pay-pin-{time.strftime('%H%M%S')}.png")
            print("결제 비밀번호 화면입니다. 직접 입력하세요.")
            return "pin"

        agree = locate_label(root, PAY_AGREE_LABELS)
        if agree is not None and not is_checked(root, agree[1]):
            print(tap_located(device, agree[1], agree[2], agree[0]))
            time.sleep(0.35)
            continue

        card = locate_label(root, CARD_LABELS)
        if card is not None and not tapped_card:
            print(tap_located(device, card[1], card[2], card[0]))
            tapped_card = True
            time.sleep(0.45)
            continue

        pay = locate_label(root, PAY_LABELS, pick="bottom")
        if pay is not None and pay[1].attrib.get("enabled", "true") != "false":
            pay_taps += 1
            print(tap_located(device, pay[1], pay[2], pay[0]))
            time.sleep(0.8)
            if pay_taps >= 3:
                break
            continue

        time.sleep(0.3)

    shot = device.capture(png=True)
    shot.save(shot_dir / f"pay-stop-{time.strftime('%H%M%S')}.png")
    print("결제 비밀번호 화면을 못 찾았습니다. 이어서 직접 결제하세요.")
    return "timeout"


def wait_for_sheet(
    device: AndroidDevice, timeout: float = 6.0
) -> tuple[str, ET.Element, Bounds, str] | None:
    """하단 시트에서 예약 대기 신청 또는 바로 예매를 찾는다."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        try:
            root = dump_ui(device)
        except Exception:
            time.sleep(0.2)
            continue
        waitlist = locate_label(root, WAITLIST_LABELS)
        if waitlist is not None:
            label, node, bounds = waitlist
            return "waitlist", node, bounds, label
        book = locate_label(root, BOOK_LABELS)
        if book is not None:
            label, node, bounds = book
            return "book", node, bounds, label
        time.sleep(0.2)
    return None


def tap_label(
    device: AndroidDevice,
    labels: tuple[str, ...],
    *,
    fallback: tuple[int, int] | None = None,
    timeout: float = 8.0,
) -> str:
    """텍스트/설명이 일치하는 버튼을 찾아 누른다. 화면이 바뀔 때까지 잠시 기다린다."""
    deadline = time.perf_counter() + timeout
    last_error = ""
    while time.perf_counter() < deadline:
        try:
            root = dump_ui(device)
        except Exception as exc:
            last_error = str(exc)
            time.sleep(0.2)
            continue
        found = locate_label(root, labels)
        if found is None:
            time.sleep(0.2)
            continue
        label, node, bounds = found
        return tap_located(device, node, bounds, label)

    if fallback is not None:
        device.tap(*fallback)
        return f"{labels[0]} 없음({last_error}) → 기본 좌표 {fallback}"
    return f"{labels[0]} 버튼을 찾지 못했습니다. {last_error}".strip()


def run_check(device: AndroidDevice) -> int:
    image, _rgb, card = wait_for_card(device, timeout=2.0)
    out = Path(__file__).parent / "screenshots"
    out.mkdir(exist_ok=True)
    if card is None:
        image.save(out / "check_failed.png")
        print("[오류] KTX 333 카드를 찾지 못했습니다. 열차 목록 화면인지 확인하세요.")
        return 1
    overlay = out / "check_overlay.png"
    save_overlay(image, card, overlay)
    print(f"KTX 333 뱃지: {card.badge}")
    for seat in (card.standard, card.special):
        print(
            f"{seat.name}: {seat_label(seat)}  "
            f"(빨강 {seat.red_pixels}, 초록 {seat.green_pixels}, 탭 {seat.tap})"
        )
    print(f"표시: {overlay}")
    return 0


def run_watch(
    device: AndroidDevice, dry_run: bool, interval: float, phone: tuple[str, str, str]
) -> int:
    shot_dir = Path(__file__).parent / "screenshots"
    shot_dir.mkdir(exist_ok=True)
    store = AdSkipStore.load()
    monitor = TouchMonitor(device)
    monitor_ok = monitor.start()
    if store.sequences:
        print(f"기억한 광고 닫기 좌표 {len(store.sequences)}개: {store.sequences[0]}")

    refresh = REFRESH_FALLBACK
    try:
        root = dump_ui(device)
        found = find_node(root, desc="새로고침")
        if found:
            refresh = clickable_bounds(root, found[0]).center
            print(f"새로고침 버튼: {refresh}")
        else:
            print(f"새로고침 버튼을 못 찾아 기본 좌표 {refresh} 사용")
    except Exception as exc:
        print(f"초기 화면 구조 읽기 실패({exc}). 기본 좌표로 새로고침합니다.")

    print("KTX 333 감시 시작. 중지하려면 Ctrl+C")
    cycle = 0
    try:
        while True:
            cycle += 1
            device.tap(*refresh)
            time.sleep(0.55)
            image, _rgb, card = wait_for_card(device, timeout=3.5)
            both_unknown = (
                card is not None
                and card.standard.kind == "unknown"
                and card.special.kind == "unknown"
            )
            if card is None or both_unknown:
                dismissed = dismiss_sancheon_notice(device)
                if dismissed:
                    print(f"[{cycle}] {dismissed}")
                    image, _rgb, card = wait_for_card(device, timeout=2.0)
            if card is None:
                why = try_skip_ad(device, store)
                if why:
                    print(f"[{cycle}] {why}")
                    image, _rgb, card = wait_for_card(device, timeout=2.0)
                if card is None:
                    if monitor_ok and len(store.sequences) < 5:
                        print(f"[{cycle}] 목록 없음. 광고면 폰에서 닫으세요. 그 터치를 기억합니다.")
                        learned = wait_and_learn_skip(device, monitor, store, timeout=6.0)
                        if learned:
                            print(f"[{cycle}] 직접 닫은 좌표를 다음에 씁니다.")
                            continue
                    print(f"[{cycle}] 목록을 아직 못 찾음. 다시 새로고침")
                    continue

            target = pick_target(card)
            std = seat_label(card.standard)
            spc = seat_label(card.special)
            print(f"[{cycle}] 일반실={std}  특실={spc}")

            if target is None:
                time.sleep(interval)
                continue

            print(f"[{cycle}] 감지: {target.name} {target.kind}")
            save_overlay(image, card, shot_dir / f"{target.kind}-{time.strftime('%H%M%S')}.png")
            alert_found(device, target.kind)

            if dry_run:
                print("dry-run: 클릭하지 않고 종료합니다.")
                return 0

            print(f"{target.name} 탭 {target.tap}")
            device.tap(*target.tap)
            time.sleep(0.35)
            dismissed = dismiss_sancheon_notice(device)
            if dismissed:
                print(f"[{cycle}] {dismissed} — 산천 안내라서 예매를 진행하지 않습니다.")
                continue
            after = device.capture(png=True)
            after.save(shot_dir / f"after-seat-{time.strftime('%H%M%S')}.png")

            sheet = wait_for_sheet(device, timeout=6.0)
            action = sheet[0] if sheet is not None else target.kind
            if action == "waitlist" or (sheet is None and target.kind == "waitlist"):
                if not phone[0]:
                    print("[오류] 예약 대기를 찾았지만 --phone 이 없어 신청하지 않습니다.")
                    beep_pc()
                    return 2
                if sheet is not None and sheet[0] == "waitlist":
                    _kind, node, bounds, label = sheet
                    print(tap_located(device, node, bounds, label))
                else:
                    print(tap_label(device, WAITLIST_LABELS, fallback=WAITLIST_FALLBACK, timeout=4.0))
                time.sleep(0.5)
                try:
                    fill_waitlist_form(device, phone, shot_dir)
                    print("예약 대기 신청서 제출까지 전달했습니다.")
                except Exception as exc:
                    err = device.capture(png=True)
                    err.save(shot_dir / f"waitlist-error-{time.strftime('%H%M%S')}.png")
                    print(f"예약 대기 신청서 작성 실패: {exc}")
                    raise
                beep_pc()
                return 0

            if sheet is not None and sheet[0] == "book":
                _kind, node, bounds, label = sheet
                print(tap_located(device, node, bounds, label))
            else:
                print(tap_label(device, BOOK_LABELS, fallback=BOOK_FALLBACK, timeout=4.0))
            time.sleep(0.4)
            booked = device.capture(png=True)
            booked.save(shot_dir / f"after-book-{time.strftime('%H%M%S')}.png")

            cart = tap_label(device, CART_LABELS, timeout=10.0)
            print(cart)
            time.sleep(0.35)
            final = device.capture(png=True)
            final.save(shot_dir / f"after-cart-{time.strftime('%H%M%S')}.png")
            print("바로 예매 → 장바구니 담기. 등록 카드 결제를 이어서 진행합니다.")
            pay_until_password(device, shot_dir)
            beep_pc()
            return 0
    except KeyboardInterrupt:
        print("\n중지했습니다.")
        return 130
    finally:
        monitor.stop()


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="KTX 333 매진 해제 감시")
    parser.add_argument("--check", action="store_true", help="현재 화면에서 333 위치만 확인")
    parser.add_argument("--dry-run", action="store_true", help="발견해도 클릭하지 않음")
    parser.add_argument("--interval", type=float, default=0.15, help="새로고침 사이 대기(초)")
    parser.add_argument("--serial", help="adb 기기 시리얼")
    parser.add_argument("--learn-ad", action="store_true", help="광고 닫기 버튼을 화면에서 찍어 저장")
    parser.add_argument(
        "--ad-tap",
        action="append",
        default=[],
        metavar="X,Y",
        help="광고 닫기 좌표를 직접 추가 (여러 번 주면 순서대로 누름)",
    )
    parser.add_argument(
        "--phone",
        default=os.environ.get("KTX_WAITLIST_PHONE", ""),
        help="예약 대기 안내 번호 11자리 (또는 환경변수 KTX_WAITLIST_PHONE)",
    )
    args = parser.parse_args(argv)

    digits = re.sub(r"\D", "", args.phone)
    if len(digits) != 11:
        phone = ("", "", "")
        if not args.check and not args.learn_ad and not args.ad_tap:
            print(
                "[경고] --phone 이 없습니다. 매진 감시와 바로 예매는 진행하고, "
                "예약 대기는 신청하지 않습니다.",
                file=sys.stderr,
            )
    else:
        phone = (digits[:3], digits[3:7], digits[7:])

    adb = find_adb()
    serial = pick_serial(adb, args.serial)
    device = AndroidDevice(adb, serial)
    try:
        if args.ad_tap:
            taps: list[tuple[int, int]] = []
            for raw in args.ad_tap:
                parts = re.split(r"[,\sx]+", raw.strip())
                if len(parts) != 2:
                    print(f"[오류] --ad-tap 형식은 X,Y 입니다: {raw}", file=sys.stderr)
                    return 2
                taps.append((int(parts[0]), int(parts[1])))
            store = AdSkipStore.load()
            store.add_sequence(taps, source="ad-tap")
            if not args.learn_ad:
                return 0
        if args.learn_ad:
            return run_learn_ad(device)
        if args.check:
            return run_check(device)
        return run_watch(
            device, dry_run=args.dry_run, interval=max(args.interval, 0.0), phone=phone
        )
    finally:
        device.close()


if __name__ == "__main__":
    sys.exit(main())
