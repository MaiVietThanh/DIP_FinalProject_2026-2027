"""Nhật ký sự kiện xe: mỗi lần chạy ghi 1 file CSV trong tracking_result/, in ra màn hình,
và giữ các dòng gần nhất (câu tiếng Việt có dấu) để hiện trong khung log của giao diện."""
import csv
import os
import threading
import time
from collections import deque

from core.lot import in_row

GATE_VI = {"cong trai": "cổng trái", "cong phai": "cổng phải"}


def describe(car_id, ev, gate, day, x, y, note):
    """Câu mô tả sự kiện cho người xem (viết hoa đầu câu, có dấu)."""
    xe = f"Xe #{car_id}"
    pos = f"(x = {x:.0f} mm, y = {y:.0f} mm)"
    day_vi = {"tren": "dãy trên", "duoi": "dãy dưới"}.get(day, "")
    if ev == "co_san":
        return f"{xe} đã có sẵn trong bãi" + (f" ở {day_vi}" if day_vi else "") + f" {pos}"
    if ev == "vao":
        return f"{xe} vào bãi qua {GATE_VI.get(gate, gate)}"
    if ev == "xuat_hien":
        return f"{xe} xuất hiện giữa bãi, không đi qua ô cam {pos}"
    if ev == "do":
        return f"{xe} đã đỗ ở {day_vi or 'bãi'} {pos}"
    if ev == "roi_cho":
        why = note.replace("xe đang đỗ ", "")
        return f"{xe} rời chỗ đỗ ({why})"
    if ev == "roi_bai":
        if gate:
            return f"{xe} rời bãi qua {GATE_VI.get(gate, gate)}"
        if "đang đỗ" in note:
            return f"{xe} bị mất dấu khi đang đỗ, coi như đã rời bãi"
        return f"{xe} rời bãi"
    return f"{xe}: {note[:1].upper() + note[1:]}"


class EventLog:
    """Gọi như hàm: log(xe, su_kien, cong, ghi_chu). `frame_no` là hàm trả về số khung YOLO hiện tại.
    note(text, kind) ghi thêm thông báo hệ thống (chỉ hiện trên giao diện, không ghi CSV).
    entries() trả về bản sao các dòng gần nhất: dict {time, kind, text}."""

    def __init__(self, out_dir, frame_no=lambda: 0, keep=500):
        self.path = os.path.join(out_dir, f"{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8-sig")
        self.w = csv.writer(self.file)
        self.w.writerow(["thoi_gian", "khung_hinh", "xe_id", "su_kien", "cong", "day", "x_mm", "y_mm", "ghi_chu"])
        self.file.flush()
        self.frame_no = frame_no
        self.recent = deque(maxlen=keep)
        self.total = 0                         # tổng số dòng đã thêm (để giao diện biết có dòng mới)
        self.lock = threading.Lock()

    def _add(self, kind, text):
        with self.lock:
            self.recent.append({"time": time.strftime("%H:%M:%S"), "kind": kind, "text": text})
            self.total += 1

    def __call__(self, car, ev, gate, note):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        row = in_row(car.c)
        day = "" if row is None else ("tren" if row == 0 else "duoi")
        x, y = float(car.c[0]), float(car.c[1])
        self.w.writerow([now, self.frame_no(), car.id, ev, gate, day, round(x, 1), round(y, 1), note])
        self.file.flush()
        print(f"[{now}] ID {car.id}: {ev}" + (f" ({gate})" if gate else "") + (f" day {day}" if day else "")
              + f" x={x:.0f} y={y:.0f} mm - {note}")
        self._add(ev, describe(car.id, ev, gate, day, x, y, note))

    def note(self, text, kind="he_thong"):
        self._add(kind, text)

    def entries(self):
        with self.lock:
            return list(self.recent), self.total

    def close(self):
        self.file.close()
