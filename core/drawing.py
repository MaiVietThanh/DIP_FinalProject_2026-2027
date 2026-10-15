"""Vẽ kết quả lên khung hình: viền tờ giấy, dải đỗ, ô cam, ô chỗ trống, xe kèm nhãn bên trong."""
import cv2
import numpy as np

from core.config import BLUE, CYAN, GREEN, MAGENTA, RED, WHITE, YELLOW
from core.lot import GATES_MM, gate_at
from helpers.detect_lot import ROWS_MM
from helpers.find_spaces import S


_static_cache = {}


def _static_overlay(H):
    """Hàm đổi mm -> pixel khung hình và các khung cố định (dải đỗ, ô cam): chỉ tính lại khi khung bãi đổi."""
    key = id(H)
    if key not in _static_cache:
        _static_cache.clear()
        Hinv = np.linalg.inv(H)

        def img(pts_mm):
            return cv2.perspectiveTransform(np.float32(pts_mm).reshape(-1, 1, 2) * S, Hinv).reshape(-1, 2)

        rect = lambda x0, y0, x1, y1: img([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]).astype(np.int32)
        rows = [rect(*r) for r in ROWS_MM]
        gates = [(rect(*r), name) for name, r in GATES_MM.items()]
        _static_cache[key] = (img, rows, gates)
    return _static_cache[key]


def label_inside(out, poly, lines, outline, lw, dim=0.55):
    """Viền poly bằng màu outline, làm tối nhẹ bên trong rồi ghi các dòng chữ ở giữa hình."""
    poly = np.int32(poly).reshape(-1, 2)
    Hh, Ww = out.shape[:2]
    x, y, w, h = cv2.boundingRect(poly)
    x0, y0, x1, y1 = max(0, x), max(0, y), min(Ww, x + w), min(Hh, y + h)
    if x1 > x0 and y1 > y0:
        roi = out[y0:y1, x0:x1]
        m = np.zeros(roi.shape[:2], np.uint8)
        cv2.fillPoly(m, [poly - (x0, y0)], 1)
        sel = m.astype(bool)
        roi[sel] = (roi[sel] * dim).astype(np.uint8)
    cv2.polylines(out, [poly.reshape(-1, 1, 2)], True, outline, lw)
    if not lines:
        return
    fs, th = lw / 4.5, max(1, lw - 1)
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in lines]
    lh = max(s[1] for s in sizes) + 8
    cx, cy = poly.mean(0)
    ty = cy - lh * len(lines) / 2 + lh - 4
    for t, (tw, _) in zip(lines, sizes):
        cv2.putText(out, t, (int(cx - tw / 2), int(ty)), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, th, cv2.LINE_AA)
        ty += lh


def draw_overlay(out, snap, info, lw):
    """Vẽ kết quả detect gần nhất (snap) lên khung hình out (sửa trực tiếp out).
    info=None: không ghi dòng thông số lên video (giao diện mới hiện thông số ở hai cột bên)."""
    H = snap["H"] if snap else None
    spaces = snap["spaces"] if snap and H is not None else []
    if H is not None:
        img, rows, gates = _static_overlay(H)
        cv2.polylines(out, [snap["q"].astype(np.int32)], True, WHITE, lw)
        cv2.polylines(out, rows, True, YELLOW, lw)
        busy = {gate_at(c["hull"].mean(0), margin=0) for c in snap["cars"]}
        for g, name in gates:                    # tên ô cam chỉ ghi khi không có xe trong ô
            label_inside(out, g, [] if name in busy else [name], MAGENTA, lw, dim=0.8)
        for k, sp in enumerate(spaces, 1):       # ô xanh phủ hết đoạn trống: số thứ tự, độ dài khe, số xe đặt được
            x1, y1, x2, y2 = sp["slot_mm"]
            n = sp["capacity"]
            car_s = f"{n} car" + ("s" if n > 1 else "")    # cv2.putText không viết được dấu tiếng Việt
            text = [f"#{k}  {sp['gap_mm']:.0f}mm", f"fits {car_s}"] if sp["gap_mm"] >= 90 else \
                [f"#{k}", f"{sp['gap_mm']:.0f}mm", car_s]
            label_inside(out, img([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), text, GREEN, lw)
        for car in snap["cars"]:
            parked = car["state"] == "da_do"
            new = car["origin"].startswith("cong")
            col = BLUE if parked else (CYAN if new else RED)
            text = [f"ID {car['id']}", "DO" if parked else ("MOI" if new else "")]
            if new and not parked:
                text.append(car["origin"])
            if not car["seen"]:
                text.append("bi che")
            if len(car["trail"]) > 1:
                cv2.polylines(out, [img(car["trail"]).astype(np.int32).reshape(-1, 1, 2)], False, col, max(1, lw - 1))
            label_inside(out, img(car["hull"]), [t for t in text if t], col, lw)
    if info is None:
        return out
    n_do = sum(c["state"] == "da_do" for c in snap["cars"]) if snap else 0
    n_mv = len(snap["cars"]) - n_do if snap else 0
    text = info + [f"Dang do: {n_do} | dang di chuyen: {n_mv} | da vao: {snap['n_in'] if snap else 0}"
                   f" | da roi: {snap['n_out'] if snap else 0} | cho trong: {len(spaces)}"]
    for i, t in enumerate(text):
        cv2.putText(out, t, (20, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(out, t, (20, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    return out
