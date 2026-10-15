"""YOLO-seg: nạp model GPU, ảnh nắn thẳng tờ giấy, lấy hình xe từ mask và lọc phát hiện không giống xe."""
import sys

import cv2
import numpy as np

from core.lot import PRINT_MM
from helpers.find_spaces import S


REC_PX_MM = 3.0                 # độ phân giải ảnh "nắn thẳng" đưa vào YOLO (px/mm), ~1320x940 px
REC_MARGIN_MM = 8.0             # lấy rộng ra ngoài vùng in một chút


def rect_map(H):
    """Homography khung hình -> ảnh nắn thẳng (tờ giấy nằm ngang, mọi xe trong dải đỗ đứng dọc như
    lúc chụp dataset) và kích thước ảnh nắn. Gốc ảnh nắn = góc trên-trái vùng in trừ lề."""
    lo = PRINT_MM.min(0) - REC_MARGIN_MM
    hi = PRINT_MM.max(0) + REC_MARGIN_MM
    A = np.array([[REC_PX_MM / S, 0, -lo[0] * REC_PX_MM],
                  [0, REC_PX_MM / S, -lo[1] * REC_PX_MM], [0, 0, 1]])
    Hr = A @ H
    return Hr, np.linalg.inv(Hr), tuple(int(v) for v in np.ceil((hi - lo) * REC_PX_MM)), lo

def car_shape(seg, box):
    """Hình xe trên ảnh nắn: (viền mask để vẽ, 4 góc box xoay để tính tâm và chỗ trống).
    Box xoay = minAreaRect của mask; khi mask hỏng (không có, quá méo, hoặc lem đầy box thẳng như lúc
    2 xe đỗ sát nhau) thì dùng box thẳng, vốn đã song song với dải đỗ trên ảnh nắn."""
    x1, y1, x2, y2 = box
    quad = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if seg is None or len(seg) < 5:
        return quad, quad
    hull = cv2.convexHull(np.float32(seg)).reshape(-1, 2)
    rc = cv2.minAreaRect(hull)
    rw, rh = rc[1]
    asp = max(rw, rh) / max(1e-3, min(rw, rh))
    fill = rw * rh / max(1.0, (x2 - x1) * (y2 - y1))
    straight = abs(((rc[2] % 90) + 45) % 90 - 45) < 5
    if not 1.6 <= asp <= 3.2 or (fill > 0.9 and straight):
        return hull, quad
    return hull, np.float32(cv2.boxPoints(rc))


CAR_SHORT_MM = (18.0, 75.0)     # cạnh ngắn hợp lệ của box xe (mm); xe thật ~28-37, box thẳng khi xe xoay 45° ~70
CAR_LONG_MM = (45.0, 115.0)     # cạnh dài hợp lệ (mm); xe thật ~68


def car_like(q):
    """Loại phát hiện không giống xe: mảnh tay/bóng mảnh dài, khối tay + xe dính nhau quá to,
    hoặc tâm nằm ngoài tờ giấy (tay thò vào từ mép)."""
    a, b = np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[1]))
    short, long_ = min(a, b), max(a, b)
    c = q.mean(0)
    lo, hi = PRINT_MM.min(0), PRINT_MM.max(0)
    return (CAR_SHORT_MM[0] <= short <= CAR_SHORT_MM[1] and CAR_LONG_MM[0] <= long_ <= CAR_LONG_MM[1]
            and lo[0] <= c[0] <= hi[0] and lo[1] <= c[1] <= hi[1])


# Hình mask của xe thật, đo đúng như lúc chạy (YOLO trên ảnh nắn 3 px/mm, imgsz 640) trên 116 xe ở
# 12 ảnh TEST/dataset, 2026-10-15:
#   độ đầy hình chữ nhật (diện tích mask / hình chữ nhật xoay bao ngoài): 0.85 .. 0.94 (hình elip chỉ ~0.78)
#   độ lồi (diện tích mask / bao lồi): 0.94 .. 0.99      tỉ lệ dài / ngắn: 1.75 .. 2.31
#   cạnh ngắn 30 .. 39 mm, cạnh dài 66 .. 73 mm           độ tin cậy: 0.91 .. 0.96
# Ngưỡng dưới đây chừa rộng cho ảnh video mờ hơn; bóng tay / bóng người (méo, dài, lõm, tròn) rớt ít nhất một ngưỡng.
SHAPE_FILL_MIN = 0.80
SHAPE_SOLID_MIN = 0.92
SHAPE_ASPECT = (1.5, 2.8)
SHAPE_SHORT_MM = (24.0, 46.0)   # cạnh ngắn của hình chữ nhật xoay quanh mask
SHAPE_LONG_MM = (55.0, 90.0)    # cạnh dài
SHAPE_CONF_MIN = 0.75           # độ tin cậy tối thiểu để được cấp ID mới


def car_polygon_ok(seg, conf):
    """Mask (toạ độ ảnh nắn, REC_PX_MM px/mm) có hình giống một chiếc xe không: gần chữ nhật, lồi,
    tỉ lệ và kích thước đúng như xe mô hình. Chỉ phát hiện đạt điều kiện này mới được cấp ID mới."""
    if seg is None or len(seg) < 5 or conf < SHAPE_CONF_MIN:
        return False
    s = np.float32(seg)
    area = cv2.contourArea(s)
    hull_area = cv2.contourArea(cv2.convexHull(s))
    (_, _), (w, h), _ = cv2.minAreaRect(s)
    short, long_ = min(w, h) / REC_PX_MM, max(w, h) / REC_PX_MM
    if short <= 0 or hull_area <= 0:
        return False
    return (area / (w * h) >= SHAPE_FILL_MIN and area / hull_area >= SHAPE_SOLID_MIN
            and SHAPE_ASPECT[0] <= long_ / short <= SHAPE_ASPECT[1]
            and SHAPE_SHORT_MM[0] <= short <= SHAPE_SHORT_MM[1] and SHAPE_LONG_MM[0] <= long_ <= SHAPE_LONG_MM[1])


def load_model(a):
    """YOLO trên GPU NVIDIA (torch CUDA), FP16. Không có CUDA thì dừng và in cách cài."""
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        sys.exit(f"Không thấy GPU CUDA (torch {torch.__version__}). Cài torch bản CUDA:\n"
                 "  uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    model = YOLO(a.model)
    return model, dict(imgsz=a.imgsz, device=0, quantize=16), torch.cuda.get_device_name(0)
