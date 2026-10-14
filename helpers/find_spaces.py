"""Giai đoạn 4: tìm chỗ trống trong bãi xe trên ẢNH TĨNH.

Chạy:   python find_spaces.py --image duong_dan_anh.jpg
        python find_spaces.py --image thu_muc_anh/            (chạy cả thư mục)
Tùy chọn: --model weights/best.pt  --out ket_qua_cho_trong  --pad 5  --conf 0.5
Cần: helpers/detect_lot.py, helpers/measure_gaps.py (lấy hàm nắn ảnh), weights/best.pt

Các bước (đúng theo đề cương giai đoạn 4):
  1. YOLO -> danh sách bounding box (x1, y1, x2, y2) của các xe.
  2. Tạo ảnh đen cùng kích thước khung bãi, tô trắng vùng box xe, nới rộng thêm --pad mm.
     Làm trên mặt phẳng bãi đã nắn về góc nhìn từ trên xuống (1 px = 1/S mm), để mọi
     pixel đều cùng kích thước thật và chỉ xét bên trong 2 dải đỗ (không tính lối đi).
  3. Đảo mask -> vùng trống; opening (erosion rồi dilation) để bỏ nhiễu, khe hẹp.
  4. Connected components -> các vùng trống liên thông.
  5. Kích thước xe chuẩn = trung vị kích thước box các xe. Vùng hợp lệ nếu:
       - bề ngang và chiều cao bbox của vùng >= kích thước xe chuẩn, và
       - erosion vùng trống bằng phần tử cấu trúc hình chữ nhật có kích thước xe
         còn sót lại ít nhất 1 điểm (nghĩa là đặt VỪA trọn 1 xe vào được, kể cả khi
         vùng có hình méo do xe bên cạnh đỗ xéo). Mỗi cụm điểm còn lại = 1 chỗ trống.
     Vì vùng xe đã nới rộng --pad mm, khe giữa 2 xe cần >= xe chuẩn + 2*pad
     (box xe đo được rộng khoảng 37 mm -> khe cần khoảng 47 mm).
  6. Vẽ: xe = khung đỏ, vùng trống hợp lệ = viền xanh + 1 ô gợi ý đặt xe (khung xanh).
     Nhãn "#k Xmm xN": X = bề rộng khe thật (chỗ hẹp nhất), N = số xe đặt vừa.
Kết quả: <tên>_cho_trong.jpg (ảnh gốc) và <tên>_tren_xuong.jpg (ảnh đã nắn).
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # thư mục chứa main.py, helpers/, weights/
if ROOT not in sys.path:   # để chạy thẳng file này (python helpers/xxx.py) vẫn import được helpers.*
    sys.path.insert(0, ROOT)

from helpers.detect_lot import ROWS_MM, LOT_H, LOT_W
from helpers.measure_gaps import lot_homography

S = 8                       # phải trùng với S trong measure_gaps.py (px / mm trên mặt phẳng bãi)
OPEN_MM = 3                 # kích thước kernel opening để bỏ nhiễu (mm)
CAR_DEFAULT_MM = (37.0, 69.0)  # trung vị box xe trên 20 ảnh TEST, dùng khi ảnh không có xe nào
RED, GREEN, YELLOW = (0, 0, 255), (0, 200, 0), (0, 220, 255)


# ------------------------------------------------------------------ bước 1
def detect_cars(model, img, conf=0.5):
    """Trả về (boxes, masks): boxes = list (x1, y1, x2, y2) trên ảnh gốc,
    masks = mask từng xe (chỉ dùng để xóa vùng xe khi tìm khung bãi)."""
    r = model.predict(img, imgsz=1024, conf=conf, retina_masks=True, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], []
    boxes = r.boxes.xyxy.cpu().numpy().tolist()
    masks = []
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():
            m = (m > 0.5).astype(np.uint8)
            if m.shape != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            masks.append(m)
    return boxes, masks


def box_to_lot(box, H):
    """Box (x1, y1, x2, y2) hoặc đa giác thân xe (N x 2) trên ảnh gốc -> đa giác trên mặt phẳng bãi (px)."""
    if np.ndim(box) == 2:
        pts = np.float32(box).reshape(-1, 1, 2)
    else:
        x1, y1, x2, y2 = box
        pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


# ------------------------------------------------------------------ bước 2-6
def find_spaces(boxes, H, pad_mm=5, car_size_mm=None, s=S):
    """Lõi xử lý ảnh. Trả về dict kết quả, mọi độ dài theo mm.

    boxes: list (x1, y1, x2, y2) hoặc đa giác thân xe (N x 2) trên ảnh gốc; H: homography ảnh -> mặt phẳng bãi (S px/mm).
    s: độ phân giải mặt phẳng bãi dùng để tính (px/mm). Mặc định S; video trực tiếp dùng s nhỏ
    hơn (vd 4) cho nhanh. Khi s != S thì H được đổi tỉ lệ, slot_px và region tính theo s px/mm.
    """
    if s != S:
        H = np.diag([s / S, s / S, 1.0]) @ H
    size = (int(LOT_W * s), int(LOT_H * s))          # (rộng, cao) của mặt phẳng bãi
    rows = np.zeros(size[::-1], np.uint8)            # vùng 2 dải đỗ
    for x0, y0, x1, y1 in ROWS_MM:
        cv2.rectangle(rows, (int(x0 * s), int(y0 * s)), (int(x1 * s) - 1, int(y1 * s) - 1), 255, -1)

    # --- bước 2: ảnh đen, tô trắng vùng xe (nới rộng pad mm)
    occ = np.zeros_like(rows)
    cars, dims = [], []
    for b in boxes:
        poly = box_to_lot(b, H)
        cx, cy = poly.mean(0) / s
        in_row = any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in ROWS_MM)
        cars.append((poly, in_row))
        cv2.fillPoly(occ, [poly.astype(np.int32)], 255)
        if in_row:
            w, h = np.ptp(poly[:, 0]) / s, np.ptp(poly[:, 1]) / s
            dims.append((min(w, h), max(w, h)))      # (bề ngang, chiều dài) xe
    raw_free = cv2.bitwise_and(rows, cv2.bitwise_not(occ))   # vùng trống thật (chưa nới) để đo khe
    if pad_mm > 0:
        k = 2 * int(pad_mm * s) + 1
        occ = cv2.dilate(occ, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # kích thước xe chuẩn = trung vị (bước 5); xe đứng dọc trong dải -> ngang = cạnh ngắn
    if car_size_mm is None:
        car_size_mm = tuple(np.median(dims, axis=0)) if dims else CAR_DEFAULT_MM
    car_w, car_h = car_size_mm

    # --- bước 3: đảo mask (chỉ trong 2 dải) + opening bỏ nhiễu
    free = cv2.bitwise_and(rows, cv2.bitwise_not(occ))
    k = max(3, int(OPEN_MM * s) | 1)
    free = cv2.morphologyEx(free, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))

    # --- bước 4: connected components
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)

    # --- bước 5: lọc vùng đủ cho 1 xe
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (int(car_w * s) | 1, int(car_h * s) | 1))
    regions, spaces = [], []
    for i in range(1, n):
        x, y, w, h, area = st[i]
        region = (lab == i).astype(np.uint8) * 255
        big_enough = w / s >= car_w and h / s >= car_h          # so bbox với xe chuẩn
        regions.append({"bbox_mm": (x / s, y / s, w / s, h / s), "big_enough": big_enough})
        if not big_enough:
            continue
        # erosion bằng hình chữ nhật có kích thước xe: pixel còn lại = vị trí TÂM xe đặt vừa.
        # Một vùng trống có thể nối 2 khe qua dải hẹp sau đuôi xe, nên tách tiếp từng cụm tâm.
        centers = cv2.erode(region, se)
        m, lab2, st2, _ = cv2.connectedComponentsWithStats(centers, connectivity=8)
        for j in range(1, m):
            ys, xs = np.nonzero(lab2 == j)
            hw, hh = int(car_w * s) // 2, int(car_h * s) // 2
            # bề ngang khe = vùng tâm + 1 bề ngang xe; số xe = 1 + phần dư / (xe + đệm)
            width = (xs.max() - xs.min()) / s + car_w
            cx = xs.min()                                     # ô gợi ý: sát bên trái nhất
            cy = int(np.median(ys[xs == cx]))
            # đo khe thật (chưa trừ đệm) trên dải cao bằng 1 xe quanh ô gợi ý:
            # cột "trống" khi cả dải đều trống -> lấy đoạn cột trống liên tục chứa ô gợi ý
            band = raw_free[max(0, cy - hh):cy + hh + 1] > 0
            col = band.all(axis=0)
            l = r_ = cx
            while l > 0 and col[l - 1]:
                l -= 1
            while r_ < col.size - 1 and col[r_ + 1]:
                r_ += 1
            spaces.append({
                "row": "trên" if cy / s < ROWS_MM[1][1] else "dưới",
                "x_mm": (xs.min() / s - car_w / 2, xs.max() / s + car_w / 2),
                "width_mm": width,
                "gap_mm": (r_ - l + 1) / s,
                "capacity": 1 + int((width - car_w) // (car_w + pad_mm)),
                "slot_px": (cx - hw, cy - hh, cx + hw, cy + hh),
                # ô chữ nhật phủ hết đoạn trống thật (dài = gap_mm, cao = 1 xe) để vẽ
                "gap_px": (l, max(0, cy - hh), r_ + 1, min(raw_free.shape[0], cy + hh + 1)),
                "region": region,
                "rid": i,
            })
    spaces.sort(key=lambda sp: (sp["row"] != "trên", sp["x_mm"][0]))
    return {"cars": cars, "spaces": spaces, "regions": regions, "car_size_mm": (car_w, car_h),
            "occ": occ, "free": free, "size": size}


# ------------------------------------------------------------------ bước 6: vẽ
def draw(img, H, res):
    Hinv = np.linalg.inv(H)
    top = cv2.warpPerspective(img, H, res["size"])
    out = img.copy()
    lw = max(2, img.shape[1] // 600)

    def to_img(pts):
        return cv2.perspectiveTransform(np.float32(pts).reshape(-1, 1, 2), Hinv).reshape(-1, 2).astype(np.int32)

    for x0, y0, x1, y1 in ROWS_MM:
        r = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) * S
        cv2.polylines(top, [r.astype(np.int32)], True, YELLOW, 2)
        cv2.polylines(out, [to_img(r)], True, YELLOW, lw)
    for poly, in_row in res["cars"]:
        cv2.polylines(top, [poly.astype(np.int32)], True, RED, 3)
        cv2.polylines(out, [to_img(poly)], True, RED, lw)
    drawn = set()
    for k, sp in enumerate(res["spaces"], 1):
        if sp["rid"] not in drawn:                                   # viền vùng trống (1 lần / vùng)
            drawn.add(sp["rid"])
            cnts, _ = cv2.findContours(sp["region"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            m = sp["region"] > 0                                     # tô theo mask, không đè lên xe
            top[m] = (0.75 * top[m] + 0.25 * np.array(GREEN)).astype(np.uint8)
            cv2.drawContours(top, cnts, -1, GREEN, 3)
            for c in cnts:
                cv2.polylines(out, [to_img(c.reshape(-1, 2))], True, GREEN, lw)
        x1, y1, x2, y2 = sp["gap_px"]                                # ô xanh dài theo cả khe trống
        slot = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        cv2.rectangle(top, (x1, y1), (x2, y2), GREEN, 5)
        cv2.polylines(out, [to_img(slot)], True, GREEN, lw * 2)
        n = sp["capacity"]
        label = f"#{k} {sp['gap_mm']:.0f}mm - fits {n} car" + ("s" if n > 1 else "")
        cv2.putText(top, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREEN, 3)
        p = to_img([[x1, y1]])[0]
        cv2.putText(out, label, (int(p[0]), int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, lw / 2, GREEN, lw)
    return out, top


def process(path, model, a):
    img = cv2.imread(path)
    name = os.path.splitext(os.path.basename(path))[0]
    boxes, masks = detect_cars(model, img, a.conf)
    cars_px = cv2.dilate(np.max(masks, axis=0), np.ones((25, 25), np.uint8)) if len(masks) else None
    H, how = lot_homography(img, cars_px)
    if H is None:
        print(f"{name}: không tìm được khung bãi")
        return
    res = find_spaces(boxes, H, a.pad)
    out, top = draw(img, H, res)
    cv2.imwrite(os.path.join(a.out, f"{name}_cho_trong.jpg"), out)
    cv2.imwrite(os.path.join(a.out, f"{name}_tren_xuong.jpg"), top)
    cw, ch = res["car_size_mm"]
    print(f"{name}: {len(boxes)} xe, xe chuẩn {cw:.1f} x {ch:.1f} mm, "
          f"{len(res['regions'])} vùng trống, {len(res['spaces'])} chỗ đặt được xe (homography: {how})")
    for k, sp in enumerate(res["spaces"], 1):
        x0, x1 = sp["x_mm"]
        print(f"   #{k} dãy {sp['row']}: khe {sp['gap_mm']:.1f} mm (x = {x0:.0f}..{x1:.0f} mm), "
              f"đặt được {sp['capacity']} xe")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="1 ảnh hoặc 1 thư mục ảnh")
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "best.pt"))
    ap.add_argument("--out", default="ket_qua_cho_trong")
    ap.add_argument("--pad", type=float, default=5, help="khoảng đệm an toàn quanh mỗi xe (mm)")
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(a.model)
    paths = sorted(glob.glob(os.path.join(a.image, "*.jpg"))) if os.path.isdir(a.image) else [a.image]
    for p in paths:
        process(p, model, a)
    print(f"Ảnh kết quả trong thư mục {a.out}/")


if __name__ == "__main__":
    main()
