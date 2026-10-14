"""Do khoang trong (mm) giua cac xe tren bo TEST bang mask YOLOv8-seg, roi so voi dap an.

Can dat cung thu muc:  measure_gaps.py, detect_lot.py, best.pt, thu muc TEST/
Cai 1 lan:             pip install ultralytics opencv-python
Chay:                  python measure_gaps.py
Tuy chon:              python measure_gaps.py --model best.pt --images TEST --gt TEST/ground_truth_test.csv --out ket_qua_do

Cac buoc cho moi anh:
  1. YOLO tra ve mask tung xe tren anh GOC (retina_masks=True, imgsz=1024).
  2. detect_lot.py tim 2 dai do -> homography anh -> mat phang bai (mm).
     Neu xe che vach lam khong thay du 2 dai: tim vien ngoai ca khung bai (4 goc).
     Moi anh tu tim rieng, vi giua cac anh camera co xe dich.
  3. Nan tung mask ve mat phang bai, xe co tam nam trong dai nao thi thuoc dai do.
     Xe o o cam / loi di (tam ngoai 2 dai) khong tinh.
  4. Trong moi dai, sap xe trai -> phai. Khoang thu k = mep trai xe k+1 - mep phai xe k,
     mep lay tu hinh chieu cua mask len truc doc dai (chi phan mask nam trong dai).
     Khoang 0 = vach trai -> xe dau, khoang n = xe cuoi -> vach phai.
  5. So voi ground_truth_test.csv (file, row, gap_index, gap_mm).

Ket qua trong thu muc --out:
  do_tat_ca.csv   moi khoang do duoc (ke ca khoang khong co trong dap an)
  so_sanh.csv     dap an vs do duoc, sai so
  <anh>_tren_xuong.jpg   anh da nan, to mau tung xe, ghi so mm tung khoang
  <anh>_mask.jpg         anh goc ve vien mask
"""
import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # thư mục chứa main.py, helpers/, weights/
if ROOT not in sys.path:   # để chạy thẳng file này (python helpers/xxx.py) vẫn import được helpers.*
    sys.path.insert(0, ROOT)

from helpers.detect_lot import (LOT_H, LOT_W, PX_PER_MM, ROWS_MM, assign_rows,
                        find_rows, homography, line_mask, quad_of)

S = 8                       # do phan giai mat phang do: 8 px / mm (0,125 mm / px)
CAR_W = 28                  # be ngang xe (mm)
MARGIN = 5                  # khoang chua moi ben (mm)
NEED = CAR_W + 2 * MARGIN   # khoang toi thieu de do them 1 xe = 38 mm
ROW_NAME = {0: "tren", 1: "duoi"}
COLORS = [(255, 80, 80), (80, 200, 80), (80, 80, 255), (0, 200, 255),
          (255, 0, 255), (255, 255, 0), (0, 128, 255), (180, 105, 255)]


# ---------------------------------------------------------------- homography
def lot_homography(img, cars=None):
    """Homography anh -> mat phang bai, don vi S px/mm. Tra ve (H, cach_tim) hoac (None, "").
    Cach 1: tim 2 dai do (lo kin trong vach) nhu detect_lot.py.
    Cach 2: neu xe che vach lam dai bi ho, tim VIEN NGOAI cua ca khung bai (bao loi
            cua vach, da xoa vung xe) -> 4 goc khung, khong can vach giua."""
    lm = line_mask(img)
    rows = find_rows(lm)
    if len(rows) >= 2:
        M = homography(assign_rows(rows), lm)               # don vi PX_PER_MM px/mm
        return np.diag([S / PX_PER_MM, S / PX_PER_MM, 1.0]) @ M, "2 dai"
    r = frame_quad(lm, cars)
    if r is None:
        return None, ""
    dst = np.float32([[0, 0], [LOT_W, 0], [LOT_W, LOT_H], [0, LOT_H]]) * S
    return cv2.getPerspectiveTransform(np.float32(r[0]), dst), "vien khung"


def _side_line(pts, p, q):
    d = q - p; L = np.linalg.norm(d); n = np.array([-d[1], d[0]]) / L
    t = (pts - p) @ d / L**2
    dist = np.abs((pts - p) @ n)
    sel = pts[(dist < 12) & (t > 0.05) & (t < 0.95)]
    if len(sel) < 20:
        return None
    return cv2.fitLine(sel.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()


def _inter(l1, l2):
    vx1, vy1, x1, y1 = l1; vx2, vy2, x2, y2 = l2
    t = np.linalg.solve(np.array([[vx1, -vx2], [vy1, -vy2]]), np.array([x2 - x1, y2 - y1]))[0]
    return x1 + t * vx1, y1 + t * vy1


def _right_angles(q, tol=20):
    for i in range(4):
        a, b = q[i - 1] - q[i], q[(i + 1) % 4] - q[i]
        ang = np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))
        if abs(ang - 90) > tol:
            return False
    return True


def frame_quad(lm, cars=None):
    """Tim vien NGOAI cua khung bai (2 dai gop lai) bang bao loi cua thanh phan vach.
    Khong can vach giua hay 4 goc tung dai con nguyen, nen it bi xe che anh huong."""
    m = lm.copy()
    if cars is not None:
        m[cars > 0] = 0
    H, W = m.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    big = [k for k in range(1, n) if st[k, 2] > W / 3 and st[k, 3] > H / 8]
    # ung vien: tung thanh phan lon, va tung cap (khung bi xe che dut lam 2 manh)
    cands = [[k] for k in big] + [[a, b] for i, a in enumerate(big) for b in big[i + 1:]]
    best = None
    for ks in cands:
        pts = np.concatenate([np.stack(np.nonzero(lab == k)[::-1], 1) for k in ks]).astype(np.int32)
        hull = cv2.convexHull(pts)
        q = quad_of(hull)
        if q is None:
            continue
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if not 1.6 < w / h < 2.5:
            continue
        # vien phai thuc su la vach: lay mau tren 4 canh, phan lon phai trung mask vach goc
        hits = 0; tot = 0
        for i in range(4):
            p, r = q[i], q[(i + 1) % 4]
            for t in np.linspace(0.05, 0.95, 60):
                x, y = (p + t * (r - p)).astype(int)
                c = (q.mean(0) - (x, y)); c = c / np.linalg.norm(c)
                ok = any(lm[int(y + c[1] * d), int(x + c[0] * d)] for d in range(0, 25, 3)
                         if 0 <= int(y + c[1] * d) < H and 0 <= int(x + c[0] * d) < W)
                hits += ok; tot += 1
        fill = hits / tot
        if fill < 0.85 or not _right_angles(q):
            continue
        area = cv2.contourArea(q)
        if best is None or area > best[0]:
            best = (area, q, hull.reshape(-1, 2), fill)
    if best is None:
        return None
    _, q, hull, fill = best
    # tinh lai 4 canh bang fitLine tren diem vien ngoai cua vach
    cnt_pts = hull.astype(np.float64)
    lines = [_side_line(cnt_pts, q[i], q[(i + 1) % 4]) for i in range(4)]
    if all(l is not None for l in lines):
        q = np.array([_inter(lines[i - 1], lines[i]) for i in range(4)], np.float32)
    return q, fill


# ---------------------------------------------------------------- do khoang
def car_masks(model, img, conf):
    r = model.predict(img, imgsz=1024, conf=conf, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    out = []
    for m in r.masks.data.cpu().numpy():
        m = (m > 0.5).astype(np.uint8)
        if m.shape != img.shape[:2]:
            m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m)
        if n > 2:                                   # giu manh lon nhat, bo hat nhieu
            m = (lab == 1 + st[1:, cv2.CC_STAT_AREA].argmax()).astype(np.uint8)
        out.append(m)
    return out


def measure(masks, H):
    """Tra ve dict row -> list xe (sap trai->phai), moi xe: (x_trai, x_phai, mask_nan) theo mm."""
    size = (int(LOT_W * S), int(LOT_H * S))
    rows = {0: [], 1: []}
    for m in masks:
        w = cv2.warpPerspective(m * 255, H, size, flags=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(w)
        if len(xs) == 0:
            continue
        cx, cy = xs.mean() / S, ys.mean() / S
        for i, (x0, y0, x1, y1) in enumerate(ROWS_MM):
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                band = w[int(y0 * S):int(y1 * S)]           # chi phan mask nam trong dai
                cols = np.nonzero(band.any(axis=0))[0]
                if len(cols):
                    rows[i].append((cols.min() / S, (cols.max() + 1) / S, w))
                break
    for i in rows:
        rows[i].sort(key=lambda c: c[0])
    return rows


def gaps_of(cars, row):
    """Khoang 0..n theo mm. Khoang am (xe xeo chong nhau) -> 0."""
    x0, _, x1, _ = ROWS_MM[row]
    edges = [x0] + [v for c in cars for v in (c[0], c[1])] + [x1]
    return [max(0.0, edges[2 * k + 1] - edges[2 * k]) for k in range(len(cars) + 1)]


def fits(g, edge):
    """So xe moi do them duoc vao khoang g. Khoang sat vach chi can chua 1 ben."""
    need_first = CAR_W + MARGIN if edge else NEED
    if g < need_first:
        return 0
    return 1 + int((g - need_first) // (CAR_W + MARGIN))


# ---------------------------------------------------------------- ve
def draw_top(rows, gaps, H_img, img):
    size = (int(LOT_W * S), int(LOT_H * S))
    top = cv2.warpPerspective(img, H_img, size)
    k = 0
    for i, cars in rows.items():
        x0, y0, x1, y1 = ROWS_MM[i]
        cv2.rectangle(top, (int(x0 * S), int(y0 * S)), (int(x1 * S), int(y1 * S)), (0, 255, 0), 3)
        for c in cars:
            col = np.array(COLORS[k % len(COLORS)], np.uint8)
            top[c[2] > 0] = (0.5 * top[c[2] > 0] + 0.5 * col).astype(np.uint8)
            k += 1
        e = [x0] + [v for c in cars for v in (c[0], c[1])] + [x1]
        ymid = int((y0 + y1) / 2 * S)
        for j, g in enumerate(gaps[i]):
            a, b = int(e[2 * j] * S), int(e[2 * j + 1] * S)
            ok = fits(g, j in (0, len(gaps[i]) - 1)) > 0
            colr = (0, 220, 0) if ok else (0, 0, 255)
            if b > a:
                cv2.arrowedLine(top, (a, ymid), (b, ymid), colr, 3, tipLength=0.08)
                cv2.arrowedLine(top, (b, ymid), (a, ymid), colr, 3, tipLength=0.08)
            cv2.putText(top, f"{j}:{g:.1f}", (a + 4, ymid - 12 - 40 * (j % 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, colr, 3)
    return top


def draw_masks(img, masks):
    out = img.copy()
    for k, m in enumerate(masks):
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cnts, -1, COLORS[k % len(COLORS)], 4)
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "weights", "best.pt"))
    ap.add_argument("--images", default="TEST")
    ap.add_argument("--gt", default=None, help="mac dinh: <images>/ground_truth_test.csv")
    ap.add_argument("--out", default="ket_qua_do")
    ap.add_argument("--conf", type=float, default=0.5)
    a = ap.parse_args()
    gt_path = a.gt or os.path.join(a.images, "ground_truth_test.csv")
    os.makedirs(a.out, exist_ok=True)

    from ultralytics import YOLO
    model = YOLO(a.model)

    gt = {}
    if os.path.exists(gt_path):
        with open(gt_path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                gt.setdefault(r["file"], []).append((r["row"], int(r["gap_index"]), float(r["gap_mm"])))
    else:
        print("Khong thay file dap an:", gt_path)

    files = sorted(f for f in glob.glob(os.path.join(a.images, "*.jpg")) if "empty" not in f)
    all_rows, cmp_rows = [], []
    for p in files:
        name = os.path.basename(p)
        img = cv2.imread(p)
        masks = car_masks(model, img, a.conf)
        cars = None
        if masks:
            cars = cv2.dilate(np.max(masks, axis=0), np.ones((25, 25), np.uint8))
        H, src = lot_homography(img, cars)
        if H is None:
            print(f"{name}: khong tim duoc khung bai, bo qua")
            continue
        rows = measure(masks, H)
        gaps = {i: gaps_of(rows[i], i) for i in rows}
        n_out = len(masks) - sum(len(c) for c in rows.values())
        print(f"{name}: {len(masks)} xe (tren {len(rows[0])}, duoi {len(rows[1])}, ngoai dai {n_out}), "
              f"homography {src}")
        for i in rows:
            for j, g in enumerate(gaps[i]):
                edge = j in (0, len(gaps[i]) - 1)
                all_rows.append([name, ROW_NAME[i], j, f"{g:.1f}", "vach" if edge else "giua 2 xe", fits(g, edge)])
        for row, idx, mm in gt.get(name, []):
            i = 0 if row == "tren" else 1
            meas = gaps[i][idx] if idx < len(gaps[i]) - 1 else None
            err = None if meas is None else meas - mm
            cmp_rows.append([name, row, idx, mm, "" if meas is None else f"{meas:.1f}",
                             "" if err is None else f"{err:+.1f}", len(rows[i]),
                             "" if meas is None else ("dung" if (meas >= NEED) == (mm >= NEED) else "SAI")])
        cv2.imwrite(os.path.join(a.out, name.replace(".jpg", "_tren_xuong.jpg")), draw_top(rows, gaps, H, img))
        cv2.imwrite(os.path.join(a.out, name.replace(".jpg", "_mask.jpg")), draw_masks(img, masks))

    with open(os.path.join(a.out, "do_tat_ca.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "row", "gap_index", "do_mm", "loai", "so_xe_do_them"])
        w.writerows(all_rows)
    with open(os.path.join(a.out, "so_sanh.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "row", "gap_index", "dap_an_mm", "do_mm", "sai_so_mm", "so_xe_trong_dai", "ket_luan_do_xe"])
        w.writerows(cmp_rows)

    # ---- tong ket
    ok = [r for r in cmp_rows if r[5] != ""]
    print(f"\n=== So sanh voi dap an: {len(ok)}/{len(cmp_rows)} khoang do duoc ===")
    if not ok:
        return
    e = np.array([float(r[5]) for r in ok])
    print(f"Sai so trung binh (co dau): {e.mean():+.2f} mm")
    print(f"MAE: {np.abs(e).mean():.2f} mm   RMSE: {np.sqrt((e ** 2).mean()):.2f} mm   "
          f"lon nhat: {np.abs(e).max():.2f} mm")
    for mm in sorted({r[3] for r in ok}):
        ee = np.array([float(r[5]) for r in ok if r[3] == mm])
        print(f"  khoang {mm:.0f} mm: n={len(ee):2d}  trung binh {ee.mean():+.2f}  do lech chuan {ee.std():.2f}")
    right = sum(r[7] == "dung" for r in ok)
    print(f"Ket luan do duoc / khong do duoc (nguong {NEED} mm) dung: {right}/{len(ok)}")
    bad = [r for r in ok if abs(float(r[5])) > 5]
    if bad:
        print("Khoang lech > 5 mm (nen xem anh *_tren_xuong.jpg, co the thieu/thua xe nen lech so thu tu):")
        for r in bad:
            print("  ", r[:6])
    miss = [r for r in cmp_rows if r[5] == ""]
    if miss:
        print("Khong do duoc (dai co it xe hon so thu tu khoang trong dap an):")
        for r in miss:
            print("  ", r[:4], "so xe trong dai:", r[6])
    print(f"\nChi tiet: {a.out}/so_sanh.csv, {a.out}/do_tat_ca.csv, anh trong {a.out}/")


if __name__ == "__main__":
    sys.exit(main())
