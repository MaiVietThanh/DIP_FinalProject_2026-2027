"""Tu dong do vach trang (2 dai do + vach giua) va o cam tren anh bai xe mo hinh,
roi nan phoi canh ve goc nhin tu tren xuong theo don vi mm.

Chay:  python detect_lot.py anh1.jpg [anh2.jpg ...]
Xuat:  anh_debug.jpg (ve vung tim duoc), anh_warp.jpg (anh da nan, 1 mm = PX_PER_MM px)

Y tuong:
  1. Tach vach trang bang white top-hat (sang hon nen xung quanh) + do bao hoa thap,
     roi opening voi kernel ngang/doc de bo hat nhieu cua nen nhua duong.
  2. Moi dai do la mot "lo" (vung toi) duoc vach trang bao kin, hinh chu nhat dai.
  3. Chi can 1 dai do thay ro du 4 goc la tinh duoc homography cho ca bai,
     vi biet kich thuoc that (mm) cua tung dai trong ban thiet ke.
"""
import sys
import cv2
import numpy as np

# Kich thuoc that (mm), do tu ban thiet ke A3. Nen do lai bang thuoc tren ban in.
LOT_W, LOT_H = 359, 179.5                 # khung ngoai cua bai
ROWS_MM = [(1.5, 2.0, 357.5, 87.0),       # dai tren: x0, y0, x1, y1 (mep trong vach)
           (1.5, 92.5, 357.5, 177.5)]     # dai duoi
PX_PER_MM = 4


def line_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))   # rong hon be ngang vach
    tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, k)
    m = ((tophat > 35) & (hsv[..., 1] < 70)).astype(np.uint8) * 255
    horiz = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, 31), np.uint8))
    vert = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((21, 1), np.uint8))
    return cv2.morphologyEx(horiz | vert, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))


def order_corners(pts):
    """TL, TR, BR, BL."""
    pts = pts.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]])


def quad_of(cnt):
    hull = cv2.convexHull(cnt)
    for eps in (0.01, 0.02, 0.03, 0.05):
        q = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
        if len(q) == 4:
            return order_corners(q)
    return None


def find_rows(m):
    """Tra ve list (quad, chieu_cao_trung_binh) cua cac dai do tim duoc, sap tren -> duoi."""
    H, W = m.shape
    cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:                       # anh khong co vach trang nao
        return []
    rows = []
    for c, h in zip(cnts, hier[0]):
        if h[3] < 0:                       # chi lay contour "lo" ben trong vach
            continue
        area = cv2.contourArea(c)
        if area < 0.02 * H * W:
            continue
        if area / cv2.contourArea(cv2.convexHull(c)) < 0.9:   # bi xe che / ho vach
            continue
        q = quad_of(c)
        if q is None:
            continue
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        hh = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if 2.5 < w / hh < 8:
            rows.append(q)
    rows.sort(key=lambda q: q[:, 1].mean())
    return rows


def assign_rows(rows):
    """Gan dai tim duoc vao dai tren/duoi. Neu chi thay 1 dai, doan theo vi tri vach
    con lai: dai o nua tren anh la dai tren."""
    if len(rows) >= 2:
        return {0: rows[0], 1: rows[1]}
    if len(rows) == 1:
        return {"?": rows[0]}
    return {}


def homography(found, mask):
    """Tinh homography anh -> mat phang bai (px = mm * PX_PER_MM)."""
    if "?" in found:
        # Chi thay ro 1 dai: thu gia thiet no la dai tren hoac dai duoi, chieu vien
        # dai con lai len anh va chon gia thiet co nhieu pixel vach trang trung nhat.
        q, best = found["?"], None
        for cand in (0, 1):
            M = homography({cand: q}, mask)
            other = rect_pts(ROWS_MM[1 - cand]).reshape(-1, 1, 2)
            pts = cv2.perspectiveTransform(other, np.linalg.inv(M)).astype(np.int32)
            ring = np.zeros_like(mask)
            cv2.polylines(ring, [pts], True, 255, 15)
            score = cv2.countNonZero(ring & mask) / max(cv2.countNonZero(ring), 1)
            if best is None or score > best[0]:
                best = (score, M)
        return best[1]
    src = np.concatenate([found[i] for i in sorted(found)])
    dst = np.concatenate([rect_pts(ROWS_MM[i]) for i in sorted(found)])
    M, _ = cv2.findHomography(src, dst)
    return M


def rect_pts(r):
    x0, y0, x1, y1 = r
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32) * PX_PER_MM


def find_gates(img, M):
    """O cam: vien cam hinh chu nhat, nam ngoai khung bai."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (8, 90, 120), (28, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gates = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 0.004 * m.size or cv2.countNonZero(m[y:y + h, x:x + w]) > 0.5 * w * h:
            continue                        # qua nho, hoac dac ruot (sticker cam tren xe)
        q = quad_of(c)
        if q is None:
            continue
        mm = cv2.perspectiveTransform(q.reshape(-1, 1, 2), M).reshape(-1, 2) / PX_PER_MM
        cx, cy = mm.mean(0)
        if 0 < cx < LOT_W and 0 < cy < LOT_H:
            continue                        # nam trong bai -> khong phai cong
        gw, gh = np.ptp(mm[:, 0]), np.ptp(mm[:, 1])
        if not (60 < gw < 100 and 25 < gh < 55):
            continue                        # sai kich thuoc o cam (~79 x 39 mm)
        gates.append((q, mm))
    return gates


def main(path):
    img = cv2.imread(path)
    stem = path.rsplit(".", 1)[0]
    mask = line_mask(img)
    rows = find_rows(mask)
    found = assign_rows(rows)
    print(f"{path}: thay ro {len(rows)}/2 dai do")
    if not found:
        print("  khong tim thay dai nao, thu chup lai anh bai trong")
        return
    M = homography(found, mask)
    Minv = np.linalg.inv(M)

    dbg = img.copy()
    for i, r in enumerate(ROWS_MM):        # ve CA HAI dai, ke ca dai suy ra tu homography
        q = cv2.perspectiveTransform(rect_pts(r).reshape(-1, 1, 2), Minv).reshape(-1, 2)
        cv2.polylines(dbg, [q.astype(int)], True, (0, 255, 0), 6)
        cv2.putText(dbg, f"row {i}", tuple(q[0].astype(int) + [15, 60]),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    for q in rows:
        cv2.polylines(dbg, [q.astype(int)], True, (255, 255, 0), 3)
    gates = find_gates(img, M)
    for q, mm in gates:
        cv2.polylines(dbg, [q.astype(int)], True, (255, 0, 255), 6)
        print("  o cam (mm, toa do bai):", np.round(mm).astype(int).tolist())
    cv2.imwrite(stem + "_debug.jpg", dbg)

    pad = 70 * PX_PER_MM                    # nan rong them 60 mm moi phia de thay ca o cam
    T = np.array([[1, 0, pad], [0, 1, pad], [0, 0, 1]], np.float64)
    size = (int(LOT_W * PX_PER_MM + 2 * pad), int(LOT_H * PX_PER_MM + 2 * pad))
    warp = cv2.warpPerspective(img, T @ M, size)
    for i, r in enumerate(ROWS_MM):
        p = (rect_pts(r) + pad).astype(int)
        cv2.rectangle(warp, tuple(p[0]), tuple(p[2]), (0, 255, 0), 3)
        cv2.putText(warp, f"row {i}", tuple(p[0] + [10, 40]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    for k, (_, mm) in enumerate(gates):     # o cam: doi tu mm sang px cua anh nan
        p = (mm * PX_PER_MM + pad).astype(np.int32)
        cv2.polylines(warp, [p], True, (255, 0, 255), 4)
        cv2.putText(warp, f"gate {k}", tuple(p[0] + [8, -12]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
    cv2.imwrite(stem + "_warp.jpg", warp)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
