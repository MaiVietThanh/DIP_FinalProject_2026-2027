"""Tờ giấy A3 -> mặt phẳng bãi (mm): tìm 4 góc vùng in, khoá khung (LotLock), vị trí dải đỗ / ô cam."""
import time

import cv2
import numpy as np

from helpers.detect_lot import ROWS_MM
from helpers.find_spaces import S


# Vùng in tối của tờ A3 và 2 ô cam, theo mm trên mặt phẳng bãi (gốc = góc trên-trái khung bãi,
# giống ROWS_MM). Đo trung bình trên 4 ảnh TEST đã nắn bằng 2 dải đỗ, lệch giữa các ảnh < 1.5 mm.
PRINT_MM = np.float32([[-31.6, -60.1], [390.2, -59.3], [391.6, 238.3], [-32.4, 237.8]])   # TL, TR, BR, BL
GATES_MM = {"cong trai": (-19.3, 187.8, 62.3, 228.2),           # x0, y0, x1, y1 của ô cam
            "cong phai": (299.9, -48.9, 380.9, -8.3)}


def _order(p):
    p = p.reshape(-1, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    return np.float32([p[s.argmin()], p[d.argmin()], p[s.argmax()], p[d.argmax()]])


def _paper_quad_at(sm, vmin):
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    white = ((hsv[..., 2] >= vmin) & (hsv[..., 1] < 50)).astype(np.uint8) * 255   # lề giấy trắng
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts, hier = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return None
    H, W = white.shape
    best = None
    for c, h in zip(cnts, hier[0]):
        if h[3] < 0:                       # chỉ lấy "lỗ" bên trong lề trắng = vùng in tối
            continue
        area = cv2.contourArea(c)
        if area < 0.08 * H * W:
            continue
        hull = cv2.convexHull(c)
        for eps in (0.01, 0.02, 0.03, 0.05):
            q = cv2.approxPolyDP(hull, eps * cv2.arcLength(hull, True), True)
            if len(q) == 4:
                break
        if len(q) != 4 or area / cv2.contourArea(q) < 0.9:
            continue
        q = _order(q)
        w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
        h_ = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
        if not 1.15 < max(w, h_) / min(w, h_) < 1.75:      # A3 = 1.41, cho phép méo phối cảnh
            continue
        if best is None or area > best[0]:
            best = (area, q)
    return None if best is None else best[1]


def paper_quad(frame, work_w=1280):
    """4 góc vùng in tối của tờ giấy (TL, TR, BR, BL theo hướng cạnh dài nằm ngang), hoặc None.
    Lề giấy trắng tạo thành một vòng sáng khép kín; lỗ bên trong vòng đó chính là vùng in."""
    f = work_w / frame.shape[1]
    sm = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    for vmin in (235, 215, 195, 175, 155):                   # thử dần ngưỡng sáng theo ánh sáng phòng
        q = _paper_quad_at(sm, vmin)
        if q is not None:
            q = q / f
            if np.linalg.norm(q[1] - q[0]) < np.linalg.norm(q[3] - q[0]):
                q = np.roll(q, 1, axis=0)                     # giấy đặt dọc -> xoay để cạnh dài là trục x
            return q
    return None


class LotLock:
    """Khoá khung bãi theo 4 góc tờ giấy. Trạng thái:
      "tim"  : chưa có khung, thử paper_quad mỗi 1 s
      "khoa" : đã khoá, mỗi khung chỉ kiểm tra dải viền quanh tờ giấy (ảnh nhỏ 320 px, rất nhẹ)
      "cho"  : viền bị thay đổi -> chờ ảnh đứng yên `settle_s` giây rồi tìm lại 4 góc
    update() trả về "moi" khi khung bãi mới khác khung cũ, ngược lại None."""

    SW = 320                                   # chiều rộng ảnh nhỏ dùng để so chuyển động

    def __init__(self, settle_s=1.0, still_diff=3.0, band=0.025, side_frac=0.5):
        self.q = self.H = None
        self.state = "tim"
        self.settle_s, self.still_diff, self.band, self.side_frac = settle_s, still_diff, band, side_frac
        self.prev = self.ref = self.sides = None
        self.t_still = self.t_try = 0.0

    def _small(self, frame):
        f = self.SW / frame.shape[1]
        g = cv2.cvtColor(cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(g, (5, 5), 0), f

    def _make_sides(self, shape, qs):
        """4 mặt nạ dải hẹp dọc theo 4 cạnh vùng in (gồm mép vùng in và mép tờ giấy), trên ảnh nhỏ."""
        c = qs.mean(0)
        outer = c + (qs - c) * (1 + self.band)
        inner = c + (qs - c) * (1 - self.band)
        ring = np.zeros(shape, np.uint8)
        cv2.fillPoly(ring, [outer.astype(np.int32)], 255)
        cv2.fillPoly(ring, [inner.astype(np.int32)], 0)
        sides = []
        for i in range(4):                     # tam giác tâm - 2 góc kề nhau cắt ra 1 cạnh
            tri = np.zeros(shape, np.uint8)
            a, b = c + (qs[i] - c) * 3, c + (qs[(i + 1) % 4] - c) * 3
            cv2.fillPoly(tri, [np.int32([c, a, b])], 255)
            sides.append((ring > 0) & (tri > 0))
        return sides

    def _lock(self, frame, q):
        new = self.q is None or np.abs(q - self.q).mean() > 5
        if new:
            self.q = q
            self.H = cv2.getPerspectiveTransform(q, PRINT_MM * S)          # khung hình -> bãi (S px/mm)
        g, f = self._small(frame)
        e = cv2.Canny(g, 40, 120) > 0
        # mỗi cạnh: các điểm biên (mép giấy, mép vùng in) lúc khoá
        self.sides = [np.nonzero(e & m) for m in self._make_sides(g.shape, self.q * f)]
        self.sides = [ix for ix in self.sides if len(ix[0]) > 15]
        self.ref = g
        self.state = "khoa"
        return "moi" if new else None

    def _try_find(self, frame):
        self.t_try = time.time()
        q = paper_quad(frame)
        if q is None or not cv2.isContourConvex(q.astype(np.int32)):
            return None
        x, y, w, h = cv2.boundingRect(q.astype(np.int32))
        if w < 0.2 * frame.shape[1] or h < 0.2 * frame.shape[0]:          # quá nhỏ -> bỏ
            return None
        return self._lock(frame, q)

    def update(self, frame):
        g, _ = self._small(frame)
        still = self.prev is not None and g.shape == self.prev.shape and \
            cv2.absdiff(g, self.prev).mean() < self.still_diff
        self.prev = g
        now = time.time()
        if self.state == "khoa":
            if g.shape != self.ref.shape:
                changed = 4
            else:                              # tỉ lệ điểm biên lúc khoá không còn biên quanh nó
                e = cv2.dilate(cv2.Canny(g, 40, 120), np.ones((3, 3), np.uint8)) > 0
                changed = sum(1 - e[ix].mean() > self.side_frac for ix in self.sides)
            if changed >= 2:                   # >= 2 cạnh lệch: giấy/máy xê dịch (tay chỉ che 1 cạnh)
                self.state, self.t_still = "cho", now
            return None
        if not still:
            self.t_still = now
            return None
        if self.state == "cho" and now - self.t_still < self.settle_s:
            return None
        if now - self.t_try < 1.0:
            return None
        ev = self._try_find(frame)
        if self.state != "khoa":
            self.state = "tim"
        return ev


def to_img(pts_lot_px, H):
    return cv2.perspectiveTransform(np.float32(pts_lot_px).reshape(-1, 1, 2), np.linalg.inv(H)).reshape(-1, 2)


def in_row(c):
    return next((k for k, (x0, y0, x1, y1) in enumerate(ROWS_MM) if x0 <= c[0] <= x1 and y0 <= c[1] <= y1), None)


def gate_at(c, margin=15.0):
    for name, (x0, y0, x1, y1) in GATES_MM.items():
        if x0 - margin <= c[0] <= x1 + margin and y0 - margin <= c[1] <= y1 + margin:
            return name
    return None
