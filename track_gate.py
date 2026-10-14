"""Giai đoạn 5: theo dõi xe vào / đỗ / rời bãi trên stream IP Webcam (Full HD), chạy YOLO bằng GPU.

Chạy:   uv run ./track_gate.py                                   (URL ở STREAM_URL)
        uv run ./track_gate.py --source http://<ip>:8080/video   (nhập URL trực tiếp)
Tùy chọn: --model best.pt  --conf 0.5  --imgsz 640  --pad 5  --do_yen 10  --mat_xe 3  --mat_xe_do 8
          --mjpeg (bỏ qua H.264, chỉ dùng MJPEG)  --chat_luong 80
Cần cùng thư mục: detect_lot.py, find_spaces.py, measure_gaps.py, best.pt. Cần torch bản CUDA.

Stream: mặc định đọc H.264 qua RTSP của IP Webcam (rtsp://<ip>:8080/h264_ulaw.sdp, phần audio bị bỏ qua);
link h264.sdp không audio của app hay treo nên chỉ thử sau. Không được thì tự chuyển sang MJPEG (/video).
Vài dòng "error while decoding MB" lúc mới kết nối là bình thường (chưa gặp keyframe).

Khung bãi: lề giấy trắng bao quanh vùng in tối; tìm 4 góc vùng in rồi ánh xạ về kích thước đã đo
(PRINT_MM) -> homography khung hình -> mặt phẳng bãi (mm). Khung được KHOÁ; mỗi khung chỉ so dải viền
quanh tờ giấy với lúc khoá. Nếu từ 2 cạnh trở lên thay đổi (giấy/máy xê dịch) thì tạm dừng YOLO, chờ
ảnh đứng yên ~1 s, tìm lại 4 góc rồi detect tiếp. Tay che 1 cạnh khi đặt xe không làm mất khoá.

Các bước:
  1. YOLO-seg chạy trên ảnh NẮN THẲNG vùng tờ giấy (3 px/mm): xe luôn đứng dọc như ảnh train dù
     camera/giấy bị xoay, không nhận nhầm đồ vật ngoài tờ giấy. Xe được vẽ bằng viền mask (polygon).
  2. ParkingRegistry cấp ID và ghép xe giữa các khung theo khoảng cách tâm (mm):
       - xe xuất hiện ở ô cam -> "vao" (ghi cổng); xe có sẵn lúc bắt đầu -> "co_san"
       - xe đứng yên trong dải >= --do_yen giây -> "do": lưu ID + vị trí, hình xe đứng im
       - xe đang đỗ bị dời/nhấc lên -> "roi_cho", theo dõi tiếp với đúng ID cũ (tay che không đổi ID)
       - xe mất dấu quá --mat_xe giây (đang đỗ: --mat_xe_do) -> "roi_bai", xoá khỏi danh sách đỗ
  3. Mỗi giây chạy lõi giai đoạn 4 (find_spaces, 4 px/mm) trên các xe đang đứng yên để vẽ chỗ trống.

YOLO chạy ở thread riêng trên khung MỚI NHẤT; cửa sổ hiện mọi khung của stream kèm kết quả gần nhất.
Phím tắt: q = thoát, space = tạm dừng.
Kết quả: ket_qua_tracking/su_kien_<ngày_giờ>.csv (nhật ký mỗi lần chạy, cạnh file track_gate.py).
"""
import argparse
import base64
import csv
import json
import os
import sys
import threading
import time
import traceback
import urllib.request
from collections import deque
from urllib.parse import unquote, urlsplit, urlunsplit

import cv2
import numpy as np

from detect_lot import ROWS_MM
from find_spaces import S, find_spaces

PHIEN_BAN = "2026-10-14 v5.3 (GPU + H264 ulaw mặc định + polygon)"   # in ra khi chạy để biết đang chạy bản nào
STREAM_URL = "http://172.18.0.200:8080/video"   # <-- URL IP Webcam (có thể dạng http://user:pass@ip:8080/video)
FULL_HD = (1920, 1080)
RED, GREEN, YELLOW, CYAN, MAGENTA, WHITE = (0, 0, 255), (0, 200, 0), (0, 220, 255), (255, 200, 0), (255, 0, 255), (255, 255, 255)
BLUE = (255, 90, 0)
HERE = os.path.dirname(os.path.abspath(__file__))


class Rate:
    """Đếm số sự kiện trong 2 giây gần nhất -> FPS thật (không bị thổi phồng khi khung đến dồn cục)."""

    def __init__(self, window=2.0):
        self.t, self.window = deque(), window

    def tick(self):
        now = time.time()
        self.t.append(now)
        while self.t and now - self.t[0] > self.window:
            self.t.popleft()

    def get(self):
        if len(self.t) < 2:
            return 0.0
        return (len(self.t) - 1) / max(time.time() - self.t[0], 1e-3)


# ------------------------------------------------------------------ IP Webcam
def _request(url):
    """urllib Request có Basic Auth nếu URL chứa user:pass@."""
    p = urlsplit(url)
    host = p.hostname + (f":{p.port}" if p.port else "")
    req = urllib.request.Request(urlunsplit((p.scheme, host, p.path, p.query, "")))
    if p.username:
        cred = f"{unquote(p.username)}:{unquote(p.password or '')}"
        req.add_header("Authorization", "Basic " + base64.b64encode(cred.encode()).decode())
    return req


def ipcam_full_hd(url, quality=80):
    """Đặt IP Webcam về 1920x1080 và chất lượng JPEG cao qua HTTP API của app.
    Trả về chuỗi mô tả để in ra màn hình (không dừng chương trình nếu lỗi)."""
    base = url.split("/video")[0].rstrip("/")
    try:
        with urllib.request.urlopen(_request(base + "/status.json?show_avail=1"), timeout=8) as r:
            st = json.loads(r.read().decode("utf-8", "ignore"))
        cur = st.get("curvals", {})
        avail = st.get("avail", {}).get("video_size", [])
        want = f"{FULL_HD[0]}x{FULL_HD[1]}"
        msg = [f"IP Webcam đang để: {cur.get('video_size')} , chất lượng {cur.get('quality')}%"]
        if avail and want not in avail:
            msg.append(f"Camera không có {want}. Các cỡ hỗ trợ: {', '.join(avail)}")
        elif cur.get("video_size") != want:
            urllib.request.urlopen(_request(f"{base}/settings/video_size?set={want}"), timeout=8).read()
            msg.append(f"Đã chuyển sang {want}")
        if str(cur.get("quality")) != str(quality):
            urllib.request.urlopen(_request(f"{base}/settings/quality?set={quality}"), timeout=8).read()
            msg.append(f"Đã đặt chất lượng JPEG {quality}%")
        return " | ".join(msg)
    except Exception as e:
        return f"Không đọc/đặt được cấu hình IP Webcam ({e}). Hãy chỉnh tay trong app: Video resolution 1920x1080."


class StreamReader(threading.Thread):
    """Đọc stream ở thread riêng, luôn giữ khung hình MỚI NHẤT ở độ phân giải gốc (không thu nhỏ).
    Thử lần lượt các URL (H.264 RTSP trước, rồi MJPEG); URL nào đọc được thì dùng."""

    def __init__(self, urls):
        super().__init__(daemon=True)
        self.urls, self.frame, self.fid = urls, None, 0
        self.lock, self.running, self.info = threading.Lock(), True, "Đang kết nối..."
        self.rate = Rate()                     # FPS thật của stream điện thoại gửi về
        self.using = None

    def run(self):
        while self.running:
            urls = ([self.using] if self.using else []) + [u for u in self.urls if u != self.using]
            for u in urls:                     # URL đã chạy được thì thử lại trước
                name = u.split("@")[-1]
                self.info = f"Đang thử {name}..."
                print(self.info)
                url, _, tr = u.partition(" (")
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (f"rtsp_transport;{tr.rstrip(')')}|fflags;nobuffer|flags;low_delay"
                                                            if tr else "")
                cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000,
                                                           cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000])
                if not cap.isOpened():
                    print(f"  -> không mở được {name}")
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                got, fails = 0, 0
                # chưa nhận khung nào thì bỏ URL sau 3 lần đọc lỗi; đang chạy thì cho phép rớt tới 30 lần
                while self.running and fails < (30 if got else 3):
                    ok, f = cap.read()
                    if not ok:
                        fails += 1
                        time.sleep(0.05)
                        continue
                    fails, got = 0, got + 1
                    if self.using != u:
                        self.using = u
                        print(f"Đang đọc stream: {name}")
                    with self.lock:
                        self.frame, self.fid = f, self.fid + 1
                    self.rate.tick()
                    self.info = "Stream OK"
                cap.release()
                if not got:
                    print(f"  -> mở được {name} nhưng không nhận được khung hình")
                if not self.running:
                    return
            self.info = "Mất kết nối stream, đang thử lại..."
            time.sleep(1)

    def get(self):
        with self.lock:
            return (None, 0) if self.frame is None else (self.frame, self.fid)


def stream_urls(url, h264=True):
    """Danh sách URL thử: RTSP H.264 + ulaw rồi H.264 qua TCP (trừ khi --mjpeg); sau đó MJPEG.
    Đuôi " (tcp)" / " (udp)" là cách truyền RTSP, được tách ra trước khi mở."""
    p = urlsplit(url)
    base = url.split("/video")[0].rstrip("/")
    urls = []
    if h264 and p.scheme == "http":
        rtsp = urlunsplit(("rtsp", p.netloc, "/h264.sdp", "", ""))
        # IP Webcam: link có audio (h264_ulaw) mới ra hình ổn định; link h264.sdp chỉ có video hay treo
        ulaw = urlunsplit(("rtsp", p.netloc, "/h264_ulaw.sdp", "", ""))
        urls += [ulaw + " (tcp)", rtsp + " (tcp)"]
    urls += [url, base + "/video", base + "/videofeed"]
    return list(dict.fromkeys(urls))


# ------------------------------------------------------------------ tờ giấy -> mặt phẳng bãi
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


# ------------------------------------------------------------------ YOLO
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


def load_model(a):
    """YOLO trên GPU NVIDIA (torch CUDA), FP16. Không có CUDA thì dừng và in cách cài."""
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        sys.exit(f"Không thấy GPU CUDA (torch {torch.__version__}). Cài torch bản CUDA:\n"
                 "  uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    model = YOLO(a.model)
    return model, dict(imgsz=a.imgsz, device=0, quantize=16), torch.cuda.get_device_name(0)


class Car:
    """Một xe trong bãi với ID cố định do chương trình cấp.
    Toạ độ đều theo mm trên mặt phẳng bãi nên không đổi khi tờ giấy/camera xê dịch rồi khoá lại."""

    def __init__(self, cid, det, now, origin):
        self.id, self.origin = cid, origin                 # origin: "co_san" | "cong trai" | "cong phai" | "xuat_hien"
        self.state = "di_chuyen"                           # "di_chuyen" | "da_do"
        self.quad = self.hull = self.c = None
        self.hist = deque()                                # (t, tâm mm) để xét đứng yên
        self.trail = deque(maxlen=60)
        self.t_seen = self.t_start = now
        self.t_park = None
        self.update(det, now)

    def update(self, det, now):
        self.quad, self.hull, self.c = det["quad"], det["hull"], det["c"]
        self.t_seen = now
        self.hist.append((now, self.c))
        self.trail.append(self.c)

    def still_for(self, now, still_mm):
        """Bao lâu (s) tâm xe không lệch quá still_mm so với vị trí hiện tại."""
        t0 = now
        for t, c in reversed(self.hist):
            if np.hypot(*(c - self.c)) > still_mm:
                break
            t0 = t
        return now - t0


def in_row(c):
    return next((k for k, (x0, y0, x1, y1) in enumerate(ROWS_MM) if x0 <= c[0] <= x1 and y0 <= c[1] <= y1), None)


def gate_at(c, margin=15.0):
    for name, (x0, y0, x1, y1) in GATES_MM.items():
        if x0 - margin <= c[0] <= x1 + margin and y0 - margin <= c[1] <= y1 + margin:
            return name
    return None


class ParkingRegistry:
    """Theo dõi xe theo trạng thái (ý tưởng TTL):
      - Xe mới xuất hiện ở ô cam -> sự kiện "vao" (ghi cổng). Xe có sẵn lúc bắt đầu -> "co_san".
      - Xe đứng yên trong dải đỗ >= park_s giây -> "da_do": lưu ID + vị trí đỗ, KHÔNG cập nhật hình xe nữa
        (đứng im, không giật). Tay che xe đang đỗ không làm mất ID.
      - Xe đang đỗ bị nhấc/đẩy đi (phát hiện lệch khỏi chỗ đỗ, hoặc biến mất rồi hiện lại gần đó)
        -> "roi_cho", theo dõi lại liên tục với ĐÚNG ID cũ.
      - Xe đang di chuyển biến mất quá lost_s giây (hoặc xe đang đỗ biến mất quá lost_park_s)
        -> "roi_bai": xoá khỏi danh sách đỗ, ghi lại ID đã rời bãi (kèm cổng nếu rời ở ô cam).
    Ghép xe giữa các khung theo khoảng cách tâm (mm), cho phép xe nhảy xa khi bị tay che / FPS thấp,
    nên cầm xe di chuyển không bị đổi ID."""

    TIGHT_MM = 8.0          # lệch <= mức này coi là cùng chỗ (nhiễu mask)
    STILL_MM = 5.0          # đứng yên nếu tâm không lệch quá mức này
    MOVE_GATE_MM = 70.0     # xe đang chạy: ghép với phát hiện trong bán kính này (theo khung liền kề)
    PARK_PICK_MM = 45.0     # xe đang đỗ: phát hiện lệch 8..45 mm quanh chỗ đỗ = xe đang bị dời đi

    def __init__(self, park_s, lost_s, lost_park_s, startup_s, log_event):
        self.park_s, self.lost_s, self.lost_park_s, self.startup_s = park_s, lost_s, lost_park_s, startup_s
        self.cars, self.next_id, self.t0 = {}, 1, None
        self.log = log_event                       # log(xe, su_kien, cong, ghi_chu)
        self.n_in = self.n_out = 0

    def _new(self, det, now):
        if self.t0 is None:
            self.t0 = now
        g = gate_at(det["c"])
        origin = "co_san" if now - self.t0 < self.startup_s and g is None else (g or "xuat_hien")
        car = Car(self.next_id, det, now, origin)
        self.cars[car.id] = car
        self.next_id += 1
        if origin == "co_san":
            self.log(car, "co_san", "", "xe có sẵn khi bắt đầu chạy")
        elif g:
            self.n_in += 1
            self.log(car, "vao", g, "xe mới vào bãi")
        else:
            self.log(car, "xuat_hien", "", "xe mới không đi qua ô cam (có thể bị mất dấu quá lâu)")
        return car

    def _start_moving(self, car, det, now, why):
        car.state, car.t_park = "di_chuyen", None
        car.hist.clear()
        car.update(det, now)
        self.log(car, "roi_cho", "", why)

    def step(self, dets, now):
        if self.t0 is None:
            self.t0 = now
        free = list(range(len(dets)))
        cars = list(self.cars.values())

        def take(pairs, limit):
            """Ghép tham lam theo khoảng cách tăng dần; trả về các cặp (car, det) đã ghép."""
            pairs = sorted((d, c.id, j) for c, j, d in pairs if d <= limit)
            used_c, out = set(), []
            for d, cid, j in pairs:
                if cid in used_c or j not in free:
                    continue
                used_c.add(cid)
                free.remove(j)
                out.append((self.cars[cid], j))
            return out

        dist = lambda c, j: float(np.hypot(*(dets[j]["c"] - c.c)))
        matched = set()
        # 1) mọi xe (đỗ hoặc chạy) có phát hiện gần như trùng chỗ
        for car, j in take([(c, j, dist(c, j)) for c in cars for j in free], self.TIGHT_MM):
            matched.add(car.id)
            if car.state == "da_do":
                car.t_seen = now                  # vẫn ở chỗ đỗ: giữ nguyên box đã lưu
            else:
                car.update(dets[j], now)
        # 2) xe đang chạy (vừa thấy ở khung trước) -> phát hiện trong bán kính MOVE_GATE_MM
        moving = [c for c in cars if c.id not in matched and c.state == "di_chuyen" and now - c.t_seen < 1.0]
        for car, j in take([(c, j, dist(c, j)) for c in moving for j in free], self.MOVE_GATE_MM):
            matched.add(car.id)
            car.update(dets[j], now)
        # 3) xe đang đỗ mà có phát hiện lệch khỏi chỗ đỗ một chút -> đang bị dời đi
        parked = [c for c in cars if c.id not in matched and c.state == "da_do"]
        for car, j in take([(c, j, dist(c, j)) for c in parked for j in free], self.PARK_PICK_MM):
            matched.add(car.id)
            self._start_moving(car, dets[j], now, "xe đang đỗ bị dời khỏi chỗ")
        # 4) phát hiện còn thừa: nhận lại xe vừa mất dấu gần đó (bán kính tăng theo thời gian mất dấu)
        lost = [c for c in cars if c.id not in matched]
        pairs = []
        for c in lost:
            r = min(250.0, 60.0 + 150.0 * (now - c.t_seen))
            pairs += [(c, j, dist(c, j)) for j in free if dist(c, j) <= r]
        for car, j in take(pairs, 250.0):
            matched.add(car.id)
            if car.state == "da_do":
                self._start_moving(car, dets[j], now, "xe đang đỗ được nhấc lên đặt chỗ khác")
            else:
                car.update(dets[j], now)
        # 5) phát hiện không thuộc xe nào -> xe mới
        for j in list(free):
            self._new(dets[j], now)
        # 6) chuyển trạng thái
        for car in list(self.cars.values()):
            gone = now - car.t_seen
            if car.state == "di_chuyen":
                while car.hist and now - car.hist[0][0] > self.park_s + 2:
                    car.hist.popleft()
                if car.id in matched and in_row(car.c) is not None and \
                        car.still_for(now, self.STILL_MM) >= self.park_s:
                    car.state, car.t_park = "da_do", now
                    self.log(car, "do", "", f"đứng yên {self.park_s:.0f} s trong dãy {'trên' if in_row(car.c) == 0 else 'dưới'}")
                elif gone > self.lost_s:
                    self._leave(car, now)
            elif gone > self.lost_park_s:
                self._leave(car, now)

    def _leave(self, car, now):
        g = gate_at(car.c, margin=25.0)
        self.n_out += 1
        self.log(car, "roi_bai", g or "", "mất dấu khi đang đỗ" if car.state == "da_do" else "xe rời khỏi bãi")
        del self.cars[car.id]

    def shift(self, dt):
        """Dời mốc thời gian của mọi xe đi dt giây (sau khi tạm dừng detect)."""
        for car in self.cars.values():
            car.t_seen += dt
            car.hist = deque((t + dt, c) for t, c in car.hist)
        if self.t0 is not None:
            self.t0 += dt

    def still_now(self, car, now, s=1.0):
        return car.state == "da_do" or car.still_for(now, self.STILL_MM) >= s


class Detector:
    """Toàn bộ phần xử lý của 1 khung hình: khoá tờ giấy, YOLO trên ảnh nắn, ParkingRegistry, find_spaces.
    step() trả về một "ảnh chụp" kết quả (dict) để vẽ; không sửa khung hình đưa vào."""

    SPACE_S = 4                                # px/mm khi chạy find_spaces trên video

    def __init__(self, model, track_kw, a, log_event):
        self.model, self.kw, self.a = model, track_kw, a
        self.paper = LotLock()
        self.reg = ParkingRegistry(a.do_yen, a.mat_xe, a.mat_xe_do, 3.0, log_event)
        self.spaces, self.t_space, self.n = [], 0.0, 0
        self.det_rate = Rate()
        self._rmap = None
        self.t_hold = None                     # lúc bắt đầu tạm dừng detect (giấy/máy xê dịch)
        self.snap = None

    def _detect(self, frame, H):
        """YOLO trên ảnh nắn thẳng -> list xe {quad, hull, c} theo mm trên mặt phẳng bãi."""
        if self._rmap is None or self._rmap[0] is not H:
            self._rmap = (H,) + rect_map(H)
        _, Hr, _, size, lo = self._rmap
        rect = cv2.warpPerspective(frame, Hr, size, flags=cv2.INTER_LINEAR)
        r = self.model.predict(rect, conf=self.a.conf, verbose=False, **self.kw)[0]
        dets = []
        if r.boxes is not None and len(r.boxes):
            segs = r.masks.xy if r.masks is not None else []
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                hull, quad = car_shape(segs[i] if i < len(segs) else None, box)
                q = quad / REC_PX_MM + lo
                dets.append({"quad": q, "hull": hull / REC_PX_MM + lo, "c": q.mean(0)})
        return dets

    def step(self, frame):
        self.n += 1
        paper, now = self.paper, time.time()
        prev_state = paper.state
        if paper.update(frame) == "moi" and self.n > 1:
            # toạ độ xe lưu theo mm trên mặt phẳng bãi nên vẫn đúng sau khi khoá lại, không cần xoá
            self.spaces, self.t_space = [], 0.0
            print("Da ve lai khung bai theo vi tri moi cua to giay")
        if prev_state == "khoa" and paper.state == "cho":
            print("Giay/may xe dich -> tam dung detect, cho on dinh...")
        H = paper.H if paper.state == "khoa" else None
        if H is None:
            if self.t_hold is None:
                self.t_hold = now
        else:
            if self.t_hold is not None:        # bỏ qua thời gian tạm dừng để xe không bị coi là mất dấu
                self.reg.shift(now - self.t_hold)
                self.t_hold = None
            self.reg.step(self._detect(frame, H), now)
            # lõi giai đoạn 4 mỗi giây, chỉ tính các xe đang đứng yên (xe đang cầm trên tay không chặn chỗ)
            if time.time() - self.t_space > 1.0:
                quads = [to_img(c.quad * S, H) for c in self.reg.cars.values() if self.reg.still_now(c, now)]
                s = self.SPACE_S
                res = find_spaces(quads, H, self.a.pad, s=s)
                self.spaces = [{"slot_mm": np.float32(sp["slot_px"]) / s, "gap_mm": sp["gap_mm"],
                                "capacity": sp["capacity"]} for sp in res["spaces"]]
                self.t_space = time.time()
        self.det_rate.tick()
        self.snap = {
            "cars": [{"id": c.id, "state": c.state, "origin": c.origin, "hull": c.hull,
                      "trail": list(c.trail) if c.state == "di_chuyen" else [],
                      "seen": now - c.t_seen < 0.5} for c in self.reg.cars.values()],
            "spaces": self.spaces, "state": paper.state,
            "q": None if paper.q is None else paper.q.copy(), "H": H,
            "n_in": self.reg.n_in, "n_out": self.reg.n_out, "det_fps": self.det_rate.get(),
        }
        return self.snap


class DetectWorker(threading.Thread):
    """Chạy Detector.step liên tục trên khung MỚI NHẤT của stream; lỗi được in ra rồi chạy tiếp."""

    def __init__(self, det, reader):
        super().__init__(daemon=True)
        self.det, self.reader = det, reader
        self.running, self.paused, self.errors = True, False, 0

    def run(self):
        last = 0
        while self.running:
            f, fid = self.reader.get()
            if self.paused or f is None or fid == last:
                time.sleep(0.003)
                continue
            last = fid
            try:
                self.det.step(f)
            except Exception:
                self.errors += 1
                print(f"Loi khi detect (lan {self.errors}), bo qua khung nay:")
                traceback.print_exc()
                time.sleep(0.2)


# ------------------------------------------------------------------ vẽ
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
    """Vẽ kết quả detect gần nhất (snap) lên khung hình out (sửa trực tiếp out)."""
    H = snap["H"] if snap else None
    spaces = snap["spaces"] if snap and H is not None else []
    if H is not None:
        img, rows, gates = _static_overlay(H)
        cv2.polylines(out, [snap["q"].astype(np.int32)], True, WHITE, lw)
        cv2.polylines(out, rows, True, YELLOW, lw)
        busy = {gate_at(c["hull"].mean(0), margin=0) for c in snap["cars"]}
        for g, name in gates:                    # tên ô cam chỉ ghi khi không có xe trong ô
            label_inside(out, g, [] if name in busy else [name], MAGENTA, lw, dim=0.8)
        for k, sp in enumerate(spaces, 1):
            x1, y1, x2, y2 = sp["slot_mm"]
            label_inside(out, img([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]),
                         [f"#{k}", f"{sp['gap_mm']:.0f}mm", f"x{sp['capacity']}"], GREEN, lw)
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
    n_do = sum(c["state"] == "da_do" for c in snap["cars"]) if snap else 0
    n_mv = len(snap["cars"]) - n_do if snap else 0
    text = info + [f"Dang do: {n_do} | dang di chuyen: {n_mv} | da vao: {snap['n_in'] if snap else 0}"
                   f" | da roi: {snap['n_out'] if snap else 0} | cho trong: {len(spaces)}"]
    for i, t in enumerate(text):
        cv2.putText(out, t, (20, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
        cv2.putText(out, t, (20, 40 + i * 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, WHITE, 2)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=STREAM_URL, help="URL IP Webcam, vd http://<ip>:8080/video")
    ap.add_argument("--mjpeg", action="store_true", help="bỏ qua H.264 (RTSP), chỉ đọc MJPEG /video")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640, help="cỡ ảnh đưa vào YOLO (ảnh nắn ~1320x940 px)")
    ap.add_argument("--pad", type=float, default=5, help="khoảng đệm quanh xe khi tìm chỗ trống (mm)")
    ap.add_argument("--out", default=os.path.join(HERE, "ket_qua_tracking"))
    ap.add_argument("--do_yen", type=float, default=10,
                    help="xe đứng yên trong dải bao nhiêu giây thì coi là ĐÃ ĐỖ (lưu ID + vị trí)")
    ap.add_argument("--mat_xe", type=float, default=3,
                    help="xe đang di chuyển mất dấu bao nhiêu giây thì coi là ĐÃ RỜI BÃI")
    ap.add_argument("--mat_xe_do", type=float, default=8,
                    help="xe đang đỗ mất dấu bao nhiêu giây thì coi là đã rời bãi (tay che ngắn hơn thì giữ ID)")
    ap.add_argument("--chat_luong", type=int, default=80, help="chất lượng JPEG đặt cho IP Webcam (phần trăm, chỉ ảnh hưởng MJPEG)")
    a = ap.parse_args()
    if not os.path.isabs(a.model) and not os.path.exists(a.model):
        a.model = os.path.join(HERE, a.model)
    os.makedirs(a.out, exist_ok=True)

    # Luồng YOLO giữ GIL của Python lâu; đổi luồng thường hơn để luồng hiển thị không bị chậm.
    sys.setswitchinterval(0.0005)
    print(f"track_gate.py phien ban: {PHIEN_BAN}")
    model, kw, gpu = load_model(a)
    print(f"YOLO: GPU {gpu} (FP16)")
    print(ipcam_full_hd(a.source, a.chat_luong))
    reader = StreamReader(stream_urls(a.source, not a.mjpeg))
    reader.start()

    frame, t0 = None, time.time()
    while frame is None:
        frame, _ = reader.get()
        if frame is None:
            if time.time() - t0 > 60:
                reader.running = False
                sys.exit(f"Không đọc được khung hình nào: {reader.info}")
            time.sleep(0.05)
    h0, w0 = frame.shape[:2]
    win = "Giai doan 5 - theo doi xe"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1600, w0), int(min(1600, w0) * h0 / w0))
    lw = max(2, w0 // 640)
    print(f"Khung hinh {w0}x{h0}" + ("" if (w0, h0) == FULL_HD else " - CHUA PHAI FULL HD!"))
    cv2.imwrite(os.path.join(a.out, "khung_dau.jpg"), frame)   # lưu khung đầu để kiểm tra góc máy

    # nhật ký sự kiện: mỗi lần chạy 1 file riêng
    log_path = os.path.join(a.out, f"su_kien_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    log = open(log_path, "w", newline="", encoding="utf-8-sig")
    logw = csv.writer(log)
    logw.writerow(["thoi_gian", "khung_hinh", "xe_id", "su_kien", "cong", "day", "x_mm", "y_mm", "ghi_chu"])
    log.flush()
    print(f"Nhat ky su kien: {log_path}")
    holder = {}

    def log_event(car, ev, gate, note):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        row = in_row(car.c)
        day = "" if row is None else ("tren" if row == 0 else "duoi")
        logw.writerow([now, holder["det"].n, car.id, ev, gate, day, round(float(car.c[0]), 1),
                       round(float(car.c[1]), 1), note])
        log.flush()
        print(f"[{now}] ID {car.id}: {ev}" + (f" ({gate})" if gate else "") + (f" day {day}" if day else "")
              + f" x={car.c[0]:.0f} y={car.c[1]:.0f} mm - {note}")

    det = holder["det"] = Detector(model, kw, a, log_event)
    det.paper.update(frame)
    det.paper.update(frame)                      # 2 khung giống nhau = đứng yên -> thử tìm ngay
    print("To giay: " + ("tim thay 4 goc, da khoa khung bai" if det.paper.state == "khoa" else "CHUA THAY, se tim lai moi 1 s"))

    worker = DetectWorker(det, reader)
    worker.start()
    shown, paused, show_rate = -1, False, Rate()
    try:
        while True:
            new = False
            if not paused:
                frame, fid = reader.get()
                new = frame is not None and fid != shown
                if new:
                    shown = fid
                    snap = det.snap
                    show_rate.tick()
                    state = snap["state"] if snap else det.paper.state
                    info = [f"{w0}x{h0} | stream {reader.rate.get():.1f} | hien thi {show_rate.get():.1f}"
                            f" | YOLO {snap['det_fps'] if snap else 0.0:.1f} FPS | {len(snap['cars']) if snap else 0} xe"
                            + ("" if (w0, h0) == FULL_HD else " | CHUA FULL HD")
                            + {"khoa": "", "cho": " | GIAY/MAY XE DICH - CHO ON DINH", "tim": " | DANG TIM TO GIAY"}[state]]
                    if reader.info != "Stream OK":
                        info.append(reader.info)
                    cv2.imshow(win, draw_overlay(frame.copy(), snap, info, lw))
            k = cv2.waitKey(1 if new else 5) & 0xFF
            if k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord(" "):
                paused = worker.paused = not paused
    except KeyboardInterrupt:
        pass
    finally:
        worker.running = False
        worker.join(timeout=2)
        reader.running = False
        log.close()
        cv2.destroyAllWindows()
    print(f"Xe da vao: {det.reg.n_in}, da roi: {det.reg.n_out}, dang do: "
          f"{sum(c.state == 'da_do' for c in det.reg.cars.values())}. Nhat ky: {log_path}")


if __name__ == "__main__":
    main()