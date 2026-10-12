"""
SmartPark - demo báo cáo tiến độ
Học object từ stream IP Webcam -> nhận diện bounding box -> đo chiều rộng -> đo khoảng cách (gap) giữa các object

Cài đặt:  pip install opencv-python numpy pillow
Chạy:     python smartpark_demo.py
Nhớ sửa STREAM_URL bên dưới thành URL IP Webcam của bạn (thường là http://<ip>:8080/video)

Ý tưởng:
  - Học (Learning): dùng GrabCut tách object khỏi nền (object đặt trong khung xanh) -> lưu ảnh crop + mask
                    + (tùy chọn) chiều rộng thật (cm) vào thư mục learned_objects/<tên>/
  - Nhận diện (Detect): SIFT + FLANN matching + RANSAC homography -> chiếu đường viền (contour) đã học lên
                        khung hình hiện tại -> bounding box chữ nhật bao sát object
  - Chiều rộng: tính bằng pixel, và đổi sang cm nếu đã nhập chiều rộng thật lúc học
                (quy đổi theo tỉ lệ thu/phóng của homography)
  - Từ 2 object trở lên: đo gap giữa MỌI cặp object kề nhau; khoảng cách = khoảng trống giữa 2 mép gần nhau nhất,
                         hướng đo tự xoay (ngang nếu 2 object nằm cạnh nhau, dọc nếu xếp trên/dưới,
                         chéo nếu lệch nhau theo cả 2 trục)
"""

import os
import json
import glob
import time
import shutil
import threading
import urllib.request
import tkinter as tk
from tkinter import simpledialog

import cv2
import numpy as np
from PIL import Image, ImageTk

# ======================= CAU HINH =======================
STREAM_URL = "http://9.165.230.239:8080/video"   # <-- DAN URL IP WEBCAM CUA BAN VAO DAY
SAVE_DIR = "learned_objects"
PROC_W = 640            # chieu rong khung hinh xu ly (nho hon = nhanh hon)
NUM_VIEWS = 6           # so khung hinh hoc cho moi object
MIN_GOOD_MATCHES = 10   # so match tot toi thieu de xet homography
MIN_INLIERS = 10        # so inlier RANSAC toi thieu de chap nhan nhan dien
# ========================================================

COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0)]


# ---------------------------------------------------------------- stream
class StreamReader(threading.Thread):
    """Doc stream o thread rieng. Tu thu: URL ban nhap -> /videofeed -> /video -> chup /shot.jpg lien tuc."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url.strip()
        self.base = self.url.split("/video")[0].rstrip("/")
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.info = "Starting..."

    def _set(self, frame):
        with self.lock:
            self.frame = frame

    def _run_cv(self, url):
        self.info = f"Trying OpenCV: {url}"
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            cap.release()
            return False
        got, fails = False, 0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    break
                time.sleep(0.05)
                continue
            fails, got = 0, True
            self.info = f"Stream OK: {url}"
            self._set(frame)
        cap.release()
        return got

    def _run_snapshot(self):
        url = self.base + "/shot.jpg"
        fails = 0
        while self.running:
            try:
                with urllib.request.urlopen(url, timeout=3) as r:
                    data = r.read()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("khong giai ma duoc anh")
                self._set(frame)
                self.info = f"Snapshot mode OK: {url}"
                fails = 0
            except Exception as e:
                fails += 1
                self.info = f"Loi {url}: {e}"
                if fails > 5:
                    return
                time.sleep(0.5)

    def run(self):
        candidates = []
        for u in (self.url, self.base + "/videofeed", self.base + "/video"):
            if u not in candidates:
                candidates.append(u)
        while self.running:
            for u in candidates:
                if not self.running:
                    return
                if self._run_cv(u):
                    break
            else:
                self._run_snapshot()
            time.sleep(1.0)

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.running = False


# ---------------------------------------------------------------- model
class ObjectModel:
    def __init__(self, name, real_w_cm, views):
        self.name = name
        self.real_w_cm = real_w_cm
        self.views = views  # list dict: kp_pts, des, contour, area, width_px


class Detector:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1500)
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))

    def load_model(self, folder):
        meta_path = os.path.join(folder, "meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        views = []
        for img_path in sorted(glob.glob(os.path.join(folder, "view_*.png"))):
            mask_path = img_path.replace("view_", "mask_")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            kp, des = self.sift.detectAndCompute(img, mask)
            if des is None or len(kp) < 8:
                continue
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            c = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
            views.append(dict(
                pts=np.float32([k.pt for k in kp]),
                des=des,
                contour=c.astype(np.float32),
                area=float(cv2.contourArea(c)),
                width_px=float(img.shape[1]),
            ))
        if not views:
            return None
        return ObjectModel(meta["name"], meta.get("real_width_cm"), views)

    def detect(self, frame_bgr, models):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        results = {}
        if des is None or len(kp) < 8:
            return results
        pts = np.float32([k.pt for k in kp])
        H_img, W_img = gray.shape[:2]

        for model in models:
            best = None
            for v in model.views:
                try:
                    knn = self.flann.knnMatch(v["des"], des, k=2)
                except cv2.error:
                    continue
                good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.72 * m[1].distance]
                if len(good) < MIN_GOOD_MATCHES:
                    continue
                src = v["pts"][[m.queryIdx for m in good]].reshape(-1, 1, 2)
                dst = pts[[m.trainIdx for m in good]].reshape(-1, 1, 2)
                H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                if H is None:
                    continue
                n = int(inl.sum())
                if n < MIN_INLIERS:
                    continue
                if best is None or n > best[0]:
                    best = (n, H, v)
            if best is None:
                continue

            n, H, v = best
            proj = cv2.perspectiveTransform(v["contour"], H)
            area = cv2.contourArea(proj.astype(np.float32))
            if area < 100 or v["area"] <= 0:
                continue
            x, y, w, h = cv2.boundingRect(proj.astype(np.float32))
            if w < 10 or h < 10 or w > W_img * 1.5 or h > H_img * 1.5:
                continue
            # clip vao khung hinh
            x2, y2 = min(x + w, W_img), min(y + h, H_img)
            x, y = max(x, 0), max(y, 0)
            w, h = x2 - x, y2 - y
            if w < 10 or h < 10:
                continue

            px_per_cm = None
            if model.real_w_cm:
                s = np.sqrt(area / v["area"])  # ti le thu/phong so voi luc hoc (phan anh khoang cach)
                px_per_cm = (v["width_px"] / model.real_w_cm) * s
            results[model.name] = dict(box=(x, y, w, h), inliers=n, px_per_cm=px_per_cm)
        return results


# ---------------------------------------------------------------- segmentation (learning)
def segment_object(frame, guide):
    """GrabCut khoi tao bang khung guide. Tra ve dict(img, mask, box, contour) hoac None."""
    gx, gy, gw, gh = guide
    mask = np.zeros(frame.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, mask, guide, bgd, fgd, 4, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.03 * gw * gh:
        return None
    clean = np.zeros_like(fg)
    cv2.drawContours(clean, [c], -1, 255, -1)
    x, y, w, h = cv2.boundingRect(c)
    return dict(
        img=frame[y:y + h, x:x + w].copy(),
        mask=clean[y:y + h, x:x + w].copy(),
        box=(x, y, w, h),
        contour=c,
    )



# ---------------------------------------------------------------- gap giua 2 box
def _axis_nearest(a1, a2, b1, b2):
    """Tren 1 truc: tra ve (khoang trong, toa do diem tren A, toa do diem tren B).
    Neu 2 doan chong nhau -> khoang trong 0, ca 2 diem nam giua vung chong nhau."""
    if a2 < b1:
        return b1 - a2, a2, b1
    if b2 < a1:
        return a1 - b2, a1, b2
    m = (max(a1, b1) + min(a2, b2)) / 2.0
    return 0.0, m, m


def rect_gap(ra, rb):
    """Khoang cach giua 2 bounding box (x, y, w, h), tu chon huong do.
    Tra ve (gap_px, p_on_A, p_on_B, axis) voi axis = horizontal | vertical | diagonal | overlap.
      - tach nhau theo x, chong theo y  -> do ngang
      - tach nhau theo y, chong theo x  -> do doc
      - tach ca 2 truc                  -> do cheo giua 2 goc gan nhat
    """
    ax, ay, aw, ah = ra
    bx, by, bw, bh = rb
    gx, pax, pbx = _axis_nearest(ax, ax + aw, bx, bx + bw)
    gy, pay, pby = _axis_nearest(ay, ay + ah, by, by + bh)
    if gx > 0 and gy > 0:
        axis = "diagonal"
    elif gx > 0:
        axis = "horizontal"
    elif gy > 0:
        axis = "vertical"
    else:
        axis = "overlap"
    gap = float(np.hypot(gx, gy))
    return gap, (int(round(pax)), int(round(pay))), (int(round(pbx)), int(round(pby))), axis



def neighbor_pairs(boxes):
    """Chon cac cap object KE NHAU (Relative Neighborhood Graph theo khoang gap):
    cap (i, j) bi loai neu co object k nam 'giua' - tuc gap(i,k) va gap(j,k) deu nho hon gap(i,j).
    VD 3 xe xep hang A-B-C -> do A-B va B-C (khong do A-C vi B nam giua)."""
    n = len(boxes)
    G = {}
    for i in range(n):
        for j in range(i + 1, n):
            G[(i, j)] = G[(j, i)] = rect_gap(boxes[i], boxes[j])
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            dij = G[(i, j)][0]
            blocked = any(max(G[(i, k)][0], G[(j, k)][0]) < dij
                          for k in range(n) if k != i and k != j)
            if not blocked:
                pairs.append((i, j, G[(i, j)]))
    return pairs


def draw_label(vis, lines, x, y, color, scale=0.5, anchor="tl"):
    """Ve nhieu dong chu co nen den mo, tu kep trong khung hinh.
    anchor: 'tl' = (x,y) la goc tren-trai, 'c' = (x,y) la tam."""
    font, th_ = cv2.FONT_HERSHEY_SIMPLEX, 1
    sizes = [cv2.getTextSize(t, font, scale, th_)[0] for t in lines]
    lh = max(h for _, h in sizes) + 6
    bw = max(w for w, _ in sizes) + 8
    bh = lh * len(lines) + 2
    if anchor == "c":
        x, y = x - bw // 2, y - bh // 2
    H, W = vis.shape[:2]
    x = int(np.clip(x, 0, max(0, W - bw)))
    y = int(np.clip(y, 0, max(0, H - bh)))
    roi = vis[y:y + bh, x:x + bw]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for k, t in enumerate(lines):
        cv2.putText(vis, t, (x + 4, y + lh * (k + 1) - 4), font, scale, color, th_ + 1, cv2.LINE_AA)
    return x, y, bw, bh


# ---------------------------------------------------------------- app
class App:
    def __init__(self, root):
        self.root = root
        root.title("SmartPark - Object width & gap demo")
        root.geometry("1000x720")

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bar = tk.Frame(root, height=70, bg="#222")
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.btn_learn = tk.Button(bar, text="Learning", width=14, height=2,
                                   bg="#2e7d32", fg="white", command=self.on_learning)
        self.btn_learn.pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(bar, text="Reset objects", width=14, height=2,
                  command=self.on_reset).pack(side=tk.LEFT, padx=8, pady=8)
        tk.Button(bar, text="Quit", width=10, height=2,
                  command=self.on_quit).pack(side=tk.LEFT, padx=8, pady=8)
        self.status_var = tk.StringVar(value="Connecting to stream...")
        tk.Label(bar, textvariable=self.status_var, bg="#222", fg="white",
                 anchor="w", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=16, fill=tk.X, expand=True)

        os.makedirs(SAVE_DIR, exist_ok=True)
        self.detector = Detector()
        self.models = []
        self.load_all_models()

        self.reader = StreamReader(STREAM_URL)
        self.reader.start()

        self.proc_frame = None
        self.proc_lock = threading.Lock()
        self.detections = {}
        self.tracks = {}          # name -> dict(box, px_per_cm, last_seen)
        self.mode = "detect"      # "detect" | "learning"
        self.learn_status = ""
        self.learn_preview = None
        self.running = True

        self.guide = None
        threading.Thread(target=self.detect_loop, daemon=True).start()
        self.tick()

    # ---------- models
    def load_all_models(self):
        models = []
        for folder in sorted(glob.glob(os.path.join(SAVE_DIR, "*"))):
            if os.path.isdir(folder):
                m = self.detector.load_model(folder)
                if m is not None:
                    models.append(m)
        self.models = models

    def on_reset(self):
        if self.mode == "learning":
            return
        shutil.rmtree(SAVE_DIR, ignore_errors=True)
        os.makedirs(SAVE_DIR, exist_ok=True)
        self.models = []
        self.tracks = {}
        self.detections = {}

    # ---------- learning
    def on_learning(self):
        if self.mode == "learning" or self.proc_frame is None:
            return
        default = f"obj{len(self.models) + 1}"
        name = simpledialog.askstring("Learning", "Ten object:", initialvalue=default, parent=self.root)
        if not name:
            return
        name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "-_") or default
        w_str = simpledialog.askstring(
            "Learning", "Chieu rong THAT cua object (cm).\nDe trong neu chi muon hien pixel:",
            parent=self.root)
        real_w = None
        if w_str:
            try:
                real_w = float(w_str.replace(",", "."))
                if real_w <= 0:
                    real_w = None
            except ValueError:
                real_w = None
        self.mode = "learning"
        self.btn_learn.config(state=tk.DISABLED)
        threading.Thread(target=self.learn_worker, args=(name, real_w), daemon=True).start()

    def learn_worker(self, name, real_w):
        try:
            for t in range(3, 0, -1):
                self.learn_status = f"Dat '{name}' vao khung xanh... {t}"
                time.sleep(1.0)

            views, tries = [], 0
            while len(views) < NUM_VIEWS and tries < NUM_VIEWS * 4:
                tries += 1
                with self.proc_lock:
                    frame = None if self.proc_frame is None else self.proc_frame.copy()
                if frame is None:
                    time.sleep(0.2)
                    continue
                res = segment_object(frame, self.guide)
                if res is None:
                    self.learn_status = "Khong tach duoc object - dung nen don gian, dat object ro hon"
                    self.learn_preview = None
                    time.sleep(0.4)
                    continue
                views.append(res)
                self.learn_preview = res
                self.learn_status = f"Learning '{name}' {len(views)}/{NUM_VIEWS} - xoay/dich nhe object"
                time.sleep(0.9)

            if len(views) >= 2:
                folder = os.path.join(SAVE_DIR, name)
                shutil.rmtree(folder, ignore_errors=True)
                os.makedirs(folder, exist_ok=True)
                for i, v in enumerate(views):
                    cv2.imwrite(os.path.join(folder, f"view_{i:02d}.png"), v["img"])
                    cv2.imwrite(os.path.join(folder, f"mask_{i:02d}.png"), v["mask"])
                with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump({"name": name, "real_width_cm": real_w}, f)
                model = self.detector.load_model(folder)
                if model is not None:
                    self.models = [m for m in self.models if m.name != name] + [model]
                    self.learn_status = f"Da hoc xong '{name}'"
                else:
                    self.learn_status = "Hoc that bai (qua it dac trung) - thu object co hoa van ro hon"
            else:
                self.learn_status = "Hoc that bai - khong tach duoc object"
            time.sleep(1.5)
        finally:
            self.learn_preview = None
            self.learn_status = ""
            self.mode = "detect"
            self.root.after(0, lambda: self.btn_learn.config(state=tk.NORMAL))

    # ---------- detection thread
    def detect_loop(self):
        while self.running:
            if self.mode != "detect" or not self.models:
                time.sleep(0.05)
                continue
            with self.proc_lock:
                frame = None if self.proc_frame is None else self.proc_frame.copy()
            if frame is None:
                time.sleep(0.05)
                continue
            try:
                self.detections = self.detector.detect(frame, list(self.models))
            except Exception as e:  # khong de thread chet
                print("detect error:", e)
                time.sleep(0.1)

    # ---------- main loop
    def tick(self):
        frame = self.reader.get()
        if frame is not None:
            h, w = frame.shape[:2]
            s = PROC_W / float(w)
            proc = cv2.resize(frame, (PROC_W, int(h * s)))
            with self.proc_lock:
                self.proc_frame = proc
            ph, pw = proc.shape[:2]
            self.guide = (int(pw * 0.25), int(ph * 0.2), int(pw * 0.5), int(ph * 0.6))
            vis = proc.copy()
            self.draw(vis)
            self.show(vis)
        else:
            self.status_var.set(self.reader.info)
        self.root.after(30, self.tick)

    def draw(self, vis):
        now = time.time()
        if self.mode == "learning":
            gx, gy, gw, gh = self.guide
            cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)
            prev = self.learn_preview
            if prev is not None:
                x, y, w, h = prev["box"]
                cv2.drawContours(vis, [prev["contour"]], -1, (0, 255, 255), 2)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(vis, f"w = {w}px", (x, max(y - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            self.status_var.set(self.learn_status)
            return

        # cap nhat tracks (lam muot + giu box 0.5s khi mat dau)
        for name, d in self.detections.items():
            old = self.tracks.get(name)
            box = np.array(d["box"], np.float32)
            if old is not None:
                box = 0.6 * old["box"] + 0.4 * box
            self.tracks[name] = dict(box=box, px_per_cm=d["px_per_cm"], last_seen=now,
                                     inliers=d["inliers"])
        self.tracks = {n: t for n, t in self.tracks.items() if now - t["last_seen"] < 0.5}

        items = []
        for i, (name, t) in enumerate(sorted(self.tracks.items())):
            x, y, w, h = [int(round(v)) for v in t["box"]]
            col = COLORS[i % len(COLORS)]
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 2)
            wtxt = f"w={w}px"
            if t["px_per_cm"]:
                wtxt += f" ({w / t['px_per_cm']:.1f}cm)"
            # nhan nam ben trong goc tren box -> cac box khong chong nhau thi nhan cung khong de nhau
            draw_label(vis, [name, wtxt], x + 2, y + 2, col, scale=0.45)
            # dim chieu rong duoi box
            yy = min(y + h + 12, vis.shape[0] - 5)
            cv2.arrowedLine(vis, (x + w // 2, yy), (x, yy), col, 1, tipLength=0.08)
            cv2.arrowedLine(vis, (x + w // 2, yy), (x + w, yy), col, 1, tipLength=0.08)
            items.append((name, x, y, w, h, t["px_per_cm"], col))

        msg = f"Objects learned: {len(self.models)} | Detected: {len(items)}"
        if len(items) >= 2:
            pairs = neighbor_pairs([it[1:5] for it in items])
            gap_msgs = []
            for i, j, (gap, p1, p2, axis) in pairs:
                a, b = items[i], items[j]
                scales = [sc for sc in (a[5], b[5]) if sc]
                cm = f"{gap / (sum(scales) / len(scales)):.1f}cm" if scales else None
                if gap > 0:
                    cv2.arrowedLine(vis, p1, p2, (0, 0, 255), 2, tipLength=0.08)
                    cv2.arrowedLine(vis, p2, p1, (0, 0, 255), 2, tipLength=0.08)
                    mx, my = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                    lines = [f"{gap:.0f}px"] + ([cm] if cm else [])
                    # dat nhan gap lech khoi mui ten (tren voi do ngang, ben phai voi do doc)
                    if axis == "vertical":
                        draw_label(vis, lines, mx + 8, my - 14, (0, 0, 255), scale=0.5)
                    else:
                        draw_label(vis, lines, mx, my - 26, (0, 0, 255), scale=0.5, anchor="c")
                gtxt = f"{gap:.0f}px" + (f"/{cm}" if cm else "")
                gap_msgs.append(f"{a[0]}<->{b[0]}: {gtxt}")
            msg += " | " + "  ".join(gap_msgs)
        self.status_var.set(msg)

    def show(self, bgr):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        nw, nh = int(w * s), int(h * s)
        rgb = cv2.cvtColor(cv2.resize(bgr, (nw, nh)), cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.photo)

    def on_quit(self):
        self.running = False
        self.reader.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.protocol("WM_DELETE_WINDOW", app.on_quit)
    root.mainloop()