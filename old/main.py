"""
main.py  -  SmartPark demo (YOLO-World / YOLOv8 + Roboflow)
Cai dat:  pip install ultralytics roboflow opencv-python numpy pillow
Chay:     python main.py        (de main.py va yolo_model.py chung 1 thu muc)

Quy trinh:
  1. Learning  : chup anh tu stream IP Webcam (dat/di chuyen object) -> luu dataset_raw/<ten>/ -> tu upload Roboflow
  2. (Tren web Roboflow) ve bounding box cho anh -> Generate Version
  3. Train     : tai dataset tu Roboflow -> train YOLOv8 -> luu weights/best.pt -> tu nap model
  4. Tu dong   : nhan dien box, chieu rong, khoang cach ngang giua cac object ben canh nhau
  Calibrate    : (tuy chon) nhap chieu rong that cua 1 object dang thay -> hien them cm
"""

import os
import re
import json
import time
import base64
import threading
import traceback
import urllib.request
import tkinter as tk
from tkinter import simpledialog
from urllib.parse import urlsplit, urlunsplit, unquote

import cv2
import numpy as np
from PIL import Image, ImageTk

import yolo_model as ym

# ============================== CAU HINH ==============================
STREAM_URL = "http://skibidi:toilet@172.16.31.20:8080/video"   # co login: http://user:pass@ip:8080/video

# --- Roboflow ---
ROBOFLOW_API_KEY = "ijD0pUfNQVKOdEXma1UK"        # Roboflow -> Settings -> API Keys
ROBOFLOW_WORKSPACE = None                # None = workspace mac dinh; hoac dien workspace id (slug)
ROBOFLOW_PROJECT = "smartpark"           # project id (slug) tren Roboflow (loai Object Detection)
ROBOFLOW_VERSION = 1                     # so version da Generate tren Roboflow dung de train

# --- Che do nhan dien ---
DETECTOR_MODE = "world"                  # "world" = YOLO-World (khong can train) | "custom" = YOLOv8 Roboflow
WORLD_WEIGHTS = "yolov8s-worldv2.pt"
WORLD_PROMPTS = ["toy motorbike"]        # mo ta object bang tieng Anh; doi bang nut "Prompts" luc dang chay
WORLD_CONF = 0.25                        # YOLO-World cho diem thap hon YOLOv8, tang len neu bi nhan nham

# --- Model custom (YOLOv8 + Roboflow) ---
WEIGHTS_PATH = "weights/best.pt"
BASE_WEIGHTS = "yolov8n.pt"
FALLBACK_CLASSES = ["motorcycle"]        # chi dung khi chua train (model COCO san) - dat None de hien tat ca
CONF = 0.4
IMG_SIZE = 640
EPOCHS = 50
BATCH = 8

# --- Thu thap du lieu ---
RAW_DIR = "dataset_raw"
CAPTURE_FRAMES = 40
CAPTURE_INTERVAL = 0.4                   # giay giua 2 anh

PROC_W = 960                             # chieu rong khung hinh hien thi/xu ly
CALIB_FILE = "calibration.json"
# =====================================================================

COLORS = [(0, 255, 0), (0, 165, 255), (255, 0, 255), (255, 255, 0), (0, 255, 255), (255, 128, 0)]


def mask_url(u):
    return re.sub(r"//([^:/@]+):[^@]*@", r"//\1:****@", u)


def roboflow_ready():
    return bool(ROBOFLOW_API_KEY) and ROBOFLOW_API_KEY != "YOUR_API_KEY"


# ------------------------------------------------------------------ stream
class StreamReader(threading.Thread):
    """Doc stream o thread rieng. Tu thu: URL nhap -> /videofeed -> /video -> chup /shot.jpg."""

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url.strip()
        self.base = self.url.split("/video")[0].rstrip("/")
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.running = True
        self.info = "Starting..."

    def _set(self, frame):
        with self.lock:
            self.frame = frame
            self.frame_id += 1

    def _run_cv(self, url):
        self.info = f"Trying OpenCV: {mask_url(url)}"
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
            self.info = f"Stream OK: {mask_url(url)}"
            self._set(frame)
        cap.release()
        return got

    def _run_snapshot(self):
        parts = urlsplit(self.base + "/shot.jpg")
        host = parts.hostname + (f":{parts.port}" if parts.port else "")
        clean = urlunsplit((parts.scheme, host, parts.path, "", ""))
        req = urllib.request.Request(clean)
        if parts.username:
            cred = f"{unquote(parts.username)}:{unquote(parts.password or '')}"
            req.add_header("Authorization", "Basic " + base64.b64encode(cred.encode()).decode())
        fails = 0
        while self.running:
            try:
                with urllib.request.urlopen(req, timeout=3) as r:
                    data = r.read()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("khong giai ma duoc anh")
                self._set(frame)
                self.info = f"Snapshot mode OK: {clean}"
                fails = 0
            except Exception as e:
                fails += 1
                self.info = f"Loi {clean}: {e}"
                if fails > 5:
                    return
                time.sleep(0.5)

    def run(self):
        cands = []
        for u in (self.url, self.base + "/videofeed", self.base + "/video"):
            if u not in cands:
                cands.append(u)
        while self.running:
            for u in cands:
                if not self.running:
                    return
                if self._run_cv(u):
                    break
            else:
                self._run_snapshot()
            time.sleep(1.0)

    def get(self):
        with self.lock:
            return (None, 0) if self.frame is None else (self.frame.copy(), self.frame_id)

    def stop(self):
        self.running = False


# ------------------------------------------------------------------ app
class App:
    def __init__(self, root):
        self.root = root
        root.title("SmartPark - YOLOv8 width & gap demo")
        root.geometry("1100x760")

        self.canvas = tk.Canvas(root, bg="black", highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bar = tk.Frame(root, height=70, bg="#222")
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        self.buttons = []
        for text, cmd, bg in (("Learning", self.on_learning, "#2e7d32"),
                              ("Train", self.on_train, "#1565c0"),
                              ("Calibrate", self.on_calibrate, None),
                              ("Prompts", self.on_prompts, None),
                              ("Mode", self.on_mode, None)):
            b = tk.Button(bar, text=text, width=12, height=2, command=cmd,
                          **({"bg": bg, "fg": "white"} if bg else {}))
            b.pack(side=tk.LEFT, padx=6, pady=8)
            self.buttons.append(b)
        self.btn_mode = self.buttons[-1]
        self.btn_mode.config(text=f"Mode: {DETECTOR_MODE}")
        tk.Button(bar, text="Quit", width=8, height=2, command=self.on_quit).pack(side=tk.LEFT, padx=6, pady=8)
        self.status_var = tk.StringVar(value="Starting...")
        tk.Label(bar, textvariable=self.status_var, bg="#222", fg="white", anchor="w",
                 font=("Segoe UI", 10), justify="left").pack(side=tk.LEFT, padx=12, fill=tk.X, expand=True)

        # model + tracker
        self.model = ym.ParkingYOLO(
            mode=DETECTOR_MODE, weights_path=WEIGHTS_PATH, fallback_weights=BASE_WEIGHTS,
            world_weights=WORLD_WEIGHTS, prompts=WORLD_PROMPTS, conf=CONF, world_conf=WORLD_CONF,
            imgsz=IMG_SIZE, fallback_classes=FALLBACK_CLASSES)
        self.tracker = ym.BoxTracker()
        self.tracks_out = []        # list (name, conf, box)
        self.px_per_cm = self.load_calibration()

        self.mode = "detect"        # "detect" | "busy"
        self.task_msg = ""
        self.running = True
        self.proc_frame = None
        self.proc_id = 0
        self.proc_lock = threading.Lock()

        self.reader = StreamReader(STREAM_URL)
        self.reader.start()

        threading.Thread(target=self.reload_model, daemon=True).start()
        threading.Thread(target=self.detect_loop, daemon=True).start()
        self.tick()

    # ---------------- helpers
    def set_busy(self, busy):
        self.mode = "busy" if busy else "detect"
        state = tk.DISABLED if busy else tk.NORMAL
        self.root.after(0, lambda: [b.config(state=state) for b in self.buttons])

    def load_calibration(self):
        try:
            with open(CALIB_FILE, "r", encoding="utf-8") as f:
                return float(json.load(f)["px_per_cm"])
        except Exception:
            return None

    def reload_model(self):
        self.task_msg = "Dang nap model YOLO..."
        try:
            path = self.model.load()
            self.tracker.reset()
            self.task_msg = ""
            print("Da nap model:", path, "| classes:", self.model.class_names()[:10])
            return True
        except Exception as e:
            traceback.print_exc()
            self.task_msg = f"Loi nap model: {e}"
            return False

    # ---------------- Learning: chup anh + upload Roboflow
    def on_learning(self):
        if self.mode != "detect" or self.proc_frame is None:
            return
        name = simpledialog.askstring("Learning", "Ten nhom anh / object (chu, so, - _):",
                                      initialvalue="obj1", parent=self.root)
        if not name:
            return
        name = "".join(ch for ch in name.strip() if ch.isalnum() or ch in "-_") or "obj1"
        self.set_busy(True)
        threading.Thread(target=self.capture_worker, args=(name,), daemon=True).start()

    def capture_worker(self, name):
        try:
            folder = os.path.join(RAW_DIR, name)
            os.makedirs(folder, exist_ok=True)
            for t in range(3, 0, -1):
                self.task_msg = f"Dat object '{name}' vao khung hinh... {t}"
                time.sleep(1.0)
            stamp = time.strftime("%H%M%S")
            paths = []
            for i in range(CAPTURE_FRAMES):
                frame, _ = self.reader.get()
                if frame is None:
                    time.sleep(0.2)
                    continue
                p = os.path.join(folder, f"{name}_{stamp}_{i:03d}.jpg")
                cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                paths.append(p)
                self.task_msg = f"Dang chup {i + 1}/{CAPTURE_FRAMES} - xoay / doi vi tri / doi anh sang"
                time.sleep(CAPTURE_INTERVAL)

            if roboflow_ready():
                self.task_msg = "Dang upload len Roboflow..."
                ok = ym.upload_images(
                    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, paths, name,
                    progress_cb=lambda a, b: setattr(self, "task_msg", f"Upload Roboflow {a}/{b}"))
                self.task_msg = (f"Da chup {len(paths)} anh, upload {ok}. Vao Roboflow ve box -> "
                                 f"Generate Version -> bam Train")
            else:
                self.task_msg = (f"Da chup {len(paths)} anh vao {folder}. Chua cau hinh API key nen hay "
                                 f"upload thu cong len Roboflow")
            time.sleep(4)
        except Exception as e:
            traceback.print_exc()
            self.task_msg = f"Loi Learning: {e}"
            time.sleep(4)
        finally:
            self.task_msg = ""
            self.set_busy(False)

    # ---------------- Train: tai dataset Roboflow + train YOLOv8
    def on_train(self):
        if self.mode != "detect":
            return
        if not roboflow_ready():
            self.task_msg = "Chua dien ROBOFLOW_API_KEY trong main.py"
            self.root.after(4000, lambda: setattr(self, "task_msg", ""))
            return
        self.set_busy(True)
        threading.Thread(target=self.train_worker, daemon=True).start()

    def train_worker(self):
        try:
            self.task_msg = "Dang tai dataset tu Roboflow..."
            data_yaml = ym.download_dataset(ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT,
                                            ROBOFLOW_VERSION)
            self.task_msg = "Bat dau train (xem tien do chi tiet o terminal)..."
            ym.train(data_yaml, WEIGHTS_PATH, BASE_WEIGHTS, EPOCHS, IMG_SIZE, BATCH,
                     progress_cb=lambda e, n: setattr(self, "task_msg", f"Training epoch {e}/{n} ..."))
            self.task_msg = "Train xong, dang nap model moi..."
            self.reload_model()
            self.task_msg = "Train xong! Dang nhan dien bang model moi."
            time.sleep(3)
        except Exception as e:
            traceback.print_exc()
            self.task_msg = f"Loi Train: {e}"
            time.sleep(6)
        finally:
            self.task_msg = ""
            self.set_busy(False)

    # ---------------- Calibrate: quy doi pixel -> cm
    def on_calibrate(self):
        if self.mode != "detect":
            return
        tracks = list(self.tracks_out)
        if not tracks:
            self.task_msg = "Calibrate: chua thay object nao. Dat 1 object ro trong khung hinh."
            self.root.after(3500, lambda: setattr(self, "task_msg", ""))
            return
        name, conf, box = max(tracks, key=lambda t: t[1])
        w_px = ym.box_width(box)
        val = simpledialog.askfloat("Calibrate", f"Chieu rong THAT cua '{name}' (cm):", parent=self.root,
                                    minvalue=0.1)
        if not val:
            return
        self.px_per_cm = w_px / val
        with open(CALIB_FILE, "w", encoding="utf-8") as f:
            json.dump({"px_per_cm": self.px_per_cm, "proc_w": PROC_W}, f)
        self.task_msg = f"Calibrate xong: {self.px_per_cm:.2f} px/cm (camera co dinh, cung mat phang)"
        self.root.after(3500, lambda: setattr(self, "task_msg", ""))

    # ---------------- Prompts / Mode (YOLO-World)
    def on_prompts(self):
        if self.mode != "detect":
            return
        if self.model.mode != "world":
            self.task_msg = "Prompts chi dung o che do world (bam nut Mode de chuyen)"
            self.root.after(3500, lambda: setattr(self, "task_msg", ""))
            return
        s = simpledialog.askstring(
            "Prompts", "Mo ta object bang tieng Anh, cach nhau dau phay\n"
                       "(vd: toy motorbike  hoac  red toy motorbike, blue toy motorbike):",
            initialvalue=", ".join(self.model.prompts), parent=self.root)
        prompts = [p.strip() for p in (s or "").split(",") if p.strip()]
        if not prompts:
            return
        self.set_busy(True)
        threading.Thread(target=self.prompts_worker, args=(prompts,), daemon=True).start()

    def prompts_worker(self, prompts):
        try:
            self.task_msg = "Dang cap nhat prompts..."
            self.model.set_prompts(prompts)
            self.tracker.reset()
            self.task_msg = ""
        except Exception as e:
            traceback.print_exc()
            self.task_msg = f"Loi prompts: {e}"
            time.sleep(4)
            self.task_msg = ""
        finally:
            self.set_busy(False)

    def on_mode(self):
        if self.mode != "detect":
            return
        new = "custom" if self.model.mode == "world" else "world"
        self.set_busy(True)
        threading.Thread(target=self.mode_worker, args=(new,), daemon=True).start()

    def mode_worker(self, new):
        old = self.model.mode
        try:
            self.model.mode = new
            if self.reload_model():
                self.root.after(0, lambda: self.btn_mode.config(text=f"Mode: {new}"))
            else:
                self.model.mode = old
                time.sleep(5)
                self.task_msg = ""
        finally:
            self.set_busy(False)

    # ---------------- detection thread
    def detect_loop(self):
        last_id = -1
        while self.running:
            if self.mode != "detect" or not self.model.ready:
                self.tracks_out = []
                self.tracker.reset()
                time.sleep(0.1)
                continue
            with self.proc_lock:
                frame, fid = (None, 0) if self.proc_frame is None else (self.proc_frame.copy(), self.proc_id)
            if frame is None or fid == last_id:
                time.sleep(0.01)
                continue
            last_id = fid
            try:
                dets = self.model.detect(frame)
                tracks = self.tracker.update(dets, time.time())
                self.tracks_out = [(t["name"], t["conf"], t["box"].copy()) for t in tracks]
            except Exception:
                traceback.print_exc()
                time.sleep(0.2)

    # ---------------- GUI loop
    def tick(self):
        frame, fid = self.reader.get()
        if frame is not None:
            h, w = frame.shape[:2]
            proc = cv2.resize(frame, (PROC_W, int(h * PROC_W / float(w))))
            with self.proc_lock:
                self.proc_frame, self.proc_id = proc, fid
            vis = proc.copy()
            self.draw(vis)
            self.show(vis)
        else:
            self.status_var.set(self.reader.info)
        self.root.after(30, self.tick)

    def draw(self, vis):
        tracks = list(self.tracks_out) if self.mode == "detect" else []
        names = sorted({t[0] for t in tracks})
        boxes = []
        for name, conf, box in tracks:
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            col = COLORS[names.index(name) % len(COLORS)]
            w = ym.box_width(box)
            txt = f"{name} {conf:.2f}  w={w:.0f}px"
            if self.px_per_cm:
                txt += f" ({w / self.px_per_cm:.1f}cm)"
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, txt, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            yy = min(y2 + 14, vis.shape[0] - 5)
            cx = (x1 + x2) // 2
            cv2.arrowedLine(vis, (cx, yy), (x1, yy), col, 1, tipLength=0.08)
            cv2.arrowedLine(vis, (cx, yy), (x2, yy), col, 1, tipLength=0.08)
            boxes.append((x1, y1, x2, y2))

        parts = []
        for a, b, gap in ym.horizontal_gaps(boxes):
            ym_ = int((boxes[a][1] + boxes[a][3] + boxes[b][1] + boxes[b][3]) / 4)
            xa, xb = boxes[a][2], boxes[b][0]
            if xb > xa:
                cv2.arrowedLine(vis, (xa, ym_), (xb, ym_), (0, 0, 255), 2, tipLength=0.05)
                cv2.arrowedLine(vis, (xb, ym_), (xa, ym_), (0, 0, 255), 2, tipLength=0.05)
            txt = f"gap={gap:.0f}px"
            if self.px_per_cm:
                txt += f" ({gap / self.px_per_cm:.1f}cm)"
            cv2.putText(vis, txt, (int((xa + xb) / 2) - 45, ym_ - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            parts.append(txt)

        if self.task_msg:
            self.status_var.set(self.task_msg)
        else:
            src = self.model.describe()
            msg = f"Model: {src} | Detected: {len(tracks)}"
            if not self.px_per_cm:
                msg += " | (Calibrate de hien cm)"
            if parts:
                msg += " | " + " ; ".join(parts)
            self.status_var.set(msg)

    def show(self, bgr):
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        h, w = bgr.shape[:2]
        s = min(cw / w, ch / h)
        rgb = cv2.cvtColor(cv2.resize(bgr, (int(w * s), int(h * s))), cv2.COLOR_BGR2RGB)
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