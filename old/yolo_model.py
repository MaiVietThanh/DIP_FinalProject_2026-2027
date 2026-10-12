"""
yolo_model.py  -  Phan MODEL cua SmartPark (YOLO-World zero-shot + YOLOv8/Roboflow)

Chua:
  - ParkingYOLO      : 2 che do: "world" (YOLO-World, mo ta object bang chu, KHONG can train)
                       va "custom" (YOLOv8 da train tu Roboflow, hoac COCO san neu chua train)
  - BoxTracker       : lam muot box + giu box vai khung hinh khi YOLO bo sot (chong "chap chon")
  - box_width / horizontal_gaps : tinh chieu rong va khoang cach ngang
  - upload_images    : day anh len Roboflow de gan nhan
  - download_dataset : tai dataset da gan nhan tu Roboflow (dinh dang YOLOv8)
  - train            : fine-tune YOLOv8 tren dataset do, luu best.pt

File main.py chi lo giao dien / cau hinh va goi cac ham trong file nay.
"""

import os
import shutil
import threading

import numpy as np


# ====================================================================== DETECT
class ParkingYOLO:
    def __init__(self, mode="world", weights_path="weights/best.pt", fallback_weights="yolov8n.pt",
                 world_weights="yolov8s-worldv2.pt", prompts=None, conf=0.4, world_conf=0.25,
                 imgsz=640, fallback_classes=None):
        self.mode = mode                          # "world" | "custom"
        self.weights_path = weights_path
        self.fallback_weights = fallback_weights
        self.world_weights = world_weights
        self.prompts = list(prompts) if prompts else ["motorbike"]
        self.conf = conf                          # nguong cho che do custom
        self.world_conf = world_conf              # nguong cho YOLO-World (diem zero-shot thuong thap hon)
        self.imgsz = imgsz
        self.fallback_classes = fallback_classes  # vd ["motorcycle"], chi ap dung khi dung model COCO san
        self.model = None
        self.class_ids = None
        self.using_fallback = False
        self.lock = threading.Lock()

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        """world: nap YOLO-World + dat prompts. custom: nap weights/best.pt, chua co thi dung COCO san."""
        from ultralytics import YOLO, YOLOWorld
        class_ids, fallback = None, False
        if self.mode == "world":
            model = YOLOWorld(self.world_weights)   # tu tai ve lan dau (can internet)
            model.set_classes(self.prompts)         # can CLIP, chi chay 1 lan luc nap
            path = self.world_weights
        else:
            if os.path.exists(self.weights_path):
                path = self.weights_path
            else:
                path, fallback = self.fallback_weights, True
            model = YOLO(path)
            if fallback and self.fallback_classes:
                inv = {v: k for k, v in model.names.items()}
                class_ids = [inv[n] for n in self.fallback_classes if n in inv] or None
        with self.lock:
            self.model, self.class_ids, self.using_fallback = model, class_ids, fallback
        return path

    def set_prompts(self, prompts):
        """Doi mo ta object luc dang chay (chi co tac dung o che do world)."""
        prompts = [p.strip() for p in prompts if p and p.strip()]
        if not prompts:
            raise ValueError("prompts rong")
        with self.lock:
            if self.model is not None and self.mode == "world":
                self.model.set_classes(prompts)
            self.prompts = prompts

    def describe(self):
        if self.mode == "world":
            return "YOLO-World: " + ", ".join(self.prompts)
        return "COCO (chua train)" if self.using_fallback else os.path.basename(self.weights_path)

    def class_names(self):
        return list(self.model.names.values()) if self.model is not None else []

    def detect(self, frame_bgr):
        """Tra ve list dict(name, conf, box=np.array([x1,y1,x2,y2]))."""
        with self.lock:
            model = self.model
            if model is None:
                return []
            conf = self.world_conf if self.mode == "world" else self.conf
            res = model.predict(frame_bgr, imgsz=self.imgsz, conf=conf, classes=self.class_ids,
                                agnostic_nms=(self.mode == "world"), verbose=False)[0]
            names = model.names
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append(dict(name=names[int(b.cls[0])], conf=float(b.conf[0]),
                            box=np.array([x1, y1, x2, y2], np.float32)))
        return out


# ====================================================================== TRACK / SMOOTH
def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class BoxTracker:
    """Ghep detection voi track cu theo IoU, lam muot bang EMA, giu track max_age giay khi mat dau."""

    def __init__(self, alpha=0.55, max_age=0.8, iou_thr=0.25, min_hits=2):
        self.alpha, self.max_age, self.iou_thr, self.min_hits = alpha, max_age, iou_thr, min_hits
        self.tracks = []
        self._next_id = 1

    def reset(self):
        self.tracks = []

    def update(self, dets, now):
        used = set()
        for t in self.tracks:
            best = None
            for i, d in enumerate(dets):
                if i in used or d["name"] != t["name"]:
                    continue
                v = iou(t["box"], d["box"])
                if v >= self.iou_thr and (best is None or v > best[0]):
                    best = (v, i)
            if best is not None:
                d = dets[best[1]]
                used.add(best[1])
                t["box"] = self.alpha * d["box"] + (1 - self.alpha) * t["box"]
                t["conf"], t["last"] = d["conf"], now
                t["hits"] += 1
        for i, d in enumerate(dets):
            if i not in used:
                self.tracks.append(dict(id=self._next_id, name=d["name"], conf=d["conf"],
                                        box=d["box"].copy(), last=now, hits=1))
                self._next_id += 1
        self.tracks = [t for t in self.tracks if now - t["last"] <= self.max_age]
        return [t for t in self.tracks if t["hits"] >= self.min_hits]


# ====================================================================== GEOMETRY
def box_width(box):
    return float(box[2] - box[0])


def horizontal_gaps(boxes):
    """boxes: list (x1,y1,x2,y2). Tra ve list (i_trai, i_phai, gap_px) cho cac cap lien ke theo truc ngang."""
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    gaps = []
    for a, b in zip(order, order[1:]):
        gaps.append((a, b, max(0.0, float(boxes[b][0] - boxes[a][2]))))
    return gaps


# ====================================================================== ROBOFLOW
def _project(api_key, workspace, project_name):
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    return ws.project(project_name)


def upload_images(api_key, workspace, project_name, image_paths, batch_name, progress_cb=None):
    """Day anh len Roboflow (80% train / 20% valid). Tra ve so anh upload thanh cong."""
    project = _project(api_key, workspace, project_name)
    ok = 0
    for i, p in enumerate(image_paths):
        split = "valid" if i % 5 == 0 else "train"
        try:
            if hasattr(project, "upload"):
                project.upload(image_path=p, split=split, batch_name=batch_name)
            else:
                project.upload_image(p, split=split, batch_name=batch_name)
            ok += 1
        except Exception as e:
            print("upload loi:", p, e)
        if progress_cb:
            progress_cb(i + 1, len(image_paths))
    return ok


def _fix_yaml(location):
    """Sua data.yaml cua Roboflow: dung duong dan tuyet doi, tranh loi '../train/images'."""
    import yaml
    yaml_path = os.path.join(location, "data.yaml")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data["path"] = os.path.abspath(location)
    for key, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        if os.path.isdir(os.path.join(location, folder, "images")):
            data[key] = f"{folder}/images"
        elif key in data:
            data.pop(key)
    if "val" not in data and "train" in data:
        data["val"] = data["train"]
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)
    return yaml_path


def download_dataset(api_key, workspace, project_name, version, location="roboflow_dataset"):
    """Tai version da generate tren Roboflow o dinh dang YOLOv8. Tra ve duong dan data.yaml."""
    project = _project(api_key, workspace, project_name)
    ds = project.version(int(version)).download("yolov8", location=location, overwrite=True)
    return _fix_yaml(ds.location)


# ====================================================================== TRAIN
def train(data_yaml, out_weights="weights/best.pt", base="yolov8n.pt", epochs=50, imgsz=640,
          batch=8, progress_cb=None):
    """Fine-tune YOLOv8 tren dataset Roboflow, copy best.pt vao out_weights. Tra ve out_weights."""
    from ultralytics import YOLO
    model = YOLO(base)
    if progress_cb:
        model.add_callback("on_train_epoch_end",
                           lambda tr: progress_cb(tr.epoch + 1, tr.epochs))
    model.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch,
                project=os.path.abspath("runs"), name="smartpark", exist_ok=True,
                workers=0, patience=20, verbose=False)
    best = str(model.trainer.best)
    os.makedirs(os.path.dirname(os.path.abspath(out_weights)), exist_ok=True)
    shutil.copy(best, out_weights)
    return out_weights