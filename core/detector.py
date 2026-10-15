"""Xử lý 1 khung hình (khoá giấy -> YOLO -> tracker -> chỗ trống) và thread chạy liên tục trên stream."""
import threading
import time
import traceback

import cv2
import numpy as np

from core.lot import LotLock, to_img
from core.rate import Rate
from core.tracker import ParkingRegistry
from core.vision import REC_PX_MM, car_like, car_polygon_ok, car_shape, rect_map
from helpers.find_spaces import S, find_spaces


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
        self.n_reject = 0                      # số phát hiện bị loại vì không giống xe ở khung gần nhất

    def _note(self, text, kind="he_thong"):
        """Ghi một dòng thông báo hệ thống vào nhật ký (nếu nhật ký hỗ trợ)."""
        note = getattr(self.reg.log, "note", None)
        if note:
            note(text, kind)

    def _detect(self, frame, H):
        """YOLO trên ảnh nắn thẳng -> list xe {quad, hull, c} theo mm trên mặt phẳng bãi."""
        if self._rmap is None or self._rmap[0] is not H:
            self._rmap = (H,) + rect_map(H)
        _, Hr, _, size, lo = self._rmap
        rect = cv2.warpPerspective(frame, Hr, size, flags=cv2.INTER_LINEAR)
        r = self.model.predict(rect, conf=self.a.conf, verbose=False, **self.kw)[0]
        dets, self.n_reject = [], 0
        if r.boxes is not None and len(r.boxes):
            segs = r.masks.xy if r.masks is not None else []
            confs = r.boxes.conf.cpu().numpy() if getattr(r.boxes, "conf", None) is not None else []
            for i, box in enumerate(r.boxes.xyxy.cpu().numpy()):
                seg = segs[i] if i < len(segs) else None
                hull, quad = car_shape(seg, box)
                q = quad / REC_PX_MM + lo
                if not car_like(q):
                    self.n_reject += 1
                    continue
                ok = car_polygon_ok(seg, float(confs[i]) if i < len(confs) else 1.0)
                self.n_reject += not ok
                # car_ok=False: hình không giống xe (bóng tay/người...) -> chỉ dùng để giữ xe đã có ID, không cấp ID mới
                dets.append({"quad": q, "hull": hull / REC_PX_MM + lo, "c": q.mean(0), "car_ok": ok})
        return dets

    def step(self, frame):
        self.n += 1
        paper, now = self.paper, time.time()
        prev_state = paper.state
        if paper.update(frame) == "moi" and self.n > 1:
            # toạ độ xe lưu theo mm trên mặt phẳng bãi nên vẫn đúng sau khi khoá lại, không cần xoá
            self.spaces, self.t_space = [], 0.0
            print("Da ve lai khung bai theo vi tri moi cua to giay")
            self._note("Đã khoá lại khung bãi theo vị trí mới của tờ giấy")
        if prev_state == "khoa" and paper.state == "cho":
            print("Giay/may xe dich -> tam dung detect, cho on dinh...")
            self._note("Tờ giấy hoặc camera bị xê dịch, tạm dừng nhận diện để chờ ổn định", "canh_bao")
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
                self.spaces = [{"slot_mm": np.float32(sp["gap_px"]) / s, "gap_mm": sp["gap_mm"],
                                "capacity": sp["capacity"], "row": sp["row"]} for sp in res["spaces"]]
                self.t_space = time.time()
        self.det_rate.tick()
        self.snap = {
            "cars": [{"id": c.id, "state": c.state, "origin": c.origin, "hull": c.hull,
                      "trail": list(c.trail) if c.state == "di_chuyen" else [],
                      "seen": now - c.t_seen < 0.5} for c in self.reg.cars.values()],
            "spaces": self.spaces, "state": paper.state,
            "q": None if paper.q is None else paper.q.copy(), "H": H,
            "n_in": self.reg.n_in, "n_out": self.reg.n_out, "det_fps": self.det_rate.get(),
            "n_reject": self.n_reject,
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
