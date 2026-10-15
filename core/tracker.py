"""Cấp ID và theo dõi trạng thái xe (vào / đỗ / rời chỗ / rời bãi) theo toạ độ mm trên mặt phẳng bãi."""
from collections import deque

import numpy as np

from core.lot import gate_at, in_row


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
    HOLD_MM = 15.0          # xe đang đỗ: phát hiện lệch <= mức này vẫn là xe đó đứng yên (tay/bóng làm mask lệch)
    STILL_MM = 5.0          # đứng yên nếu tâm không lệch quá mức này
    MOVE_GATE_MM = 70.0     # xe đang chạy: ghép với phát hiện trong bán kính này (theo khung liền kề)
    PARK_PICK_MM = 45.0     # xe đang đỗ: phát hiện lệch 15..45 mm quanh chỗ đỗ = xe đang bị dời đi
    DUP_MM = 25.0           # phát hiện thừa có tâm gần xe đã có hơn mức này = mảnh tay/bóng dính vào xe, bỏ
    CAND_MM = 40.0          # ứng viên xe mới: ghép giữa các khung trong bán kính này
    CAND_HITS = 4           # ứng viên phải được thấy (với hình giống xe) ở >= ngần này khung YOLO ...
    CAND_S = 0.5            # ... tồn tại >= ngần này giây ...
    CAND_RATIO = 0.7        # ... và có mặt ở >= 70% số khung kể từ lúc xuất hiện thì mới cấp ID (tay chập chờn thì không)
    CAND_TTL = 0.6          # ứng viên không thấy lại quá ngần này giây thì bỏ

    def __init__(self, park_s, lost_s, lost_park_s, startup_s, log_event):
        self.park_s, self.lost_s, self.lost_park_s, self.startup_s = park_s, lost_s, lost_park_s, startup_s
        self.cars, self.next_id, self.t0 = {}, 1, None
        self.cands = []                            # ứng viên xe mới chưa cấp ID: {c, det, hits, t0, t}
        self.log = log_event                       # log(xe, su_kien, cong, ghi_chu)
        self.n_in = self.n_out = 0

    def _new(self, det, now, t_first=None, gate=None):
        if self.t0 is None:
            self.t0 = now
        g = gate or gate_at(det["c"])           # gate: ô cam nơi ứng viên được thấy lần đầu (nếu có)
        t_first = now if t_first is None else t_first
        origin = "co_san" if t_first - self.t0 < self.startup_s and g is None else (g or "xuat_hien")
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
        car_ok = lambda j: dets[j].get("car_ok", True)    # False: hình không giống xe (bóng tay/người)
        matched = set()
        # 0) xe đang đỗ: phát hiện lệch ít (tay che / bóng đổ làm mask xê dịch) -> vẫn đứng yên, giữ box đã lưu.
        #    Làm TRƯỚC để xe khác (hay "xe ma" do tay) không giành mất phát hiện của xe đang đỗ.
        parked = [c for c in cars if c.state == "da_do"]
        for car, j in take([(c, j, dist(c, j)) for c in parked for j in free], self.HOLD_MM):
            matched.add(car.id)
            car.t_seen = now
        # 1) xe đang chạy có phát hiện gần như trùng chỗ
        moving = [c for c in cars if c.id not in matched and c.state == "di_chuyen"]
        for car, j in take([(c, j, dist(c, j)) for c in moving for j in free], self.TIGHT_MM):
            matched.add(car.id)
            car.update(dets[j], now)
        # 2) xe đang chạy (vừa thấy < 1 s) trong bán kính MOVE_GATE_MM và xe đang đỗ bị lệch HOLD..PARK_PICK mm
        #    ghép CHUNG theo khoảng cách: phát hiện thuộc về xe gần nhất, không ưu tiên xe đang chạy nữa
        pairs = [(c, j, dist(c, j)) for c in cars if c.id not in matched for j in free
                 if (c.state == "di_chuyen" and now - c.t_seen < 1.0 and dist(c, j) <= self.MOVE_GATE_MM)
                 or (c.state == "da_do" and car_ok(j) and dist(c, j) <= self.PARK_PICK_MM)]
        for car, j in take(pairs, self.MOVE_GATE_MM):
            matched.add(car.id)
            if car.state == "da_do":
                self._start_moving(car, dets[j], now, "xe đang đỗ bị dời khỏi chỗ")
            else:
                car.update(dets[j], now)
        # 3) phát hiện còn thừa: nhận lại xe vừa mất dấu gần đó (bán kính tăng theo thời gian mất dấu).
        #    Chỉ nhận phát hiện có hình giống xe, để bóng tay đi ngang không "kéo" xe đã mất dấu theo.
        lost = [c for c in cars if c.id not in matched]
        pairs = []
        for c in lost:
            r = min(250.0, 60.0 + 150.0 * (now - c.t_seen))
            pairs += [(c, j, dist(c, j)) for j in free if car_ok(j) and dist(c, j) <= r]
        for car, j in take(pairs, 250.0):
            matched.add(car.id)
            if car.state == "da_do":
                self._start_moving(car, dets[j], now, "xe đang đỗ được nhấc lên đặt chỗ khác")
            else:
                car.update(dets[j], now)
        # 4) phát hiện không thuộc xe nào: bỏ phát hiện có hình không giống xe (bóng tay/người: méo, dài, lõm)
        #    và mảnh dính sát xe đã có (2 xe thật không thể cách tâm < DUP_MM); phần còn lại thành ứng viên,
        #    chỉ cấp ID khi ứng viên được thấy đủ CAND_HITS khung / CAND_S giây
        rest = [j for j in free if car_ok(j) and all(dist(c, j) > self.DUP_MM for c in self.cars.values())]
        used = set()
        for j in rest:
            c_j = dets[j]["c"]
            best = min((k for k in range(len(self.cands)) if k not in used),
                       key=lambda k: np.hypot(*(self.cands[k]["c"] - c_j)), default=None)
            if best is not None and np.hypot(*(self.cands[best]["c"] - c_j)) <= self.CAND_MM:
                cd = self.cands[best]
                cd.update(c=c_j, det=dets[j], t=now, hits=cd["hits"] + 1, gate=cd["gate"] or gate_at(c_j))
            else:
                self.cands.append({"c": c_j, "det": dets[j], "hits": 1, "n": 0, "t0": now, "t": now,
                                   "gate": gate_at(c_j)})
                best = len(self.cands) - 1
            used.add(best)
        keep = []
        for cd in self.cands:
            cd["n"] += 1                                    # số khung YOLO kể từ lúc ứng viên xuất hiện
            ratio = cd["hits"] / cd["n"]
            if cd["t"] == now and cd["hits"] >= self.CAND_HITS and now - cd["t0"] >= self.CAND_S \
                    and ratio >= self.CAND_RATIO:
                self._new(cd["det"], now, cd["t0"], cd["gate"])
            elif now - cd["t"] <= self.CAND_TTL and (cd["n"] < 5 or ratio >= 0.5):
                keep.append(cd)
        self.cands = keep
        # 5) chuyển trạng thái
        for car in list(self.cars.values()):
            gone = now - car.t_seen
            if car.state == "di_chuyen":
                while car.hist and now - car.hist[0][0] > self.park_s + 2:
                    car.hist.popleft()
                if car.id in matched and in_row(car.c) is not None and \
                        car.still_for(now, self.STILL_MM) >= self.park_s:
                    car.state, car.t_park = "da_do", now
                    self.log(car, "do", "", f"đứng yên {self.park_s:.0f} s trong dãy {'trên' if in_row(car.c) == 0 else 'dưới'}")
                # xe chỉ rời bãi qua ô cam: mất dấu giữa bãi (tay cầm che) thì chờ lâu như xe đang đỗ
                elif gone > (self.lost_s if gate_at(car.c, margin=25.0) else self.lost_park_s):
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
        for cd in self.cands:
            cd["t0"] += dt
            cd["t"] += dt
        for car in self.cars.values():
            car.t_seen += dt
            car.hist = deque((t + dt, c) for t, c in car.hist)
        if self.t0 is not None:
            self.t0 += dt

    def still_now(self, car, now, s=1.0):
        return car.state == "da_do" or car.still_for(now, self.STILL_MM) >= s
