"""Đo FPS thật theo cửa sổ thời gian."""
import time
from collections import deque


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
