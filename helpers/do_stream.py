"""Đo tốc độ stream MJPEG của IP Webcam ở vài mức chất lượng JPEG (không chạy YOLO).

Chạy:   uv run ./helpers/do_stream.py                       (URL lấy từ STREAM_URL trong main.py)
        uv run ./helpers/do_stream.py --source http://<ip>:8080/video --giay 6 --muc 90 70 50 30
Mỗi mức in ra: số khung/giây điện thoại gửi về và băng thông (Mbit/s).
  - FPS tăng rõ khi hạ chất lượng, Mbit/s gần như không đổi -> nghẽn ở Wi-Fi
  - FPS gần như không đổi dù hạ chất lượng                  -> nghẽn ở điện thoại (camera/bộ nén)
Cuối cùng trả chất lượng về như lúc đầu.
"""
import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # thư mục chứa main.py, helpers/, weights/
if ROOT not in sys.path:   # để chạy thẳng file này (python helpers/xxx.py) vẫn import được helpers.*
    sys.path.insert(0, ROOT)

from main import STREAM_URL, _request


def measure(url, secs):
    """Đọc thẳng byte MJPEG (không giải mã), đếm số khung JPEG và số byte trong secs giây."""
    n = nbytes = 0
    tail = b""
    with urllib.request.urlopen(_request(url), timeout=10) as r:
        t0 = time.time()
        while time.time() - t0 < secs:
            chunk = r.read(65536)
            if not chunk:
                break
            nbytes += len(chunk)
            buf = tail + chunk
            n += buf.count(b"\xff\xd8\xff")            # mỗi ảnh JPEG bắt đầu bằng FF D8 FF
            tail = buf[-2:]
        dt = time.time() - t0
    return n / dt, nbytes * 8 / dt / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=STREAM_URL)
    ap.add_argument("--giay", type=float, default=6, help="số giây đo cho mỗi mức")
    ap.add_argument("--muc", type=int, nargs="+", default=[90, 70, 50, 30], help="các mức chất lượng JPEG (%)")
    a = ap.parse_args()
    base = a.source.split("/video")[0].rstrip("/")
    with urllib.request.urlopen(_request(base + "/status.json"), timeout=8) as r:
        cur = json.loads(r.read().decode("utf-8", "ignore")).get("curvals", {})
    q0 = cur.get("quality")
    print(f"IP Webcam: {cur.get('video_size')}, chất lượng hiện tại {q0}%, FPS limit {cur.get('fps_limit')}")
    print(f"{'chat luong':>10} | {'FPS':>5} | {'Mbit/s':>6}")
    try:
        for q in a.muc:
            urllib.request.urlopen(_request(f"{base}/settings/quality?set={q}"), timeout=8).read()
            time.sleep(1.5)                                # chờ app áp dụng
            fps, mbps = measure(base + "/video", a.giay)
            print(f"{q:>9}% | {fps:5.1f} | {mbps:6.1f}")
    finally:
        if q0 is not None:
            urllib.request.urlopen(_request(f"{base}/settings/quality?set={q0}"), timeout=8).read()
            print(f"Đã trả chất lượng về {q0}%")


if __name__ == "__main__":
    main()
