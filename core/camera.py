"""Đọc stream IP Webcam: đặt Full HD qua HTTP API, thread đọc khung mới nhất, danh sách URL thử."""
import base64
import json
import os
import threading
import time
import urllib.request
from urllib.parse import unquote, urlsplit, urlunsplit

import cv2

from core.config import FULL_HD
from core.rate import Rate


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


def stream_urls(url, h264=False):
    """Danh sách URL thử: RTSP H.264 + ulaw rồi H.264 qua TCP (chỉ khi --h264); sau đó MJPEG.
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
