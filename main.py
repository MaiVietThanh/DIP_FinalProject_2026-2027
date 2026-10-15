"""Giai đoạn 5: theo dõi xe vào / đỗ / rời bãi trên stream IP Webcam (Full HD), chạy YOLO bằng GPU.

Chạy:   uv run ./main.py                                   (URL ở STREAM_URL)
        uv run ./main.py --source http://<ip>:8080/video   (nhập URL trực tiếp)
Tùy chọn: --model weights/best.pt  --conf 0.5  --imgsz 640  --pad 5  --do_yen 10  --mat_xe 3  --mat_xe_do 8
          --h264 (thử RTSP H.264 trước rồi mới MJPEG)  --chat_luong 80

Cấu trúc thư mục:
  main.py            đọc tham số, mở stream, chạy vòng lặp cửa sổ chính (file này)
  core/config.py     hằng số dùng chung (Full HD, màu vẽ)
  core/rate.py       đo FPS
  core/camera.py     IP Webcam: đặt Full HD, StreamReader đọc khung mới nhất, danh sách URL thử
  core/lot.py        tờ giấy -> mặt phẳng bãi (mm): paper_quad, LotLock, PRINT_MM, GATES_MM, in_row, gate_at
  core/vision.py     YOLO: load_model (GPU FP16), rect_map (ảnh nắn thẳng), car_shape, car_like
  core/tracker.py    Car, ParkingRegistry (cấp ID, sự kiện vao / do / roi_cho / roi_bai)
  core/detector.py   Detector (xử lý 1 khung) và DetectWorker (thread chạy YOLO)
  core/drawing.py    vẽ kết quả (xe, chỗ trống, dải đỗ) lên khung hình video
  core/ui.py         giao diện 1920x1080 trên nền assets/ui_bg.png: ô video, hai cột thông số, khung log cuộn được
  core/panels.py     nội dung hai cột thông số
  core/event_log.py  ghi nhật ký sự kiện ra CSV + câu tiếng Việt cho khung log
  assets/ui_bg.png   ảnh nền giao diện (Canva)
  helpers/           thuật toán ảnh tĩnh giai đoạn 1-4 (detect_lot, measure_gaps, find_spaces)
  weights/best.pt    model YOLOv8s-seg

Stream: mặc định đọc MJPEG (/video, cùng luồng với web player của IP Webcam) vì đã đủ FPS và ổn định.
--h264: thử RTSP H.264 trước (rtsp://<ip>:8080/h264_ulaw.sdp rồi h264.sdp), không được thì về MJPEG.
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
Phím tắt: q = thoát, space = tạm dừng, f = bật/tắt toàn màn hình. Con lăn chuột trên khung log để xem dòng cũ.
Kết quả: tracking_result/<ngày_giờ>.csv (nhật ký mỗi lần chạy).
"""
import argparse
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))   # thư mục chứa main.py
if HERE not in sys.path:   # để import core/ và helpers/ dù chạy main.py từ thư mục nào
    sys.path.insert(0, HERE)

from core.camera import StreamReader, ipcam_full_hd, stream_urls
from core.config import FULL_HD
from core.detector import DetectWorker, Detector
from core.drawing import draw_overlay
from core.event_log import EventLog
from core.panels import left_items, right_items
from core.rate import Rate
from core.ui import Dashboard
from core.vision import load_model

PHIEN_BAN = "2026-10-15 v6.1 (lọc bóng tay/người theo hình mask trước khi cấp ID)"   # in ra khi chạy để biết đang chạy bản nào
STREAM_URL = "http://192.168.1.9:8080/video"   # <-- URL IP Webcam (có thể dạng http://user:pass@ip:8080/video)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=STREAM_URL, help="URL IP Webcam, vd http://<ip>:8080/video")
    ap.add_argument("--h264", action="store_true", help="thử RTSP H.264 trước, không được mới đọc MJPEG /video (mặc định: chỉ MJPEG)")
    ap.add_argument("--model", default=os.path.join(HERE, "weights", "best.pt"))
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640, help="cỡ ảnh đưa vào YOLO (ảnh nắn ~1320x940 px)")
    ap.add_argument("--pad", type=float, default=5, help="khoảng đệm quanh xe khi tìm chỗ trống (mm)")
    ap.add_argument("--out", default=os.path.join(HERE, "tracking_result"))
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
    return a


def wait_first_frame(reader, timeout=60):
    t0 = time.time()
    while True:
        frame, _ = reader.get()
        if frame is not None:
            return frame
        if time.time() - t0 > timeout:
            reader.running = False
            sys.exit(f"Không đọc được khung hình nào: {reader.info}")
        time.sleep(0.05)


def main():
    a = parse_args()
    # Luồng YOLO giữ GIL của Python lâu; đổi luồng thường hơn để luồng hiển thị không bị chậm.
    sys.setswitchinterval(0.0005)
    print(f"main.py phien ban: {PHIEN_BAN}")
    model, kw, gpu = load_model(a)
    print(f"YOLO: GPU {gpu} (FP16)")
    print(ipcam_full_hd(a.source, a.chat_luong))
    reader = StreamReader(stream_urls(a.source, a.h264))
    reader.start()

    frame = wait_first_frame(reader)
    h0, w0 = frame.shape[:2]
    win = "Smart Unmarked Parking Assistant"
    ui = Dashboard(os.path.join(HERE, "assets", "ui_bg.png"))
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1440, 810)
    cv2.setMouseCallback(win, ui.on_mouse)
    lw = max(2, w0 // 640)
    print(f"Khung hinh {w0}x{h0}" + ("" if (w0, h0) == FULL_HD else " - CHUA PHAI FULL HD!"))
    cv2.imwrite(os.path.join(a.out, "khung_dau.jpg"), frame)   # lưu khung đầu để kiểm tra góc máy

    log = EventLog(a.out)                        # nhật ký sự kiện: mỗi lần chạy 1 file riêng
    print(f"Nhat ky su kien: {log.path}")
    det = Detector(model, kw, a, log)
    log.frame_no = lambda: det.n
    det.paper.update(frame)
    det.paper.update(frame)                      # 2 khung giống nhau = đứng yên -> thử tìm ngay
    print("To giay: " + ("tim thay 4 goc, da khoa khung bai" if det.paper.state == "khoa" else "CHUA THAY, se tim lai moi 1 s"))
    log.note("Bắt đầu theo dõi bãi xe")
    log.note("Đã khoá khung bãi theo tờ giấy" if det.paper.state == "khoa" else
             "Chưa thấy tờ giấy, đang tìm lại mỗi giây", "he_thong" if det.paper.state == "khoa" else "canh_bao")

    worker = DetectWorker(det, reader)
    worker.start()
    shown, paused, show_rate = -1, False, Rate()
    t_start, t_panel, last_info, video = time.time(), 0.0, "Stream OK", None
    fullscreen = False
    try:
        while True:
            frame, fid = reader.get()
            snap = det.snap
            new = not paused and frame is not None and fid != shown
            if new:
                shown = fid
                show_rate.tick()
                video = draw_overlay(frame.copy(), snap, None, lw)
            if reader.info != last_info:         # mất / có lại kết nối -> ghi vào log
                last_info = reader.info
                log.note("Kết nối stream ổn định" if last_info == "Stream OK" else last_info,
                         "he_thong" if last_info == "Stream OK" else "canh_bao")
            if time.time() - t_panel > 0.5:     # hai cột thông số: cập nhật 2 lần/giây là đủ
                t_panel = time.time()
                state = snap["state"] if snap else det.paper.state
                ui.set_panel("left", left_items(snap, state, (w0, h0), reader.rate.get(), show_rate.get(),
                                                gpu, reader.info, paused, time.time() - t_start))
                ui.set_panel("right", right_items(snap))
            ui.set_log(*log.entries())
            if new or ui.dirty:                  # chỉ vẽ lại cửa sổ khi có khung mới hoặc chữ đổi
                cv2.imshow(win, ui.compose(video))
            k = cv2.waitKey(1 if new else 5) & 0xFF
            if k == ord("q") or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            if k == ord(" "):
                paused = worker.paused = not paused
                log.note("Tạm dừng" if paused else "Tiếp tục theo dõi")
                t_panel = 0.0
            if k == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    except KeyboardInterrupt:
        pass
    finally:
        worker.running = False
        worker.join(timeout=2)
        reader.running = False
        log.close()
        cv2.destroyAllWindows()
    print(f"Xe da vao: {det.reg.n_in}, da roi: {det.reg.n_out}, dang do: "
          f"{sum(c.state == 'da_do' for c in det.reg.cars.values())}. Nhat ky: {log.path}")


if __name__ == "__main__":
    main()
