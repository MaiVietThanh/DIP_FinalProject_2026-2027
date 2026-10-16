# Smart Unmarked Parking Assistant – Bãi đỗ xe không kẻ vạch

Đồ án cuối kỳ môn Xử lý ảnh số (DIP), HCMUTE, 2026-2027.

## 1. Tổng quan về đề tài

Mô hình bãi đỗ xe in trên giấy A3, **không kẻ vạch ô đỗ** (giống bãi giữ xe máy): gồm 2 dãy đỗ dài (~359 × 87 mm) ngăn bởi một vạch trắng và 2 ô cổng màu cam (vào/ra) ở hai góc đối diện. Xe là ~15–20 xe mô hình (khoảng 30 × 67 mm).

Một điện thoại đặt trên tripod nhìn **thẳng từ trên xuống**, phát hình qua IP Webcam (MJPEG). Hệ thống:

- Phát hiện xe bằng **YOLOv8s-seg** (1 lớp `car`), chạy trên ảnh tờ giấy đã nắn phối cảnh.
- Dùng **xử lý ảnh cổ điển** để tìm tờ giấy, các dãy và ô cổng, tính **homography** đưa ảnh về mặt phẳng tính bằng mm.
- **Đo khe hở** giữa các xe trong từng dãy và **tìm chỗ trống** đủ cho một xe bằng phép toán hình thái học (mask, opening, connected components, erosion với phần tử cấu trúc cỡ một xe).
- **Theo dõi xe** qua video (bộ ParkingRegistry tự viết, ghép ID theo khoảng cách tâm tính bằng mm). Hệ thống ghi các sự kiện vào bãi, đã đỗ, rời chỗ và rời bãi ra file CSV, đồng thời hiển thị trên giao diện dashboard.

Kết quả chính trên tập TEST: mask mAP50 = 0,989, recall = 1,00; đo khe hở 42/42 khe với sai số trung bình (MAE) khoảng 1,5 mm.

## 2. Cấu trúc đề tài

```
DIP_FinalProject_2026-2027/
├── main.py            # Chương trình chính: đọc stream, phát hiện, theo dõi, hiển thị
├── core/              # Các module của main.py
│   ├── config.py      #   Hằng số (Full HD, màu sắc...)
│   ├── camera.py      #   Đọc stream IP Webcam (MJPEG, tùy chọn --h264)
│   ├── lot.py         #   Khóa tờ giấy, kích thước bãi, dãy và ô cổng
│   ├── vision.py      #   Nạp model, nắn ảnh, lọc hình dạng xe
│   ├── detector.py    #   Luồng chạy YOLO (GPU, FP16)
│   ├── tracker.py     #   ParkingRegistry: gán ID, trạng thái xe
│   ├── drawing.py     #   Vẽ polygon xe và chỗ trống lên video
│   ├── event_log.py   #   Ghi sự kiện ra CSV
│   ├── ui.py, panels.py  # Giao diện dashboard (Pillow, tiếng Việt)
│   └── rate.py        #   Đo FPS
├── helpers/           # Xử lý ảnh tĩnh: tìm bãi, đo khe hở, tìm chỗ trống (find_spaces.py...)
├── weights/best.pt    # Trọng số YOLOv8s-seg đã huấn luyện
├── assets/ui_bg.png   # Nền giao diện 1920×1080
├── old/               # Các phiên bản cũ để đối chiếu
└── README.md
```

Chạy chương trình (cần GPU CUDA, điện thoại bật IP Webcam ở Full HD):

```bash
uv run ./main.py
```

Phím `F` để bật/tắt toàn màn hình.

## 3. Các công cụ sử dụng

| Nhóm | Công cụ |
|---|---|
| Ngôn ngữ, môi trường | Python, `uv` |
| Xử lý ảnh | OpenCV, NumPy, Pillow (chữ tiếng Việt trên giao diện) |
| Học sâu | Ultralytics YOLOv8s-seg 8.4, PyTorch 2.11 (CUDA 12.8, FP16) |
| Gán nhãn, dữ liệu | Roboflow (Auto Label SAM 3, rồi duyệt tay từng ảnh) |
| Huấn luyện | Google Colab (GPU T4), khoảng 10 phút |
| Phần cứng | Laptop RTX 3050 6 GB; điện thoại Xiaomi 11T 5G + tripod; app IP Webcam |

## 4. Mô tả về dataset

Link: **https://universe.roboflow.com/le-phat-a8j7p/baixe-dip**

- Ảnh chụp trực tiếp mô hình bãi xe bằng Xiaomi 11T từ trên xuống, độ phân giải 4000 × 3000.
- 5 điều kiện ánh sáng L1–L5 (L5 rất tối), xe đặt tự do ở nhiều vị trí, có cả ảnh bãi trống.
- 1 lớp duy nhất `car`, nhãn dạng polygon (segmentation).
- Chia tập: **train 69** (gồm 10 ảnh trống), **valid 15**, **test 22**. Tập test có các khe hở chuẩn 20/38/60 mm để đo sai số và không dùng để huấn luyện.
- Tiền xử lý trên Roboflow: auto-orient, Fit 1024 (viền đen). Tăng cường ×3: lật ngang và dọc, xoay ±10°, độ sáng ±25%, phơi sáng ±15%.
- Kết quả trên TEST (22 ảnh, 208 xe): box mAP50 0,989; mask mAP50 0,989; mask mAP50-95 0,970; precision 0,985; recall 1,00.
