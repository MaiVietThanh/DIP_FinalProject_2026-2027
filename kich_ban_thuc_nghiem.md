# Kịch bản thực nghiệm cho Phần 4 – 5

Tài liệu này chia thực nghiệm thành 3 nhóm:

- **Nhóm A:** Claude tự chạy trên ảnh TEST. Nhóm không cần làm gì.
- **Nhóm B:** lấy kết quả huấn luyện có sẵn trên Google Drive và Roboflow.
- **Nhóm C:** nhóm chạy `main.py` trên video thật rồi chụp màn hình.

Nhóm chỉ cần lo Nhóm B và Nhóm C.

**Nơi lưu file:** tạo thư mục `anh_thuc_nghiem\` trong `DIP_FinalProject_2026-2027`. Đặt tên ảnh theo mã kịch bản, ví dụ `C2_1_o_cong.png`, `C2_2_dang_di.png`. File nhật ký CSV của mỗi lần chạy tự lưu trong `tracking_result\`. Nhóm ghi lại lần chạy nào ứng với kịch bản nào, ví dụ `C2 → 20261015_101500.csv`. Mình đọc được thẳng hai thư mục này nên không cần gửi lại.

**Cách chụp màn hình:** dùng `Win + Shift + S` và chụp nguyên cửa sổ "Giai doan 5 - theo doi xe", để thấy cả dòng trạng thái FPS ở trên. Với các bước đang thao tác tay (kịch bản C2, C3, C5), nên quay thêm màn hình bằng `Win + Alt + R` (Xbox Game Bar), rồi mình cắt khung hình từ video.

---

## 0. Việc làm trước

| # | Việc | Ghi lại |
|---|------|---------|
| 0.1 | Đo bằng thước 2–3 xe mô hình: bề ngang (tính cả bánh xe) và chiều dài | Ví dụ: 34 × 69 mm. Số này quyết định xe danh nghĩa 28 mm trong báo cáo có đúng không |
| 0.2 | Chụp 1 ảnh giá đỡ camera nhìn từ bên cạnh: chồng sách, tấm bìa, điện thoại, đèn | `0_gia_camera.jpg`, thay hình vẽ minh hoạ ở Hình 3.3 |
| 0.3 | Ghi cấu hình máy: CPU, GPU, RAM, tên điện thoại, phiên bản Python / torch / ultralytics | Lấy từ dòng đầu khi chạy `main.py` (in tên GPU) và lệnh `uv pip list` |

---

## Nhóm A. Claude tự chạy trên ảnh TEST

Phần này không cần nhóm làm gì. Nhóm chỉ cần trả lời "ok" là mình chạy.

| Mã | Thực nghiệm | Kết quả đưa vào báo cáo |
|----|-------------|-------------------------|
| A1 | `measure_gaps.py` trên 20 ảnh TEST | Bảng sai số theo cỡ thẻ 20 / 38 / 60 mm (MAE, RMSE, sai số lớn nhất, độ lệch trung bình), biểu đồ đo được so với đáp án, số lần kết luận đúng/sai "đỗ được", ảnh minh hoạ ca đúng và ca sai |
| A2 | `find_spaces.py` trên 20 ảnh TEST | Tỉ lệ khe 20 / 38 / 60 mm được gợi ý, so sánh quy tắc 38 mm với quy tắc thận trọng ~47 mm |
| A3 | Sức chứa trên ảnh bãi trống (`TEST_empty_1`) | Số chỗ gợi ý mỗi dãy (đã biết là 8 xe/dãy) |
| A4 | Ảnh xe vẫn được phân đoạn đúng ở ánh sáng L1–L5 (lấy từ tập valid) | Lưới 5 ảnh, mỗi ảnh một điều kiện chiếu sáng |
| A5 | Ca khó: xe xéo, xe chạm nhau, xe trắng sát vạch, xe che vạch phải dùng viền khung | 4–6 ảnh có chú thích |
| A6 | Thời gian xử lý từng bước của chế độ ảnh tĩnh | Bảng thời gian (ms) |

---

## Nhóm B. Kết quả huấn luyện (lấy từ Drive `MyDrive/dip_baixe/` và Roboflow)

| Mã | Lấy file / chụp gì | Dùng cho |
|----|--------------------|----------|
| B1 | `train/results.png`, `train/MaskPR_curve.png`, `train/BoxPR_curve.png`, `train/confusion_matrix.png`, `train/val_batch0_pred.jpg`, `train/args.yaml` | Đường cong loss và mAP, tham số huấn luyện |
| B2 | Thư mục `test_eval/` (các file `*.png`). Chụp màn hình output của Cell 7 (dòng Box/Mask mAP) | Bảng chỉ số trên tập TEST |
| B3 | 3–4 ảnh trong `test_pred/` | Ảnh mặt nạ dự đoán |
| B4 | Roboflow: trang **Versions**, chụp màn hình phần Preprocessing và Augmentation của version đã train, cùng trang **Health Check** (số ảnh, số nhãn) | Xác nhận Bảng 3.4 và thống kê dataset |
| B5 | Thời gian train: dòng cuối của Cell 5 (`xx epochs completed in x.xxx hours`) | Bảng cấu hình huấn luyện |

Chép tất cả vào `anh_thuc_nghiem\B\`.

---

## Nhóm C. Thực nghiệm video với `main.py`

Thiết lập chung cho mọi kịch bản:

- Đèn L1, giá camera cố định, điện thoại đã mở IP Webcam ở Full HD.
- Chạy `uv run ./main.py` và chờ dòng trạng thái không còn chữ "DANG TIM TO GIAY".
- Mỗi kịch bản là **một lần chạy riêng**, để mỗi kịch bản có một file CSV riêng. Thoát bằng phím `q`.
- Với các kịch bản có cột "Lặp", lặp lại thao tác trong cùng lần chạy rồi đếm số lần đúng/sai.

### C1. Khởi động với bãi có sẵn xe

1. Xếp sẵn khoảng 10 xe vào hai dãy, chừa ít nhất 2 khe rộng.
2. Chạy chương trình và chờ khoảng 15 giây.
3. **Chụp** `C1_1`: tất cả xe có nhãn `ID x / DO`, màu xanh dương, và các khung xanh lá ghi độ rộng khe.
4. Đo bằng thước 2–3 khe có khung xanh, ghi số mm thật để so với số trên màn hình.

**Ghi lại:** số xe thật, số xe hệ thống thấy, số chỗ trống hiển thị, mm thật và mm hiển thị của các khe đã đo.

### C2. Xe vào bãi qua ô cổng rồi đỗ (lặp 5 lần, xen kẽ cổng trái và cổng phải)

1. Đặt 1 xe vào ô cam phải. **Chụp** `C2_1`: xe màu lục lam, nhãn `MOI`, `cong phai`.
2. Đẩy xe tới khe được gợi ý. **Chụp** `C2_2`: có vệt đường đi.
3. Để yên 10 giây. **Chụp** `C2_3`: xe chuyển sang `DO`, khung xanh của khe đó biến mất hoặc nhỏ lại.

**Ghi lại** cho mỗi lần: có sự kiện `vao` đúng cổng không, xe có giữ một ID suốt quá trình không, có sự kiện `do` không, chỗ trống có cập nhật đúng không.

### C3. Xe rời chỗ rồi ra khỏi bãi (lặp 5 lần)

1. Đẩy một xe đang đỗ ra lối đi. Nhãn `DO` mất, sinh sự kiện `roi_cho`, ID giữ nguyên. **Chụp** `C3_1`.
2. Đẩy xe vào ô cam rồi nhấc xe khỏi giấy. Sau khoảng 3 giây sinh sự kiện `roi_bai` kèm tên cổng. **Chụp** `C3_2`: bộ đếm "da roi" tăng 1, khe cũ hiện khung xanh.

**Ghi lại:** đúng/sai của từng sự kiện, thời gian từ lúc nhấc xe tới lúc có `roi_bai`.

### C4. Tay và bóng tay (lặp mỗi thao tác 5 lần)

1. Đặt bàn tay úp lên một xe đang đỗ 3–5 giây rồi rút tay ra. Kỳ vọng: ID giữ nguyên, không có ID mới. **Chụp** `C4_1` lúc đang che (nhãn `bi che`).
2. Quơ tay qua lại trên bãi khoảng 5 giây, không cầm xe. Kỳ vọng: không sinh ID mới (không có "xe ma").
3. Che một cạnh tờ giấy bằng cánh tay. Kỳ vọng: khung bãi không mất khoá.

**Ghi lại:** số lần sinh ID giả, số lần mất hoặc đổi ID.

### C5. Nhấc xe đặt sang chỗ khác (lặp 5 lần)

1. Cầm một xe đang đỗ ở dãy trên, nhấc lên khoảng 2 giây, đặt xuống một khe ở dãy dưới.
2. Kỳ vọng: cùng ID, các sự kiện `roi_cho` rồi `do` ở dãy dưới.
3. **Chụp** `C5_1` khi xe vừa đỗ xong ở vị trí mới.

**Ghi lại:** số lần giữ đúng ID.

### C6. Lấp đầy bãi từ trống tới kín

1. Bắt đầu với bãi trống. **Chụp** `C6_0`: mỗi dãy một khung xanh dài, ghi `fits 8 cars`.
2. Lần lượt đưa xe vào từ cổng và đỗ theo gợi ý, mỗi lần 1 xe. Sau mỗi 2 xe, **chụp** một ảnh (`C6_2`, `C6_4`, …).
3. Dừng khi `cho trong: 0`. **Chụp** `C6_end`.

**Ghi lại:** bảng "số xe đã đỗ, số chỗ trống, tổng sức chứa còn lại hiển thị". Tổng số xe đỗ được khi hết chỗ chính là sức chứa thực tế.

### C7. Tờ giấy hoặc camera bị xê dịch

1. Bãi có khoảng 6 xe đã `DO`. Đẩy cả tờ giấy lệch khoảng 2 cm (hoặc xoay nhẹ khoảng 10°).
2. **Chụp** `C7_1` khi dòng trạng thái hiện `GIAY/MAY XE DICH - CHO ON DINH`.
3. Để yên. **Chụp** `C7_2` sau khi khung khoá lại.

**Ghi lại:** thời gian khoá lại, các ID có giữ nguyên không, có sinh sự kiện `roi_bai` sai không.

### C8. Ánh sáng (chạy lại C1 ở mỗi điều kiện)

Chạy lại C1 lần lượt với L1, L3 (vàng dịu), L4 (chiếu chếch, có bóng) và L5 (tắt đèn bàn). **Chụp** `C8_L1`, `C8_L3`, `C8_L4`, `C8_L5`.

**Ghi lại:** số xe thấy được / số xe thật, có ID giả do bóng không, FPS YOLO.

### C9. Ca khó trên video

Xếp lần lượt 3 tình huống: 2 xe chạm nhau; 1 xe đỗ xéo khoảng 30°; xe trắng sát vạch trắng. **Chụp** `C9_1`, `C9_2`, `C9_3`.

**Ghi lại:** các xe có tách đúng, khe bên cạnh có được tính đúng không.

### C10. Tốc độ

1. Chạy C1 khoảng 2 phút, không thao tác.
2. **Chụp** 3 lần dòng trạng thái: FPS của stream, FPS hiển thị, FPS của YOLO.
3. Chạy `uv run ./helpers/do_stream.py` rồi chụp bảng FPS và Mbit/s ở các mức chất lượng 90 / 70 / 50 / 30.
4. Mở Task Manager, tab Performance, chụp mức sử dụng GPU khi `main.py` đang chạy.

### C11. Giao diện và hướng dẫn sử dụng

Chụp các ảnh: màn hình terminal lúc khởi động (dòng phiên bản, GPU, IP Webcam, "To giay: tim thay 4 goc"), cửa sổ chương trình với chú thích màu đầy đủ, ứng dụng IP Webcam trên điện thoại, và 1 file CSV mở bằng Excel.

---

## Bảng ghi kết quả cho Nhóm C (chép vào Excel hoặc ghi tay)

| Kịch bản | Số lần thử | Số lần đúng | Lỗi gặp phải (mô tả ngắn) | File CSV |
|----------|-----------:|------------:|---------------------------|----------|
| C2 vào → đỗ | 5 | | | |
| C3 rời chỗ → rời bãi | 5 | | | |
| C4 tay che xe đỗ | 5 | | | |
| C4 quơ tay (ID giả) | 5 | | | |
| C5 nhấc đặt chỗ khác | 5 | | | |
| C7 xê dịch giấy | 3 | | | |

Từ các ảnh, CSV và bảng này, mình sẽ viết Phần 4: thiết lập thực nghiệm, kết quả huấn luyện, độ chính xác đo, kết quả gợi ý chỗ trống, theo dõi trên video, tốc độ, giao diện và hướng dẫn chạy. Sau đó mình viết Phần 5: kết quả đạt được, hạn chế và hướng phát triển.
