"""Nội dung hai cột thông số hai bên video (gạch đầu dòng). Cách vẽ nằm ở core/ui.py."""
from core.config import BLUE, CYAN, GREEN, MAGENTA, RED, YELLOW
from core.ui import bgr

OK, WARN, BAD = (22, 163, 74), (202, 138, 4), (220, 38, 38)        # RGB
PAPER = {"khoa": ("Đã khoá khung bãi", OK), "cho": ("Xê dịch, chờ ổn định", WARN),
         "tim": ("Đang tìm tờ giấy", WARN)}


def hms(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def left_items(snap, paper_state, size, fps_stream, fps_show, gpu, stream_info, paused, uptime):
    """Cột trái: tình trạng hệ thống và chú thích màu."""
    w0, h0 = size
    fhd = (w0, h0) == (1920, 1080)
    paper_txt, paper_col = PAPER.get(paper_state, ("?", WARN))
    conn = "Ổn định" if stream_info == "Stream OK" else stream_info
    return [
        ("title", "HỆ THỐNG"),
        ("bullet", "Trạng thái", "Tạm dừng" if paused else "Đang chạy", WARN if paused else OK),
        ("bullet", "Camera", f"{w0}×{h0}" + ("" if fhd else " (chưa Full HD)"), None if fhd else BAD),
        ("bullet", "Kết nối", conn, OK if conn == "Ổn định" else WARN),
        ("bullet", "Stream", f"{fps_stream:.1f} FPS"),
        ("bullet", "Hiển thị", f"{fps_show:.1f} FPS"),
        ("bullet", "YOLO", f"{snap['det_fps'] if snap else 0.0:.1f} FPS"),
        ("bullet", "Bị loại (không giống xe)", str(snap.get("n_reject", 0) if snap else 0)),
        ("bullet", "GPU", gpu.replace("NVIDIA ", "").replace("GeForce ", "")),
        ("bullet", "Tờ giấy", paper_txt, paper_col),
        ("bullet", "Thời gian chạy", hms(uptime)),
        ("title", "CHÚ THÍCH MÀU"),
        ("swatch", bgr(BLUE), "Xe đã đỗ"),
        ("swatch", bgr(CYAN), "Xe mới vào qua cổng"),
        ("swatch", bgr(RED), "Xe đang di chuyển"),
        ("swatch", bgr(GREEN), "Chỗ trống"),
        ("swatch", bgr(YELLOW), "Dải đỗ xe"),
        ("swatch", bgr(MAGENTA), "Ô cam (cổng vào/ra)"),
    ]


def right_items(snap):
    """Cột phải: số liệu bãi xe và danh sách chỗ trống."""
    cars = snap["cars"] if snap else []
    n_do = sum(c["state"] == "da_do" for c in cars)
    spaces = snap["spaces"] if snap and snap["H"] is not None else []
    items = [
        ("title", "BÃI XE"),
        ("bullet", "Xe trong bãi", str(len(cars))),
        ("bullet", "Đã đỗ", str(n_do)),
        ("bullet", "Đang di chuyển", str(len(cars) - n_do)),
        ("bullet", "Lượt vào", str(snap["n_in"] if snap else 0)),
        ("bullet", "Lượt ra", str(snap["n_out"] if snap else 0)),
        ("title", f"CHỖ TRỐNG ({len(spaces)})"),
    ]
    if not snap or snap["H"] is None:
        items.append(("text", "Chưa khoá được khung bãi"))
    elif not spaces:
        items.append(("text", "Bãi đã kín, không còn chỗ trống", BAD))
    else:
        for k, sp in enumerate(spaces[:8], 1):            # cột cao có hạn: tối đa 8 khe
            items.append(("bullet", f"#{k} dãy {sp.get('row', '')}".strip(),
                          f"{sp['gap_mm']:.0f} mm, vừa {sp['capacity']} xe"))
        if len(spaces) > 8:
            items.append(("text", f"... và {len(spaces) - 8} khe khác"))
        items.append(("bullet", "Có thể đỗ thêm", f"{sum(sp['capacity'] for sp in spaces)} xe", OK))
    items += [("gap", 14), ("foot", "Phím tắt: Q thoát · Space tạm dừng · F toàn màn hình")]
    return items
