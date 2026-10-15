"""Giao diện 1920x1080 trên nền assets/ui_bg.png (thiết kế bằng Canva):
  - video đặt vào ô xám (x 320..1599, y 150..869, 1280x720)
  - cột trái (dưới logo) và cột phải: thông số dạng gạch đầu dòng
  - dải xanh nhạt phía dưới: khung nhật ký sự kiện, cuộn bằng con lăn chuột hoặc kéo thanh cuộn
Chữ tiếng Việt có dấu vẽ bằng Pillow (cv2.putText không có dấu). Các khung chữ chỉ vẽ lại khi nội dung đổi."""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
VIDEO = (320, 150, 1280, 720)                    # x, y, rộng, cao của ô video
LEFT = (14, 262, 292, 600)                       # cột trái, bắt đầu dưới bong bóng logo
RIGHT = (1614, 162, 292, 700)                    # cột phải
LOG = (20, 880, 1880, 190)                       # khung log trong dải xanh nhạt

# màu RGB (Pillow)
NAVY = (40, 46, 140)                             # gần màu logo HCMUTE
TEXT = (31, 41, 55)
MUTED = (107, 114, 128)
LINE = (203, 213, 225)
PANEL_BG = (255, 255, 255)
STRIP_BG = (194, 233, 255)
BAR_BG = (226, 232, 240)
BAR_FG = (100, 116, 139)

# màu nhãn sự kiện (RGB) và chữ hiện trên nhãn
KIND = {
    "vao": ("VÀO BÃI", (8, 145, 178)),
    "do": ("ĐÃ ĐỖ", (37, 99, 235)),
    "roi_cho": ("RỜI CHỖ", (234, 88, 12)),
    "roi_bai": ("RỜI BÃI", (220, 38, 38)),
    "co_san": ("CÓ SẴN", (100, 116, 139)),
    "xuat_hien": ("XUẤT HIỆN", (147, 51, 234)),
    "he_thong": ("HỆ THỐNG", (71, 85, 105)),
    "canh_bao": ("CẢNH BÁO", (202, 138, 4)),
}

_FONT_FILES = {
    "regular": ["segoeui.ttf", "arial.ttf", "tahoma.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"],
    "bold": ["segoeuib.ttf", "arialbd.ttf", "tahomabd.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
}
_FONT_DIRS = [os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
              os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "fonts"),
              "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation",
              "/System/Library/Fonts/Supplemental"]
_font_cache = {}


def font(size, weight="regular"):
    """Font có đủ dấu tiếng Việt: Segoe UI trên Windows, không có thì Arial / DejaVu."""
    key = (size, weight)
    if key not in _font_cache:
        f = None
        for name in _FONT_FILES[weight]:
            for d in _FONT_DIRS:
                p = os.path.join(d, name)
                if os.path.exists(p):
                    f = ImageFont.truetype(p, size)
                    break
            if f:
                break
        _font_cache[key] = f or ImageFont.load_default(size)
    return _font_cache[key]


def bgr(rgb):
    return rgb[2], rgb[1], rgb[0]


def _imread(path):
    """Đọc ảnh kể cả khi đường dẫn có dấu tiếng Việt (cv2.imread trên Windows không đọc được)."""
    try:
        return cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None


def _fallback_bg():
    """Nền thay thế khi không có assets/ui_bg.png."""
    img = Image.new("RGB", (W, H), PANEL_BG)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 149], fill=STRIP_BG)
    d.rectangle([0, 870, W, H], fill=STRIP_BG)
    d.text((24, 40), "SMART UNMARKED PARKING ASSISTANT", font=font(44, "bold"), fill=(0, 0, 0))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def wrap(draw, text, fnt, width):
    """Ngắt dòng theo độ rộng pixel."""
    words, lines, cur = text.split(" "), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if draw.textlength(t, font=fnt) <= width or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def render_panel(base_bgr, items, width):
    """Vẽ một cột thông số lên ảnh nền (BGR) của cột đó. items là list tuple:
      ("title", text)                 tiêu đề mục (chữ đậm, gạch chân)
      ("bullet", label, value[, rgb]) gạch đầu dòng "label: value" (value đậm, có thể tô màu)
      ("text", text[, rgb])           dòng chữ thường có gạch đầu dòng
      ("swatch", rgb, text)           ô màu + chú thích
      ("gap", px)                     khoảng trống
      ("foot", text)                  chữ nhỏ màu xám"""
    img = Image.fromarray(cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    f_title, f_txt, f_bold, f_small = font(21, "bold"), font(19), font(19, "bold"), font(16)
    y, lh, ind = 2, 27, 20
    for it in items:
        kind = it[0]
        if kind == "title":
            if y > 2:
                y += 10
            d.text((0, y), it[1], font=f_title, fill=NAVY)
            y += 30
            d.line([0, y, width, y], fill=NAVY, width=2)
            y += 10
        elif kind in ("bullet", "text"):
            col = (it[3] if kind == "bullet" and len(it) > 3 else (it[2] if kind == "text" and len(it) > 2 else None)) or TEXT
            d.ellipse([4, y + 10, 11, y + 17], fill=NAVY)
            if kind == "bullet":
                label = f"{it[1]}: "
                lw_ = d.textlength(label, font=f_txt)
                d.text((ind, y), label, font=f_txt, fill=MUTED)
                lines = wrap(d, str(it[2]), f_bold, width - ind - lw_)
                if len(lines) > 1 and lw_ > (width - ind) * 0.45:      # nhãn dài: giá trị xuống dòng dưới
                    lines = wrap(d, str(it[2]), f_bold, width - ind)
                    y += lh
                    for ln in lines:
                        d.text((ind, y), ln, font=f_bold, fill=col)
                        y += lh
                else:
                    for k, ln in enumerate(lines):
                        d.text((ind + (lw_ if k == 0 else 0), y), ln, font=f_bold, fill=col)
                        y += lh
            else:
                for ln in wrap(d, it[1], f_txt, width - ind):
                    d.text((ind, y), ln, font=f_txt, fill=col)
                    y += lh
        elif kind == "swatch":
            d.rounded_rectangle([2, y + 5, 26, y + 21], radius=4, outline=it[1], width=3, fill=PANEL_BG)
            d.text((36, y), it[2], font=f_txt, fill=TEXT)
            y += lh
        elif kind == "gap":
            y += it[1]
        elif kind == "foot":
            for ln in wrap(d, it[1], f_small, width):
                d.text((0, y), ln, font=f_small, fill=MUTED)
                y += 22
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class Dashboard:
    """Ghép nền + video + hai cột + khung log thành một ảnh 1920x1080 để cv2.imshow."""

    LINE_H = 28                                   # chiều cao 1 dòng log
    HEAD_H = 36                                   # dòng tiêu đề của khung log
    BAR_W = 12                                    # thanh cuộn

    def __init__(self, bg_path):
        bg = _imread(bg_path) if os.path.exists(bg_path) else None
        if bg is None:
            print(f"Khong thay {bg_path}, dung nen mac dinh")
            bg = _fallback_bg()
        elif bg.shape[:2] != (H, W):
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_AREA)
        self.bg = bg
        self.base = bg.copy()                     # nền + hai cột + log (video dán sau)
        self._panel_key = {}
        self.scroll = 0                           # số dòng đang cuộn lên so với dòng mới nhất
        self._log_key = None
        self._seen_total = 0
        self._n_entries = 0
        self._drag = None                         # (y chuột lúc bấm, scroll lúc bấm) khi kéo thanh cuộn
        self._thumb = None                        # (y0, y1) của thanh cuộn trên màn hình
        self.dirty = True                         # nền/cột/log vừa đổi -> cần vẽ lại cửa sổ
        self._redraw_log([], 0)

    # ---------------------------------------------------------------- cột thông số
    def set_panel(self, side, items):
        x, y, w, h = LEFT if side == "left" else RIGHT
        key = repr(items)
        if self._panel_key.get(side) == key:
            return
        self._panel_key[side] = key
        self.base[y:y + h, x:x + w] = render_panel(self.bg[y:y + h, x:x + w], items, w)
        self.dirty = True

    # ---------------------------------------------------------------- khung log
    @property
    def _rows(self):
        return (LOG[3] - self.HEAD_H - 8) // self.LINE_H

    def _max_scroll(self):
        return max(0, self._n_entries - self._rows)

    def set_log(self, entries, total):
        new = total - self._seen_total
        self._seen_total = total
        self._n_entries = len(entries)
        if self.scroll > 0 and new > 0:           # đang xem log cũ: giữ nguyên chỗ đang xem
            self.scroll += new
        self.scroll = min(self.scroll, self._max_scroll())
        key = (total, len(entries), self.scroll)
        if key != self._log_key:
            self._log_key = key
            self._redraw_log(entries, total)

    def _redraw_log(self, entries, total):
        x, y, w, h = LOG
        img = Image.fromarray(cv2.cvtColor(self.bg[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=12, fill=PANEL_BG, outline=LINE, width=1)
        d.text((16, 6), "NHẬT KÝ SỰ KIỆN", font=font(20, "bold"), fill=NAVY)
        hint = (f"{len(entries)} dòng  ·  Cuộn chuột hoặc kéo thanh bên phải để xem lại"
                if self.scroll == 0 else f"Đang xem dòng cũ (lùi {self.scroll} dòng)  ·  Cuộn xuống để về mới nhất")
        f_small = font(16)
        d.text((w - 28 - d.textlength(hint, font=f_small), 10), hint, font=f_small, fill=MUTED)
        d.line([12, self.HEAD_H, w - 12, self.HEAD_H], fill=LINE, width=1)
        rows = self._rows
        end = len(entries) - self.scroll
        show = entries[max(0, end - rows):end]
        f_txt, f_time, f_tag = font(19), font(18), font(14, "bold")
        ty = self.HEAD_H + 4
        if not entries:
            d.text((16, ty), "Chưa có sự kiện nào.", font=f_txt, fill=MUTED)
        for e in show:
            tag, col = KIND.get(e["kind"], KIND["he_thong"])
            cy = ty + self.LINE_H // 2                      # căn giữa theo chiều dọc của dòng
            d.text((16, cy), e["time"], font=f_time, fill=MUTED, anchor="lm")
            tx = 106
            d.rounded_rectangle([tx, cy - 10, tx + 104, cy + 10], radius=10, fill=col)
            d.text((tx + 52, cy), tag, font=f_tag, fill=(255, 255, 255), anchor="mm")
            d.text((tx + 118, cy), e["text"], font=f_txt, fill=TEXT, anchor="lm")
            ty += self.LINE_H
        # thanh cuộn
        bx0, by0, by1 = w - 20, self.HEAD_H + 6, h - 8
        d.rounded_rectangle([bx0, by0, bx0 + self.BAR_W - 4, by1], radius=4, fill=BAR_BG)
        n = max(len(entries), 1)
        if len(entries) > rows:
            th = max(24, (by1 - by0) * rows / n)
            top = by0 + (by1 - by0 - th) * (1 - self.scroll / max(1, self._max_scroll()))
            d.rounded_rectangle([bx0, top, bx0 + self.BAR_W - 4, top + th], radius=4, fill=BAR_FG)
            self._thumb = (y + top, y + top + th)
        else:
            self._thumb = None
        self.base[y:y + h, x:x + w] = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        self.dirty = True

    def on_mouse(self, event, mx, my, flags, _param=None):
        """Gắn bằng cv2.setMouseCallback: con lăn chuột để cuộn log, kéo thanh cuộn bằng chuột trái."""
        x, y, w, h = LOG
        inside = x <= mx <= x + w and y <= my <= y + h
        if event == cv2.EVENT_MOUSEWHEEL:              # khung log là phần duy nhất cuộn được nên lăn ở đâu cũng cuộn log
            step = 3 if flags > 0 else -3        # dấu của flags = hướng lăn; lăn lên = xem dòng cũ hơn
            self.scroll = int(np.clip(self.scroll + step, 0, self._max_scroll()))
        elif event == cv2.EVENT_LBUTTONDOWN and inside and mx >= x + w - 30 and self._thumb:
            if not self._thumb[0] <= my <= self._thumb[1]:               # bấm vào rãnh: nhảy tới đó
                self._jump(my)
            self._drag = (my, self.scroll)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag and flags & cv2.EVENT_FLAG_LBUTTON:
            y0, s0 = self._drag
            track = (h - 8) - (self.HEAD_H + 6)
            th = self._thumb[1] - self._thumb[0] if self._thumb else 24
            per_px = self._max_scroll() / max(1.0, track - th)
            self.scroll = int(np.clip(round(s0 - (my - y0) * per_px), 0, self._max_scroll()))   # kéo lên = dòng cũ
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _jump(self, my):
        _, y, _, h = LOG
        by0, by1 = y + self.HEAD_H + 6, y + h - 8
        frac = np.clip((my - by0) / max(1, by1 - by0), 0, 1)
        self.scroll = int(round((1 - frac) * self._max_scroll()))

    # ---------------------------------------------------------------- ghép ảnh
    def compose(self, frame):
        self.dirty = False
        out = self.base.copy()
        x, y, w, h = VIDEO
        if frame is not None:
            fh, fw = frame.shape[:2]
            s = min(w / fw, h / fh)               # giữ tỉ lệ, khung không phải 16:9 thì có viền đen
            nw, nh = int(round(fw * s)), int(round(fh * s))
            vid = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            if (nw, nh) != (w, h):
                out[y:y + h, x:x + w] = 0
            ox, oy = x + (w - nw) // 2, y + (h - nh) // 2
            out[oy:oy + nh, ox:ox + nw] = vid
        return out
