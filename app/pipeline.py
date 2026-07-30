import math
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

class ComicTranslationPipeline:
    def __init__(self, source_lang, target_lang, ocr_reader, translator):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.ocr_reader = ocr_reader
        self.translator = translator
        self._font_cache = {}
        self._font_path = self._find_font(target_lang)

    def process_image(self, image_path, forced_bboxes=None, progress_cb=None):
        def _step(msg):
            if progress_cb:
                progress_cb(msg)

        pil_orig = Image.open(str(image_path)).convert("RGB")
        pil_orig = ImageOps.exif_transpose(pil_orig)
        img_np = np.array(pil_orig)

        _cjk = self.source_lang in {"ch_sim", "ch_tra", "ja", "ja_manga", "ko"}

        results = []

        if forced_bboxes:
            _step("Processing user‑marked zones…")
            for bbox in forced_bboxes:
                xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
                x1 = max(0, int(min(xs))); y1 = max(0, int(min(ys)))
                x2 = min(img_np.shape[1], int(max(xs)))
                y2 = min(img_np.shape[0], int(max(ys)))
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                crop = img_np[y1:y2, x1:x2]
                try:
                    for r in self.ocr_reader.readtext(crop, detail=1, paragraph=False):
                        adj = [[p[0] + x1, p[1] + y1] for p in r[0]]
                        if any(c.isalnum() for c in r[1]):
                            results.append((adj, r[1], r[2]))
                except Exception:
                    pass
        else:
            _step("Running OCR…")
            results = self._ocr_with_fallback(img_np, _cjk, _step)

        if not results:
            _step("No text found — skipping.")
            return pil_orig, pil_orig.copy(), []

        results = self._merge_text_boxes(results, img_np.shape[1], img_np.shape[0])

        texts = [r[1] for r in results]
        _step(f"Translating {len(texts)} text region(s)…")
        translated = self._translate_texts(texts)

        _step("Inpainting original text…")
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask = self._build_mask(img_cv.shape[:2], results)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        img_inpainted = cv2.inpaint(img_cv, mask, 5, cv2.INPAINT_TELEA)

        _step("Rendering translation…")
        img_pil = Image.fromarray(cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB))
        base_pil = img_pil.copy()
        draw = ImageDraw.Draw(img_pil)

        regions = []
        for result, trans in zip(results, translated):
            if trans:
                fs = self._render(draw, img_pil, trans, result[0])
                regions.append({
                    'bbox': result[0],
                    'original_text': result[1],
                    'text': trans,
                    'font_size': fs
                })
            else:
                regions.append({
                    'bbox': result[0],
                    'original_text': result[1],
                    'text': '',
                    'font_size': 12
                })

        return img_pil, base_pil, regions

    def _ocr_with_fallback(self, img_np, _cjk, _step):
        img_h, img_w = img_np.shape[:2]

        if self.source_lang == "auto":
            kwargs = dict(text_threshold=0.5, low_text=0.3, paragraph=False)
            min_conf = 0.15
        elif _cjk:
            kwargs = dict(text_threshold=0.6, low_text=0.35, paragraph=False)
            min_conf = 0.1
        else:
            kwargs = dict(paragraph=False)
            min_conf = 0.3

        results = self.ocr_reader.readtext(img_np, detail=1, **kwargs)
        if results is None:
            results = []
        results = self._filter(results, min_conf, max_box_area=img_h * img_w * 0.12, alphanumeric=True)

        if results:
            return results

        _step("No text found — retrying with relaxed settings…")
        results = self.ocr_reader.readtext(
            img_np, detail=1, paragraph=False,
            text_threshold=0.4, low_text=0.3,
        )
        if results is None:
            results = []
        results = self._filter(results, min_conf=0.1, max_box_area=img_h * img_w * 0.5, alphanumeric=False)

        if results:
            return results

        _step("Retrying with rotation detection…")
        try:
            results = self.ocr_reader.readtext(
                img_np, detail=1, paragraph=False,
                text_threshold=0.4, low_text=0.3,
                rotation_info=[90, 180, 270],
            )
        except TypeError:
            results = []
        if results is None:
            results = []
        results = self._filter(results, min_conf=0.1, max_box_area=img_h * img_w * 0.5, alphanumeric=False)
        return results

    def _filter(self, results, min_conf, max_box_area, alphanumeric):
        out = []
        for r in results:
            if not isinstance(r, (list, tuple)) or len(r) < 3:
                continue
            bbox, text, conf = r[0], r[1], r[2]
            if conf <= min_conf:
                continue
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            if area >= max_box_area:
                continue
            if alphanumeric and not any(c.isalnum() for c in text):
                continue
            out.append(r)
        return out

    def _merge_text_boxes(self, results, img_w, img_h):
        """
        Cluster boxes using centroid distance.
        """
        if len(results) <= 1:
            return results

        boxes = []
        for bbox, text, conf in results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            boxes.append({
                'bbox': bbox,
                'text': text,
                'conf': conf,
                'cx': cx,
                'cy': cy,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1,
                'height': y2 - y1
            })

        heights = [b['height'] for b in boxes]
        median_h = sorted(heights)[len(heights)//2] if heights else 20
        radius = 2.0 * median_h

        n = len(boxes)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i+1, n):
                dx = boxes[i]['cx'] - boxes[j]['cx']
                dy = boxes[i]['cy'] - boxes[j]['cy']
                dist = math.hypot(dx, dy)
                if dist < radius:
                    merged_x1 = min(boxes[i]['x1'], boxes[j]['x1'])
                    merged_y1 = min(boxes[i]['y1'], boxes[j]['y1'])
                    merged_x2 = max(boxes[i]['x2'], boxes[j]['x2'])
                    merged_y2 = max(boxes[i]['y2'], boxes[j]['y2'])
                    area = (merged_x2 - merged_x1) * (merged_y2 - merged_y1)
                    if area < 0.20 * img_w * img_h:
                        union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(i)

        merged = []
        for indices in groups.values():
            if len(indices) == 1:
                idx = indices[0]
                merged.append((boxes[idx]['bbox'], boxes[idx]['text'], boxes[idx]['conf']))
            else:
                group_boxes = [boxes[i] for i in indices]
                merged_text = ' '.join([b['text'] for b in group_boxes])
                all_x = [b['x1'] for b in group_boxes] + [b['x2'] for b in group_boxes]
                all_y = [b['y1'] for b in group_boxes] + [b['y2'] for b in group_boxes]
                new_bbox = [
                    [min(all_x), min(all_y)],
                    [max(all_x), min(all_y)],
                    [max(all_x), max(all_y)],
                    [min(all_x), max(all_y)]
                ]
                avg_conf = sum(b['conf'] for b in group_boxes) / len(group_boxes)
                merged.append((new_bbox, merged_text, avg_conf))

        return merged

    # ---------- rest unchanged except _render (no rectangle) ----------
    _CJK_RANGES = [
        ('一', '鿿'),
        ('぀', 'ゟ'),
        ('゠', 'ヿ'),
        ('가', '힯'),
        ('㐀', '䶿'),
    ]

    def _has_cjk(self, text):
        return any(low <= c <= high for c in text for low, high in self._CJK_RANGES)

    _LATIN_TARGETS = {"en", "es", "fr", "de", "pt", "ru", "it", "vi", "th"}

    def _translate_texts(self, texts):
        out = []
        for text in texts:
            text = text.strip()
            if not text:
                out.append("")
                continue
            try:
                result = self.translator.translate(text)
                translated = result.strip() if result else ""
                if translated and self._has_cjk(translated) and self.target_lang in self._LATIN_TARGETS:
                    translated = ""
                out.append(translated)
            except Exception:
                out.append("")
        return out

    def _build_mask(self, shape, results):
        mask = np.zeros(shape, dtype=np.uint8)
        for r in results:
            pts = np.array([[int(p[0]), int(p[1])] for p in r[0]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def _quad_angle(self, pts):
        if len(pts) < 2:
            return 0.0
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        return math.degrees(math.atan2(dy, dx))

    def _is_vertical_box(self, x1, y1, x2, y2):
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return False
        cjk_sources = {"ja", "ja_manga", "ko", "ch_sim", "ch_tra"}
        latin_targets = {"en", "es", "fr", "de", "pt", "ru", "it", "vi", "th"}
        if self.source_lang in cjk_sources and self.target_lang in latin_targets:
            if height > 1.5 * width:
                return True
        return False

    def _render(self, draw, img_pil, text, bbox, font_size=None):
        pts = [(int(p[0]), int(p[1])) for p in bbox]
        angle = self._quad_angle(pts)

        if abs(angle) > 5:
            return self._render_rotated(img_pil, text, pts, angle, font_size)

        x1 = max(0, min(p[0] for p in pts))
        y1 = max(0, min(p[1] for p in pts))
        x2 = min(img_pil.width,  max(p[0] for p in pts))
        y2 = min(img_pil.height, max(p[1] for p in pts))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 2 or box_h <= 2:
            return font_size or 10

        is_vertical = self._is_vertical_box(x1, y1, x2, y2)
        MIN_RENDER_W = 60
        if not is_vertical and box_w < MIN_RENDER_W:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - MIN_RENDER_W // 2)
            x2 = min(img_pil.width, x1 + MIN_RENDER_W)
            box_w = x2 - x1

        if is_vertical:
            max_expand = 1.5
            new_w = min(img_pil.width, int(box_w * max_expand))
            if new_w > box_w:
                cx = (x1 + x2) // 2
                x1 = max(0, cx - new_w // 2)
                x2 = min(img_pil.width, x1 + new_w)
                box_w = x2 - x1

        # Determine text color based on the region's dominant color (for contrast)
        region = img_pil.crop((x1, y1, x2, y2))
        bg = self._dominant_color(region)
        fg = self._contrast_color(bg)

        min_size = 5 if is_vertical else 7

        if font_size:
            font = self._get_font(font_size)
            lines = self._wrap(text, font, box_w, draw)
        else:
            font, lines = self._fit_text(text, box_w, box_h, draw, min_size=min_size)

        if not lines:
            return font_size or 10

        actual_size = font.size if hasattr(font, 'size') else 10
        if is_vertical:
            line_h = int(actual_size * 1.4)
        else:
            line_h = self._line_height(font)

        total_h = line_h * len(lines)
        cur_y = y1 + max(0, (box_h - total_h) // 2)

        for line in lines:
            if cur_y + line_h > y2:
                break
            try:
                text_w = draw.textbbox((0, 0), line, font=font)[2]
            except Exception:
                text_w = len(line) * 7
            draw.text((x1 + max(0, (box_w - text_w) // 2), cur_y), line, fill=fg, font=font)
            cur_y += line_h

        return actual_size

    def _render_rotated(self, img_pil, text, pts, angle, font_size=None):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x1 = max(0, min(xs)); y1 = max(0, min(ys))
        x2 = min(img_pil.width, max(xs)); y2 = min(img_pil.height, max(ys))
        box_w = x2 - x1; box_h = y2 - y1
        if box_w <= 2 or box_h <= 2:
            return font_size or 10

        region = img_pil.crop((x1, y1, x2, y2))
        bg = self._dominant_color(region)
        fg = self._contrast_color(bg)

        tile = Image.new("RGBA", (box_w, box_h), (0,0,0,0))  # transparent tile
        tile_draw = ImageDraw.Draw(tile)

        if font_size:
            font = self._get_font(font_size)
            lines = self._wrap(text, font, box_w, tile_draw)
        else:
            font, lines = self._fit_text(text, box_w, box_h, tile_draw, min_size=7)

        actual_size = font.size if hasattr(font, 'size') else 10
        line_h = self._line_height(font)
        total_h = line_h * len(lines)
        cur_y = max(0, (box_h - total_h) // 2)

        for line in lines:
            if cur_y + line_h > box_h:
                break
            try:
                text_w = tile_draw.textbbox((0, 0), line, font=font)[2]
            except Exception:
                text_w = len(line) * 7
            tile_draw.text((max(0, (box_w - text_w) // 2), cur_y), line, fill=fg, font=font)
            cur_y += line_h

        rotated = tile.rotate(-angle, resample=Image.BICUBIC, expand=True)

        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        px = cx - rotated.width // 2; py = cy - rotated.height // 2
        img_pil.paste(rotated, (px, py), rotated)

        return actual_size

    def render_from_regions(self, base_img, regions):
        img = base_img.copy()
        draw = ImageDraw.Draw(img)
        for r in regions:
            if r.get('text'):
                self._render(draw, img, r['text'], r['bbox'], font_size=r.get('font_size'))
        return img

    def _dominant_color(self, region):
        a = np.array(region.convert("RGB"))
        return (int(a[:, :, 0].mean()), int(a[:, :, 1].mean()), int(a[:, :, 2].mean()))

    def _contrast_color(self, bg):
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        return (0, 0, 0) if lum > 128 else (255, 255, 255)

    def _line_height(self, font):
        try:
            return font.size + 3
        except AttributeError:
            return 13

    def _get_font(self, size):
        if size not in self._font_cache:
            try:
                self._font_cache[size] = (
                    ImageFont.truetype(self._font_path, size)
                    if self._font_path else ImageFont.load_default()
                )
            except Exception:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _fit_text(self, text, max_w, max_h, draw, min_size=7):
        for size in range(18, min_size - 1, -1):
            font = self._get_font(size)
            lines = self._wrap(text, font, max_w, draw)
            if self._line_height(font) * len(lines) <= max_h:
                return font, lines
        font = self._get_font(min_size)
        short = text if len(text) <= 20 else text[:17] + "..."
        return font, [short]

    def _wrap(self, text, font, max_w, draw):
        words = text.split()
        if not words:
            return [""]
        lines, current = [], []
        for word in words:
            test = " ".join(current + [word])
            try:
                w = draw.textbbox((0, 0), test, font=font)[2]
            except Exception:
                w = len(test) * 7
            if w > max_w and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(" ".join(current))
        return lines or [""]

    def _find_font(self, target_lang=None):
        CJK_FONTS = {
            "zh-CN": [
                r"C:\Windows\Fonts\msyh.ttc",
                r"C:\Windows\Fonts\simsun.ttc",
                r"C:\Windows\Fonts\simhei.ttf",
            ],
            "zh-TW": [
                r"C:\Windows\Fonts\msjh.ttc",
                r"C:\Windows\Fonts\mingliu.ttc",
            ],
            "ja": [
                r"C:\Windows\Fonts\meiryo.ttc",
                r"C:\Windows\Fonts\msgothic.ttc",
                r"C:\Windows\Fonts\msmincho.ttc",
            ],
            "ko": [
                r"C:\Windows\Fonts\malgun.ttf",
                r"C:\Windows\Fonts\gulim.ttc",
                r"C:\Windows\Fonts\batang.ttc",
            ],
        }

        if target_lang in CJK_FONTS:
            for p in CJK_FONTS[target_lang]:
                if os.path.exists(p):
                    return p

        for p in [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
        ]:
            if os.path.exists(p):
                return p

        return None