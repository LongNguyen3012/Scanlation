import math
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


class ComicTranslationPipeline:
    _NO_SPACE_SOURCES = {"ja", "ja_manga", "ch_sim", "ch_tra"}

    BUBBLE_H_CLOSE       = 15
    BUBBLE_V_CLOSE       = 25
    BUBBLE_OPEN_K        = 5
    BUBBLE_MIN_AREA_FRAC = 0.0005
    BUBBLE_MAX_AREA_FRAC = 0.40
    BUBBLE_MIN_FILL      = 0.30
    BUBBLE_UNIFY_MARGIN  = 12
    BUBBLE_UNIFY_ENABLED = False

    OCR_TARGET_LONG_SIDE = 2200
    OCR_MAX_UPSCALE      = 2.0

    CROP_MARGIN_X_FRAC        = 0.06
    CROP_MARGIN_Y_FRAC        = 0.20
    CROP_MIN_PIXELS           = 8
    CROP_MIN_CONF_TO_SKIP     = 0.75
    CROP_MIN_CONF_GAIN        = 0.15
    CROP_REPLACEMENT_MIN_CONF = 0.40

    RAW_TOP_MARGIN_FRAC    = 1.50
    RAW_SIDE_MARGIN_FRAC   = 0.30
    RAW_BOTTOM_MARGIN_FRAC = 0.30
    RAW_TOP_MIN_PX         = 100
    RAW_SIDE_MIN_PX        = 60
    RAW_BOTTOM_MIN_PX      = 30
    RAW_NEIGHBOR_GAP_PX    = 8
    RAW_RECHECK_MIN_CONF   = 0.55

    def __init__(self, source_lang, target_lang, ocr_reader, translator):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.ocr_reader = ocr_reader
        self.translator = translator
        self._font_cache = {}
        self._font_path = self._find_font(target_lang)

    def _preprocess_for_ocr(self, img_rgb):
        if self.source_lang not in {"ko", "ja", "ja_manga", "ch_sim", "ch_tra"}:
            return img_rgb
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        smoothed = cv2.bilateralFilter(gray, d=5, sigmaColor=40, sigmaSpace=40)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        bg = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, kernel)
        bg = np.clip(bg.astype(np.float32), 1.0, None)
        norm = np.clip(smoothed.astype(np.float32) / bg * 230.0, 0, 255).astype(np.uint8)
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)

    def _compute_ocr_scale(self, img_np, _cjk):
        if not _cjk:
            return 1.0
        long_side = max(img_np.shape[:2])
        if long_side >= self.OCR_TARGET_LONG_SIDE:
            return 1.0
        return min(self.OCR_MAX_UPSCALE,
                   self.OCR_TARGET_LONG_SIDE / long_side)

    def _text_fits_box(self, text, box_width, box_height):
        text_len = sum(1 for c in text if not c.isspace())
        if text_len <= 0 or box_height <= 0:
            return False
        avg_char_width = box_width / text_len
        return avg_char_width <= 1.6 * box_height

    def _refine_lines_with_crops(self, lines, ocr_input, ocr_scale, _cjk):
        if not lines:
            return lines
        refined = []
        debug = bool(os.environ.get("CT_DEBUG"))
        h_img, w_img = ocr_input.shape[:2]

        for line in lines:
            old_conf = float(line.get('conf', 0.0))
            try:
                box_confs = [float(b.get('conf', 0.0)) for b in line.get('boxes', [])]
            except Exception:
                box_confs = []
            min_box_conf = min(box_confs) if box_confs else old_conf

            x1, y1, x2, y2 = line['x1'], line['y1'], line['x2'], line['y2']
            w_line = x2 - x1
            h_line = y2 - y1

            plausible_old = self._text_fits_box(
                line.get('text', ''), w_line, h_line
            )

            if min_box_conf >= self.CROP_MIN_CONF_TO_SKIP and plausible_old:
                refined.append(line)
                continue

            if min_box_conf >= self.CROP_MIN_CONF_TO_SKIP and not plausible_old and debug:
                print(f"[pipeline] suspicious high-conf OCR: "
                      f"{line.get('text','')!r} conf={min_box_conf:.2f} "
                      f"box={int(w_line)}x{int(h_line)} → forcing crop refinement")

            mx = max(4, int(w_line * self.CROP_MARGIN_X_FRAC))
            my = max(4, int(h_line * self.CROP_MARGIN_Y_FRAC))

            cx1 = max(0, int(x1) - mx)
            cy1 = max(0, int(y1) - my)
            cx2 = min(w_img, int(x2) + mx)
            cy2 = min(h_img, int(y2) + my)

            if cx2 - cx1 < self.CROP_MIN_PIXELS or cy2 - cy1 < self.CROP_MIN_PIXELS:
                refined.append(line)
                continue

            crop = ocr_input[cy1:cy2, cx1:cx2]

            if ocr_scale != 1.0:
                crop = cv2.resize(crop, None, fx=ocr_scale, fy=ocr_scale,
                                  interpolation=cv2.INTER_CUBIC)

            try:
                crop_results = self.ocr_reader.readtext(
                    crop, detail=1, paragraph=False
                )
            except Exception:
                crop_results = []

            if not crop_results:
                refined.append(line)
                continue

            def _cx(r):
                try:
                    xs = [float(p[0]) for p in r[0]]
                    return sum(xs) / len(xs)
                except Exception:
                    return 0.0

            try:
                crop_results_sorted = sorted(crop_results, key=_cx)
            except Exception:
                crop_results_sorted = crop_results

            pieces, confs = [], []
            for r in crop_results_sorted:
                if isinstance(r, (list, tuple)) and len(r) >= 3:
                    t = str(r[1]).strip()
                    if t:
                        pieces.append(t)
                        confs.append(float(r[2]))

            if not pieces:
                refined.append(line)
                continue

            crop_text = ' '.join(pieces)
            crop_min_conf = min(confs) if confs else 0.0
            plausible_crop = self._text_fits_box(crop_text, w_line, h_line)

            if not plausible_old:
                accept = (
                    plausible_crop
                    and crop_min_conf >= self.CROP_REPLACEMENT_MIN_CONF
                    and crop_text.strip() != line.get('text', '').strip()
                )
            else:
                accept = (
                    plausible_crop
                    and crop_min_conf >= min_box_conf + self.CROP_MIN_CONF_GAIN
                )

            if accept:
                new_line = dict(line)
                new_line['text'] = crop_text
                new_line['conf'] = crop_min_conf
                if debug:
                    tag = "forced" if not plausible_old else "refined"
                    print(f"[pipeline] crop-{tag} (conf {min_box_conf:.2f}→"
                          f"{crop_min_conf:.2f}): {line.get('text','')!r} → {crop_text!r}")
                refined.append(new_line)
            else:
                refined.append(line)

        return refined

    def _dialogue_bbox(self, dlg):
        all_boxes = [b for line in dlg for b in line.get('boxes', [])]
        if not all_boxes:
            return None
        xs = [b['x1'] for b in all_boxes] + [b['x2'] for b in all_boxes]
        ys = [b['y1'] for b in all_boxes] + [b['y2'] for b in all_boxes]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def _augment_dialogues_with_raw(self, dialogues, raw_img, ocr_scale):
        if not dialogues:
            return dialogues

        debug = bool(os.environ.get("CT_DEBUG"))
        h_img, w_img = raw_img.shape[:2]
        out = []

        bboxes = [self._dialogue_bbox(d) for d in dialogues]

        for idx, dlg in enumerate(dialogues):
            bb = bboxes[idx]
            if bb is None:
                out.append(dlg)
                continue
            x1, y1, x2, y2 = bb
            w_line = x2 - x1
            h_line = y2 - y1
            if w_line <= 0 or h_line <= 0:
                out.append(dlg)
                continue

            # Find nearest dialogue above and below
            top_limit = 0
            for j in range(idx):
                if bboxes[j] is not None and bboxes[j][3] < y1:
                    top_limit = max(top_limit, int(bboxes[j][3]))
            bot_limit = int(h_img)
            for j in range(idx + 1, len(dialogues)):
                if bboxes[j] is not None and bboxes[j][1] > y2:
                    bot_limit = min(bot_limit, int(bboxes[j][1]))

            side_margin   = max(self.RAW_SIDE_MIN_PX,
                                int(self.RAW_SIDE_MARGIN_FRAC * w_line))
            top_margin    = max(self.RAW_TOP_MIN_PX,
                                int(self.RAW_TOP_MARGIN_FRAC * h_line))
            bottom_margin = max(self.RAW_BOTTOM_MIN_PX,
                                int(self.RAW_BOTTOM_MARGIN_FRAC * h_line))

            ex1 = max(0, x1 - side_margin)
            ex2 = min(w_img, x2 + side_margin)

            naive_top = y1 - top_margin
            if top_limit > 0:
                clipped_top = top_limit + int(self.RAW_NEIGHBOR_GAP_PX)
                ey1 = max(0, max(naive_top, clipped_top))
            else:
                ey1 = max(0, naive_top)

            naive_bot = y2 + bottom_margin
            if bot_limit < h_img:
                clipped_bot = bot_limit - int(self.RAW_NEIGHBOR_GAP_PX)
                ey2 = min(h_img, min(naive_bot, clipped_bot))
            else:
                ey2 = min(h_img, naive_bot)

            ey1 = int(ey1)
            ey2 = int(ey2)
            ex1 = int(ex1)
            ex2 = int(ex2)

            if ey2 - ey1 < 16 or ex2 - ex1 < 16:
                if debug:
                    print(f"[pipeline] raw-augment: dialogue bbox="
                          f"({x1},{y1})-({x2},{y2}) SKIPPED (crop too small)")
                out.append(dlg)
                continue

            crop = raw_img[ey1:ey2, ex1:ex2]

            if ocr_scale != 1.0:
                crop_work = cv2.resize(crop, None, fx=ocr_scale, fy=ocr_scale,
                                       interpolation=cv2.INTER_CUBIC)
            else:
                crop_work = crop

            try:
                raw_results = self.ocr_reader.readtext(
                    crop_work, detail=1, paragraph=False,
                    text_threshold=0.5, low_text=0.3,
                )
            except Exception:
                raw_results = []

            if not raw_results:
                if debug:
                    print(f"[pipeline] raw-augment: dialogue bbox="
                          f"({x1},{y1})-({x2},{y2}) crop=({ex1},{ey1})-({ex2},{ey2}) "
                          f"raw OCR returned 0 boxes")
                out.append(dlg)
                continue

            raw_page = []
            for bb_, t, c in raw_results:
                try:
                    xs_b = [float(p[0]) / ocr_scale + ex1 for p in bb_]
                    ys_b = [float(p[1]) / ocr_scale + ey1 for p in bb_]
                    x1_b = min(xs_b); x2_b = max(xs_b)
                    y1_b = min(ys_b); y2_b = max(ys_b)
                    quad = [[x1_b, y1_b], [x2_b, y1_b],
                            [x2_b, y2_b], [x1_b, y2_b]]
                except Exception:
                    continue
                if not any(ch.isalnum() for ch in str(t)):
                    continue
                raw_page.append((quad, str(t), float(c)))

            if not raw_page:
                if debug:
                    print(f"[pipeline] raw-augment: dialogue bbox="
                          f"({x1},{y1})-({x2},{y2}) crop=({ex1},{ey1})-({ex2},{ey2}) "
                          f"all boxes filtered out (no alnum)")
                out.append(dlg)
                continue

            raw_lines = self._group_lines(raw_page)

            if debug:
                print(f"[pipeline] raw-augment: dialogue bbox="
                      f"({x1},{y1})-({x2},{y2}) crop=({ex1},{ey1})-({ex2},{ey2}) "
                      f"top_limit={top_limit} bot_limit={bot_limit} "
                      f"raw_lines={len(raw_lines)}")
                for rl in raw_lines:
                    print(f"    raw line: {rl['text']!r} "
                          f"conf={float(rl.get('conf', 0.0)):.2f} "
                          f"bbox=({int(rl['x1'])},{int(rl['y1'])})-"
                          f"({int(rl['x2'])},{int(rl['y2'])})")

            existing_rects = [(l['x1'], l['y1'], l['x2'], l['y2']) for l in dlg]
            existing_texts = {l['text'].strip() for l in dlg}

            def _overlaps(rect_a, rect_b):
                ax1, ay1, ax2, ay2 = rect_a
                bx1, by1, bx2, by2 = rect_b
                ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
                if ix2 <= ix1 or iy2 <= iy1:
                    return 0.0
                inter = (ix2 - ix1) * (iy2 - iy1)
                area_a = (ax2 - ax1) * (ay2 - ay1)
                area_b = (bx2 - bx1) * (by2 - by1)
                min_area = min(area_a, area_b)
                return inter / min_area if min_area > 0 else 0.0

            new_lines = list(dlg)
            added_any = False
            for rline in raw_lines:
                rtext = rline['text'].strip()
                rrect = (rline['x1'], rline['y1'], rline['x2'], rline['y2'])

                if not rtext or rtext in existing_texts:
                    if debug:
                        print(f"      REJECT {rtext!r}: already in dialogue")
                    continue
                if rline.get('conf', 0.0) < self.RAW_RECHECK_MIN_CONF:
                    if debug:
                        print(f"      REJECT {rtext!r}: conf too low")
                    continue
                if not self._text_fits_box(rtext, rline['width'], rline['height']):
                    if debug:
                        print(f"      REJECT {rtext!r}: geometry doesn't fit")
                    continue
                overlaps_existing = any(_overlaps(rrect, er) > 0.30
                                        for er in existing_rects)
                if overlaps_existing:
                    if debug:
                        print(f"      REJECT {rtext!r}: overlaps existing line")
                    continue

                r_center_y = (rline['y1'] + rline['y2']) / 2.0
                if r_center_y < y1 - 0.5 * h_line or r_center_y > y2 + 0.5 * h_line:
                    if debug:
                        print(f"      REJECT {rtext!r}: center-y "
                              f"{int(r_center_y)} outside dialogue range "
                              f"({y1}-{y2})")
                    continue

                new_lines.append(rline)
                existing_rects.append(rrect)
                existing_texts.add(rtext)
                added_any = True
                if debug:
                    print(f"[pipeline] raw-augmented dialogue: added "
                          f"{rtext!r} at "
                          f"({int(rrect[0])},{int(rrect[1])})-"
                          f"({int(rrect[2])},{int(rrect[3])})")

            if added_any:
                new_lines.sort(key=lambda l: l['cy'])
            out.append(new_lines)

        return out

    def process_image(self, image_path, forced_bboxes=None, progress_cb=None):
        def _step(msg):
            if progress_cb:
                progress_cb(msg)

        pil_orig = Image.open(str(image_path)).convert("RGB")
        pil_orig = ImageOps.exif_transpose(pil_orig)
        img_np = np.array(pil_orig)

        _cjk = self.source_lang in {"ch_sim", "ch_tra", "ja", "ja_manga", "ko"}

        h, w = img_np.shape[:2]
        max_side = 3500
        if max(h, w) > max_side:
            scale = max_side / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
            pil_orig = Image.fromarray(img_np)
            if os.environ.get("CT_DEBUG"):
                print(f"[pipeline] downscaled page {w}x{h} → {new_w}x{new_h}")

        ocr_input = self._preprocess_for_ocr(img_np) if _cjk else img_np
        ocr_scale = self._compute_ocr_scale(img_np, _cjk)

        if ocr_scale != 1.0:
            work = cv2.resize(ocr_input, None, fx=ocr_scale, fy=ocr_scale,
                              interpolation=cv2.INTER_CUBIC)
        else:
            work = ocr_input

        if os.environ.get("CT_DEBUG"):
            print(f"[pipeline] OCR scale ×{ocr_scale:.2f} "
                  f"({img_np.shape[1]}×{img_np.shape[0]})")

        if forced_bboxes:
            _step("Processing user‑marked zones…")
            results = []
            for bbox in forced_bboxes:
                xs = [p[0] * ocr_scale for p in bbox]
                ys = [p[1] * ocr_scale for p in bbox]
                x1 = max(0, int(min(xs)))
                y1 = max(0, int(min(ys)))
                x2 = min(work.shape[1], int(max(xs)))
                y2 = min(work.shape[0], int(max(ys)))
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                crop = work[y1:y2, x1:x2]
                try:
                    for r in self.ocr_reader.readtext(crop, detail=1, paragraph=False):
                        adj = [[p[0] + x1, p[1] + y1] for p in r[0]]
                        if any(c.isalnum() for c in r[1]):
                            results.append((adj, r[1], r[2]))
                except Exception:
                    pass
        else:
            _step("Running OCR…")
            results = self._ocr_with_fallback(work, _cjk, _step)

        if os.environ.get("CT_DEBUG"):
            print(f"[pipeline] OCR OUTPUT ({len(results)} boxes, from preprocessed image):")
            for i, (bb, t, c) in enumerate(results):
                try:
                    xs = [float(p[0]) for p in bb]
                    ys = [float(p[1]) for p in bb]
                    x1, y1 = int(min(xs)), int(min(ys))
                    x2, y2 = int(max(xs)), int(max(ys))
                except Exception:
                    x1 = y1 = x2 = y2 = -1
                print(f"  [{i:02d}] ({x1},{y1})-({x2},{y2}) "
                      f"conf={float(c):.2f}  {str(t)!r}")

        if ocr_scale != 1.0 and results:
            results = [
                ([[p[0] / ocr_scale, p[1] / ocr_scale] for p in bbox], text, conf)
                for bbox, text, conf in results
            ]

        if not results:
            _step("No text found — skipping.")
            return pil_orig, pil_orig.copy(), []

        lines = self._group_lines(results)
        if not lines:
            _step("No text found — skipping.")
            return pil_orig, pil_orig.copy(), []

        if not forced_bboxes:
            _step(f"Checking {len(lines)} line(s)…")
            lines = self._refine_lines_with_crops(lines, ocr_input, ocr_scale, _cjk)

        if forced_bboxes:
            bubbles = self._forced_as_bubbles(forced_bboxes)
        else:
            bubbles = self._detect_bubbles(img_np)

        dialogues = self._group_dialogue(lines, bubbles)

        if not forced_bboxes:
            _step(f"Raw-image recheck on {len(dialogues)} dialogue(s)…")
            dialogues = self._augment_dialogues_with_raw(dialogues, img_np, ocr_scale)

        region_specs = self._build_regions_from_dialogues(dialogues)

        if not region_specs:
            _step("No text found — skipping.")
            return pil_orig, pil_orig.copy(), []

        if os.environ.get("CT_DEBUG"):
            print(f"[pipeline] OCR boxes: {len(results)} → lines: {len(lines)} "
                  f"→ bubbles: {len(bubbles)} → dialogues: {len(region_specs)}")
            self._dump_debug_v2(img_np, bubbles, region_specs, lines, image_path)

        source_texts = [r['text'] for r in region_specs]
        _step(f"Translating {len(source_texts)} dialogue block(s)…")
        translated = self._translate_texts(source_texts)

        if not any(t.strip() for t in translated):
            _step("No translations returned — check console output (set CT_DEBUG=1).")

        _step("Inpainting original text…")
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        mask = self._build_mask_from_quads(
            img_cv.shape[:2], [r['quads'] for r in region_specs]
        )
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        img_inpainted = cv2.inpaint(img_cv, mask, 5, cv2.INPAINT_TELEA)

        _step("Rendering translation…")
        img_pil = Image.fromarray(cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB))
        base_pil = img_pil.copy()
        draw = ImageDraw.Draw(img_pil)

        regions = []
        for spec, trans in zip(region_specs, translated):
            entry = {
                'bbox':          spec['bbox'],
                'original_text': spec['text'],
                'source_text':   spec['text'],
                'lines':         spec['lines'],
                'line_bboxes':   spec['line_bboxes'],
                'line_quads':    spec['line_quads'],
                'text':          trans if trans else '',
                'font_size':     12,
            }
            if trans:
                entry['font_size'] = self._render(draw, img_pil, trans, spec['bbox'])
            regions.append(entry)

        return img_pil, base_pil, regions

    def _ocr_with_fallback(self, img_np, _cjk, _step):
        img_h, img_w = img_np.shape[:2]

        if _cjk:
            kwargs = dict(text_threshold=0.5, low_text=0.3, paragraph=False)
            min_conf = 0.1
        else:
            kwargs = dict(paragraph=False)
            min_conf = 0.3

        results = self.ocr_reader.readtext(img_np, detail=1, **kwargs)
        if results is None:
            results = []
        results = self._filter(results, min_conf,
                               max_box_area=img_h * img_w * 0.12,
                               alphanumeric=True)
        if results:
            return results

        if _step:
            _step("Retrying with relaxed settings…")
        results = self.ocr_reader.readtext(
            img_np, detail=1, paragraph=False,
            text_threshold=0.4, low_text=0.3,
        )
        if results is None:
            results = []
        return self._filter(results, min_conf=0.05,
                            max_box_area=img_h * img_w * 0.5,
                            alphanumeric=False)

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

    def _is_plausible_region(self, text):
        stripped = text.strip()
        if len(stripped) < 3:
            return False
        if self.source_lang == "ko":
            hangul = sum(1 for c in stripped if '\uac00' <= c <= '\ud7a3')
            latin = sum(1 for c in stripped if c.isascii() and c.isalpha())
            if hangul < 3:
                return False
            if latin > hangul:
                return False
        elif self.source_lang in {"ja", "ja_manga"}:
            if not any('\u3040' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff'
                       for c in stripped):
                return False
        elif self.source_lang in {"ch_sim", "ch_tra"}:
            if not any('\u4e00' <= c <= '\u9fff' for c in stripped):
                return False
        return True

    def _detect_bubbles(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (self.BUBBLE_H_CLOSE, 1)))
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.BUBBLE_V_CLOSE)))
        bright = cv2.morphologyEx(
            bright, cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (self.BUBBLE_OPEN_K, self.BUBBLE_OPEN_K)))
        num, _labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        bubbles = []
        for i in range(1, num):
            x, y, bw, bh, area = stats[i]
            if area < h * w * self.BUBBLE_MIN_AREA_FRAC:
                continue
            if area > h * w * self.BUBBLE_MAX_AREA_FRAC:
                continue
            fill = area / float(bw * bh) if bw * bh else 0.0
            if fill < self.BUBBLE_MIN_FILL:
                continue
            bubbles.append((x, y, x + bw, y + bh))
        if self.BUBBLE_UNIFY_ENABLED:
            bubbles = self._unify_bubbles(bubbles, margin=self.BUBBLE_UNIFY_MARGIN)
        return bubbles

    def _unify_bubbles(self, bubbles, margin=12):
        if len(bubbles) <= 1:
            return bubbles
        n = len(bubbles)
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
            for j in range(i + 1, n):
                x1a, y1a, x2a, y2a = bubbles[i]
                x1b, y1b, x2b, y2b = bubbles[j]
                if (x1a - margin <= x2b and x1b - margin <= x2a and
                        y1a - margin <= y2b and y1b - margin <= y2a):
                    union(i, j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        out = []
        for indices in groups.values():
            xs1 = [bubbles[i][0] for i in indices]
            ys1 = [bubbles[i][1] for i in indices]
            xs2 = [bubbles[i][2] for i in indices]
            ys2 = [bubbles[i][3] for i in indices]
            out.append((min(xs1), min(ys1), max(xs2), max(ys2)))
        return out

    def _bubble_for_box(self, bbox, bubbles):
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        cx = (min(xs) + max(xs)) / 2.0
        cy = (min(ys) + max(ys)) / 2.0
        for i, (x1, y1, x2, y2) in enumerate(bubbles):
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return i
        return None

    def _forced_as_bubbles(self, forced_bboxes):
        bubbles = []
        for bbox in forced_bboxes:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            bubbles.append((min(xs), min(ys), max(xs), max(ys)))
        return bubbles

    def _group_lines(self, ocr_results):
        if not ocr_results:
            return []
        boxes = []
        for bbox, text, conf in ocr_results:
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
            boxes.append({
                'bbox': bbox, 'text': text, 'conf': conf,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
                'width': x2 - x1, 'height': y2 - y1,
            })
        boxes.sort(key=lambda b: (b['cy'], b['cx']))
        lines = []
        for b in boxes:
            placed = False
            for line in lines:
                v_close = abs(b['cy'] - line['cy']) < 0.6 * max(b['height'], line['height'])
                if not v_close:
                    continue
                overlap = min(b['x2'], line['x2']) - max(b['x1'], line['x1'])
                narrower = min(b['width'], line['width'])
                if narrower > 0 and overlap > 0.3 * narrower:
                    line['boxes'].append(b)
                    line['x1'] = min(line['x1'], b['x1'])
                    line['y1'] = min(line['y1'], b['y1'])
                    line['x2'] = max(line['x2'], b['x2'])
                    line['y2'] = max(line['y2'], b['y2'])
                    line['cx'] = (line['x1'] + line['x2']) / 2
                    line['cy'] = (line['y1'] + line['y2']) / 2
                    line['width']  = line['x2'] - line['x1']
                    line['height'] = line['y2'] - line['y1']
                    placed = True
                    break
            if not placed:
                lines.append({
                    'boxes': [b],
                    'x1': b['x1'], 'y1': b['y1'], 'x2': b['x2'], 'y2': b['y2'],
                    'cx': b['cx'], 'cy': b['cy'],
                    'width': b['width'], 'height': b['height'],
                })
        for line in lines:
            line['boxes'].sort(key=lambda b: b['cx'])
            line['text'] = ' '.join(b['text'] for b in line['boxes'])
            line['conf'] = (min(b['conf'] for b in line['boxes'])
                            if line['boxes'] else 0.0)
            line['bbox'] = [
                [line['x1'], line['y1']],
                [line['x2'], line['y1']],
                [line['x2'], line['y2']],
                [line['x1'], line['y2']],
            ]
        lines.sort(key=lambda l: l['cy'])
        return lines

    def _group_dialogue(self, lines, bubbles):
        if not lines:
            return []
        for line in lines:
            line['bubble'] = self._bubble_for_box(line['bbox'], bubbles)
        bubble_line_count = {}
        for line in lines:
            if line['bubble'] is not None:
                bubble_line_count[line['bubble']] = bubble_line_count.get(line['bubble'], 0) + 1
        strong_bubbles = {bid for bid, n in bubble_line_count.items() if n >= 2}
        heights = [l['height'] for l in lines]
        median_h = sorted(heights)[len(heights) // 2] if heights else 20
        n = len(lines)
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
        def v_gap(a, b):
            if a['y1'] < b['y1']:
                return b['y1'] - a['y2']
            return a['y1'] - b['y2']
        for i in range(n):
            for j in range(i + 1, n):
                a, b = lines[i], lines[j]
                ba, bb = a['bubble'], b['bubble']
                gap = v_gap(a, b)
                if ba is not None and ba == bb:
                    if gap < 4.0 * median_h:
                        union(i, j)
                    continue
                if (ba is not None and bb is not None
                        and ba != bb
                        and ba in strong_bubbles and bb in strong_bubbles):
                    continue
                overlap = min(a['x2'], b['x2']) - max(a['x1'], b['x1'])
                narrower = min(a['width'], b['width'])
                if narrower <= 0 or overlap < 0.4 * narrower:
                    continue
                if gap < 1.6 * median_h:
                    union(i, j)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        dialogues = []
        for indices in groups.values():
            group = [lines[i] for i in indices]
            group.sort(key=lambda l: l['cy'])
            dialogues.append(group)
        dialogues.sort(key=lambda g: g[0]['cy'])
        return dialogues

    def _join_lines(self, line_texts):
        if not line_texts:
            return ""
        if self.source_lang in self._NO_SPACE_SOURCES:
            return "".join(line_texts)
        return " ".join(line_texts)

    def _build_regions_from_dialogues(self, dialogues):
        out = []
        for dlg in dialogues:
            all_boxes = []
            for line in dlg:
                all_boxes.extend(line['boxes'])
            if not all_boxes:
                continue
            all_x = [b['x1'] for b in all_boxes] + [b['x2'] for b in all_boxes]
            all_y = [b['y1'] for b in all_boxes] + [b['y2'] for b in all_boxes]
            bbox = [
                [min(all_x), min(all_y)],
                [max(all_x), min(all_y)],
                [max(all_x), max(all_y)],
                [min(all_x), max(all_y)],
            ]
            line_texts = [l['text'] for l in dlg]
            joined = self._join_lines(line_texts)
            if not self._is_plausible_region(joined):
                if os.environ.get("CT_DEBUG"):
                    print(f"[pipeline] dropped implausible region: {joined!r}")
                continue
            out.append({
                'bbox':        bbox,
                'text':        joined,
                'lines':       line_texts,
                'line_bboxes': [l['bbox'] for l in dlg],
                'line_quads':  [[b['bbox'] for b in l['boxes']] for l in dlg],
                'quads':       [b['bbox'] for b in all_boxes],
            })
        return out

    _CJK_RANGES = [
        ('一', '鿿'), ('぀', 'ゟ'), ('゠', 'ヿ'), ('가', '힯'), ('㐀', '䶿'),
    ]

    def _has_cjk(self, text):
        return any(low <= c <= high for c in text for low, high in self._CJK_RANGES)

    _LATIN_TARGETS = {"en", "es", "fr", "de", "pt", "ru", "it", "vi", "th"}

    def _translate_texts(self, texts):
        out = []
        debug = bool(os.environ.get("CT_DEBUG"))
        for i, text in enumerate(texts):
            text = text.strip()
            if not text:
                if debug:
                    print(f"[pipeline] translate[{i:02d}] (empty input)")
                out.append("")
                continue
            try:
                result = self.translator.translate(text)
                if result is None:
                    if debug:
                        print(f"[pipeline] translate[{i:02d}] {text!r} -> None")
                    out.append("")
                    continue
                translated = result.strip()
                blanked = False
                if (translated
                        and self._has_cjk(translated)
                        and self.target_lang in self._LATIN_TARGETS):
                    blanked = True
                    translated = ""
                if debug:
                    note = "  [BLANKED: still CJK]" if blanked else ""
                    print(f"[pipeline] translate[{i:02d}] {text!r} -> {translated!r}{note}")
                out.append(translated)
            except Exception as e:
                if debug:
                    print(f"[pipeline] translate[{i:02d}] FAILED {text!r}: {e!r}")
                out.append("")
        return out

    def _build_mask_from_quads(self, shape, quads_per_region):
        mask = np.zeros(shape, dtype=np.uint8)
        for quads in quads_per_region:
            for q in quads:
                pts = np.array([[int(p[0]), int(p[1])] for p in q], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
        return mask

    def _dump_debug_v2(self, img_np, bubbles, region_specs, lines, image_path):
        try:
            vis = img_np.copy()
            for (x1, y1, x2, y2) in bubbles:
                cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 128, 255), 1)
            for l in lines:
                cv2.rectangle(vis,
                              (int(l['x1']), int(l['y1'])),
                              (int(l['x2']), int(l['y2'])),
                              (255, 200, 0), 1)
            for i, r in enumerate(region_specs):
                pts = np.array([[int(p[0]), int(p[1])] for p in r['bbox']], dtype=np.int32)
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                cv2.putText(vis, str(i),
                            (int(pts[0][0]), int(pts[0][1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            out = Path(image_path).with_name(Path(image_path).stem + "_debug.png")
            Image.fromarray(vis).save(str(out))
            print(f"[pipeline] debug image → {out}")
        except Exception as e:
            print(f"[pipeline] debug dump failed: {e}")

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
            draw.text((x1 + max(0, (box_w - text_w) // 2), cur_y),
                      line, fill=fg, font=font)
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
        tile = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 0))
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
            tile_draw.text((max(0, (box_w - text_w) // 2), cur_y),
                           line, fill=fg, font=font)
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
            "zh-CN": [r"C:\Windows\Fonts\msyh.ttc",
                      r"C:\Windows\Fonts\simsun.ttc",
                      r"C:\Windows\Fonts\simhei.ttf"],
            "zh-TW": [r"C:\Windows\Fonts\msjh.ttc",
                      r"C:\Windows\Fonts\mingliu.ttc"],
            "ja":    [r"C:\Windows\Fonts\meiryo.ttc",
                      r"C:\Windows\Fonts\msgothic.ttc",
                      r"C:\Windows\Fonts\msmincho.ttc"],
            "ko":    [r"C:\Windows\Fonts\malgun.ttf",
                      r"C:\Windows\Fonts\gulim.ttc",
                      r"C:\Windows\Fonts\batang.ttc"],
        }
        if target_lang in CJK_FONTS:
            for p in CJK_FONTS[target_lang]:
                if os.path.exists(p):
                    return p
        for p in [r"C:\Windows\Fonts\arial.ttf",
                  r"C:\Windows\Fonts\calibri.ttf",
                  r"C:\Windows\Fonts\segoeui.ttf",
                  r"C:\Windows\Fonts\tahoma.ttf"]:
            if os.path.exists(p):
                return p
        return None