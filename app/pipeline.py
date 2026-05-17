"""
pipeline.py — Core image processing logic.

Flow for each image:
  1. Load the image via PIL (handles more formats than cv2 alone)
  2. Run EasyOCR to find every text region + read the text
  3. Translate each detected string
  4. Paint over the original text regions using OpenCV inpainting
     (TELEA fills the hole by sampling surrounding pixels)
  5. Draw the translated strings back in the same spots using Pillow
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ComicTranslationPipeline:
    def __init__(self, source_lang, target_lang, ocr_reader, translator):
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.ocr_reader = ocr_reader      # EasyOCR Reader instance
        self.translator = translator      # deep_translator instance
        self._font_cache = {}             # size → ImageFont, avoids re-loading the same font repeatedly
        self._font_path = self._find_font(target_lang)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def process_image(self, image_path, forced_bboxes=None):
        """
        Translate all text in one image file.
        Returns a PIL Image with original text replaced by translated text.
        """
        # Load via PIL so exotic formats (TIFF, WEBP, GIF, BMP) all work
        pil_orig = Image.open(str(image_path)).convert("RGB")
        img_np = np.array(pil_orig)  # EasyOCR and OpenCV both work with numpy arrays

        # --- Detect and read text ---
        # detail=1  → always returns (bbox, text, confidence) tuples with paragraph=False.
        # We keep paragraph=False for all languages:
        #   • paragraph=True merges nearby regions including separate speech bubbles
        #     into one huge bounding box, which overwrites large parts of the image.
        #   • EasyOCR's CRAFT detector already groups characters within a single
        #     text line into one detection without paragraph mode.
        _cjk = self.source_lang in {"ch_sim", "ch_tra", "ja", "ja_manga", "ko"}

        # For CJK we nudge the CRAFT detector to be slightly more aggressive
        # (text_threshold / low_text just below their defaults of 0.7 / 0.4).
        # mag_ratio is intentionally left at 1 — upscaling the image before
        # detection increases false positives on artwork/texture patterns.
        cjk_kwargs = dict(text_threshold=0.6, low_text=0.35) if _cjk else {}
        results = self.ocr_reader.readtext(img_np, detail=1, paragraph=False,
                                           **cjk_kwargs)

        # ── Filter 1: confidence ────────────────────────────────────────── #
        # CJK comic fonts score lower than Latin script, so the threshold is
        # more permissive.  The bbox-size and alphanumeric filters below catch
        # the false positives that a too-low threshold would otherwise let through.
        min_conf = 0.1 if _cjk else 0.3
        results = [r for r in results if r[2] > min_conf]

        # ── Filter 2: bounding-box size cap ─────────────────────────────── #
        # Any box covering more than 12 % of the image is almost certainly a
        # false positive (panel border, large artwork texture, full-page image).
        # Real speech-bubble text never fills that much of a page.
        img_h, img_w = img_np.shape[:2]
        max_box_area = img_h * img_w * 0.12
        results = [
            r for r in results
            if (max(p[0] for p in r[0]) - min(p[0] for p in r[0])) *
               (max(p[1] for p in r[0]) - min(p[1] for p in r[0])) < max_box_area
        ]

        # ── Filter 3: must contain at least one alphanumeric character ───── #
        # Detections that are entirely punctuation, symbols, or whitespace
        # (e.g. "—", "...", decorative borders) are not translatable.
        results = [r for r in results if any(c.isalnum() for c in r[1])]

        # ── Forced regions (user-marked zones, bypass all filters) ────────── #
        # Crop each marked region and run OCR on just that crop so the
        # recognised text can go through the normal translate → render flow.
        if forced_bboxes:
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

        if not results:
            return pil_orig, pil_orig.copy(), []  # Nothing detected, return original untouched

        # --- Translate ---
        texts = [r[1] for r in results]
        translated = self._translate_texts(texts)

        # --- Inpaint (erase original text) ---
        # Convert RGB numpy → BGR for OpenCV
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Build a binary mask: white = pixels to remove (the text areas)
        mask = self._build_mask(img_cv.shape[:2], results)

        # Slightly expand the mask so stray edge pixels of letters are also covered
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        # INPAINT_TELEA: fast, good-looking algorithm that fills the white mask areas
        # by sampling colours from the surrounding pixels
        img_inpainted = cv2.inpaint(img_cv, mask, 5, cv2.INPAINT_TELEA)

        # --- Draw translated text ---
        # Back to PIL (RGB) so we can use ImageDraw for text rendering
        img_pil = Image.fromarray(cv2.cvtColor(img_inpainted, cv2.COLOR_BGR2RGB))
        base_pil = img_pil.copy()   # inpainted image without any text drawn — kept for re-editing
        draw = ImageDraw.Draw(img_pil)

        regions = []
        for result, trans in zip(results, translated):
            if trans:
                fs = self._render(draw, img_pil, trans, result[0])
                regions.append({'bbox': result[0], 'text': trans, 'font_size': fs})

        return img_pil, base_pil, regions

    # ------------------------------------------------------------------ #
    # Translation                                                          #
    # ------------------------------------------------------------------ #

    # Unicode ranges that constitute CJK script output
    _CJK_RANGES = [
        ('一', '鿿'),   # CJK Unified Ideographs (core Han)
        ('぀', 'ゟ'),   # Hiragana
        ('゠', 'ヿ'),   # Katakana
        ('가', '힯'),   # Hangul Syllables
        ('㐀', '䶿'),   # CJK Extension A
    ]

    def _has_cjk(self, text):
        return any(low <= c <= high for c in text for low, high in self._CJK_RANGES)

    # Target languages whose scripts cannot render CJK glyphs with standard
    # Latin fonts — used to detect "translation returned the original" failures.
    _LATIN_TARGETS = {"en", "es", "fr", "de", "pt", "ru", "it", "vi", "th"}

    def _translate_texts(self, texts):
        """Translate a list of strings.

        Returns '' when translation fails so the inpainted background shows
        cleanly.  Also discards any result that still contains CJK characters
        when the target language uses a Latin font — that means the translator
        returned the original unchanged (common for very short or ambiguous
        OCR output), and rendering those glyphs with Arial produces □ squares.
        """
        out = []
        for text in texts:
            text = text.strip()
            if not text:
                out.append("")
                continue
            try:
                result = self.translator.translate(text)
                translated = result.strip() if result else ""
                # Discard if the translation came back as CJK when we expected Latin
                if translated and self._has_cjk(translated) and \
                        self.target_lang in self._LATIN_TARGETS:
                    translated = ""
                out.append(translated)
            except Exception:
                out.append("")
        return out

    # ------------------------------------------------------------------ #
    # Mask building                                                        #
    # ------------------------------------------------------------------ #

    def _build_mask(self, shape, results):
        """Create a binary mask image with each detected text region filled white."""
        mask = np.zeros(shape, dtype=np.uint8)
        for r in results:
            # EasyOCR returns bboxes as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corners)
            pts = np.array([[int(p[0]), int(p[1])] for p in r[0]], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        return mask

    # ------------------------------------------------------------------ #
    # Text rendering                                                       #
    # ------------------------------------------------------------------ #

    def _render(self, draw, img_pil, text, bbox, font_size=None):
        """
        Draw translated text into the bounding box region.
        Returns the actual font size used (int).

        font_size: if given, use exactly that size (still word-wraps); if None, auto-fit.
        """
        pts = [(int(p[0]), int(p[1])) for p in bbox]
        x1 = max(0, min(p[0] for p in pts))
        y1 = max(0, min(p[1] for p in pts))
        x2 = min(img_pil.width,  max(p[0] for p in pts))
        y2 = min(img_pil.height, max(p[1] for p in pts))

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w <= 2 or box_h <= 2:
            return font_size or 10

        # CJK vertical text columns are tall and narrow (e.g. 20 × 150 px).
        # Horizontal English replacement needs a minimum usable width.
        # We expand the render area to at least 60 px wide (clamped to image edges)
        # while keeping the same centre X so we don't drift into unrelated panels.
        MIN_RENDER_W = 60
        if box_w < MIN_RENDER_W:
            cx = (x1 + x2) // 2
            x1 = max(0, cx - MIN_RENDER_W // 2)
            x2 = min(img_pil.width, x1 + MIN_RENDER_W)
            box_w = x2 - x1

        # Sample the average colour of the region to use as the background fill
        # (usually white inside speech bubbles, but can be any colour on colour pages)
        region = img_pil.crop((x1, y1, x2, y2))
        bg = self._dominant_color(region)

        # Choose black or white text depending on how bright the background is
        fg = self._contrast_color(bg)

        # Fill the region with the background colour (clears the inpainted area cleanly)
        draw.rectangle([x1, y1, x2, y2], fill=bg)

        if font_size:
            font  = self._get_font(font_size)
            lines = self._wrap(text, font, box_w, draw)
        else:
            font, lines = self._fit_text(text, box_w, box_h, draw)

        if not lines:
            return font_size or 10

        actual_size = font.size if hasattr(font, 'size') else 10

        line_h = self._line_height(font)
        total_h = line_h * len(lines)

        # Centre the block of text vertically inside the box
        cur_y = y1 + max(0, (box_h - total_h) // 2)

        for line in lines:
            if cur_y + line_h > y2:
                break  # No more vertical space
            try:
                text_w = draw.textbbox((0, 0), line, font=font)[2]
            except Exception:
                text_w = len(line) * 7  # rough fallback if textbbox isn't available

            # Centre each line horizontally inside the box
            draw.text((x1 + max(0, (box_w - text_w) // 2), cur_y), line, fill=fg, font=font)
            cur_y += line_h

        return actual_size

    def render_from_regions(self, base_img, regions):
        """
        Re-render a previously processed image using an updated regions list.
        base_img: the inpainted PIL Image (text erased, no translation drawn yet).
        regions: list of {'bbox', 'text', 'font_size'} dicts.
        Returns a new PIL Image.
        """
        img = base_img.copy()
        draw = ImageDraw.Draw(img)
        for r in regions:
            if r.get('text'):
                self._render(draw, img, r['text'], r['bbox'], font_size=r.get('font_size'))
        return img

    # ------------------------------------------------------------------ #
    # Colour helpers                                                       #
    # ------------------------------------------------------------------ #

    def _dominant_color(self, region):
        """Return the average (R, G, B) colour of a PIL image crop."""
        a = np.array(region.convert("RGB"))
        return (int(a[:, :, 0].mean()), int(a[:, :, 1].mean()), int(a[:, :, 2].mean()))

    def _contrast_color(self, bg):
        """Return black or white, whichever contrasts better against bg."""
        # Standard luminance formula: perceived brightness
        lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
        return (0, 0, 0) if lum > 128 else (255, 255, 255)

    # ------------------------------------------------------------------ #
    # Font / text sizing helpers                                           #
    # ------------------------------------------------------------------ #

    def _line_height(self, font):
        """Line height = font size + small leading gap."""
        try:
            return font.size + 3
        except AttributeError:
            return 13  # default font has no .size attribute

    def _get_font(self, size):
        """Load (and cache) a TrueType font at the given pixel size."""
        if size not in self._font_cache:
            try:
                if self._font_path:
                    self._font_cache[size] = ImageFont.truetype(self._font_path, size)
                else:
                    self._font_cache[size] = ImageFont.load_default()
            except Exception:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _fit_text(self, text, max_w, max_h, draw):
        """
        Try decreasing font sizes from 18pt down to 7pt until the wrapped text
        fits inside the box. Returns (font, list_of_lines).
        """
        for size in range(18, 6, -1):
            font = self._get_font(size)
            lines = self._wrap(text, font, max_w, draw)
            if self._line_height(font) * len(lines) <= max_h:
                return font, lines

        # Last resort: truncate with ellipsis at smallest size
        font = self._get_font(7)
        short = text if len(text) <= 20 else text[:17] + "..."
        return font, [short]

    def _wrap(self, text, font, max_w, draw):
        """
        Word-wrap text to fit within max_w pixels.
        Returns a list of lines.
        """
        words = text.split()
        if not words:
            return [""]

        lines, current = [], []
        for word in words:
            test = " ".join(current + [word])
            try:
                w = draw.textbbox((0, 0), test, font=font)[2]
            except Exception:
                w = len(test) * 7  # rough pixel estimate
            if w > max_w and current:
                lines.append(" ".join(current))
                current = [word]
            else:
                current.append(word)

        if current:
            lines.append(" ".join(current))
        return lines or [""]

    # ------------------------------------------------------------------ #
    # Font discovery                                                       #
    # ------------------------------------------------------------------ #

    def _find_font(self, target_lang=None):
        """
        Return a font that can render the target language.
        Arial/Calibri don't include CJK glyphs — without the right font,
        Chinese/Japanese/Korean output renders as empty boxes.
        """
        # Windows built-in fonts that cover CJK scripts.
        # Listed best-first: prefer modern/clean designs.
        CJK_FONTS = {
            "zh-CN": [
                r"C:\Windows\Fonts\msyh.ttc",      # Microsoft YaHei — modern Simplified Chinese
                r"C:\Windows\Fonts\simsun.ttc",     # SimSun — classic Simplified Chinese
                r"C:\Windows\Fonts\simhei.ttf",     # SimHei — another option
            ],
            "zh-TW": [
                r"C:\Windows\Fonts\msjh.ttc",       # Microsoft JhengHei — Traditional Chinese
                r"C:\Windows\Fonts\mingliu.ttc",    # MingLiU — classic Traditional Chinese
            ],
            "ja": [
                r"C:\Windows\Fonts\meiryo.ttc",     # Meiryo — clean Japanese
                r"C:\Windows\Fonts\msgothic.ttc",   # MS Gothic
                r"C:\Windows\Fonts\msmincho.ttc",   # MS Mincho
            ],
            "ko": [
                r"C:\Windows\Fonts\malgun.ttf",     # Malgun Gothic — modern Korean
                r"C:\Windows\Fonts\gulim.ttc",      # Gulim
                r"C:\Windows\Fonts\batang.ttc",     # Batang
            ],
        }

        # If rendering CJK, try the language-specific font list first
        if target_lang in CJK_FONTS:
            for p in CJK_FONTS[target_lang]:
                if os.path.exists(p):
                    return p

        # Latin-script fallback (also used when a CJK font wasn't found above)
        for p in [
            r"C:\Windows\Fonts\arial.ttf",
            r"C:\Windows\Fonts\calibri.ttf",
            r"C:\Windows\Fonts\segoeui.ttf",
            r"C:\Windows\Fonts\tahoma.ttf",
        ]:
            if os.path.exists(p):
                return p

        return None  # PIL's built-in bitmap font as last resort
