import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

GOOGLE_TARGET_CODES = {
    "en": "en", "ja": "ja", "ko": "ko",
    "zh-CN": "zh-CN", "zh-TW": "zh-TW",
    "es": "es", "fr": "fr", "de": "de", "pt": "pt",
    "ru": "ru", "it": "it", "vi": "vi", "th": "th",
}


class _ThrottledTranslator:
    def __init__(self, translator, delay: float = 0.20, timeout: float = 30.0):
        self._t = translator
        self._delay = delay
        self._timeout = timeout
        self._lock = threading.Lock()

    def translate(self, text: str) -> str:
        result = [None]
        error = [None]

        def _call():
            try:
                result[0] = self._t.translate(text)
            except Exception as e:
                error[0] = e

        with self._lock:
            t = threading.Thread(target=_call, daemon=True)
            t.start()
            t.join(timeout=self._timeout)
            if t.is_alive():
                raise TimeoutError(f"translate timed out after {self._timeout}s")
            if self._delay:
                time.sleep(self._delay)

        if error[0] is not None:
            raise error[0]
        return result[0] or ""


class PaddleOCRWrapper:
    def __init__(self, lang: str = "korean", use_gpu: bool = False):
        from paddleocr import PaddleOCR

        paddle_lang_map = {
            "ko": "korean",
            "ja": "japan",
            "ch_sim": "ch",
            "ch_tra": "chinese_cht",
            "en": "en",
        }
        self._lang = paddle_lang_map.get(lang, "korean")

        try:
            self._ocr = PaddleOCR(
                lang=self._lang,
                enable_mkldnn=False,
                use_textline_orientation=True,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
        except TypeError:
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self._lang,
                use_gpu=use_gpu,
                show_log=False,
                enable_mkldnn=False,
            )

        print(f"[PaddleOCR] Initialised with lang='{self._lang}' (mkldnn disabled)")

    def readtext(self, img_np, **kwargs):
        if img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = np.ascontiguousarray(img_np[:, :, ::-1])
        else:
            img_bgr = img_np

        try:
            results = self._ocr.ocr(img_bgr)
        except TypeError:
            results = self._ocr.ocr(img_bgr, cls=True)

        if not results:
            return []
        first = results[0]
        if first is None:
            return []
        out = []

        if isinstance(first, dict):
            texts  = first.get("rec_texts")
            scores = first.get("rec_scores")
            boxes  = first.get("rec_boxes")
            if boxes is None:
                boxes = first.get("dt_polys")

            texts_list  = list(texts) if texts is not None else []
            scores_list = list(scores) if scores is not None else []
            boxes_list  = list(boxes) if boxes is not None else []

            for i, txt in enumerate(texts_list):
                conf = float(scores_list[i]) if i < len(scores_list) else 0.0
                if i >= len(boxes_list):
                    continue
                box = boxes_list[i]
                if box is None:
                    continue
                try:
                    arr = np.asarray(box).reshape(-1).astype(float)
                    if arr.size == 4:
                        x1, y1, x2, y2 = arr[0], arr[1], arr[2], arr[3]
                    elif arr.size == 8:
                        xs = arr[0::2]; ys = arr[1::2]
                        x1, y1 = float(xs.min()), float(ys.min())
                        x2, y2 = float(xs.max()), float(ys.max())
                    else:
                        continue
                    quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                except Exception:
                    continue
                out.append((quad, str(txt), conf))
            return out

        for line in first:
            bbox_points = line[0]
            text_info = line[1]
            out.append((bbox_points, text_info[0], float(text_info[1])))
        return out


class MangaOCRWrapper:
    def __init__(self):
        import easyocr
        from manga_ocr import MangaOcr
        try:
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        self._detector = easyocr.Reader(["ja", "en"], gpu=use_gpu)
        self._recogniser = MangaOcr()

    def readtext(self, img_np, **kwargs):
        from PIL import Image
        detections = self._detector.readtext(img_np, detail=1, paragraph=False, **kwargs)
        if not detections:
            return []
        pil_img = Image.fromarray(img_np)
        out = []
        for bbox, _easyocr_text, conf in detections:
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            x1 = max(0, int(min(xs))); y1 = max(0, int(min(ys)))
            x2 = min(pil_img.width, int(max(xs)))
            y2 = min(pil_img.height, int(max(ys)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            try:
                text = self._recogniser(pil_img.crop((x1, y1, x2, y2)))
            except Exception:
                text = _easyocr_text
            if text:
                out.append((bbox, text, float(conf)))
        return out


_OPUS_SRC_CODES = {
    "ja": "ja", "ja_manga": "ja", "ko": "ko",
    "ch_sim": "zh", "ch_tra": "zh", "en": "en",
}
_OPUS_TGT_CODES = {
    "en": "en", "ja": "ja", "ko": "ko",
    "zh-CN": "zh", "zh-TW": "zh",
    "es": "es", "fr": "fr", "de": "de", "pt": "pt",
    "ru": "ru", "it": "it", "vi": "vi", "th": "th",
}


class OpusMTTranslator:
    def __init__(self, from_ocr_code: str, to_code: str):
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as e:
            raise RuntimeError(
                "Marian classes not available. Install sentencepiece:\n"
                "  pip install sentencepiece sacremoses"
            ) from e

        self._torch = torch
        src = _OPUS_SRC_CODES.get(from_ocr_code, "en")
        tgt = _OPUS_TGT_CODES.get(to_code, "en")

        candidates = (
            f"Helsinki-NLP/opus-mt-{src}-{tgt}",
            f"Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}",
        )

        errors = []
        for candidate in candidates:
            try:
                print(f"[OpusMT] Loading {candidate}…")
                self._tokenizer = MarianTokenizer.from_pretrained(candidate)
                self._model = MarianMTModel.from_pretrained(candidate)
                self._model.eval()
                print(f"[OpusMT] Loaded {candidate}")
                return
            except Exception as e:
                msg = f"{candidate}: {type(e).__name__}: {e}"
                print(f"[OpusMT] Failed — {msg}")
                errors.append(msg)

        raise RuntimeError(
            f"Could not load any OPUS-MT model for {src}→{tgt}:\n  "
            + "\n  ".join(errors)
        )

    def translate(self, text: str) -> str:
        tokens = self._tokenizer(
            [text], return_tensors="pt", truncation=True,
            padding=True, max_length=512
        )
        with self._torch.no_grad():
            output = self._model.generate(**tokens, max_length=512)
        return self._tokenizer.decode(output[0], skip_special_tokens=True)


class NLLBTranslator:
    def __init__(self, from_ocr_code: str, to_code: str,
                 model_name: str = "facebook/nllb-200-3.3B"):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()
        self.src_lang = self._get_flores_code(from_ocr_code)
        self.tgt_lang = self._get_flores_code(to_code)

    def _get_flores_code(self, code: str) -> str:
        mapping = {
            "ja": "jpn_Jpan", "ko": "kor_Hang",
            "ch_sim": "zho_Hans", "ch_tra": "zho_Hant",
            "en": "eng_Latn", "zh-CN": "zho_Hans", "zh-TW": "zho_Hant",
            "es": "spa_Latn", "fr": "fra_Latn", "de": "deu_Latn",
            "pt": "por_Latn", "ru": "rus_Cyrl", "it": "ita_Latn",
            "vi": "vie_Latn", "th": "tha_Thai",
        }
        return mapping.get(code, "eng_Latn")

    def translate(self, text: str) -> str:
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.src_lang

        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=1,
                do_sample=False,
                early_stopping=True,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class TranslationWorker(QThread):
    progress = Signal(int, str)
    step = Signal(str)
    image_done = Signal(str, object)
    finished = Signal()
    error = Signal(str)
    fatal_error = Signal(str)

    def __init__(self, files, settings):
        super().__init__()
        self.files = files
        self.settings = settings
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            self.error.emit(f"Fatal error: {e}")
            traceback.print_exc()
        finally:
            self.finished.emit()

    def _run_pipeline(self):
        from app.pipeline import ComicTranslationPipeline

        source_lang = self.settings["source_lang"]
        target_lang = self.settings["target_lang"]
        service = self.settings["service"]

        if source_lang == "ja_manga":
            self.progress.emit(0, "Loading Manga OCR — first run downloads ~450 MB…")
            reader = MangaOCRWrapper()
        else:
            self.progress.emit(0, "Loading PaddleOCR — first run downloads ~200 MB…")
            try:
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
            reader = PaddleOCRWrapper(lang=source_lang, use_gpu=use_gpu)

        self.progress.emit(0, "Setting up translator…")
        try:
            translator = self._build_translator(service, source_lang, target_lang)
        except Exception as e:
            print(f"[TranslationWorker] Translator init failed: {e}")
            traceback.print_exc()
            self.fatal_error.emit(str(e))
            return

        pipeline = ComicTranslationPipeline(
            source_lang=source_lang,
            target_lang=target_lang,
            ocr_reader=reader,
            translator=translator,
        )

        manual_zones = self.settings.get("manual_zones", {})

        for i, file_path in enumerate(self.files):
            if self._stopped:
                break
            name = Path(file_path).name
            self.progress.emit(i, f"Page {i + 1} of {len(self.files)}  —  {name}")
            self.step.emit("")
            forced = manual_zones.get(file_path, [])
            print(f"[TranslationWorker] Processing {name}  forced_zones={len(forced)}")
            try:
                result_img, base_img, regions = pipeline.process_image(
                    file_path,
                    forced_bboxes=forced or None,
                    progress_cb=lambda s: self.step.emit(s),
                )
                print(f"[TranslationWorker] Done {name}  regions={len(regions)}")
                self.image_done.emit(file_path, {
                    'image': result_img,
                    'base': base_img,
                    'regions': regions,
                })
            except Exception as e:
                print(f"[TranslationWorker] Error on {name}: {e}")
                self.error.emit(f"Skipped {name}: {e}")
                traceback.print_exc()

        self.progress.emit(len(self.files), "Done!")

    def _build_translator(self, service, source_lang, target_lang):
        from deep_translator import GoogleTranslator

        print(f"[TranslationWorker] Building translator: service={service!r} "
              f"src={source_lang!r} tgt={target_lang!r}")

        if service == "Google Translate":
            raw = GoogleTranslator(
                source="auto",
                target=GOOGLE_TARGET_CODES.get(target_lang, target_lang),
            )
            print("[TranslationWorker] Google Translate ready (throttled)")
            return _ThrottledTranslator(raw, delay=2.5, timeout=30.0)

        elif service == "OPUS-MT (Offline)":
            opus_src = source_lang
            self.progress.emit(
                0,
                f"Loading OPUS-MT {opus_src}→{target_lang} — first run downloads ~350 MB…"
            )
            print(f"[TranslationWorker] Loading OPUS-MT {opus_src}→{target_lang}")
            t = OpusMTTranslator(opus_src, target_lang)
            print("[TranslationWorker] OPUS-MT ready")
            return t

        elif service == "NLLB (Offline)":
            self.progress.emit(
                0,
                "Loading NLLB-200 3.3B — first run downloads ~6.5 GB…"
            )
            print("[TranslationWorker] Loading NLLB-200 3.3B")
            t = NLLBTranslator(source_lang, target_lang)
            print("[TranslationWorker] NLLB ready")
            return t

        raise ValueError(f"Unknown service: {service!r}")