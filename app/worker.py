import threading
import time
import traceback
from pathlib import Path

import torch
from PySide6.QtCore import QThread, Signal

GOOGLE_TARGET_CODES = {
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "zh-CN": "zh-CN",
    "zh-TW": "zh-TW",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "ru": "ru",
    "it": "it",
    "vi": "vi",
    "th": "th",
}

EASYOCR_LANG_MAP = {
    "ja": ["ja", "en"],
    "ko": ["ko", "en"],
    "ch_sim": ["ch_sim", "en"],
    "ch_tra": ["ch_tra", "en"],
    "en": ["en"],
}

class _ThrottledTranslator:
    def __init__(self, translator, delay: float = 0.0, timeout: float = 20.0):
        self._t = translator
        self._delay = delay
        self._timeout = timeout

    def translate(self, text: str) -> str:
        result = [None]

        def _call():
            try:
                result[0] = self._t.translate(text)
            except Exception:
                pass

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=self._timeout)

        if self._delay:
            time.sleep(self._delay)

        return result[0] or ""

class RapidOCRWrapper:
    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()

    def readtext(self, img_np, **kwargs):
        result, _ = self._ocr(img_np)
        if result is None:
            return []
        out = []
        for item in result:
            bbox = item[0]
            text = item[1]
            conf = float(item[2])
            out.append((bbox, text, conf))
        return out

class MangaOCRWrapper:
    def __init__(self):
        import easyocr
        from manga_ocr import MangaOcr
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
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
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            x1 = max(0, int(min(xs)))
            y1 = max(0, int(min(ys)))
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
    "ja": "ja",
    "ja_manga": "ja",
    "ko": "ko",
    "ch_sim": "zh",
    "ch_tra": "zh",
    "en": "en",
}
_OPUS_TGT_CODES = {
    "en": "en",
    "ja": "ja",
    "ko": "ko",
    "zh-CN": "zh",
    "zh-TW": "zh",
    "es": "es",
    "fr": "fr",
    "de": "de",
    "pt": "pt",
    "ru": "ru",
    "it": "it",
    "vi": "vi",
    "th": "th",
}

class OpusMTTranslator:
    def __init__(self, from_ocr_code: str, to_code: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        self._torch = torch
        src = _OPUS_SRC_CODES.get(from_ocr_code, "en")
        tgt = _OPUS_TGT_CODES.get(to_code, "en")

        for candidate in (
            f"Helsinki-NLP/opus-mt-{src}-{tgt}",
            f"Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}",
        ):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(candidate)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(candidate)
                self._model.eval()
                return
            except Exception:
                pass

        raise ValueError(
            f"No OPUS-MT model found for {src}→{tgt}.  "
            f"Switch to 'NLLB (Offline)' for this language pair."
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
    def __init__(self, from_ocr_code: str, to_code: str, model_name: str = "facebook/nllb-200-3.3B"):
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
            "ja": "jpn_Jpan", "ko": "kor_Hang", "ch_sim": "zho_Hans",
            "ch_tra": "zho_Hant", "en": "eng_Latn", "zh-CN": "zho_Hans",
            "zh-TW": "zho_Hant", "es": "spa_Latn", "fr": "fra_Latn",
            "de": "deu_Latn", "pt": "por_Latn", "ru": "rus_Cyrl",
            "it": "ita_Latn", "vi": "vie_Latn", "th": "tha_Thai"
        }
        return mapping.get(code, "eng_Latn")

    def translate(self, text: str) -> str:
        if hasattr(self.tokenizer, "src_lang"):
            self.tokenizer.src_lang = self.src_lang

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
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
                early_stopping=True
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
        import easyocr
        from app.pipeline import ComicTranslationPipeline

        source_lang = self.settings["source_lang"]
        target_lang = self.settings["target_lang"]
        service = self.settings["service"]

        if source_lang in {"ch_sim", "ch_tra"}:
            self.progress.emit(0, "Loading RapidOCR Chinese model…")
            reader = RapidOCRWrapper()
        elif source_lang == "ja_manga":
            self.progress.emit(
                0,
                "Loading Manga OCR — first run downloads ~450 MB to HuggingFace cache, please wait…"
            )
            reader = MangaOCRWrapper()
        else:
            try:
                import torch
                torch_available = True
            except ImportError:
                torch_available = False

            if service == "NLLB (Offline)":
                use_gpu = False
            else:
                use_gpu = torch.cuda.is_available() if torch_available else False

            if source_lang == "auto":
                self.progress.emit(0, "Loading multi-language OCR model — first run may download ~400 MB, please wait…")
                ocr_langs = ["ja", "ch_sim", "ko", "en"]
            else:
                self.progress.emit(0, "Loading OCR model — first run may download ~200 MB, please wait…")
                ocr_langs = EASYOCR_LANG_MAP.get(source_lang, ["en"])
            reader = easyocr.Reader(ocr_langs, gpu=use_gpu)

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
            t = GoogleTranslator(
                source="auto",
                target=GOOGLE_TARGET_CODES.get(target_lang, target_lang),
            )
            print("[TranslationWorker] Google Translate ready")
            return t

        elif service == "OPUS-MT (Offline)":
            opus_src = "ja" if source_lang == "auto" else source_lang
            self.progress.emit(
                0,
                f"Loading OPUS-MT {opus_src}→{target_lang} model "
                f"— first run downloads ~350 MB to HuggingFace cache, please wait…"
            )
            print(f"[TranslationWorker] Loading OPUS-MT {opus_src}→{target_lang}")
            t = OpusMTTranslator(opus_src, target_lang)
            print("[TranslationWorker] OPUS-MT ready")
            return t

        elif service == "NLLB (Offline)":
            self.progress.emit(
                0,
                "Loading NLLB-200 3.3B model — first run downloads ~6.5 GB to HuggingFace cache, please wait…"
            )
            print("[TranslationWorker] Loading NLLB-200 3.3B")
            t = NLLBTranslator(source_lang, target_lang)
            print("[TranslationWorker] NLLB ready")
            return t

        raise ValueError(f"Unknown service: {service!r}")