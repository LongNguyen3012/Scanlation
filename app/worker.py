"""
worker.py — Background thread that runs the translation pipeline.

Why a separate thread?
  PySide6 (and all Qt apps) freeze the UI if you do heavy work on the main thread.
  QThread lets us run OCR + translation in the background while the window stays
  responsive. We communicate back to the UI using Qt Signals.
"""

import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal


GOOGLE_TARGET_CODES = {
    "en":    "en",
    "ja":    "ja",
    "ko":    "ko",
    "zh-CN": "zh-CN",
    "zh-TW": "zh-TW",
    "es":    "es",
    "fr":    "fr",
    "de":    "de",
    "pt":    "pt",
    "ru":    "ru",
    "it":    "it",
    "vi":    "vi",
    "th":    "th",
}

# EasyOCR HARD CONSTRAINT: every CJK script (ja, ko, ch_sim, ch_tra) can ONLY
# be paired with English in the same Reader.  Mixing any two CJK scripts raises a
# ValueError.  There is therefore no valid multi-CJK "auto-detect" list — the UI
# exposes explicit per-script options only.
EASYOCR_LANG_MAP = {
    "ja":     ["ja", "en"],
    "ko":     ["ko", "en"],
    "ch_sim": ["ch_sim", "en"],
    "ch_tra": ["ch_tra", "en"],
    "en":     ["en"],
}


# ── Throttle wrapper ────────────────────────────────────────────────────── #

class _ThrottledTranslator:
    """
    Wraps any translator object to add:
      - a hard per-call timeout (prevents DeepL/MyMemory hanging the worker
        when the server stops responding after hitting a rate limit)
      - a post-call delay (keeps request rate inside free-tier limits)

    The timeout is implemented with a daemon thread so a hung network call is
    simply abandoned rather than blocking the pipeline forever.
    """

    def __init__(self, translator, delay: float = 0.0, timeout: float = 20.0):
        self._t       = translator
        self._delay   = delay
        self._timeout = timeout

    def translate(self, text: str) -> str:
        result = [None]

        def _call():
            try:
                result[0] = self._t.translate(text)
            except Exception:
                pass   # leave result as None; caller treats None as ""

        t = threading.Thread(target=_call, daemon=True)
        t.start()
        t.join(timeout=self._timeout)
        # If t is still alive here the call timed out; the daemon thread is
        # abandoned (it will die when the process exits).

        if self._delay:
            time.sleep(self._delay)

        return result[0] or ""




# ── RapidOCR wrapper (Chinese OCR) ──────────────────────────────────────── #

class RapidOCRWrapper:
    """
    Wraps rapidocr-onnxruntime for Chinese OCR.

    Why RapidOCR instead of PaddleOCR?
      PaddleOCR v3 requires the full PaddlePaddle framework (~1 GB) as a
      separate install.  RapidOCR packages the same DBNet+CRNN models as
      ONNX files and runs them via onnxruntime — no framework needed, much
      smaller install, and compatible with PyInstaller packaging.

    Model caching:
      Models are bundled inside the rapidocr-onnxruntime wheel — no download
      required at runtime.  This makes the first run as fast as every other.
    """

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._ocr = RapidOCR()

    def readtext(self, img_np, **kwargs):
        """
        kwargs accepted for API compatibility with EasyOCR but ignored.
        Returns list of (bbox, text, confidence) matching EasyOCR's format.
        """
        result, _ = self._ocr(img_np)
        if result is None:
            return []
        out = []
        for item in result:
            bbox = item[0]           # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            text = item[1]
            conf = float(item[2])
            out.append((bbox, text, conf))
        return out


# ── MangaOCR wrapper (Japanese manga-specific OCR) ──────────────────────── #

class MangaOCRWrapper:
    """
    Hybrid OCR for Japanese manga: EasyOCR CRAFT detector + manga-ocr recogniser.

    Why hybrid?
      manga-ocr is a recognition-only model (VisionEncoderDecoder fine-tuned on
      manga pages).  It expects a pre-cropped text region, not a full page.
      We use EasyOCR's CRAFT to find bounding boxes, then replace EasyOCR's
      built-in CRNN recogniser with manga-ocr for each crop — dramatically
      improving accuracy on stylised manga fonts.

    First run downloads ~450 MB of model weights to the HuggingFace cache
    (%USERPROFILE%\\.cache\\huggingface\\) and reuses them on every subsequent run.
    """

    def __init__(self):
        import easyocr
        from manga_ocr import MangaOcr
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        self._detector   = easyocr.Reader(["ja", "en"], gpu=use_gpu)
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
            x1 = max(0, int(min(xs)));  y1 = max(0, int(min(ys)))
            x2 = min(pil_img.width, int(max(xs)))
            y2 = min(pil_img.height, int(max(ys)))
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            try:
                text = self._recogniser(pil_img.crop((x1, y1, x2, y2)))
            except Exception:
                text = _easyocr_text   # fall back to EasyOCR text if manga-ocr fails
            if text:
                out.append((bbox, text, float(conf)))
        return out




# ── OPUS-MT language code maps ───────────────────────────────────────────── #

# Helsinki-NLP model naming uses short ISO codes; zh covers both Simplified and
# Traditional Chinese (their zh models are trained on mixed scripts).
_OPUS_SRC_CODES = {
    "ja":       "ja",
    "ja_manga": "ja",
    "ko":       "ko",
    "ch_sim":   "zh",
    "ch_tra":   "zh",
    "en":       "en",
}
_OPUS_TGT_CODES = {
    "en":    "en",
    "ja":    "ja",
    "ko":    "ko",
    "zh-CN": "zh",
    "zh-TW": "zh",
    "es":    "es",
    "fr":    "fr",
    "de":    "de",
    "pt":    "pt",
    "ru":    "ru",
    "it":    "it",
    "vi":    "vi",
    "th":    "th",
}


class OpusMTTranslator:
    """
    Offline neural machine translation using Helsinki-NLP OPUS-MT (MarianMT) models.

    Per-language-pair model (~300–400 MB), downloaded to HuggingFace cache on first
    use.  Inference is 3–8× faster than NLLB on CPU; quality is similar or better for
    common pairs like ja→en, ko→en, zh→en.

    Not every src→tgt pair exists.  The constructor tries the standard name, then the
    larger 'tc-big' variant, and raises ValueError if neither is available — switch to
    NLLB (Offline) for that pair instead.
    """

    def __init__(self, from_ocr_code: str, to_code: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self._torch = torch
        src = _OPUS_SRC_CODES.get(from_ocr_code, "en")
        tgt = _OPUS_TGT_CODES.get(to_code, "en")

        # Try standard model first, then the higher-quality 'tc-big' variant
        for candidate in (
            f"Helsinki-NLP/opus-mt-{src}-{tgt}",
            f"Helsinki-NLP/opus-mt-tc-big-{src}-{tgt}",
        ):
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(candidate)
                self._model     = AutoModelForSeq2SeqLM.from_pretrained(candidate)
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


# ── Local GGUF LLM translator (Qwen3, Qwen2.5, Sakura, any instruction model) ── #

# Human-readable language names for system prompt construction
_LLM_LANG_NAMES = {
    "ja":       "Japanese",
    "ja_manga": "Japanese",
    "ko":       "Korean",
    "ch_sim":   "Chinese (Simplified)",
    "ch_tra":   "Chinese (Traditional)",
    "en":       "English",
    "zh-CN":    "Chinese (Simplified)",
    "zh-TW":    "Chinese (Traditional)",
    "es":       "Spanish",
    "fr":       "French",
    "de":       "German",
    "pt":       "Portuguese",
    "ru":       "Russian",
    "it":       "Italian",
    "vi":       "Vietnamese",
    "th":       "Thai",
}

# Sakura-style system prompt used when the model filename contains "sakura" or "galtransl"
# (case-insensitive).  For all other models a generic manga-translation prompt is used.
_SAKURA_SYSTEM_PROMPT = (
    "你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，"
    "并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。"
)


class LocalLLMTranslator:
    """Translates using a local GGUF model via llama-cpp-python."""

    def __init__(self, model_path: str, from_ocr_code: str, to_code: str,
                 n_gpu_layers: int = 0):
        import re
        self._re_think = re.compile(r"<think>.*?</think>", re.DOTALL)

        try:
            from llama_cpp import Llama
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "llama-cpp-python is not installed.\n\n"
                "CPU only:    pip install llama-cpp-python\n"
                "NVIDIA GPU:  pip install llama-cpp-python "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(
                f"GGUF model file not found:\n{model_path}\n\n"
                "Use 'Manage Models…' to download one, or browse to an existing .gguf file."
            )

        print(f"[LocalLLMTranslator] Loading {Path(model_path).name} "
              f"(gpu_layers={n_gpu_layers})…")
        try:
            self._llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                verbose=False,
            )
        except OSError as e:
            # 0xc000001d = STATUS_ILLEGAL_INSTRUCTION: llama-cpp-python was compiled
            # with CPU instructions (usually AVX-512) that this CPU does not support.
            # Note: 12th/13th gen Intel Core CPUs support AVX2 but NOT AVX-512.
            if getattr(e, "winerror", None) in (-1073741795, 3221225501):
                raise RuntimeError(
                    "CPU instruction error (STATUS_ILLEGAL_INSTRUCTION / 0xc000001d)\n\n"
                    "The installed llama-cpp-python wheel was compiled with CPU instructions\n"
                    "(likely AVX-512) that your processor does not support.\n"
                    "This commonly affects 12th/13th gen Intel Core CPUs, which support AVX2\n"
                    "but not AVX-512.\n\n"
                    "Fix option 1 — try the pre-built CPU wheel (easiest):\n"
                    "  pip install llama-cpp-python --force-reinstall "
                    "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu\n\n"
                    "Fix option 2 — rebuild without AVX-512 "
                    "(needs CMake + VS Build Tools with C++):\n"
                    "  set CMAKE_ARGS=-DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF "
                    "-DGGML_AVX512_VNNI=OFF\n"
                    "  pip install llama-cpp-python --no-binary llama-cpp-python "
                    "--force-reinstall\n\n"
                    "After reinstalling, restart the app."
                ) from e
            raise

        src_name = _LLM_LANG_NAMES.get(from_ocr_code, from_ocr_code)
        tgt_name = _LLM_LANG_NAMES.get(to_code, to_code)
        fname    = Path(model_path).name.lower()

        if any(k in fname for k in ("sakura", "galtransl")):
            self._system    = _SAKURA_SYSTEM_PROMPT
            self._user_tmpl = "将下面的日文文本翻译成中文：{text}"
        else:
            self._system = (
                f"You are a professional manga and comic translator. "
                f"Translate the following {src_name} text into {tgt_name}. "
                f"Preserve tone, emotion, and manga dialogue style. "
                f"Output ONLY the translated text — no explanations, notes, or alternatives."
            )
            self._user_tmpl = "{text}"

        print("[LocalLLMTranslator] Model loaded.")

    def translate(self, text: str) -> str:
        resp = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self._system},
                {"role": "user",   "content": self._user_tmpl.format(text=text)},
            ],
            temperature=0.1,
            top_p=0.3,
            max_tokens=512,
        )
        result = resp["choices"][0]["message"]["content"].strip()
        return self._re_think.sub("", result).strip()


# ── GGUF model download worker ───────────────────────────────────────────── #

class ModelDownloadWorker(QThread):
    """Downloads a GGUF file from HuggingFace with streaming progress."""
    progress = Signal(int, int)   # downloaded_mb, total_mb
    finished = Signal(str)        # path to completed file
    error    = Signal(str)

    def __init__(self, url: str, dest_path: str):
        super().__init__()
        self._url     = url
        self._dest    = dest_path
        self._stopped = False

    def stop(self):
        self._stopped = True

    def run(self):
        import requests
        try:
            print(f"[ModelDownloadWorker] GET {self._url}")
            resp = requests.get(self._url, stream=True, timeout=60)
            resp.raise_for_status()
            total_b = int(resp.headers.get("content-length", 0))
            done_b  = 0
            Path(self._dest).parent.mkdir(parents=True, exist_ok=True)
            with open(self._dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if self._stopped:
                        print("[ModelDownloadWorker] Cancelled")
                        try:
                            Path(self._dest).unlink(missing_ok=True)
                        except Exception:
                            pass
                        return
                    if chunk:
                        f.write(chunk)
                        done_b += len(chunk)
                        self.progress.emit(
                            done_b  // (1024 * 1024),
                            total_b // (1024 * 1024),
                        )
            print(f"[ModelDownloadWorker] Complete: {self._dest}")
            self.finished.emit(self._dest)
        except Exception as e:
            print(f"[ModelDownloadWorker] Error: {e}")
            try:
                Path(self._dest).unlink(missing_ok=True)
            except Exception:
                pass
            self.error.emit(str(e))


# ── Worker thread ───────────────────────────────────────────────────────── #

class TranslationWorker(QThread):
    """
    Signals emitted during processing:
      progress(count, message) — how many files done so far + a status string
      step(message) — current sub-step within the active image
      image_done(path, pil_image) — a finished translated image
      finished() — all files processed (or stopped)
      error(message) — a non-fatal error on one file, or a fatal setup error
    """
    progress    = Signal(int, str)
    step        = Signal(str)
    image_done  = Signal(str, object)   # str path, PIL Image
    finished    = Signal()
    error       = Signal(str)
    fatal_error = Signal(str)           # setup failure — shown as a popup

    def __init__(self, files, settings):
        super().__init__()
        self.files    = files     # list of file path strings
        self.settings = settings  # dict from MainWindow
        self._stopped = False

    def stop(self):
        """Ask the thread to stop after the current image finishes."""
        self._stopped = True

    # ------------------------------------------------------------------ #

    def run(self):
        """Entry point — called automatically by QThread.start()."""
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
        service     = self.settings["service"]

        # --- Load OCR ---
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
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            if source_lang == "auto":
                self.progress.emit(0, "Loading multi-language OCR model — first run may download ~400 MB, please wait…")
                ocr_langs = ["ja", "ch_sim", "ko", "en"]
            else:
                self.progress.emit(0, "Loading OCR model — first run may download ~200 MB, please wait…")
                ocr_langs = EASYOCR_LANG_MAP.get(source_lang, ["en"])
            reader = easyocr.Reader(ocr_langs, gpu=use_gpu)

        # --- Load translator ---
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

        # --- Process each file ---
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
                    'image':   result_img,
                    'base':    base_img,
                    'regions': regions,
                })
            except Exception as e:
                print(f"[TranslationWorker] Error on {name}: {e}")
                self.error.emit(f"Skipped {name}: {e}")
                traceback.print_exc()

        self.progress.emit(len(self.files), "Done!")

    # ------------------------------------------------------------------ #

    def _build_translator(self, service, source_lang, target_lang):
        """Instantiate and return the right translator; raises on any failure."""
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
            # OPUS-MT needs an explicit source language; "auto" defaults to Japanese
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

        elif service == "Local LLM (GGUF)":
            model_path = self.settings.get("gguf_path", "").strip()
            gpu_layers = int(self.settings.get("gguf_gpu_layers", 0))
            print(f"[TranslationWorker] Loading GGUF: path={model_path!r} gpu_layers={gpu_layers}")
            self.progress.emit(0, "Loading GGUF model — this may take a moment…")
            t = LocalLLMTranslator(model_path, source_lang, target_lang, n_gpu_layers=gpu_layers)
            print("[TranslationWorker] Local LLM ready")
            return t

        raise ValueError(f"Unknown service: {service!r}")
