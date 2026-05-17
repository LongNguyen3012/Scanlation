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


# Maps our friendly language-selector codes to the format each translator expects.
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

# MyMemory needs locale-style codes like 'ja-JP'
# ch_sim / ch_tra are EasyOCR internal codes; MyMemory uses zh-CN / zh-TW
MYMEMORY_CODES = {
    "en":     "en-US",
    "ja":     "ja-JP",
    "ko":     "ko-KR",
    "zh-CN":  "zh-CN",
    "zh-TW":  "zh-TW",
    "ch_sim": "zh-CN",
    "ch_tra": "zh-TW",
    "es":     "es-ES",
    "fr":     "fr-FR",
    "de":     "de-DE",
    "pt":     "pt-PT",
    "ru":     "ru-RU",
    "it":     "it-IT",
    "vi":     "vi-VN",
    "th":     "th-TH",
}

# Argos Translate uses plain ISO codes; no distinction between zh-CN and zh-TW
ARGOS_SOURCE_CODES = {
    "ja":     "ja",
    "ko":     "ko",
    "ch_sim": "zh",
    "ch_tra": "zh",
    "en":     "en",
}
ARGOS_TARGET_CODES = {
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


# ── translators library (no API key) ────────────────────────────────────── #

# Each backend in the `translators` library uses slightly different target codes.
# Baidu in particular diverges from standard BCP-47 (jp not ja, kor not ko, etc.)
_EXT_BACKENDS = {
    "Baidu (Free)": {
        "backend": "baidu",
        "targets": {
            "en": "en",   "ja": "jp",      "ko": "kor",  "zh-CN": "zh",
            "zh-TW": "cht","es": "spa",    "fr": "fra",  "de": "de",
            "pt": "pt",   "ru": "ru",      "it": "it",   "vi": "vie",
            "th": "th",
        },
    },
    "Bing (Free)": {
        "backend": "bing",
        "targets": {
            "en": "en",       "ja": "ja",       "ko": "ko",
            "zh-CN": "zh-Hans","zh-TW": "zh-Hant","es": "es",
            "fr": "fr",       "de": "de",       "pt": "pt",
            "ru": "ru",       "it": "it",       "vi": "vi",
            "th": "th",
        },
    },
}


class ExternalTranslator:
    """
    Uses the `translators` library to reach Baidu, Bing, etc. without an API key.
    The library scrapes the same web endpoints your browser would use, so no
    registration is required.  Quality matches the real service.

    Caveats:
      • Requires internet.
      • Can break if the provider changes their web API (library updates fix this).
      • First call per session is slow (~2-3 s) while the library initialises cookies.
    """

    def __init__(self, backend: str, to_code: str):
        self._backend = backend
        self._to      = to_code

    def translate(self, text: str) -> str:
        import translators as ts
        return ts.translate_text(
            text,
            translator=self._backend,
            from_language="auto",
            to_language=self._to,
        ) or ""


# ── Argos Translate (offline) ───────────────────────────────────────────── #

class ArgosTranslator:
    """
    Offline machine translation using Argos Translate.

    First use for a language pair downloads the model package (~100-300 MB)
    to the user's Argos data directory — subsequent uses load from cache.
    No API key, no internet required after the first download, no rate limits.

    Translation quality is generally lower than Google Translate, especially
    for CJK languages, but it will never block or exhaust a quota.
    """

    def __init__(self, from_code: str, to_code: str):
        import argostranslate.package
        import argostranslate.translate

        self._from = from_code
        self._to   = to_code

        # Check if this pair is already installed
        installed = argostranslate.package.get_installed_packages()
        already = any(p.from_code == from_code and p.to_code == to_code
                      for p in installed)

        if not already:
            # Download the package index then install the required pair
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()
            pkg = next(
                (p for p in available
                 if p.from_code == from_code and p.to_code == to_code),
                None,
            )
            if pkg is None:
                raise ValueError(
                    f"Argos Translate has no '{from_code}' → '{to_code}' package.\n"
                    f"This language pair may not be supported offline yet."
                )
            argostranslate.package.install_from_path(pkg.download())

    def translate(self, text: str) -> str:
        import argostranslate.translate
        return argostranslate.translate.translate(text, self._from, self._to) or ""


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


# ── NLLB offline translator ──────────────────────────────────────────────── #

# Facebook NLLB-200 Flores-200 language codes
_NLLB_SRC_CODES = {
    "ja":     "jpn_Jpan",
    "ja_manga": "jpn_Jpan",
    "ko":     "kor_Hang",
    "ch_sim": "zho_Hans",
    "ch_tra": "zho_Hant",
    "en":     "eng_Latn",
}
_NLLB_TGT_CODES = {
    "en":    "eng_Latn",
    "ja":    "jpn_Jpan",
    "ko":    "kor_Hang",
    "zh-CN": "zho_Hans",
    "zh-TW": "zho_Hant",
    "es":    "spa_Latn",
    "fr":    "fra_Latn",
    "de":    "deu_Latn",
    "pt":    "por_Latn",
    "ru":    "rus_Cyrl",
    "it":    "ita_Latn",
    "vi":    "vie_Latn",
    "th":    "tha_Thai",
}


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
    """
    Universal local-LLM translator — loads any GGUF instruction model via
    llama-cpp-python and translates manga dialogue with an auto-generated system prompt.

    Compatible models (user downloads the .gguf manually):
      • Qwen3-8B / Qwen3-14B (bartowski or Unsloth GGUF)              ~5–9 GB
      • Qwen3-8B uncensored / Qwen3.5-9B-abliterated (lukey03 etc.)   ~5 GB
      • Qwen2.5-7B-Instruct-GGUF (official Qwen)                      ~4–8 GB
      • Qwen3.6-27B / 35B-A3B (Unsloth GGUF, needs 16+ GB RAM)        ~15–20 GB
      • SakuraLLM Sakura-1.5B / 7B (JP→ZH specialist)                 ~1–4 GB
      • Any other instruction-tuned GGUF in ChatML / Qwen format

    Install runtime:  pip install llama-cpp-python
    GPU acceleration: pip install llama-cpp-python --extra-index-url
                        https://abetlen.github.io/llama-cpp-python/whl/cu121
    Pre-built Windows wheels: https://github.com/abetlen/llama-cpp-python/releases

    Qwen3 tip: the model outputs <think>…</think> reasoning by default.
    This class strips those blocks automatically — you always get only the translation.
    """

    def __init__(self, model_path: str, from_ocr_code: str, to_code: str,
                 n_gpu_layers: int = 0):
        import os, re
        from llama_cpp import Llama

        self._re_think = re.compile(r"<think>.*?</think>", re.DOTALL)

        src_name = _LLM_LANG_NAMES.get(from_ocr_code, from_ocr_code)
        tgt_name = _LLM_LANG_NAMES.get(to_code, to_code)

        # Auto-select system prompt: Sakura specialist prompt or generic manga prompt
        fname = os.path.basename(model_path).lower()
        if any(k in fname for k in ("sakura", "galtransl")):
            self._system = _SAKURA_SYSTEM_PROMPT
            self._user_tmpl = "将下面的日文文本翻译成中文：{text}"
        else:
            self._system = (
                f"You are a professional manga and comic translator. "
                f"Translate the following {src_name} text into {tgt_name}. "
                f"Preserve the tone, emotion, and style of manga dialogue. "
                f"Output ONLY the translated text — no explanations, notes, or alternatives."
            )
            self._user_tmpl = "{text}"

        self._llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=max(1, os.cpu_count() - 1),
            n_gpu_layers=n_gpu_layers,   # set >0 to offload layers to GPU
            verbose=False,
        )

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
        # Strip Qwen3 chain-of-thought blocks if the model emits them
        result = self._re_think.sub("", result).strip()
        return result


class NLLBTranslator:
    """
    Fully offline multilingual translation using Facebook NLLB-200-distilled-600M.

    First run downloads ~2.4 GB of model weights to the HuggingFace cache
    (%USERPROFILE%\\.cache\\huggingface\\).  All subsequent runs load from cache —
    no internet required.

    Quality is noticeably better than Argos Translate, especially for
    Japanese → English and Chinese → English.  CPU inference is roughly
    1–4 seconds per sentence on a modern processor.
    """

    def __init__(self, from_ocr_code: str, to_code: str):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self._torch = torch
        src = _NLLB_SRC_CODES.get(from_ocr_code, "jpn_Jpan")
        self._tgt = _NLLB_TGT_CODES.get(to_code, "eng_Latn")

        model_id = "facebook/nllb-200-distilled-600M"
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang=src)
        self._model     = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._model.eval()

    def translate(self, text: str) -> str:
        inputs  = self._tokenizer(text, return_tensors="pt",
                                  truncation=True, max_length=256)
        tgt_id  = self._tokenizer.convert_tokens_to_ids(self._tgt)
        with self._torch.no_grad():
            output = self._model.generate(
                **inputs, forced_bos_token_id=tgt_id, max_length=256
            )
        return self._tokenizer.decode(output[0], skip_special_tokens=True)


# ── Model download worker ────────────────────────────────────────────────── #

class ModelDownloadWorker(QThread):
    """
    Downloads one GGUF file from HuggingFace with per-chunk progress signals.
    Writes to a .part temp file and renames on success so the app never sees a
    half-written file as "downloaded".
    """
    progress = Signal(int, int)   # downloaded_bytes, total_bytes
    finished = Signal(str)        # absolute local path on success
    error    = Signal(str)

    def __init__(self, repo: str, filename: str, dest_dir: str):
        super().__init__()
        self._repo     = repo
        self._filename = filename
        self._dest_dir = Path(dest_dir)
        self._stopped  = False

    def stop(self):
        self._stopped = True

    def run(self):
        import requests
        url  = f"https://huggingface.co/{self._repo}/resolve/main/{self._filename}"
        dest = self._dest_dir / self._filename
        tmp  = self._dest_dir / (self._filename + ".part")
        self._dest_dir.mkdir(parents=True, exist_ok=True)

        try:
            with requests.get(url, stream=True, timeout=30,
                              headers={"User-Agent": "comic-translator/1.0"}) as r:
                if r.status_code == 404:
                    self.error.emit(
                        f"404 — file not found on HuggingFace.\n"
                        f"Repo: {self._repo}\n"
                        f"Expected file: {self._filename}\n\n"
                        f"Visit the repo on HuggingFace, download the file manually,\n"
                        f"then use the Browse button in settings."
                    )
                    return
                r.raise_for_status()
                total      = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):   # 1 MB chunks
                        if self._stopped:
                            tmp.unlink(missing_ok=True)
                            return
                        f.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total)
            tmp.rename(dest)
            self.finished.emit(str(dest))
        except Exception as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            self.error.emit(str(exc))


# ── Worker thread ───────────────────────────────────────────────────────── #

class TranslationWorker(QThread):
    """
    Signals emitted during processing:
      progress(count, message) — how many files done so far + a status string
      image_done(path, pil_image) — a finished translated image
      finished() — all files processed (or stopped)
      error(message) — a non-fatal error on one file, or a fatal setup error
    """
    progress   = Signal(int, str)
    image_done = Signal(str, object)   # str path, PIL Image
    finished   = Signal()
    error      = Signal(str)

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
            self.progress.emit(0, "Loading OCR model — first run may download ~200 MB, please wait…")
            ocr_langs = EASYOCR_LANG_MAP.get(source_lang, ["en"])
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except ImportError:
                use_gpu = False
            reader = easyocr.Reader(ocr_langs, gpu=use_gpu)

        # --- Load translator ---
        self.progress.emit(0, "Setting up translator…")
        translator = self._build_translator(service, source_lang, target_lang)
        if translator is None:
            return  # error already emitted

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
            self.progress.emit(i, f"Translating {name}  ({i + 1} / {len(self.files)})")

            forced = manual_zones.get(file_path, [])
            try:
                result_img, base_img, regions = pipeline.process_image(
                    file_path, forced_bboxes=forced or None
                )
                self.image_done.emit(file_path, {
                    'image':   result_img,
                    'base':    base_img,
                    'regions': regions,
                })
            except Exception as e:
                self.error.emit(f"Skipped {name}: {e}")
                traceback.print_exc()

        self.progress.emit(len(self.files), "Done!")

    # ------------------------------------------------------------------ #

    def _build_translator(self, service, source_lang, target_lang):
        """
        Instantiate the right translator and wrap it in _ThrottledTranslator
        for services that have rate limits (DeepL, MyMemory).

        Delay values are chosen to stay well inside free-tier limits:
          Google:   no delay needed (generous limits, reliable)
          MyMemory: 0.5 s — 5,000 word/day free tier; bursts cause silent failures
          DeepL:    1.0 s — strict per-second rate limit on free endpoints;
                    also applies a 20 s timeout so a hung request can't stall
                    the worker (DeepL sometimes hangs instead of returning 429)
        """
        from deep_translator import GoogleTranslator, MyMemoryTranslator, DeeplTranslator

        try:
            if service == "Google Translate":
                t = GoogleTranslator(
                    source="auto",
                    target=GOOGLE_TARGET_CODES.get(target_lang, target_lang),
                )
                return t   # Google is fast and reliable — no throttling needed

            elif service == "MyMemory":
                src_code = MYMEMORY_CODES.get(source_lang, source_lang)
                tgt_code = MYMEMORY_CODES.get(target_lang, target_lang)
                t = MyMemoryTranslator(source=src_code, target=tgt_code)
                return _ThrottledTranslator(t, delay=0.5, timeout=15.0)

            elif service == "DeepL (API Key)":
                api_key = self.settings.get("deepl_key", "").strip()
                if not api_key:
                    self.error.emit("DeepL requires an API key. Enter it in the settings panel.")
                    return None
                t = DeeplTranslator(
                    api_key=api_key,
                    source="auto",
                    target=target_lang.upper(),
                    use_free_api=True,
                )
                return _ThrottledTranslator(t, delay=1.0, timeout=20.0)

            elif service in _EXT_BACKENDS:
                cfg     = _EXT_BACKENDS[service]
                to_code = cfg["targets"].get(target_lang, "en")
                t = ExternalTranslator(cfg["backend"], to_code)
                # 0.5 s between calls — scraping too fast risks a temporary block
                return _ThrottledTranslator(t, delay=0.5, timeout=20.0)

            elif service == "OPUS-MT (Offline)":
                self.progress.emit(
                    0,
                    f"Loading OPUS-MT {source_lang}→{target_lang} model "
                    f"— first run downloads ~350 MB to HuggingFace cache, please wait…"
                )
                return OpusMTTranslator(source_lang, target_lang)

            elif service == "Local LLM (GGUF)":
                model_path = self.settings.get("gguf_path", "").strip()
                if not model_path:
                    self.error.emit(
                        "Local LLM requires a GGUF model file. "
                        "Download any .gguf (Qwen3, Qwen2.5, Sakura…) and enter its path in settings."
                    )
                    return None
                n_gpu = int(self.settings.get("gguf_gpu_layers", 0))
                self.progress.emit(0, f"Loading local model from {model_path}…")
                return LocalLLMTranslator(model_path, source_lang, target_lang,
                                          n_gpu_layers=n_gpu)

            elif service == "NLLB (Offline)":
                self.progress.emit(
                    0,
                    "Loading NLLB-200 model — first run downloads ~2.4 GB to HuggingFace cache, please wait…"
                )
                return NLLBTranslator(source_lang, target_lang)

        except Exception as e:
            self.error.emit(f"Could not initialise translator: {e}")
            return None

        self.error.emit(f"Unknown service: {service}")
        return None
