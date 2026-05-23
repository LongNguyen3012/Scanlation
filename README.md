# Comic Translator

A desktop app for translating text in images. It detects text in images, translates it, and paints the translation back onto the page — all in a few clicks.

![Platform](https://img.shields.io/badge/platform-Windows-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

---

## Features

- Detects and translates text directly on comic pages
- Multiple OCR engines: EasyOCR, RapidOCR, Manga-OCR (Japanese specialist)
- Multiple translation services: Google, Bing, Baidu, MyMemory, DeepL, OPUS-MT (offline), NLLB (offline), Local LLM (GGUF)
- Built-in model manager — download GGUF models directly from the app
- Manual zone marking — draw boxes to force OCR on specific areas
- Text editor — edit translated text before saving
- Batch translate entire folders at once
- Supports JPG, PNG, WebP, BMP, TIFF, GIF

---

## Requirements

- Windows 10 / 11
- Python 3.10 or newer — download from [python.org](https://www.python.org/downloads/)
- Git (optional, for cloning) — download from [git-scm.com](https://git-scm.com)

---

## Installation

### Option 1 — Run from source (recommended for development)

**1. Clone or download the repo**
```
git clone https://github.com/LongNguyen3012/Scanlation.git
cd Scanlation
```

**2. Install dependencies**

Double-click `setup.bat`, or run in a terminal:
```
pip install -r requirements.txt
```

**3. Run the app**
```
python main.py
```

---

### Option 2 — Build a standalone .exe

Double-click `build_exe.bat`. This requires PyInstaller (`pip install pyinstaller`).

The output will be a `dist\ComicTranslator\` folder. Share or move the **entire folder** — the `.exe` alone won't work without the files next to it.

> First launch may take a minute as EasyOCR downloads its language models (~200 MB, one-time only).

---

## Quick Start

1. Launch the app with `python main.py`
2. Click **Add Files** or **Add Folder** to load your comic pages
3. Choose your **Source Language** (the language in the comic)
4. Choose your **Target Language** (what you want to translate to)
5. Pick a **Translation Service** (Google Translate works with no setup)
6. Click **Translate All**
7. Click **Save All** when done

Translated images are saved with a `_translated` suffix in the same folder as the originals, or in a custom output folder you choose in settings.

---

## Translation Services

| Service | Internet needed | API key needed | Notes |
|---|---|---|---|
| Google Translate | Yes | No | Default, works out of the box |
| Bing | Yes | No | Free tier |
| Baidu | Yes | No | Good for Chinese |
| MyMemory | Yes | No | Free with daily limit |
| DeepL | Yes | Yes | High quality, free key available at deepl.com |
| OPUS-MT | No | No | Downloads ~350 MB per language pair on first use |
| NLLB | No | No | Downloads ~2.4 GB on first use |
| Local LLM (GGUF) | No | No | Best quality offline — see below |

---

## Local LLM (GGUF) Setup

For the best offline translation quality, use a local GGUF model.

**1. Install llama-cpp-python**

CPU only:
```
pip install llama-cpp-python
```

With NVIDIA GPU (much faster):
```
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

**2. Download a model**

- Select **Local LLM (GGUF)** as the translation service
- Click **Manage Models…**
- Pick a model and quantization, then click **Download**
- Once downloaded, click **Use This Model**

Recommended models:
- **Sakura 1.5B** — best for Japanese → Chinese manga (small, CPU-friendly)
- **Qwen3-8B Q4_K_M** — best general-purpose multilingual model
- **VNTL LLaMA3 8B** — best for Japanese → English visual novels

Models are saved to `%APPDATA%\ComicTranslator\models\` by default. You can change this in the **Models folder** setting.

---

## Settings

All settings are saved automatically when you close the app and restored on next launch.

| Setting | Description |
|---|---|
| Source Language | Language the comic is written in |
| Target Language | Language to translate into |
| Translation Service | Which engine to use |
| DeepL API Key | Required only for DeepL |
| Models folder | Where GGUF models are saved (change to put them on a different drive) |
| GGUF model file | Path to the `.gguf` file to use |
| GPU layers | How many layers to offload to GPU (0 = CPU only, 999 = full GPU) |
| Output Folder | Where to save translated images (default: same folder as source) |

---

## Tips

- **Manual zones**: If the auto-detector misses text, select a file and click **Mark Zones** to draw boxes around the text manually.
- **Edit texts**: After translating, click **Edit Texts** to fix any mistranslations before saving.
- **Right-click** a file in the list for per-file options (translate, mark zones, edit).
- For Japanese manga, try **Japanese (Manga OCR)** as the source — it's more accurate than standard EasyOCR for stylized fonts.

---

## Troubleshooting

**App launches but translation fails**
- Check that the selected translation service has internet access (for online services)
- For GGUF: make sure `llama-cpp-python` is installed and the model path is set correctly

**OCR doesn't detect text**
- Try a different source language or OCR engine
- Use **Mark Zones** to manually point the OCR at the text

**GGUF model download fails with a 404 error**
- The file may have been renamed on HuggingFace — click the repo link in the error dialog, download manually, then use the **Browse** button to point the app at it

**Slow translation**
- For GGUF models, increase GPU layers (requires CUDA build of llama-cpp-python)
- Use a smaller quantization (Q4_K_M is faster than Q8_0)
