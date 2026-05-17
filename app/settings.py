import json
from pathlib import Path

APPDATA_DIR = Path.home() / "AppData" / "Roaming" / "ComicTranslator"
_FILE       = APPDATA_DIR / "settings.json"

DEFAULTS: dict = {
    "models_dir":      str(APPDATA_DIR / "models"),
    "source_lang":     "Japanese",
    "target_lang":     "English",
    "service":         "Google Translate",
    "deepl_key":       "",
    "gguf_path":       "",
    "gguf_gpu_layers": 0,
    "output_dir":      "",
}


def load() -> dict:
    if _FILE.exists():
        try:
            with open(_FILE, encoding="utf-8") as f:
                saved = json.load(f)
            return {**DEFAULTS, **saved}
        except Exception:
            pass
    return dict(DEFAULTS)


def save(data: dict) -> None:
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
