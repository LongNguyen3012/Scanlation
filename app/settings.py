import json
from pathlib import Path

APPDATA_DIR = Path.home() / "AppData" / "Roaming" / "ComicTranslator"
_FILE       = APPDATA_DIR / "settings.json"

DEFAULTS: dict = {
    "source_lang":     "Auto-detect",
    "target_lang":     "English",
    "service":         "Google Translate",
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