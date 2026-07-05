"""Small, dependency-light helpers shared across modules."""
import json
import re
from pathlib import Path

import config


def pipe_safe(text: str) -> str:
    """Global rule B: replace every | with / so no value breaks downstream parsing."""
    return str(text).replace("|", "/")


def assign_ids(reviews: list[dict], start: int = 1) -> list[dict]:
    """Give freshly scraped reviews sequential ids (the classifier echoes these back)."""
    for i, r in enumerate(reviews):
        r["id"] = start + i
    return reviews


def strip_code_fences(text: str) -> str:
    """Remove ```json ... ``` fences if the model wraps its output despite instructions."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def safe_parse_json_array(text: str) -> list[dict]:
    """Parse the model's JSON array defensively.

    1) try direct json.loads
    2) strip code fences and retry
    3) slice from the first '[' to the last ']' and retry
    Raises ValueError with the raw text if all attempts fail (so the UI can show it).
    """
    for candidate in (text, strip_code_fences(text)):
        try:
            data = json.loads(candidate)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Could not parse a JSON array from model output:\n{text[:500]}")


def load_fallback() -> list[dict]:
    """Load the bundled recent-reviews batch used when a live scrape fails."""
    p = Path(config.FALLBACK_JSON)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))
