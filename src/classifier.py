"""Gemini classification layer.

Takes a batch of scraped reviews, applies your taxonomy prompt via one Gemini REST call,
and returns validated label objects. Decoupled from Streamlit so it can be tested standalone.

Public entry point: classify_reviews(reviews) -> list[dict]
"""
import os
import re
from functools import lru_cache

import requests

import config
from src import utils


class ClassifierError(RuntimeError):
    """Raised when Gemini output cannot be parsed or validated after a retry."""


# --- credentials -------------------------------------------------------------

def get_api_key() -> str:
    """Resolve the key: env var GEMINI_API_KEY first, then .streamlit/secrets.toml.

    This lets the module run standalone (env or secrets file) AND inside Streamlit
    (app.py may instead pass the key explicitly to classify_reviews)."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    secrets_path = config.BASE_DIR / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        text = secrets_path.read_text(encoding="utf-8")
        try:
            import tomllib  # Python 3.11+
            parsed = tomllib.loads(text)
            if parsed.get("GEMINI_API_KEY"):
                return parsed["GEMINI_API_KEY"]
        except ModuleNotFoundError:
            m = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', text)
            if m:
                return m.group(1)
    raise ClassifierError(
        "No Gemini API key found. Set env var GEMINI_API_KEY or add it to .streamlit/secrets.toml."
    )


# --- prompt + request building ----------------------------------------------

@lru_cache(maxsize=1)
def load_prompt() -> str:
    """Read the taxonomy prompt once (cached)."""
    return config.PROMPT_PATH.read_text(encoding="utf-8")


def build_request_text(reviews: list[dict]) -> str:
    """Prompt + a numbered batch in the exact '{id}. {text}' format the taxonomy expects.

    Newlines inside a review are flattened to spaces so the numbered list stays one line
    per review (matches how the original 1,000 were prepared)."""
    lines = []
    for r in reviews:
        text = re.sub(r"\s+", " ", str(r["text"])).strip()
        lines.append(f'{r["id"]}. {text}')
    batch_block = "\n".join(lines)
    return f"{load_prompt()}\n\nHere is the batch of reviews to classify:\n{batch_block}"


# --- Gemini call -------------------------------------------------------------

def _call_gemini(request_text: str, api_key: str, model: str) -> str:
    """POST one generateContent request; return the raw text of the first candidate.

    Raises ClassifierError with the response body on HTTP errors (e.g. wrong model
    string -> 404, quota exhausted -> 429) so failures are easy to diagnose."""
    url = config.GEMINI_ENDPOINT.format(model=model)
    payload = {
        "contents": [{"parts": [{"text": request_text}]}],
        "generationConfig": {
            "temperature": config.GEMINI_TEMPERATURE,
            "response_mime_type": "application/json",
        },
    }
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=config.GEMINI_TIMEOUT)
    if resp.status_code != 200:
        raise ClassifierError(f"Gemini HTTP {resp.status_code}: {resp.text[:400]}")

    data = resp.json()
    candidates = data.get("candidates", [])
    if not candidates:
        raise ClassifierError(f"Gemini returned no candidates (possibly blocked): {str(data)[:400]}")
    try:
        return candidates[0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise ClassifierError(f"Unexpected Gemini response shape: {str(data)[:400]}") from exc


# --- validation --------------------------------------------------------------

_REQUIRED_KEYS = ["id"] + config.LABEL_FIELDS


def _normalize(obj: dict) -> dict:
    """Light, non-semantic cleanup: coerce discovery_related to a real bool if the
    model returned it as a string. Never touches label VALUES."""
    dr = obj.get("discovery_related")
    if isinstance(dr, str):
        obj["discovery_related"] = dr.strip().lower() == "true"
    return obj


def _validate(objs: list[dict], reviews: list[dict]) -> list[dict]:
    """Every object must carry id + all LABEL_FIELDS, and the returned ids must match
    the input ids (no dropped/added/merged reviews). Raises on any violation."""
    if not isinstance(objs, list) or not objs:
        raise ClassifierError("Parsed output is not a non-empty list.")
    for o in objs:
        missing = [k for k in _REQUIRED_KEYS if k not in o]
        if missing:
            raise ClassifierError(f"Object id={o.get('id')} missing fields: {missing}")
    got_ids = {o["id"] for o in objs}
    want_ids = {r["id"] for r in reviews}
    if got_ids != want_ids:
        raise ClassifierError(f"id mismatch. missing={want_ids - got_ids}, extra={got_ids - want_ids}")
    return [_normalize(o) for o in objs]


# --- public entry point ------------------------------------------------------

def classify_reviews(reviews: list[dict], api_key: str | None = None, model: str | None = None) -> list[dict]:
    """Classify a batch of reviews (each needs 'id' and 'text'). Retries once on a
    parse/validation failure, then raises ClassifierError. Returns list of label dicts."""
    if not reviews:
        return []
    api_key = api_key or get_api_key()
    model = model or config.GEMINI_MODEL
    request_text = build_request_text(reviews)

    last_error: Exception | None = None
    for attempt in (1, 2):  # original + one retry
        try:
            raw = _call_gemini(request_text, api_key, model)
            objs = utils.safe_parse_json_array(raw)
            return _validate(objs, reviews)
        except (ClassifierError, ValueError) as exc:
            last_error = exc
    raise ClassifierError(f"Classification failed after retry: {last_error}")


if __name__ == "__main__":  # standalone test: `python -m src.classifier`
    from src import scraper
    batch = utils.assign_ids(scraper.fetch_reviews("apple", 5), start=1)
    print(f"classifying {len(batch)} reviews on {config.GEMINI_MODEL} ...")
    for row in classify_reviews(batch):
        print(f'  id={row["id"]} discovery={row["discovery_related"]} '
              f'theme={row["theme"]} seg={row["behavior_segment"]} :: {row["evidence_phrase"]}')
