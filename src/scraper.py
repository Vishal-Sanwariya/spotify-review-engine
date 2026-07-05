"""Live review fetching. Reuses your Apple RSS + Google Play code, capped to a small
batch (config.LIVE_FETCH_COUNT) for a fast, reliable live demo.

Design: each fetcher returns a uniform list of {source, date, rating, text} and NEVER
raises on "no reviews" — it returns []. fetch_reviews() chains the two sources so an
empty primary auto-tries the other. The app layer treats an empty result as the signal
to fall back to the bundled data/fallback_reviews.json.
"""
import requests
from google_play_scraper import reviews as gp_reviews, Sort

import config

# Apple's RSS often returns a metadata-only feed (no 'entry') to the default
# python-requests User-Agent. A browser-like UA is the most common fix.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    ),
    "Accept": "application/json",
}
_APPLE_PAGES = 3  # RSS pages to walk per storefront (~50 reviews each)


def _apple_entries(payload: dict) -> list[dict]:
    """Pull review entries from one RSS JSON page, tolerating Apple's quirks:
    - 'entry' ABSENT when a storefront/page has no reviews  -> []
    - 'entry' is a single DICT (exactly one review)         -> wrap into a list
    - the first entry is app metadata (no 'im:rating')      -> filtered out
    """
    entry = payload.get("feed", {}).get("entry", [])
    if isinstance(entry, dict):
        entry = [entry]
    return [e for e in entry if "im:rating" in e and "content" in e]


def fetch_apple(count: int = config.LIVE_FETCH_COUNT, countries: list[str] | None = None) -> list[dict]:
    """Apple public RSS with a browser UA and multi-page walk. Returns [] (never raises)
    if Apple yields no reviews, so the caller can fall back to another source."""
    countries = countries or config.APPLE_COUNTRIES
    rows: list[dict] = []
    for country in countries:
        for page in range(1, _APPLE_PAGES + 1):
            url = (
                f"https://itunes.apple.com/{country}/rss/customerreviews/"
                f"page={page}/id={config.SPOTIFY_IOS_APP_ID}/sortby=mostrecent/json"
            )
            try:
                r = requests.get(url, headers=_HEADERS, timeout=config.GEMINI_TIMEOUT)
                r.raise_for_status()
                entries = _apple_entries(r.json())
            except (requests.RequestException, ValueError):
                break  # network / bad-JSON on this storefront -> try next country
            if not entries:
                break  # no (more) reviews on this storefront
            for e in entries:
                rows.append({
                    "source": f"app_store_{country}",
                    "date": e.get("updated", {}).get("label", ""),
                    "rating": int(e["im:rating"]["label"]),
                    "text": e["content"]["label"],
                })
            if len(rows) >= count:
                break
        if len(rows) >= count:
            break
    rows = [x for x in rows if str(x["text"]).strip()]
    return rows[:count]


def fetch_play(count: int = config.LIVE_FETCH_COUNT, country: str | None = None) -> list[dict]:
    """Google Play newest reviews (single pull). Returns [] (never raises) on failure.
    Labeled 'play_store_english' because Play returns an English-script pool, not a country."""
    country = country or config.PLAY_COUNTRY
    try:
        result, _ = gp_reviews(
            config.SPOTIFY_PLAY_APP_ID,
            lang="en",
            country=country,
            sort=Sort.NEWEST,
            count=count,
        )
    except Exception:
        return []
    rows = [
        {
            "source": "play_store_english",
            "date": str(x["at"]),
            "rating": int(x["score"]),
            "text": x["content"],
        }
        for x in result
        if str(x.get("content", "")).strip()
    ]
    return rows[:count]


_FETCHERS = {"apple": fetch_apple, "play": fetch_play}


def fetch_reviews(source_key: str, count: int = config.LIVE_FETCH_COUNT,
                  allow_cross_fallback: bool = True) -> list[dict]:
    """Fetch from the chosen source ('apple' | 'play'). If it returns nothing and
    allow_cross_fallback is True, try the OTHER source so the live tab still shows
    real reviews. Records keep their TRUE source label, so the UI can state honestly
    which source actually answered. Returns a list (possibly empty)."""
    if source_key not in _FETCHERS:
        raise ValueError(f"Unknown source key: {source_key!r}")
    rows = _FETCHERS[source_key](count)
    if not rows and allow_cross_fallback:
        other = "play" if source_key == "apple" else "apple"
        rows = _FETCHERS[other](count)
    return rows


if __name__ == "__main__":  # diagnose each source independently: python -m src.scraper
    for src in ("apple", "play"):
        got = _FETCHERS[src](5)
        print(f"{src}: {len(got)} reviews")
        for r in got[:5]:
            print(f"  [{r['rating']}] {r['source']}: {r['text'][:60]}")
