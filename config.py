"""Central configuration. Change values HERE, never inside the modules."""
from pathlib import Path

# --- Paths ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PROMPT_PATH = BASE_DIR / "prompts" / "taxonomy_prompt.txt"
LABELED_CSV = DATA_DIR / "reviews_labeled.csv"
FALLBACK_JSON = DATA_DIR / "fallback_reviews.json"

# --- Live fetch ---
LIVE_FETCH_COUNT = 20               # reviews pulled per live run — keep small for a reliable demo
SPOTIFY_PLAY_APP_ID = "com.spotify.music"
SPOTIFY_IOS_APP_ID = "324684580"

# Sources shown in the sidebar. Apple is default: it uses a PUBLIC RSS feed, which is more
# reliable from cloud hosts (Streamlit Cloud) than the reverse-engineered Play endpoint.
SOURCES = {
    "Apple App Store": "apple",
    "Google Play": "play",
}
DEFAULT_SOURCE = "Google Play"     # was "Apple App Store"

APPLE_COUNTRIES = ["us", "in"]      # RSS storefronts to try, in order
PLAY_COUNTRY = "in"                 # kept for parity; see note below
# NOTE (from your own scraping finding): Google Play does NOT truly filter by country.
# It returns an English-SCRIPT pool (text written in a-z), any language. The app labels
# Play reviews "play_store_english" so we never overclaim a country.

# --- Gemini (called over REST via `requests` — no SDK dependency, matches your Apple RSS style) ---
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# ↓↓↓ TO BE SOURCED: confirm the EXACT model string in Google AI Studio → your project →
# any model row → "Get code". It may carry a -preview or dated suffix. Do not assume.
#
# Your project quota (from your AI Studio screenshots, 05-07-2026), free tier, per model/day:
#   Gemini 3.1 Flash Lite : 15 RPM / 250K TPM / 500 RPD   <- USE THIS for dev AND demo
#   Gemma 4 26B / 31B     : 15 RPM / unlimited TPM / 1.5K RPD  (most RPD, but weaker at strict JSON)
#   Gemini 3.5 / 3 / 2.5 Flash : 5 RPM / 250K TPM / only 20 RPD  <- demo-only fallback if Flash-Lite mislabels
#   Gemini 2.5 Flash Lite : 10 RPM / 250K TPM / 20 RPD
# Daily quota resets at midnight Pacific. A 20-RPD model will run out fast during debugging,
# which is exactly why dev runs on 3.1 Flash Lite.
GEMINI_MODEL = "gemini-3.1-flash-lite"   # i have verified on google ai studio exact model name is "gemini-3.1-flash-lite"
GEMINI_TEMPERATURE = 0                    # deterministic classification
GEMINI_TIMEOUT = 60                       # seconds per request

# Field order the app and prompt agree on (do not reorder without editing the prompt).
LABEL_FIELDS = [
    "discovery_related", "theme", "desired_behavior", "repetition_cause",
    "behavior_segment", "primary_discovery_surface", "tier", "unmet_need",
    "sentiment", "evidence_phrase",
]
