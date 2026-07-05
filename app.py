"""Spotify Review Discovery Engine — app entry point.

Thin orchestrator. Two working surfaces plus a pipeline view:
  • Live Workflow    : fetch latest reviews -> classify with Gemini -> show labels
                       (falls back to a bundled sample if the live source is empty)
  • Research Insights: the 1,000-review dashboard (src/dashboard.py)
  • How it works     : the scrape -> classify -> dashboard pipeline (doubles as the 1-slide)

Run:  streamlit run app.py   (from the project root)
"""
import pandas as pd
import streamlit as st

import config
from src import scraper, classifier, dashboard, utils


# --- helpers (importable / testable without running Streamlit) ----------------

def _api_key() -> str | None:
    """Prefer Streamlit secrets (works locally and on Streamlit Cloud). Return None to let
    the classifier resolve the key from env / secrets.toml itself."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return None


def _merge(reviews: list[dict], labels: list[dict]) -> pd.DataFrame:
    """Join raw reviews with their labels on id; coerce discovery_related to text so the
    live table reads the same way as the historical corpus."""
    by_id = {l["id"]: l for l in labels}
    out = []
    for r in reviews:
        lab = by_id.get(r["id"], {})
        dr = lab.get("discovery_related")
        out.append({
            "id": r["id"],
            "source": r["source"],
            "rating": r["rating"],
            "discovery_related": ("true" if dr is True else "false" if dr is False else str(dr)),
            "theme": lab.get("theme", ""),
            "behavior_segment": lab.get("behavior_segment", ""),
            "tier": lab.get("tier", ""),
            "sentiment": lab.get("sentiment", ""),
            "unmet_need": lab.get("unmet_need", ""),
            "evidence_phrase": lab.get("evidence_phrase", ""),
            "text": r["text"],
        })
    return pd.DataFrame(out)


def _summary(df: pd.DataFrame) -> str:
    n = len(df)
    disc = df[df["discovery_related"] == "true"]
    if disc.empty:
        return (f"0 of {n} fetched reviews were discovery-related — expected, since discovery is "
                "mentioned in only ~7% of reviews. Discovery pain is real but under-voiced.")
    top_theme = disc["theme"].value_counts().idxmax()
    return f"{len(disc)} of {n} fetched reviews were discovery-related; the leading theme is `{top_theme}`."


# --- UI -----------------------------------------------------------------------

def _render_live(source_label: str, source_key: str, fetch_count: int) -> None:
    st.subheader("Fetch the latest reviews and classify them live")
    st.write(f"Pulls the newest reviews from **{source_label}** and applies the same taxonomy "
             "used on the 1,000-review corpus — fully automated, no manual step.")

    if st.button("Fetch latest reviews", type="primary"):
        with st.spinner(f"Fetching from {source_label}…"):
            reviews = scraper.fetch_reviews(source_key, fetch_count)  # never raises; may be []

        used_fallback = False
        if not reviews:
            reviews = utils.load_fallback()[:fetch_count]
            used_fallback = True

        if not reviews:
            st.error("No reviews from the live source or the fallback file. "
                     "Check data/fallback_reviews.json.")
            return

        reviews = utils.assign_ids(reviews, start=1)
        if used_fallback:
            st.warning("Live source returned nothing right now — showing a bundled sample batch "
                       "so the pipeline still runs end to end.", icon="⚠️")

        with st.spinner(f"Classifying {len(reviews)} reviews with {config.GEMINI_MODEL}…"):
            try:
                labels = classifier.classify_reviews(reviews, api_key=_api_key())
            except classifier.ClassifierError as exc:
                st.error(f"Classification failed: {exc}")
                st.dataframe(pd.DataFrame(reviews)[["id", "source", "rating", "text"]],
                             use_container_width=True, hide_index=True)
                return

        merged = _merge(reviews, labels)
        st.success(_summary(merged))
        st.dataframe(
            merged[["id", "source", "rating", "discovery_related",
                    "theme", "behavior_segment", "evidence_phrase"]],
            use_container_width=True, hide_index=True,
        )
        with st.expander("Show all classified fields"):
            st.dataframe(merged, use_container_width=True, hide_index=True)
    else:
        st.info("Press **Fetch latest reviews** to run the live pipeline.")


def _render_how(source_label: str) -> None:
    st.subheader("Pipeline")
    steps = [
        ("1 · Scrape", f"Newest reviews from {source_label} (Apple RSS / Google Play), source-agnostic"),
        ("2 · Prepare", "Deduplicate, sample, assign a stable id per review"),
        ("3 · Classify", f"One Gemini call ({config.GEMINI_MODEL}) applies the fixed taxonomy → strict JSON"),
        ("4 · Store", "Labelled rows (11 fields) with an evidence phrase per review"),
        ("5 · Insights", "The eight research questions, D1/D2 denominators, evidence drill-down"),
    ]
    cols = st.columns(len(steps))
    for col, (head, body) in zip(cols, steps):
        col.markdown(f"**{head}**")
        col.caption(body)
    st.divider()
    st.caption("Historical corpus (1,000 reviews) was classified with the same taxonomy; the Live "
               "Workflow tab proves the classification runs automatically on new, unseen reviews.")


def main() -> None:
    st.set_page_config(page_title="Spotify Review Discovery Engine", layout="wide")

    st.sidebar.title("⚙️ Controls")
    source_label = st.sidebar.radio(
        "Live source",
        list(config.SOURCES.keys()),
        index=list(config.SOURCES.keys()).index(config.DEFAULT_SOURCE),
    )
    source_key = config.SOURCES[source_label]
    fetch_count = st.sidebar.slider("Reviews to fetch", 5, 20, config.LIVE_FETCH_COUNT)
    st.sidebar.caption(f"Model: {config.GEMINI_MODEL}")

    st.title("🎧 Spotify Review Discovery Engine")
    st.caption("Scrapes reviews and turns them into discovery insights automatically.")

    tab_live, tab_research, tab_how = st.tabs(
        ["🔴 Live Workflow", "📊 Research Insights", "🛠️ How it works"]
    )
    with tab_live:
        _render_live(source_label, source_key, fetch_count)
    with tab_research:
        dashboard.render()
    with tab_how:
        _render_how(source_label)


if __name__ == "__main__":
    main()
