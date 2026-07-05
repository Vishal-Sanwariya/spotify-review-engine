"""Research Insights tab — the eight brief questions answered from the 1,000-review corpus.

All figures come live from data_loader (which reproduces the workbook Dashboard), so the
app can never disagree with your deck. Q1/Q2 read the theme block, Q4/Q7 the repetition-cause
block — same numbers, framed to answer each distinct question. Q8 is shown as a RATE.

Decoupled from app.py: exposes render(). Also runs standalone via:
    streamlit run src/dashboard.py
"""
import sys
from pathlib import Path

# Make the project importable whether launched from the root (app.py) or as this script.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

import config
from src import data_loader

# Themes that are about what the algorithm SERVES (used only to frame Q2; taxonomy-grounded).
_REC_THEMES = ("repetition_loop", "stale_freshness", "bad_recommendations", "unwanted_injection")


# --- small render helpers -----------------------------------------------------

def _bar(rows: list[dict], exclude: tuple = (), top: int | None = None):
    """Build a horizontal bar chart DataFrame (index = label, one 'count' column)."""
    picked = [r for r in rows if r["value"] not in exclude]
    if top:
        picked = picked[:top]
    return pd.DataFrame({"count": [r["count"] for r in picked]},
                        index=[r["value"] for r in picked])


def _phrase(r: dict) -> str:
    return f"**{r['value']}** ({r['count']}, {r['pct']:.0%})"


def _top(rows: list[dict], exclude: tuple = (), k: int = 2) -> list[dict]:
    return [r for r in rows if r["value"] not in exclude][:k]


def _card(number: str, question: str):
    """Return a bordered container with the question as its heading."""
    box = st.container(border=True)
    box.markdown(f"**{number} · {question}**")
    return box


# --- the tab ------------------------------------------------------------------

def render() -> None:
    h = data_loader.headline_stats()

    # Headline / honesty strip
    cols = st.columns(4)
    cols[0].metric("Reviews analyzed (D1)", f"{h['d1']:,}")
    cols[1].metric("Discovery-related (D2)", h["d2"])
    cols[2].metric("Discovery share", f"{h['discovery_pct']:.1%}")
    cols[3].metric("Avg rating", h["avg_rating"])
    sources = ", ".join(f"{k}: {v}" for k, v in h["sources"].items())
    st.caption(
        f"Sources — {sources}. Dates {h['date_min']} to {h['date_max']}. "
        "Discovery share uses the D1 (all-1,000) denominator. "
        "Google Play returns an English-script pool, not a country filter."
    )
    st.info(
        "Only 7.2% of reviews mention discovery — bugs, ads and payment dominate the rest. "
        "Discovery pain is real but under-voiced, which is why this labelled corpus matters.",
        icon="🔎",
    )

    # Q1 — theme distribution (D2)
    theme = data_loader.distribution("theme", "D2")
    with _card("Q1", "Why do users struggle to discover new music?"):
        t = _top(theme.rows, exclude=("other_discovery",), k=3)
        st.write("Concrete complaints center on "
                 + ", ".join(_phrase(r) for r in t) + ".")
        st.bar_chart(_bar(theme.rows), horizontal=True)
        st.caption(f"Denominator: {theme.denom} {theme.denom_label}. "
                   "'other_discovery' is the catch-all bucket.")

    # Q2 — the recommendation-specific slice of the SAME theme block
    with _card("Q2", "What are the most common frustrations with recommendations?"):
        rec_rows = [r for r in theme.rows if r["value"] in _REC_THEMES]
        lead = _top(rec_rows, k=2)
        st.write("Among recommendation-related themes, "
                 + ", ".join(_phrase(r) for r in lead)
                 + " lead; off-taste `bad_recommendations` is rare.")
        st.bar_chart(_bar(rec_rows), horizontal=True)
        st.caption(f"Same theme counts as Q1, filtered to recommendation-driven themes "
                   f"({', '.join(_REC_THEMES)}). Denominator: {theme.denom} (D2).")

    # Q3 — desired behavior (D2)
    db = data_loader.distribution("desired_behavior", "D2")
    with _card("Q3", "What listening behaviors are users trying to achieve?"):
        t = _top(db.rows, exclude=("none_stated",), k=3)
        st.write("Where a goal is stated, users most want "
                 + ", ".join(_phrase(r) for r in t) + ".")
        st.bar_chart(_bar(db.rows), horizontal=True)
        st.caption(f"Denominator: {db.denom} {db.denom_label}. "
                   "'none_stated' means the review named a complaint but no explicit goal.")

    # Q4 — repetition cause (denom 19)
    rc = data_loader.repetition_cause_distribution()
    with _card("Q4", "What causes users to repeatedly listen to the same content?"):
        lead = rc.rows[0] if rc.rows else {"value": "—", "count": 0, "pct": 0}
        st.write(f"Where a cause is stated, it is almost always {_phrase(lead)}.")
        st.bar_chart(_bar(rc.rows), horizontal=True)
        st.caption(f"Denominator: {rc.denom} {rc.denom_label}. "
                   "⚠️ Small base — directional, not a population estimate.")

    # Q5 — behavior segment (D2)
    seg = data_loader.distribution("behavior_segment", "D2")
    with _card("Q5", "Which user segments experience different discovery challenges?"):
        t = _top(seg.rows, exclude=("unclear",), k=3)
        st.write("The clearest segments are " + ", ".join(_phrase(r) for r in t) + ".")
        st.bar_chart(_bar(seg.rows), horizontal=True)
        st.caption(f"Denominator: {seg.denom} {seg.denom_label}. "
                   "'unclear' = the review's dominant behaviour couldn't be read from text.")

    # Q6 — unmet needs (fragmented)
    un = data_loader.top_unmet_needs(top_n=6)
    with _card("Q6", "What unmet needs emerge consistently across reviews?"):
        st.write("Needs are **fragmented** — no single dominant ask. The most frequent phrasing, "
                 + (_phrase(un.rows[0]) if un.rows else "—") + ", appears only a handful of times.")
        st.bar_chart(_bar(un.rows), horizontal=True)
        st.caption(f"{un.denom} of {h['d2']} discovery reviews expressed a need, spread across many "
                   "phrasings. ⚠️ Per-phrase counts are directional.")

    # Q7 — algorithm vs habit (same n=19, as a ratio)
    with _card("Q7", "Is repetition driven more by Spotify's algorithm or by users' own habits?"):
        by = {r["value"]: r["count"] for r in rc.rows}
        a, u = by.get("algorithm_narrow", 0), by.get("user_habit", 0)
        m = st.columns(2)
        m[0].metric("Algorithm (algorithm_narrow)", a)
        m[1].metric("Own habit (user_habit)", u)
        st.write(f"In reviews that state a cause, blame falls on the algorithm over habit by {a}:{u}.")
        st.caption(f"Base = {rc.denom} reviews that state a cause. ⚠️ Directional at this n; "
                   "reviewers who don't mention a cause are excluded.")

    # Q8 — free vs premium, as a RATE
    tiers = {t["tier"]: t for t in data_loader.discovery_rate_by_tier()}
    with _card("Q8", "How does discovery frustration differ between free and premium users?"):
        prem, free = tiers.get("premium"), tiers.get("free")
        m = st.columns(2)
        m[0].metric("Premium — discovery share", f"{prem['rate']:.1%}", help=f"{prem['discovery']}/{prem['total']}")
        m[1].metric("Free — discovery share", f"{free['rate']:.1%}", help=f"{free['discovery']}/{free['total']}")
        st.write("Discovery frustration is proportionally **higher among premium** reviewers than free "
                 "— the opposite of what raw counts suggest.")
        st.caption("Rate = discovery-related share within each tier's reviews. "
                   f"⚠️ Tier is 'unclear' for {tiers['unclear']['total']}/{h['d1']} reviews and is "
                   "self-reported; treat as directional.")

    # Evidence drill-down (delete this block if you want cards only)
    st.divider()
    st.markdown("**Explore the evidence**")
    c1, c2 = st.columns(2)
    theme_sel = c1.selectbox("Theme", ["All"] + data_loader.unique_values("theme"))
    seg_sel = c2.selectbox("Segment", ["All"] + data_loader.unique_values("behavior_segment"))
    hits = data_loader.filter_reviews(theme=theme_sel, behavior_segment=seg_sel)
    st.caption(f"{len(hits)} discovery-related reviews match.")
    st.dataframe(hits, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Spotify Discovery — Research Insights", layout="wide")
    st.title("Spotify Discovery — Research Insights")
    render()
