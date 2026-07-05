"""Data layer for the Research Insights tab.

Loads data/reviews_labeled.csv (your 1,000 classified reviews) and computes aggregates
that REPRODUCE the workbook Dashboard exactly (verified). Streamlit-decoupled so it can
be tested standalone. Presentation (which chart answers which question) lives in dashboard.py.

Denominator discipline:
  D1 = all 1000 classified reviews
  D2 = the 72 discovery-related reviews (discovery_related == true)
Each Distribution carries its own denom + label so the UI can never mislabel a percentage.
"""
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

import config


@dataclass
class Distribution:
    field: str
    denom: int
    denom_label: str
    rows: list  # [{'value': str, 'count': int, 'pct': float}], descending by count


@lru_cache(maxsize=1)
def load_reviews() -> pd.DataFrame:
    """Load and lightly normalize the labeled corpus (cached; the file is small)."""
    df = pd.read_csv(config.LABELED_CSV, dtype=str).fillna("")
    for col in config.LABEL_FIELDS + ["source", "tier"]:
        if col in df.columns:
            df[col] = df[col].str.strip()
    df["discovery_related"] = df["discovery_related"].str.lower()
    df["rating_num"] = pd.to_numeric(df["rating"], errors="coerce")
    return df


def _d2(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_reviews() if df is None else df
    return df[df["discovery_related"] == "true"]


def _dist(series: pd.Series, denom: int, denom_label: str, field: str) -> Distribution:
    vc = series.value_counts()
    rows = [
        {"value": v, "count": int(c), "pct": (int(c) / denom if denom else 0.0)}
        for v, c in vc.items()
    ]
    return Distribution(field, denom, denom_label, rows)


# --- headline -----------------------------------------------------------------

def headline_stats() -> dict:
    """Scale + honesty strip for the top of the Research tab."""
    df = load_reviews()
    d2 = _d2(df)
    dates = pd.to_datetime(df["date"], errors="coerce", utc=True)
    return {
        "d1": len(df),
        "d2": len(d2),
        "discovery_pct": len(d2) / len(df) if len(df) else 0.0,
        "sources": {k: int(v) for k, v in df["source"].value_counts().items()},
        "date_min": None if dates.isna().all() else dates.min().date().isoformat(),
        "date_max": None if dates.isna().all() else dates.max().date().isoformat(),
        "avg_rating": round(df["rating_num"].mean(), 2) if df["rating_num"].notna().any() else None,
    }


# --- standard D2 / D1 distributions ------------------------------------------

def distribution(field: str, among: str = "D2") -> Distribution:
    """Value distribution for a field over D2 (discovery-related) or D1 (all)."""
    df = load_reviews()
    if among == "D2":
        sub, denom, label = _d2(df), len(_d2(df)), "discovery-related reviews (D2)"
    elif among == "D1":
        sub, denom, label = df, len(df), "all classified reviews (D1)"
    else:
        raise ValueError("among must be 'D1' or 'D2'")
    return _dist(sub[field], denom, label, field)


# --- special-denominator blocks ----------------------------------------------

def repetition_cause_distribution() -> Distribution:
    """Q4/Q7. Denominator = reviews that HAVE a repetition cause (theme is
    repetition_loop or stale_freshness), NOT D2. Reproduces the workbook's n=19.
    NOTE for the UI: n is small — present as directional, never as a hard '95%'."""
    d2 = _d2()
    applicable = d2[d2["repetition_cause"].str.lower() != "not_applicable"]
    return _dist(
        applicable["repetition_cause"], len(applicable),
        "reviews with a repetition cause (theme = repetition_loop or stale_freshness)",
        "repetition_cause",
    )


def top_unmet_needs(top_n: int = 8) -> Distribution:
    """Q6. Free-text needs among D2, excluding 'none'. n is small (~14) — directional."""
    d2 = _d2()
    s = d2[d2["unmet_need"].str.lower() != "none"]["unmet_need"]
    denom = len(s)
    vc = s.value_counts().head(top_n)
    rows = [{"value": v, "count": int(c), "pct": (int(c) / denom if denom else 0.0)} for v, c in vc.items()]
    return Distribution("unmet_need", denom, "D2 reviews expressing an unmet need", rows)


def discovery_rate_by_tier() -> list[dict]:
    """Q8 (sharper framing). Discovery-frustration RATE within each tier, on D1 tier labels.
    Caveat for the UI: tier is 'unclear' for most reviews and is self-reported."""
    df = load_reviews()
    out = []
    for t in ["free", "premium", "unclear"]:
        sub = df[df["tier"].str.lower() == t]
        hits = int((sub["discovery_related"] == "true").sum())
        out.append({
            "tier": t, "discovery": hits, "total": len(sub),
            "rate": (hits / len(sub) if len(sub) else 0.0),
        })
    return out


# --- drill-down ---------------------------------------------------------------

DRILL_COLUMNS = ["id", "source", "date", "rating", "text", "theme",
                 "behavior_segment", "tier", "unmet_need", "sentiment", "evidence_phrase"]


def unique_values(field: str, among: str = "D2") -> list[str]:
    """Sorted distinct values for a field (for dashboard filter dropdowns)."""
    sub = _d2() if among == "D2" else load_reviews()
    return sorted(v for v in sub[field].unique() if v != "")


def filter_reviews(discovery_only: bool = True, **criteria) -> pd.DataFrame:
    """Return raw reviews for the drill-down. criteria are exact field matches;
    a value of None / '' / 'All' is ignored so the UI can pass 'All' to mean no filter."""
    df = _d2() if discovery_only else load_reviews()
    for field, value in criteria.items():
        if value not in (None, "", "All"):
            df = df[df[field].astype(str) == str(value)]
    cols = [c for c in DRILL_COLUMNS if c in df.columns]
    return df[cols].reset_index(drop=True)


if __name__ == "__main__":  # standalone check: python -m src.data_loader
    h = headline_stats()
    print(f"D1={h['d1']} D2={h['d2']} discovery={h['discovery_pct']:.1%} "
          f"dates={h['date_min']}..{h['date_max']} avg_rating={h['avg_rating']}")
    print("sources:", h["sources"])
    for d in (distribution("theme"), repetition_cause_distribution(),
              distribution("behavior_segment"), top_unmet_needs(5)):
        top = ", ".join(f"{r['value']}={r['count']}" for r in d.rows[:4])
        print(f"{d.field} (denom {d.denom}, {d.denom_label}): {top}")
    print("tier rate:", [(x['tier'], f"{x['rate']:.1%}", f"{x['discovery']}/{x['total']}")
                         for x in discovery_rate_by_tier()])
    print("drill repetition_loop rows:", len(filter_reviews(theme="repetition_loop")))
