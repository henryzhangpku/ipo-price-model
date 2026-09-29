"""features.py — the T-1 evidence, and nothing that was not known at T-1.

Two kinds of feature:

* **deal facts** from the calendar row itself: offer price, where it priced
  against the filed range, deal size, exchange;
* **regime**: what the *previous* listings did — the trailing first-day move of
  listings whose first close fell strictly before this one's decision date.
  A leakage assertion checks the strict ordering.

The full column list is declared in ``FEATURES`` and enforced by
``leakage.assert_features_known_at``; adding a column means adding it there,
on purpose.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .leakage import assert_features_known_at, assert_regime_uses_only_past

MAJOR = {"NYSE", "NASDAQ", "NASDAQ Global", "NASDAQ Global Select", "NASDAQ Capital", "NYSE American"}

FEATURES = [
    "log_offer", "log_deal", "priced_vs_range", "range_width", "has_range", "major_exchange",
    "regime_first_day_median", "regime_first_day_iqr", "regime_n",
]
PVR_CODE = {"below": -1.0, "within": 0.0, "top": 0.5, "above": 1.0, "unknown": 0.0}

REGIME_WINDOW_DAYS = 90
REGIME_MIN_N = 5


def priced_vs_range(row) -> str:
    lo, hi, px = row.get("price_low"), row.get("price_high"), row.get("offer_price")
    if any(v is None or pd.isna(v) for v in (lo, hi, px)):
        return "unknown"
    if px > hi:
        return "above"
    if px == hi:
        return "top"
    if px < lo:
        return "below"
    return "within"


def regime(universe_labels: pd.DataFrame, decision_dates: pd.Series,
           window_days: int = REGIME_WINDOW_DAYS) -> pd.DataFrame:
    """Trailing first-day statistics over listings whose first close is strictly before T-1.

    ``universe_labels`` needs ``first_close_date`` and ``first_day``. Pure; O(n*window)."""
    past = universe_labels.dropna(subset=["first_close_date", "first_day"]).sort_values("first_close_date")
    pdates = pd.to_datetime(past["first_close_date"]).to_numpy()
    pfd = past["first_day"].to_numpy(dtype=float)
    med, iqr, n, last = [], [], [], []
    for d in pd.to_datetime(decision_dates):
        lo = np.searchsorted(pdates, np.datetime64(d - pd.Timedelta(days=window_days)), side="left")
        hi = np.searchsorted(pdates, np.datetime64(d), side="left")   # strictly before T-1
        w = pfd[lo:hi]
        if len(w) >= REGIME_MIN_N:
            med.append(float(np.median(w)))
            iqr.append(float(np.percentile(w, 75) - np.percentile(w, 25)))
        else:
            med.append(np.nan)
            iqr.append(np.nan)
        n.append(int(len(w)))
        last.append(pdates[hi - 1] if hi > 0 else np.datetime64("NaT"))
    out = pd.DataFrame({"regime_first_day_median": med, "regime_first_day_iqr": iqr, "regime_n": n,
                        "_regime_last_date": last})
    ok = out["_regime_last_date"].notna()
    if ok.any():
        assert_regime_uses_only_past(out.loc[ok, "_regime_last_date"], pd.Series(pd.to_datetime(decision_dates))[ok])
    return out.drop(columns=["_regime_last_date"])


def build(universe: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for every row of ``universe``, aligned by symbol. Labels are used ONLY for the regime feature
    of *other* listings; the strict-before check guarantees no row sees its own or any later outcome."""
    df = universe.reset_index(drop=True)
    decision = pd.to_datetime(df["ipo_date"]) - pd.Timedelta(days=1)
    ul = df[["symbol"]].merge(labels[["symbol", "first_close_date", "first_day"]], on="symbol", how="left")
    reg = regime(ul, decision)
    pvr = df.apply(priced_vs_range, axis=1)
    width = (df["price_high"] - df["price_low"]) / df["offer_price"]
    feats = pd.DataFrame({
        "log_offer": np.log(df["offer_price"].astype(float)),
        "log_deal": np.log(df["deal_usd"].astype(float).clip(lower=1e6)),
        "priced_vs_range": pvr.map(PVR_CODE).astype(float),
        "range_width": width.fillna(0.0).astype(float),
        "has_range": df["has_range"].astype(float),
        "major_exchange": df["exchange"].isin(MAJOR).astype(float),
    })
    feats = pd.concat([feats, reg], axis=1)
    feats["regime_first_day_median"] = feats["regime_first_day_median"].fillna(0.0)
    feats["regime_first_day_iqr"] = feats["regime_first_day_iqr"].fillna(feats["regime_first_day_iqr"].median() if feats["regime_first_day_iqr"].notna().any() else 0.0)
    feats = feats[FEATURES]
    assert_features_known_at(feats, set(FEATURES))
    return feats
