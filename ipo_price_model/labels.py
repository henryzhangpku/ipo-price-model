"""labels.py — what actually happened, recorded so T-1 can never see it.

Per listing: the first close (the target), the first-day move offer -> first
close, and the close 30 sessions later (a secondary horizon). Prices from
yfinance, frozen to data/labels.parquet.

    ipo-price-model labels
"""
from __future__ import annotations

import pandas as pd

from .leakage import assert_label_after_decision
from .universe import DATA_DIR, load_universe

LABELS_PARQUET = DATA_DIR / "labels.parquet"
CLOSES_PARQUET = DATA_DIR / "closes.parquet"
HORIZON = 30
PAD_DAYS = 75


def label_one(closes: pd.Series, ipo_date: pd.Timestamp, offer_price: float | None,
              horizon: int = HORIZON) -> dict:
    """Pure: one listing's closes (index=date) -> label fields; None where the window is short."""
    s = closes.dropna()
    s = s[s.index >= pd.Timestamp(ipo_date).normalize()]
    out = {"first_close_date": None, "first_close": None, "first_day": None, f"fwd_{horizon}": None}
    if s.empty:
        return out
    d0 = s.index[0]
    assert_label_after_decision(pd.Timestamp(ipo_date) - pd.Timedelta(days=1), [d0])
    out["first_close_date"] = d0
    out["first_close"] = float(s.iloc[0])
    if offer_price:
        out["first_day"] = float(s.iloc[0] / offer_price - 1.0)
    if len(s) > horizon:
        out[f"fwd_{horizon}"] = float(s.iloc[horizon] / s.iloc[0] - 1.0)
    return out


def fetch_closes(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    if CLOSES_PARQUET.exists():
        cached = pd.read_parquet(CLOSES_PARQUET)
        if set(symbols).issubset(cached.columns):
            return cached
    import yfinance as yf
    raw = yf.download(sorted(set(symbols)), start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if isinstance(closes, pd.Series):
        closes = closes.to_frame(symbols[0])
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes = closes.sort_index()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(CLOSES_PARQUET)
    return closes


def build_labels(overwrite: bool = False) -> pd.DataFrame:
    if LABELS_PARQUET.exists() and not overwrite:
        return pd.read_parquet(LABELS_PARQUET)
    uni = load_universe()
    start = (uni["ipo_date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (uni["ipo_date"].max() + pd.Timedelta(days=PAD_DAYS)).strftime("%Y-%m-%d")
    closes = fetch_closes(uni["symbol"].tolist(), start, end)
    rows = []
    for _, r in uni.iterrows():
        sym = r["symbol"]
        series = closes[sym] if sym in closes.columns else pd.Series(dtype=float)
        lab = {"symbol": sym, "ipo_date": r["ipo_date"]}
        lab.update(label_one(series, r["ipo_date"], r.get("offer_price")))
        rows.append(lab)
    df = pd.DataFrame(rows)
    df.to_parquet(LABELS_PARQUET, index=False)
    return df


def load_labels() -> pd.DataFrame:
    if not LABELS_PARQUET.exists():
        raise SystemExit(f"{LABELS_PARQUET} missing; run `ipo-price-model labels` first")
    return pd.read_parquet(LABELS_PARQUET)
