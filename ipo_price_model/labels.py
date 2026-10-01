"""labels.py — what actually happened, recorded so T-1 can never see it.

Per listing: the first close (the target), the first-day move offer -> first
close, and the close 30 sessions later (a secondary horizon). Prices from
yfinance, frozen to data/labels.parquet.

The price source adjusts history for every split up to today, and the offer
price is not adjusted at all. A listing that later reverse-split 1-for-10
would otherwise show a first close ten times too high. So the split history is
fetched with the closes, and the first close is restored to the price that
actually printed: raw = adjusted x the product of every split ratio after that
day. The 30-session return uses adjusted closes, where splits cancel.

Fetching is sequential, in small batches, with backoff on rate limits, and
resumable: a throttled run can simply be re-run.

    ipo-price-model labels
"""
from __future__ import annotations

import json
import time

import pandas as pd

from .leakage import assert_label_after_decision
from .universe import DATA_DIR, core, load_universe

LABELS_PARQUET = DATA_DIR / "labels.parquet"
CLOSES_PARQUET = DATA_DIR / "closes.parquet"
SPLITS_PARQUET = DATA_DIR / "splits.parquet"
ATTEMPTED_JSON = DATA_DIR / "closes_attempted.json"
HORIZON = 30
BATCH = 40

# Declared data-validity bounds, fixed before any result: a label outside them is
# treated as a data error (reused ticker, missed split), not as an outcome.
MAX_FIRST_PRINT_LAG_DAYS = 7      # first close must land within a week of the listing date
MIN_RATIO, MAX_RATIO = 0.10, 30.0 # first close / offer


def split_factor(splits: pd.Series | None, after: pd.Timestamp) -> float:
    """Product of split ratios dated strictly after ``after`` (2.0 = 2-for-1, 0.1 = 1-for-10). Pure."""
    if splits is None or len(splits) == 0:
        return 1.0
    s = splits.dropna()
    s = s[(s > 0) & (pd.DatetimeIndex(s.index) > pd.Timestamp(after))]
    return float(s.prod()) if len(s) else 1.0


def label_one(closes: pd.Series, ipo_date: pd.Timestamp, offer_price: float | None,
              horizon: int = HORIZON, splits: pd.Series | None = None) -> dict:
    """Pure: split-adjusted closes (index=date) -> label fields; None where the data cannot carry one.

    ``splits`` restores the first close to the price that printed on the day."""
    out = {"first_close_date": None, "first_close": None, "first_day": None, f"fwd_{horizon}": None}
    s = closes.dropna()
    if s.empty:                                   # no price history at all (delisted, or never fetched)
        return out
    s = s[pd.DatetimeIndex(s.index) >= pd.Timestamp(ipo_date).normalize()]
    if s.empty:
        return out
    d0 = s.index[0]
    assert_label_after_decision(pd.Timestamp(ipo_date) - pd.Timedelta(days=1), [d0])
    if d0 - pd.Timestamp(ipo_date).normalize() > pd.Timedelta(days=MAX_FIRST_PRINT_LAG_DAYS):
        return out                                # the source does not have this listing's first print
    raw_first = float(s.iloc[0]) * split_factor(splits, d0)
    if not raw_first > 0:                         # a non-positive print is bad data, not a label
        return out
    if offer_price and not (MIN_RATIO <= raw_first / offer_price <= MAX_RATIO):
        return out                                # a reused ticker or a missed split, not a first day
    out["first_close_date"] = d0
    out["first_close"] = raw_first
    if offer_price:
        out["first_day"] = float(raw_first / offer_price - 1.0)
    if len(s) > horizon and s.iloc[0] > 0:
        out[f"fwd_{horizon}"] = float(s.iloc[horizon] / s.iloc[0] - 1.0)
    return out


def _field(raw: pd.DataFrame, name: str, syms: list[str]) -> pd.DataFrame:
    top = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else raw.columns
    if name not in set(top):
        return pd.DataFrame(index=raw.index)
    f = raw[name] if isinstance(raw.columns, pd.MultiIndex) else raw[[name]].rename(columns={name: syms[0]})
    if isinstance(f, pd.Series):
        f = f.to_frame(syms[0])
    f = f.copy()
    f.index = pd.to_datetime(f.index).tz_localize(None).normalize()
    return f


def _download(yf, syms: list[str], start: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-adjusted closes (not dividend-adjusted) and the split history, from ``start`` to today."""
    raw = yf.download(syms, start=start, auto_adjust=False, actions=True, progress=False,
                      group_by="column", threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    closes = _field(raw, "Close", syms)
    splits = _field(raw, "Stock Splits", syms)
    if not splits.empty:
        splits = splits.loc[:, (splits.fillna(0) != 0).any()]
    return closes, splits


def _rate_limited(yf) -> bool:
    errs = getattr(getattr(yf, "shared", None), "_ERRORS", {}) or {}
    return any("Rate" in str(v) or "Too Many" in str(v) for v in errs.values())


def _join(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    if b.empty:
        return a
    out = b if a.empty else a.join(b.loc[:, ~b.columns.isin(a.columns)], how="outer")
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def fetch_closes(symbols: list[str], start: str, pause: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Closes and splits, fetched in small sequential batches with backoff, cached and resumable.

    A symbol is marked attempted only when its batch came back without a rate-limit
    error, so a throttled run can simply be re-run and picks up where it stopped."""
    import yfinance as yf
    closes = pd.read_parquet(CLOSES_PARQUET) if CLOSES_PARQUET.exists() else pd.DataFrame()
    closes = closes.dropna(axis=1, how="all")          # an all-empty column is a failed fetch, not a result
    splits = pd.read_parquet(SPLITS_PARQUET) if SPLITS_PARQUET.exists() else pd.DataFrame()
    attempted = set(json.loads(ATTEMPTED_JSON.read_text())) if ATTEMPTED_JSON.exists() else set()
    todo = sorted(set(symbols) - attempted - set(closes.columns))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        for attempt in range(6):
            got, got_splits = _download(yf, batch, start)
            if not _rate_limited(yf):
                break
            time.sleep(30 * (attempt + 1))
        else:
            print(f"  closes: still rate-limited at batch {i // BATCH}; re-run to resume", flush=True)
            break
        closes = _join(closes, got.dropna(axis=1, how="all") if not got.empty else got)
        splits = _join(splits, got_splits)
        attempted.update(batch)
        closes.to_parquet(CLOSES_PARQUET)
        (splits if not splits.empty else pd.DataFrame({"none": pd.Series(dtype=float)})).to_parquet(SPLITS_PARQUET)
        ATTEMPTED_JSON.write_text(json.dumps(sorted(attempted)))
        print(f"  closes: {min(i + BATCH, len(todo))}/{len(todo)} fetched, {closes.shape[1]} with prices, "
              f"{splits.shape[1]} with splits", flush=True)
        time.sleep(pause)
    if not closes.empty:
        closes.index = pd.to_datetime(closes.index)
    return closes.sort_index(), splits.drop(columns=["none"], errors="ignore").sort_index()


def build_labels(overwrite: bool = False) -> pd.DataFrame:
    if LABELS_PARQUET.exists() and not overwrite:
        return pd.read_parquet(LABELS_PARQUET)
    uni = core(load_universe())   # the fitted cohort only; SPAC units are not labelled
    start = (uni["ipo_date"].min() - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    closes, splits = fetch_closes(uni["symbol"].tolist(), start)
    rows = []
    for _, r in uni.iterrows():
        sym = r["symbol"]
        series = closes[sym] if sym in closes.columns else pd.Series(dtype=float)
        spl = splits[sym] if sym in splits.columns else None
        lab = {"symbol": sym, "ipo_date": r["ipo_date"]}
        lab.update(label_one(series, r["ipo_date"], r.get("offer_price"), splits=spl))
        rows.append(lab)
    df = pd.DataFrame(rows)
    df.to_parquet(LABELS_PARQUET, index=False)
    return df


def load_labels() -> pd.DataFrame:
    if not LABELS_PARQUET.exists():
        raise SystemExit(f"{LABELS_PARQUET} missing; run `ipo-price-model labels` first")
    return pd.read_parquet(LABELS_PARQUET)
