"""universe.py — freeze the historical IPO calendar.

Source: the Finnhub IPO calendar (free tier is enough), pulled in monthly
batches and never re-pulled for a window already on disk. Only `priced` rows
are kept. SPACs, units and warrants are tagged rather than dropped so they can
be reported as their own cohorts.

    ipo-price-model universe --from 2019-01-01 --to 2025-12-31
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
UNIVERSE_PARQUET = DATA_DIR / "universe.parquet"
FINNHUB_URL = "https://finnhub.io/api/v1/calendar/ipo"

SPAC_RE = re.compile(r"acquisition|spac|blank check", re.I)
UNIT_SUFFIX_RE = re.compile(r"(U|W|WS|R)$")

COLUMNS = [
    "symbol", "name", "ipo_date", "exchange", "status", "price_raw",
    "price_low", "price_high", "offer_price", "has_range",
    "shares", "deal_usd", "is_spac", "is_unit",
]


def parse_price(raw) -> tuple[float | None, float | None, float | None]:
    """'17.00-19.00' -> (17, 19, None); '18.50' -> (None, None, 18.5); '' -> (None,)*3.

    The calendar keeps the filed range until pricing and then overwrites it with
    the final price, so a historical row usually carries one or the other."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, None
    s = str(raw).replace("$", "").replace(",", "").strip()
    if not s:
        return None, None, None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)$", s)
    if m:
        return float(m.group(1)), float(m.group(2)), None
    m = re.match(r"^(\d+(?:\.\d+)?)$", s)
    if m:
        return None, None, float(m.group(1))
    return None, None, None


def normalise(rows: list[dict]) -> pd.DataFrame:
    """Calendar rows -> the frozen schema. Pure; covered by tests."""
    out = []
    for r in rows:
        low, high, offer = parse_price(r.get("price"))
        shares = r.get("numberOfShares")
        deal = r.get("totalSharesValue")
        sym = (r.get("symbol") or "").strip().upper()
        name = (r.get("name") or "").strip()
        if not deal and shares and offer:
            deal = float(shares) * float(offer)
        out.append({
            "symbol": sym,
            "name": name,
            "ipo_date": pd.to_datetime(r.get("date"), errors="coerce"),
            "exchange": (r.get("exchange") or "").strip(),
            "status": (r.get("status") or "").strip().lower(),
            "price_raw": r.get("price"),
            "price_low": low,
            "price_high": high,
            "offer_price": offer,
            "has_range": low is not None and high is not None,
            "shares": float(shares) if shares else None,
            "deal_usd": float(deal) if deal else None,
            "is_spac": bool(SPAC_RE.search(name)),
            "is_unit": bool(UNIT_SUFFIX_RE.search(sym)) and len(sym) >= 4,
        })
    df = pd.DataFrame(out, columns=COLUMNS)
    df = df[df["symbol"].ne("") & df["ipo_date"].notna()]
    return df.sort_values(["ipo_date", "symbol"]).reset_index(drop=True)


def _month_batches(date_from: str, date_to: str):
    cur, end = pd.Timestamp(date_from), pd.Timestamp(date_to)
    while cur <= end:
        nxt = (cur + pd.offsets.MonthEnd(0)).normalize()
        yield cur.strftime("%Y-%m-%d"), min(nxt, end).strftime("%Y-%m-%d")
        cur = nxt + pd.Timedelta(days=1)


def fetch_calendar(date_from: str, date_to: str, token: str, pause: float = 1.1) -> list[dict]:
    rows: list[dict] = []
    for a, b in _month_batches(date_from, date_to):
        for attempt in range(4):
            resp = requests.get(FINNHUB_URL, params={"from": a, "to": b, "token": token}, timeout=30)
            if resp.status_code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            resp.raise_for_status()
            rows.extend(resp.json().get("ipoCalendar", []) or [])
            break
        else:
            raise RuntimeError(f"calendar {a}..{b}: rate-limited four times")
        time.sleep(pause)
    return rows


def build_universe(date_from: str, date_to: str, overwrite: bool = False) -> pd.DataFrame:
    if UNIVERSE_PARQUET.exists() and not overwrite:
        cached = pd.read_parquet(UNIVERSE_PARQUET)
        lo, hi = cached["ipo_date"].min(), cached["ipo_date"].max()
        if lo <= pd.Timestamp(date_from) and hi >= pd.Timestamp(date_to) - pd.Timedelta(days=45):
            return cached
    token = os.environ.get("FINNHUB_API_KEY", "")
    if not token:
        raise SystemExit("FINNHUB_API_KEY is not set; the universe cannot be frozen without it")
    df = normalise(fetch_calendar(date_from, date_to, token))
    df = df[df["status"].eq("priced")].reset_index(drop=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(UNIVERSE_PARQUET, index=False)
    return df


def load_universe() -> pd.DataFrame:
    if not UNIVERSE_PARQUET.exists():
        raise SystemExit(f"{UNIVERSE_PARQUET} missing; run `ipo-price-model universe` first")
    return pd.read_parquet(UNIVERSE_PARQUET)


def core(df: pd.DataFrame) -> pd.DataFrame:
    """The cohort the model is fitted on: ordinary listings with an offer price."""
    return df[~df["is_spac"] & ~df["is_unit"] & df["offer_price"].notna()].reset_index(drop=True)
