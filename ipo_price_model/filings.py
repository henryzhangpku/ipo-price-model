"""filings.py — recover the filed price range from the registration statement itself.

The IPO calendar overwrites a deal's filed range with its offer price once the
deal prices, so a historical calendar carries no ranges at all. The range is
not lost, though: it is printed on the cover of the last S-1/A (F-1/A for
foreign issuers, S-11/A for REITs) filed before pricing — "the initial public
offering price will be between $15.00 and $17.00 per share". This module reads
it from there.

Point-in-time rule: only amendments with a filing date **strictly before the
first trading day** are read, newest first, and the first cover that states a
range wins. So a re-range filed before pricing is used and the final
prospectus (424B4, filed after pricing) never is. ``leakage`` asserts it.

CIKs come from the SEC ticker map, with the Nasdaq deal record as a fallback
for names that have since delisted. Everything is cached under data/ and never
re-pulled.

    ipo-price-model ranges
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests

from .leakage import assert_filing_before_listing
from .universe import DATA_DIR, UNIVERSE_PARQUET, core, load_universe

FILINGS_DIR = DATA_DIR / "filings"
CIK_JSON = DATA_DIR / "ciks.json"
RANGES_JSON = DATA_DIR / "ranges.json"

REG_FORMS = ("S-1/A", "F-1/A", "S-11/A", "S-1", "F-1", "S-11")
HEAD_BYTES = 450_000            # the cover page is always in the first few hundred KB
MAX_DOCS = 5                    # amendments read per deal before giving up

SEC_UA = os.environ.get("SEC_USER_AGENT", "ipo-price-model research contact@sniperopt.com")
NASDAQ_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"

_NUM = r"(\d{1,4}(?:\.\d{1,2})?)"
RANGE_RE = re.compile(r"between\s+(?:US)?\$\s?" + _NUM + r"\s+and\s+(?:US)?\$\s?" + _NUM, re.I)
CONTEXT_RE = re.compile(r"\b(?:per|each)\b|offering price", re.I)


def html_to_text(raw: bytes) -> str:
    t = raw.decode("latin-1", errors="ignore")
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"&nbsp;|&#160;|&#xa0;|&#xA0;", " ", t)
    t = t.replace("&#36;", "$").replace("&amp;", "&")
    return re.sub(r"\s+", " ", t)


def parse_range(text: str) -> tuple[float, float] | None:
    """Cover-page text -> (low, high), or None. Pure; covered by tests.

    Takes the first 'between $a and $b ... per share' with a sane shape: low < high,
    high at most twice low, and a price under $1,000."""
    for m in RANGE_RE.finditer(text):
        # 'the offering price per share will be between $a and $b' or '... between $a and $b per share'
        context = text[max(0, m.start() - 160):m.start()] + " " + text[m.end():m.end() + 120].split(".")[0]
        if not CONTEXT_RE.search(context):
            continue
        lo, hi = float(m.group(1)), float(m.group(2))
        if 0 < lo < hi <= 2 * lo and hi < 1000:
            return lo, hi
    return None


class _Http:
    def __init__(self, pause: float = 0.15):
        self.s = requests.Session()
        self.pause = pause

    def get(self, url: str, ua: str, stream: bool = False, accept: str | None = None) -> requests.Response:
        headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}
        if accept:
            headers["Accept"] = accept
        for attempt in range(5):
            try:
                r = self.s.get(url, headers=headers, timeout=60, stream=stream)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(10 * (attempt + 1))
                continue
            if r.status_code in (429, 503):
                time.sleep(5 * (attempt + 1))
                continue
            time.sleep(self.pause)
            return r
        raise RuntimeError(f"gave up on {url} after five attempts")

    def head_bytes(self, url: str, ua: str, n: int = HEAD_BYTES) -> bytes:
        r = self.get(url, ua, stream=True)
        if r.status_code != 200:
            return b""
        buf = b""
        for chunk in r.iter_content(65536):
            buf += chunk
            if len(buf) >= n:
                break
        r.close()
        return buf


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _save(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str), encoding="utf-8")


def _nasdaq_deals(http: _Http, months: list[str]) -> dict[str, str]:
    """(symbol) -> Nasdaq dealID for priced deals in the given YYYY-MM months."""
    out: dict[str, str] = {}
    for mth in months:
        r = http.get(f"https://api.nasdaq.com/api/ipo/calendar?date={mth}", NASDAQ_UA, accept="application/json")
        try:
            rows = ((r.json().get("data") or {}).get("priced") or {}).get("rows") or []
        except ValueError:
            rows = []
        for row in rows:
            sym = (row.get("proposedTickerSymbol") or "").strip().upper()
            if sym and row.get("dealID"):
                out.setdefault(sym, row["dealID"])
    return out


def resolve_ciks(uni: pd.DataFrame, http: _Http) -> dict[str, str]:
    """symbol -> 10-digit CIK, cached. SEC ticker map first, Nasdaq deal record for the rest."""
    ciks: dict[str, str] = _load(CIK_JSON)
    todo = [s for s in uni["symbol"] if s not in ciks]
    if not todo:
        return ciks
    tick = http.get("https://www.sec.gov/files/company_tickers.json", SEC_UA).json()
    sec = {v["ticker"].upper(): f"{int(v['cik_str']):010d}" for v in tick.values()}
    for s in todo:
        if s in sec:
            ciks[s] = sec[s]
    _save(CIK_JSON, ciks)
    todo = [s for s in todo if s not in ciks]
    if todo:
        rows = uni[uni["symbol"].isin(todo)]
        months = sorted({d.strftime("%Y-%m") for d in pd.to_datetime(rows["ipo_date"])}
                        | {(d - pd.offsets.MonthBegin(1)).strftime("%Y-%m") for d in pd.to_datetime(rows["ipo_date"])})
        deals = _nasdaq_deals(http, months)
        for s in todo:
            deal = deals.get(s)
            if not deal:
                ciks[s] = ""
                continue
            r = http.get(f"https://api.nasdaq.com/api/ipo/overview/?dealId={deal}", NASDAQ_UA, accept="application/json")
            try:
                v = r.json()["data"]["poOverview"]["SECCIK"]["value"]
                ciks[s] = f"{int(v):010d}" if v and str(v).strip().isdigit() else ""
            except (ValueError, KeyError, TypeError):
                ciks[s] = ""
            if len(ciks) % 25 == 0:
                _save(CIK_JSON, ciks)
    _save(CIK_JSON, ciks)
    return ciks


def _registration_filings(http: _Http, cik: str) -> list[dict]:
    cache = FILINGS_DIR / f"{cik}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    r = http.get(f"https://data.sec.gov/submissions/CIK{cik}.json", SEC_UA)
    if r.status_code != 200:
        return []
    sub = r.json()
    pages = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):          # older filings for long-lived registrants
        rr = http.get(f"https://data.sec.gov/submissions/{f['name']}", SEC_UA)
        if rr.status_code == 200:
            pages.append(rr.json())
    rows = []
    for p in pages:
        for form, date, acc, doc in zip(p["form"], p["filingDate"], p["accessionNumber"], p["primaryDocument"]):
            if form in REG_FORMS:
                rows.append({"form": form, "date": date, "accession": acc, "doc": doc})
    _save(cache, rows)
    return rows


def filed_range(http: _Http, cik: str, ipo_date: pd.Timestamp) -> dict:
    """The last range stated on a registration cover filed strictly before the first trading day."""
    first_trade = pd.Timestamp(ipo_date).normalize()
    cands = [f for f in _registration_filings(http, cik) if pd.Timestamp(f["date"]) < first_trade]
    cands.sort(key=lambda f: (f["date"], f["form"].endswith("/A")), reverse=True)
    for f in cands[:MAX_DOCS]:
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{f['accession'].replace('-', '')}/{f['doc']}"
        got = parse_range(html_to_text(http.head_bytes(url, SEC_UA)))
        if got:
            assert_filing_before_listing(pd.Timestamp(f["date"]), first_trade)
            return {"low": got[0], "high": got[1], "filed": f["date"], "form": f["form"], "accession": f["accession"]}
    return {"low": None, "high": None, "filed": None, "form": None, "accession": None,
            "why": "no_cik" if not cik else ("no_filing_before_listing" if not cands else "no_range_on_cover")}


def build_ranges(overwrite: bool = False, log_every: int = 50) -> pd.DataFrame:
    """Attach the filed range to every core listing and write the universe back."""
    uni = load_universe()
    target = core(uni)
    http = _Http()
    ciks = resolve_ciks(target, http)
    ranges: dict = {} if overwrite else _load(RANGES_JSON)
    for i, r in enumerate(target.itertuples(index=False), 1):
        key = f"{r.symbol}|{pd.Timestamp(r.ipo_date).date()}"
        if key in ranges and ranges[key].get("why") not in ("no_range_on_cover", "fetch_error"):
            continue
        try:
            ranges[key] = filed_range(http, ciks.get(r.symbol, ""), r.ipo_date) if ciks.get(r.symbol) else \
                {"low": None, "high": None, "filed": None, "form": None, "accession": None, "why": "no_cik"}
        except (RuntimeError, requests.RequestException, ValueError):
            ranges[key] = {"low": None, "high": None, "filed": None, "form": None, "accession": None,
                           "why": "fetch_error"}      # retried on the next run
        if i % log_every == 0:
            _save(RANGES_JSON, ranges)
            print(f"  ranges {i}/{len(target)}: {sum(1 for v in ranges.values() if v.get('low'))} found", flush=True)
    _save(RANGES_JSON, ranges)

    keys = uni["symbol"] + "|" + pd.to_datetime(uni["ipo_date"]).dt.date.astype(str)
    got = keys.map(lambda k: ranges.get(k, {}))
    uni["price_low"] = [g.get("low") if g.get("low") else lo for g, lo in zip(got, uni["price_low"])]
    uni["price_high"] = [g.get("high") if g.get("high") else hi for g, hi in zip(got, uni["price_high"])]
    uni["range_filed"] = pd.to_datetime([g.get("filed") for g in got], errors="coerce")
    uni["range_source"] = [g.get("accession") for g in got]
    uni["has_range"] = uni["price_low"].notna() & uni["price_high"].notna()
    uni.to_parquet(UNIVERSE_PARQUET, index=False)
    return uni
