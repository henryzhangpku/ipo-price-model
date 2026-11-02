"""tracker.py — follow one listing to its first trade with a frozen model, and keep the record.

The model is not refitted here. It is read from ``docs/model.json`` (the trees,
the conformal shift and the reference band exported by ``docs/export_model.py``)
and its SHA-256 is written into every record, so anyone can check that the
model scoring the deal on the last day is the one published on the first.

What moves day to day are the T-1 inputs, each read from a public source:

* **the filing** — the issuer's latest registration statement on EDGAR: the
  filed range and the share count from its cover, and on pricing night the final
  offer price from the 424B prospectus;
* **the regime** — first-day moves of listings that closed in the 90 days before
  the decision date: the frozen research data through ``frozen_through``, then
  listings from the public Nasdaq IPO calendar with first closes from yfinance,
  filtered and labelled by the same rules as the training data;
* **the result** — after the first trade, the first close, scored against the
  band that was published before it.

Each run appends one line to ``tracker/tape.jsonl`` when anything changed and
rewrites ``docs/live.json`` for the page. The tape is append-only: a record is
never edited, a later one supersedes it.

    ipo-price-model track --deal tracker/ant.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .features import MAJOR, PVR_CODE, REGIME_MIN_N, REGIME_WINDOW_DAYS, priced_vs_range
from .gate import GateConfig
from .universe import ROOT, SPAC_RE, UNIT_SUFFIX_RE

MODEL_JSON = ROOT / "docs" / "model.json"
LIVE_JSON = ROOT / "docs" / "live.json"
TRACKER_DIR = ROOT / "tracker"
TAPE = TRACKER_DIR / "tape.jsonl"
LISTINGS_JSON = TRACKER_DIR / "regime_listings.json"

STAGES = ("filed_no_range", "range_filed", "priced", "trading", "scored")

# scenario offers relative to the filed range, fixed before the range is known
SCENARIOS = (("above", "10% above the range", lambda lo, hi: round(hi * 1.10, 2)),
             ("top", "At the top of the range", lambda lo, hi: hi),
             ("mid", "Mid-range", lambda lo, hi: round((lo + hi) / 2, 2)),
             ("below", "8% below the range", lambda lo, hi: round(lo * 0.92, 2)))


# ---------------------------------------------------------------- frozen model

@dataclass(frozen=True)
class FrozenModel:
    """The exported trees, evaluated exactly as sklearn does (float32 split test)."""
    spec: dict
    sha256: str

    @classmethod
    def load(cls, path: Path = MODEL_JSON) -> "FrozenModel":
        raw = path.read_bytes()
        return cls(json.loads(raw), hashlib.sha256(raw).hexdigest())

    def _gbr(self, name: str, x: list[float]) -> float:
        m = self.spec["models"][name]
        s = m["init"]
        x32 = [float(np.float32(v)) for v in x]
        for t in m["trees"]:
            n = 0
            while t["l"][n] != -1:
                n = t["l"][n] if x32[t["f"][n]] <= t["t"][n] else t["r"][n]
            s += m["lr"] * t["v"][n]
        return s

    def features(self, offer: float, low: float | None, high: float | None, deal: float,
                 exchange: str, regime: dict) -> dict:
        has = low is not None and high is not None
        pvr = priced_vs_range({"price_low": low, "price_high": high, "offer_price": offer})
        return {
            "log_offer": float(np.log(offer)), "log_deal": float(np.log(max(deal, 1e6))),
            "priced_vs_range": PVR_CODE[pvr], "range_width": (high - low) / offer if has else 0.0,
            "has_range": 1.0 if has else 0.0, "major_exchange": 1.0 if exchange in MAJOR else 0.0,
            "regime_first_day_median": regime["median"], "regime_first_day_iqr": regime["iqr"],
            "regime_n": float(regime["n"]),
        }

    def price(self, offer: float, low: float | None, high: float | None, deal: float,
              exchange: str, regime: dict) -> dict:
        f = self.features(offer, low, high, deal, exchange, regime)
        x = [f[k] for k in self.spec["features"]]
        q = sorted([self._gbr("q_lo", x), self._gbr("q_med", x), self._gbr("q_hi", x)])
        s = self.spec["conformal_shift"]
        lo, med, hi = min(q[0] - s, q[1]), q[1], max(q[2] + s, q[1])
        cfg = GateConfig(**self.spec["gate"])
        out = {"offer": offer, "point": offer * np.exp(med), "lo": offer * np.exp(lo), "hi": offer * np.exp(hi)}
        out = {k: round(float(v), 4) for k, v in out.items()}
        out["move"] = round(out["point"] / offer - 1, 6)
        out["move_lo"] = round(out["lo"] / offer - 1, 6)
        out["move_hi"] = round(out["hi"] / offer - 1, 6)
        width = (out["hi"] - out["lo"]) / offer
        reason = ("no_range" if not f["has_range"] else "thin_regime" if f["regime_n"] < cfg.min_regime_n
                  else "too_wide" if width > cfg.max_rel_width else None)
        out.update({"published": reason is None, "reason": reason, "rel_width": round(width, 6),
                    "priced_vs_range": pvr_name(offer, low, high)})
        return out


def pvr_name(offer: float, low: float | None, high: float | None) -> str:
    return priced_vs_range({"price_low": low, "price_high": high, "offer_price": offer})


# ---------------------------------------------------------------- the filing (EDGAR)

_N = r"(\d{1,3}(?:,\d{3}){2,})"
_OFFERING_RE = re.compile(r"initial public offering of " + _N + r" shares", re.I)
_SELLING_RE = re.compile(r"(?:are|is) selling (?:an additional )?" + _N + r" shares", re.I)
_OFFER_RES = (
    re.compile(r"initial public offering price (?:per share )?(?:is|of) (?:US)?\$\s?(\d{1,4}(?:\.\d{1,2})?)", re.I),
    re.compile(r"[Pp]rice to (?:the )?[Pp]ublic\s*(?:per share)?\s*\$\s?(\d{1,4}(?:\.\d{1,2})?)"),
)


def parse_shares(text: str) -> int | None:
    """Shares offered, read only from the cover's offering sentence; None while the cover leaves it blank. Pure.

    'This is an initial public offering of N shares' wins; otherwise the issuer's and the selling
    stockholders' 'selling N shares' are added. A share count anywhere else in the prospectus
    (a grant, a fund contribution, shares outstanding) is never read."""
    i = text.lower().find("initial public offering of")
    if i < 0:
        return None
    window = text[i:i + 1500]
    m = _OFFERING_RE.search(window)
    if m:
        return int(m.group(1).replace(",", ""))
    parts = [int(x.replace(",", "")) for x in _SELLING_RE.findall(window)]
    return sum(parts) or None


def parse_offer(text: str) -> float | None:
    """The final offer price from a 424B prospectus cover. Pure."""
    for rx in _OFFER_RES:
        m = rx.search(text)
        if m:
            v = float(m.group(1))
            if 0 < v < 10_000:
                return v
    return None


def edgar_status(cik: str) -> dict:
    """The issuer's latest registration cover and any final prospectus, read from EDGAR."""
    from .filings import SEC_UA, _Http, html_to_text, parse_range
    http = _Http(pause=0.2)
    r = http.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json", SEC_UA)
    r.raise_for_status()
    rec = r.json()["filings"]["recent"]
    rows = [{"form": f, "date": d, "accession": a, "doc": doc}
            for f, d, a, doc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"], rec["primaryDocument"])]

    def url(row):
        return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{row['accession'].replace('-', '')}/{row['doc']}"

    out: dict = {"cik": str(int(cik)), "registration": None, "prospectus": None}
    reg = [x for x in rows if x["form"] in ("S-1", "S-1/A", "F-1", "F-1/A")]
    if reg:
        latest = max(reg, key=lambda x: (x["date"], x["accession"]))
        text = html_to_text(http.head_bytes(url(latest), SEC_UA))
        rng = parse_range(text)
        out["registration"] = {"form": latest["form"], "date": latest["date"], "accession": latest["accession"],
                               "url": url(latest), "low": rng[0] if rng else None, "high": rng[1] if rng else None,
                               "shares": parse_shares(text)}
    final = [x for x in rows if x["form"] in ("424B4", "424B1")]
    if final:
        latest = max(final, key=lambda x: (x["date"], x["accession"]))
        text = html_to_text(http.head_bytes(url(latest), SEC_UA))
        out["prospectus"] = {"form": latest["form"], "date": latest["date"], "accession": latest["accession"],
                             "url": url(latest), "offer": parse_offer(text), "shares": parse_shares(text)}
    return out


# ---------------------------------------------------------------- the regime

def nasdaq_priced(months: list[str]) -> list[dict]:
    """Priced deals from the public Nasdaq IPO calendar, mapped to the universe's row shape."""
    from .filings import NASDAQ_UA, _Http
    http = _Http(pause=0.5)
    out = []
    for mth in months:
        r = http.get(f"https://api.nasdaq.com/api/ipo/calendar?date={mth}", NASDAQ_UA, accept="application/json")
        rows = ((r.json().get("data") or {}).get("priced") or {}).get("rows") or []
        for row in rows:
            out.append({"symbol": (row.get("proposedTickerSymbol") or "").strip().upper(),
                        "name": (row.get("companyName") or "").strip(),
                        "date": pd.to_datetime(row.get("pricedDate"), errors="coerce"),
                        "exchange": (row.get("proposedExchange") or "").strip(),
                        "price": row.get("proposedSharePrice")})
    return out


def eligible(row: dict) -> bool:
    """The training cohort's rule: not a SPAC, not a unit or warrant, a single offer price. Pure."""
    sym, name = row.get("symbol") or "", row.get("name") or ""
    if not sym or SPAC_RE.search(name) or (UNIT_SUFFIX_RE.search(sym) and len(sym) >= 4):
        return False
    try:
        return float(str(row.get("price")).replace("$", "").replace(",", "")) > 0
    except ValueError:
        return False


def first_day_moves(rows: list[dict]) -> list[dict]:
    """First closes from yfinance for newly priced listings, labelled by ``labels.label_one``."""
    if not rows:
        return []
    import yfinance as yf
    from .labels import _download, label_one
    start = (min(r["date"] for r in rows) - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    closes, splits = _download(yf, [r["symbol"] for r in rows], start)
    found = []
    for r in rows:
        if r["symbol"] not in closes.columns:
            continue
        offer = float(str(r["price"]).replace("$", "").replace(",", ""))
        spl = splits[r["symbol"]] if r["symbol"] in splits.columns else None
        lab = label_one(closes[r["symbol"]], r["date"], offer, splits=spl)
        if lab["first_day"] is None:
            continue
        found.append({"symbol": r["symbol"], "name": r["name"], "ipo_date": str(r["date"].date()),
                      "first_close_date": str(pd.Timestamp(lab["first_close_date"]).date()), "offer": offer,
                      "first_close": round(lab["first_close"], 4), "first_day": round(lab["first_day"], 6),
                      "source": "nasdaq+yfinance"})
    return found


def regime_stats(listings: list[dict], decision: pd.Timestamp, today: pd.Timestamp) -> dict:
    """Trailing first-day statistics as ``features.regime`` computes them: first closes in
    [decision - 90d, decision), and never later than today. Pure."""
    lo = decision - pd.Timedelta(days=REGIME_WINDOW_DAYS)
    hi = min(decision, today + pd.Timedelta(days=1))
    w = sorted(x["first_day"] for x in listings if lo <= pd.Timestamp(x["first_close_date"]) < hi)
    if len(w) >= REGIME_MIN_N:
        return {"n": len(w), "median": float(np.median(w)),
                "iqr": float(np.percentile(w, 75) - np.percentile(w, 25)), "window": [str(lo.date()), str(decision.date())]}
    return {"n": len(w), "median": 0.0, "iqr": None, "window": [str(lo.date()), str(decision.date())]}


def refresh_listings(today: pd.Timestamp, decision: pd.Timestamp, path: Path = LISTINGS_JSON) -> tuple[list[dict], dict]:
    """Add newly priced, eligible listings with a first close to the committed listings file."""
    book = json.loads(path.read_text())
    known = {x["symbol"] for x in book["listings"]}
    frozen_through = pd.Timestamp(book["frozen_through"])
    months = sorted({d.strftime("%Y-%m") for d in pd.date_range(frozen_through, today, freq="D")})
    status = {"source": "nasdaq+yfinance", "ok": True, "added": [], "error": None}
    try:
        rows = [r for r in nasdaq_priced(months)
                if eligible(r) and pd.notna(r["date"]) and r["date"] > frozen_through and r["symbol"] not in known]
        new = first_day_moves(rows)
        if new:
            book["listings"] = sorted(book["listings"] + new, key=lambda x: (x["first_close_date"], x["symbol"]))
            path.write_text(json.dumps(book, indent=1) + "\n")
        status["added"] = [x["symbol"] for x in new]
        status["pending"] = sorted({r["symbol"] for r in rows} - set(status["added"]))
    except Exception as e:                       # a source outage must not stop the record; it is written down
        status.update(ok=False, error=f"{type(e).__name__}: {e}"[:300])
    return book["listings"], status


# ---------------------------------------------------------------- the call

def stage_of(filing: dict, first_trade: pd.Timestamp, today: pd.Timestamp, result: dict | None) -> str:
    if result:
        return "scored"
    p = filing.get("prospectus") or {}
    if p.get("offer") and today >= first_trade:
        return "trading"
    if p.get("offer"):
        return "priced"
    reg = filing.get("registration") or {}
    return "range_filed" if reg.get("low") else "filed_no_range"


def make_call(model: FrozenModel, deal: dict, filing: dict, regime: dict) -> dict:
    """The record's numbers: the actual call once priced, the range scenarios once filed, else placeholders."""
    reg, pros = filing.get("registration") or {}, filing.get("prospectus") or {}
    low, high = reg.get("low"), reg.get("high")
    real_range = low is not None
    if not real_range:
        low, high = deal["placeholder_range"]
    shares = pros.get("shares") or reg.get("shares")
    out: dict = {"range": [low, high], "range_source": "filed" if real_range else "placeholder"}

    def deal_usd(offer):
        if shares:
            return shares * offer, "filed shares x offer"
        return deal["assumed_deal_usd"], deal["assumed_deal_source"]

    scen = []
    for key, label, at in SCENARIOS:
        o = at(low, high)
        d, src = deal_usd(o)
        p = model.price(o, low, high, d, deal["exchange"], regime)
        scen.append({"key": key, "label": label, "deal_usd": d, "deal_source": src, **p})
    o = (low + high) / 2
    d, src = deal_usd(o)
    nr = model.price(high, None, None, d, deal["exchange"], regime)
    scen.append({"key": "no_range", "label": "No range filed", "deal_usd": d, "deal_source": src, **nr})
    out["scenarios"] = scen
    if pros.get("offer"):
        d, src = deal_usd(pros["offer"])
        out["call"] = {"deal_usd": d, "deal_source": src,
                       **model.price(pros["offer"], reg.get("low"), reg.get("high"), d, deal["exchange"], regime)}
    else:
        out["call"] = None
    return out


def first_close(symbol: str, first_trade: pd.Timestamp) -> dict | None:
    import yfinance as yf
    from .labels import _download
    closes, _ = _download(yf, [symbol], (first_trade - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
    if symbol not in closes.columns:
        return None
    s = closes[symbol].dropna()
    s = s[pd.DatetimeIndex(s.index) >= first_trade.normalize()]
    if s.empty:
        return None
    return {"date": str(pd.Timestamp(s.index[0]).date()), "close": round(float(s.iloc[0]), 4)}


def score(call: dict, close: float) -> dict:
    """Hit or miss against the band published before the first trade. Pure."""
    return {"first_close": close, "first_day": round(close / call["offer"] - 1, 6),
            "inside_band": bool(call["lo"] <= close <= call["hi"]),
            "error_vs_median": round(close / call["point"] - 1, 6)}


# ---------------------------------------------------------------- the record

def _git_sha() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def read_tape(path: Path | None = None) -> list[dict]:
    path = path or TAPE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


_VOLATILE = ("run_utc", "code_commit", "seq")


def append_if_changed(rec: dict, path: Path | None = None) -> bool:
    """Append ``rec`` unless it says what the last record already says. Never rewrites a line."""
    path = path or TAPE
    tape = read_tape(path)
    if tape:
        last = {k: v for k, v in tape[-1].items() if k not in _VOLATILE}
        cur = {k: v for k, v in rec.items() if k not in _VOLATILE}
        if last == cur:
            return False
    rec["seq"] = len(tape) + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(rec, separators=(",", ":"), sort_keys=True) + "\n")
    return True


def headline(rec: dict) -> str:
    """One line for the commit message and the page."""
    st, sym = rec["stage"], rec["deal"]["symbol"]
    if st == "scored":
        r = rec["result"]
        return f"{sym} first close ${r['first_close']:.2f} ({r['first_day']:+.1%}): {'inside' if r['inside_band'] else 'outside'} the band"
    if rec.get("call"):
        c = rec["call"]
        return f"{sym} priced at ${c['offer']:.2f}: median {c['move']:+.1%}, 80% band {c['move_lo']:+.1%} to {c['move_hi']:+.1%}"
    top = next(s for s in rec["scenarios"] if s["key"] == "top")
    rng = rec["range"]
    if st == "range_filed":
        return f"{sym} range ${rng[0]:.2f}-${rng[1]:.2f} filed: at the top, 80% band {top['move_lo']:+.1%} to {top['move_hi']:+.1%}"
    return f"{sym} no range filed; regime {rec['regime']['n']} listings, median {rec['regime']['median']:+.1%}"


def track(deal_path: Path, today: pd.Timestamp | None = None, offline: bool = False) -> tuple[dict, bool]:
    deal = json.loads(deal_path.read_text())
    model = FrozenModel.load()
    now = datetime.now(timezone.utc)
    today = (today or pd.Timestamp(now.date())).normalize()
    first_trade = pd.Timestamp(deal["expected_first_trade"])

    tape = read_tape()
    filing = {"registration": None, "prospectus": None, "error": None}
    if not offline:
        try:
            filing = {**edgar_status(deal["cik"]), "error": None}
        except Exception as e:
            # an outage must never move the record backwards: keep the last filing read, and say so
            last = next((r["filing"] for r in reversed(tape) if (r.get("filing") or {}).get("registration")), None)
            filing = {**(last or filing), "error": f"{type(e).__name__}: {e}"[:300], "carried_forward": bool(last)}
    pros = filing.get("prospectus") or {}
    if pros.get("date"):                       # the final prospectus is filed the morning the shares first trade
        first_trade = pd.Timestamp(pros["date"])
    decision = first_trade - pd.Timedelta(days=1)

    if offline:
        listings, rstatus = json.loads(LISTINGS_JSON.read_text())["listings"], {"source": "offline", "ok": True, "added": []}
    else:
        listings, rstatus = refresh_listings(min(today, decision), decision)
    regime = regime_stats(listings, decision, today)
    if regime["iqr"] is None:
        regime["iqr"] = model.spec["regime"]["regime_first_day_iqr"]
    regime["status"] = rstatus

    numbers = make_call(model, deal, filing, regime)
    result = None
    published_call = next((r["call"] for r in reversed(tape) if r.get("call") and r["as_of"] < str(first_trade.date())), None)
    if published_call and today >= first_trade and not offline:
        fc = first_close(deal["symbol"], first_trade)
        if fc:
            result = {"first_close_date": fc["date"], **score(published_call, fc["close"]),
                      "band_published_as_of": next(r["as_of"] for r in reversed(tape) if r.get("call") == published_call)}

    rec = {
        "as_of": str(today.date()), "run_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stage": stage_of(filing, first_trade, today, result),
        "deal": {k: deal[k] for k in ("symbol", "company", "exchange", "cik")},
        "first_trade": str(first_trade.date()), "first_trade_source": "424B prospectus" if pros.get("date") else deal["first_trade_source"],
        "filing": filing, "regime": regime, **numbers, "result": result,
        "model_sha256": model.sha256, "code_commit": _git_sha(),
    }
    rec["headline"] = headline(rec)
    changed = append_if_changed(rec)
    write_live(deal)
    return rec, changed


def _note(r: dict) -> str | None:
    """What a reader needs to know about a record beyond its numbers. Pure."""
    notes = []
    f = r.get("filing") or {}
    if f.get("error"):
        notes.append("EDGAR read failed; last filing kept" if f.get("carried_forward") else "EDGAR read failed")
    st = (r.get("regime") or {}).get("status") or {}
    if st.get("ok") is False:
        notes.append("IPO calendar unavailable; regime kept")
    if st.get("added"):
        notes.append("new listings: " + ", ".join(st["added"]))
    return "; ".join(notes) or None


def write_live(deal: dict, path: Path | None = None) -> None:
    path = path or LIVE_JSON
    tape = read_tape()
    hist = [{"seq": r["seq"], "as_of": r["as_of"], "run_utc": r["run_utc"], "stage": r["stage"], "headline": r["headline"],
             "range": r["range"], "range_source": r["range_source"], "regime_n": r["regime"]["n"],
             "regime_median": r["regime"]["median"],
             "top": next(({k: s[k] for k in ("move", "move_lo", "move_hi", "published", "reason")}
                          for s in r["scenarios"] if s["key"] == "top"), None),
             "call": r.get("call"), "result": r.get("result"), "note": _note(r)} for r in tape]
    path.write_text(json.dumps({"deal": deal, "latest": tape[-1] if tape else None, "history": hist},
                               separators=(",", ":")) + "\n")
