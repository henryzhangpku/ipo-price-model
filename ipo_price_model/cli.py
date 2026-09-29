"""cli.py — `ipo-price-model universe | labels | evaluate | price`."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd


def _universe(a) -> None:
    from .universe import UNIVERSE_PARQUET, build_universe, core
    df = build_universe(a.date_from, a.date_to, overwrite=a.overwrite)
    c = core(df)
    print(f"universe: {len(df)} priced listings {a.date_from}..{a.date_to} -> {UNIVERSE_PARQUET}")
    print(f"  core: {len(c)}   spac: {int(df['is_spac'].sum())}   unit/warrant: {int(df['is_unit'].sum())}"
          f"   range kept: {int(df['has_range'].sum())}")


def _labels(a) -> None:
    from .labels import LABELS_PARQUET, build_labels
    df = build_labels(overwrite=a.overwrite)
    print(f"labels: {len(df)} listings, {int(df['first_close'].notna().sum())} with a first close -> {LABELS_PARQUET}")


def _evaluate(a) -> None:
    from .evaluate import run
    res = run(a.train_end, a.calib_end, a.test_start, alpha=a.alpha)
    path = res.pop("_path")
    m, r, g = res["model"]["all"], res["reference"]["all"], res["gate"]
    print(f"test window from {res['windows']['test_start']}: {res['windows']['n_test']} listings, nominal coverage {res['nominal_coverage']:.0%}")
    print(f"  model      coverage {m['coverage']:.3f}   mean width {m['mean_rel_width']:.3f} x offer")
    print(f"  reference  coverage {r['coverage']:.3f}   mean width {r['mean_rel_width']:.3f} x offer")
    if res["model"]["published"]:
        p = res["model"]["published"]
        print(f"  published  coverage {p['coverage']:.3f}   mean width {p['mean_rel_width']:.3f}   n {p['n']}")
    print(f"  gate: published {g['published']}/{g['n']}   withheld {g['withheld']}")
    print(f"  verdict: {'model beats reference' if res['verdict']['model_beats_reference'] else 'reference stands'} ({res['verdict']['reason']})")
    print(f"-> {path}")


def _price(a) -> None:
    """Price one hypothetical listing from T-1 facts, using a model fitted on the frozen data."""
    from . import gate as gate_mod
    from .evaluate import build, core, load_labels, load_universe
    from .model import ConformalQuantileModel
    uni = core(load_universe())
    lab = load_labels()
    df = uni.merge(lab.drop(columns=["ipo_date"]), on="symbol", how="left")
    df = df[df["first_close"].notna()].reset_index(drop=True)
    X = build(df, lab)
    y = np.log(df["first_close"].to_numpy(dtype=float) / df["offer_price"].to_numpy(dtype=float))
    d = pd.to_datetime(df["ipo_date"])
    tr, ca = d <= pd.Timestamp(a.train_end), d > pd.Timestamp(a.train_end)
    model = ConformalQuantileModel(alpha=a.alpha).fit(X[tr], y[tr], X[ca], y[ca])
    row = pd.DataFrame([{"symbol": "NEW", "name": "", "ipo_date": pd.Timestamp(a.date), "exchange": a.exchange,
                         "status": "priced", "price_raw": "", "price_low": a.low, "price_high": a.high,
                         "offer_price": a.offer, "has_range": a.low is not None and a.high is not None,
                         "shares": None, "deal_usd": a.deal, "is_spac": False, "is_unit": False}])
    Xn = build(row, lab)
    prices = model.predict_price(Xn, row["offer_price"])
    g = gate_mod.apply(Xn, prices, row["offer_price"])
    out = {"offer": a.offer, "point": float(prices["point"].iloc[0]), "lo": float(prices["lo"].iloc[0]),
           "hi": float(prices["hi"].iloc[0]), "nominal_coverage": 1 - a.alpha,
           "published": bool(g["published"].iloc[0]), "reason": g["reason"].iloc[0],
           "regime_n": int(Xn["regime_n"].iloc[0])}
    print(json.dumps(out, indent=2))
    if not out["published"]:
        print(f"withheld: {out['reason']} — the interval above is shown for audit, not for use")


def main() -> None:
    ap = argparse.ArgumentParser(prog="ipo-price-model", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("universe", help="freeze the IPO calendar (FINNHUB_API_KEY)")
    u.add_argument("--from", dest="date_from", default="2019-01-01")
    u.add_argument("--to", dest="date_to", default="2025-12-31")
    u.add_argument("--overwrite", action="store_true")
    u.set_defaults(fn=_universe)
    l = sub.add_parser("labels", help="freeze first closes and post-listing returns")
    l.add_argument("--overwrite", action="store_true")
    l.set_defaults(fn=_labels)
    e = sub.add_parser("evaluate", help="chronological fit, calibrate, test, gate")
    e.add_argument("--train-end", default="2022-12-31")
    e.add_argument("--calib-end", default="2023-12-31")
    e.add_argument("--test-start", default="2024-01-01")
    e.add_argument("--alpha", type=float, default=0.20)
    e.set_defaults(fn=_evaluate)
    p = sub.add_parser("price", help="price one listing from T-1 facts")
    p.add_argument("--date", required=True, help="first-trade date, YYYY-MM-DD")
    p.add_argument("--offer", type=float, required=True)
    p.add_argument("--low", type=float, default=None, help="filed range low")
    p.add_argument("--high", type=float, default=None, help="filed range high")
    p.add_argument("--deal", type=float, required=True, help="deal size in USD")
    p.add_argument("--exchange", default="NASDAQ")
    p.add_argument("--train-end", default="2023-12-31")
    p.add_argument("--alpha", type=float, default=0.20)
    p.set_defaults(fn=_price)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
