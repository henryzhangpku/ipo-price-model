"""evaluate.py — fit chronologically, calibrate, test, gate, and write the numbers down.

Splits are by listing date and disjoint: train < calibration < test. Coverage is
reported on the test window for the fitted model and for the reference, both
before and after the gate, overall and by year and by deal-size bucket. The
gate's refusals are counted by reason. A JSON lands in results/ either way.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import gate as gate_mod
from .features import build
from .labels import load_labels
from .leakage import assert_chronological_splits
from .model import ConformalQuantileModel, coverage, mean_width
from .universe import core, load_universe

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def deal_bucket(deal: float) -> str:
    if np.isnan(deal):
        return "unknown"
    if deal < 50e6:
        return "<50M"
    if deal < 100e6:
        return "50-100M"
    if deal < 500e6:
        return "100-500M"
    return ">=500M"


def _cov_block(lo, hi, y, offer) -> dict:
    return {"coverage": coverage(lo, hi, y), "mean_rel_width": float(np.nanmean((hi - lo) / offer)), "n": int(np.sum(~np.isnan(y)))}


def run(train_end: str = "2022-12-31", calib_end: str = "2023-12-31", test_start: str = "2024-01-01",
        alpha: float = 0.20, gate_cfg: gate_mod.GateConfig = gate_mod.GateConfig(), seed: int = 7) -> dict:
    assert_chronological_splits(pd.Timestamp(train_end), pd.Timestamp(calib_end), pd.Timestamp(test_start))
    uni = core(load_universe())
    lab = load_labels()
    df = uni.merge(lab.drop(columns=["ipo_date"]), on="symbol", how="left")
    df = df[df["first_close"].notna()].reset_index(drop=True)
    X = build(df, lab)
    y = np.log(df["first_close"].to_numpy(dtype=float) / df["offer_price"].to_numpy(dtype=float))
    d = pd.to_datetime(df["ipo_date"])
    tr = d <= pd.Timestamp(train_end)
    ca = (d > pd.Timestamp(train_end)) & (d <= pd.Timestamp(calib_end))
    te = d >= pd.Timestamp(test_start)
    if tr.sum() < 100 or ca.sum() < 50 or te.sum() < 50:
        raise SystemExit(f"windows too thin: train {int(tr.sum())}, calib {int(ca.sum())}, test {int(te.sum())}")

    model = ConformalQuantileModel(alpha=alpha, random_state=seed).fit(X[tr], y[tr], X[ca], y[ca])
    Xt, yt, off = X[te], y[te], df.loc[te, "offer_price"]
    prices = model.predict_price(Xt, off)
    gated = gate_mod.apply(Xt, prices, off, gate_cfg)
    o = off.to_numpy(dtype=float)
    fc = df.loc[te, "first_close"].to_numpy(dtype=float)
    pub = gated["published"].to_numpy()

    res = {
        "windows": {"train_end": train_end, "calib_end": calib_end, "test_start": test_start,
                    "n_train": int(tr.sum()), "n_calib": int(ca.sum()), "n_test": int(te.sum())},
        "alpha": alpha, "nominal_coverage": 1 - alpha,
        "conformal_shift_log": model.conformal_shift_,
        "model": {"all": _cov_block(prices["lo"].to_numpy(), prices["hi"].to_numpy(), fc, o),
                  "published": _cov_block(prices["lo"].to_numpy()[pub], prices["hi"].to_numpy()[pub], fc[pub], o[pub]) if pub.any() else None,
                  "withheld": _cov_block(prices["lo"].to_numpy()[~pub], prices["hi"].to_numpy()[~pub], fc[~pub], o[~pub]) if (~pub).any() else None},
        "reference": {"all": _cov_block(prices["ref_lo"].to_numpy(), prices["ref_hi"].to_numpy(), fc, o),
                      "published": _cov_block(prices["ref_lo"].to_numpy()[pub], prices["ref_hi"].to_numpy()[pub], fc[pub], o[pub]) if pub.any() else None},
        "point_mae_rel": {"model": float(np.mean(np.abs(prices["point"].to_numpy() - fc) / o)),
                          "reference": float(np.mean(np.abs(o - fc) / o))},
        "gate": gate_mod.summary(gated),
        "by_year": {}, "by_deal_bucket": {},
    }
    years = d[te].dt.year.to_numpy()
    for yr in sorted(set(years)):
        m = years == yr
        res["by_year"][int(yr)] = {"n": int(m.sum()),
                                   "model": _cov_block(prices["lo"].to_numpy()[m], prices["hi"].to_numpy()[m], fc[m], o[m]),
                                   "reference": _cov_block(prices["ref_lo"].to_numpy()[m], prices["ref_hi"].to_numpy()[m], fc[m], o[m]),
                                   "published_share": float(pub[m].mean())}
    buckets = np.array([deal_bucket(v) for v in df.loc[te, "deal_usd"].to_numpy(dtype=float)])
    for b in ["<50M", "50-100M", "100-500M", ">=500M", "unknown"]:
        m = buckets == b
        if m.any():
            res["by_deal_bucket"][b] = {"n": int(m.sum()),
                                       "model": _cov_block(prices["lo"].to_numpy()[m], prices["hi"].to_numpy()[m], fc[m], o[m]),
                                       "published_share": float(pub[m].mean())}
    mw, rw = res["model"]["all"]["mean_rel_width"], res["reference"]["all"]["mean_rel_width"]
    mc, rc = res["model"]["all"]["coverage"], res["reference"]["all"]["coverage"]
    res["verdict"] = {"model_beats_reference": bool(mw < rw and mc >= (1 - alpha) - 0.03),
                      "reason": f"width {mw:.3f} vs {rw:.3f}, coverage {mc:.3f} vs {rc:.3f} at nominal {1 - alpha:.2f}"}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    res["_path"] = str(out)
    return res
