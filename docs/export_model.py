"""export_model.py — freeze the fitted model into docs/model.json for the browser demo.

Fits exactly what `ipo-price-model price` fits (train <= train_end, calibrate after),
then writes every tree of the three quantile regressors, the conformal shift, the
reference band, the gate thresholds and the regime as of the demo's decision date.
The page evaluates those trees in JavaScript, so a price on the page is the same
number the CLI prints. Test vectors are written alongside so the page can check it.

    python docs/export_model.py --date 2026-11-16
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from ipo_price_model import gate as gate_mod
from ipo_price_model.evaluate import build, core, load_labels, load_universe
from ipo_price_model.features import FEATURES, MAJOR
from ipo_price_model.model import ConformalQuantileModel

OUT = Path(__file__).resolve().parent / "model.json"


def _trees(m) -> dict:
    X0 = pd.DataFrame([np.zeros(len(FEATURES))], columns=FEATURES)
    trees = []
    for est in m.estimators_[:, 0]:
        t = est.tree_
        trees.append({"f": t.feature.tolist(), "t": [round(float(v), 9) for v in t.threshold],
                      "l": t.children_left.tolist(), "r": t.children_right.tolist(),
                      "v": [round(float(v), 9) for v in t.value[:, 0, 0]]})
    return {"init": float(m._raw_predict_init(X0)[0, 0]), "lr": float(m.learning_rate), "trees": trees}


def _row(date, offer, low, high, deal, exchange):
    return {"symbol": "NEW", "name": "", "ipo_date": pd.Timestamp(date), "exchange": exchange, "status": "priced",
            "price_raw": "", "price_low": low, "price_high": high, "offer_price": offer,
            "has_range": low is not None and high is not None, "shares": None, "deal_usd": deal,
            "is_spac": False, "is_unit": False}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-16")
    ap.add_argument("--train-end", default="2023-12-31")
    ap.add_argument("--alpha", type=float, default=0.20)
    a = ap.parse_args()

    uni = core(load_universe())
    lab = load_labels()
    df = uni.merge(lab.drop(columns=["ipo_date"]), on="symbol", how="left")
    df = df[df["first_close"].notna()].reset_index(drop=True)
    X = build(df, lab)
    y = np.log(df["first_close"].to_numpy(dtype=float) / df["offer_price"].to_numpy(dtype=float))
    d = pd.to_datetime(df["ipo_date"])
    tr, ca = d <= pd.Timestamp(a.train_end), d > pd.Timestamp(a.train_end)
    model = ConformalQuantileModel(alpha=a.alpha).fit(X[tr], y[tr], X[ca], y[ca])

    hist = df[[c for c in _row(a.date, 1, None, None, 1, "").keys() if c in df.columns]]

    def features_for(rows: list[dict]) -> pd.DataFrame:
        r = pd.DataFrame(rows)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            out = build(pd.concat([hist, r], ignore_index=True),
                        pd.concat([lab, pd.DataFrame([{"symbol": "NEW"}] * len(r))], ignore_index=True))
        return out.iloc[-len(r):].reset_index(drop=True), r

    # regime as of the decision date (identical for every scenario on that date)
    Xr, _ = features_for([_row(a.date, 40, 36, 40, 60e9, "NASDAQ Global Select")])
    regime = {k: float(Xr[k].iloc[0]) for k in ("regime_first_day_median", "regime_first_day_iqr", "regime_n")}

    # test vectors: python's answer for a spread of scenarios, checked by the page on load
    scen = [_row(a.date, o, lo, hi, dl, ex) for o, lo, hi, dl, ex in [
        (44, 40, 44, 60e9, "NASDAQ Global Select"), (48, 40, 44, 60e9, "NASDAQ Global Select"),
        (38, 40, 44, 100e9, "NASDAQ Global Select"), (42, 40, 44, 20e9, "NASDAQ Global Select"),
        (44, None, None, 60e9, "NASDAQ Global Select"), (21, 18, 20, 450e6, "NASDAQ"),
        (4, 4, 5, 8e6, "NASDAQ Capital"), (150, 130, 150, 5e9, "NYSE")]]
    Xs, rs = features_for(scen)
    ps = model.predict_price(Xs, rs["offer_price"])
    gs = gate_mod.apply(Xs, ps, rs["offer_price"])
    vectors = [{"input": {"offer": s["offer_price"], "low": s["price_low"], "high": s["price_high"],
                          "deal": s["deal_usd"], "exchange": s["exchange"]},
                "features": {k: float(Xs[k].iloc[i]) for k in FEATURES},
                "point": float(ps["point"].iloc[i]), "lo": float(ps["lo"].iloc[i]), "hi": float(ps["hi"].iloc[i]),
                "published": bool(gs["published"].iloc[i]), "reason": gs["reason"].iloc[i]}
               for i, s in enumerate(scen)]

    # the largest listings the model has ever seen: the only evidence near a deal this size
    big = df.assign(first_day=np.exp(y) - 1).sort_values("deal_usd", ascending=False).head(15)
    largest = [{"symbol": r.symbol, "name": str(r.name)[:40], "date": str(pd.Timestamp(r.ipo_date).date()),
                "deal": float(r.deal_usd), "offer": float(r.offer_price), "first_close": float(r.first_close),
                "first_day": float(r.first_day)} for r in big.itertuples()]

    cfg = gate_mod.GateConfig()
    out = {
        "decision_date": a.date, "features": FEATURES, "major_exchanges": sorted(MAJOR),
        "alpha": a.alpha, "conformal_shift": model.conformal_shift_,
        "ref_lo": model.ref_lo_, "ref_hi": model.ref_hi_,
        "n_train": model.n_train_, "n_calib": model.n_calib_, "train_end": a.train_end,
        "data_through": str(pd.to_datetime(df["ipo_date"]).max().date()),
        "train_max_deal": float(df.loc[tr, "deal_usd"].max()), "all_max_deal": float(df["deal_usd"].max()),
        "regime": regime, "gate": {"min_regime_n": cfg.min_regime_n, "max_rel_width": cfg.max_rel_width},
        "models": {"q_lo": _trees(model.q_lo_), "q_med": _trees(model.q_med_), "q_hi": _trees(model.q_hi_)},
        "vectors": vectors, "largest": largest,
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB); regime {regime}; shift {model.conformal_shift_:.4f}")
    for v in vectors:
        print(v["input"], round(v["point"], 2), round(v["lo"], 2), round(v["hi"], 2), v["reason"])


if __name__ == "__main__":
    main()
