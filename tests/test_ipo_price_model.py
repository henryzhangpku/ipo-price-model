"""The claims in METHODOLOGY.md, as tests. No network, no keys. Synthetic data only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ipo_price_model import gate as gate_mod
from ipo_price_model.features import FEATURES, build, priced_vs_range, regime
from ipo_price_model.labels import label_one
from ipo_price_model.leakage import LeakageError, assert_chronological_splits, assert_features_known_at, assert_label_after_decision
from ipo_price_model.model import ConformalQuantileModel, coverage, mean_width
from ipo_price_model.universe import normalise, parse_price


# ------------------------------------------------------------------ synthetic world

def synthetic(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A universe + labels where the first-day move depends on priced-vs-range and a regime."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n * 2)[::2][:n]
    offer = rng.uniform(8, 40, n).round(2)
    lo = (offer * rng.uniform(0.85, 0.98, n)).round(2)
    hi = (lo * rng.uniform(1.05, 1.20, n)).round(2)
    pvr = rng.choice(["below", "within", "top", "above"], n, p=[0.15, 0.4, 0.2, 0.25])
    offer = np.where(pvr == "below", lo * 0.9, np.where(pvr == "within", (lo + hi) / 2, np.where(pvr == "top", hi, hi * 1.1))).round(2)
    regime_cycle = 0.10 * np.sin(np.arange(n) / 60)
    effect = {"below": -0.08, "within": 0.02, "top": 0.10, "above": 0.25}
    y = np.array([effect[p] for p in pvr]) + regime_cycle + rng.normal(0, 0.12, n)
    first_close = (offer * np.exp(y)).round(2)
    uni = pd.DataFrame({
        "symbol": [f"S{i:04d}" for i in range(n)], "name": [f"Co {i}" for i in range(n)], "ipo_date": dates,
        "exchange": rng.choice(["NASDAQ", "NYSE", "OTC"], n, p=[0.6, 0.35, 0.05]), "status": "priced", "price_raw": "",
        "price_low": lo, "price_high": hi, "offer_price": offer, "has_range": True,
        "shares": rng.uniform(2e6, 5e7, n), "deal_usd": rng.lognormal(18.5, 1.0, n),
        "is_spac": False, "is_unit": False,
    })
    lab = pd.DataFrame({"symbol": uni["symbol"], "ipo_date": dates, "first_close_date": dates,
                        "first_close": first_close, "first_day": first_close / offer - 1, "fwd_30": np.nan})
    return uni, lab


# ------------------------------------------------------------------ universe

def test_parse_price():
    assert parse_price("17.00-19.00") == (17.0, 19.0, None)
    assert parse_price("$18.50") == (None, None, 18.5)
    assert parse_price(None) == (None, None, None)


def test_normalise_tags_and_derives():
    df = normalise([
        {"symbol": "XYZ", "name": "XYZ Acquisition Corp", "date": "2021-05-01", "exchange": "NASDAQ",
         "status": "priced", "price": "10.00", "numberOfShares": 20000000},
        {"symbol": "ABCDU", "name": "Abcd Units", "date": "2021-05-02", "exchange": "NYSE",
         "status": "priced", "price": "10.00-12.00", "numberOfShares": 1000, "totalSharesValue": 11000},
    ])
    assert df.loc[df.symbol == "XYZ", "is_spac"].item() and df.loc[df.symbol == "XYZ", "deal_usd"].item() == 200e6
    assert df.loc[df.symbol == "ABCDU", "is_unit"].item() and df.loc[df.symbol == "ABCDU", "has_range"].item()


# ------------------------------------------------------------------ labels and leakage

def test_label_one_takes_the_first_close_on_or_after_listing():
    idx = pd.bdate_range("2024-03-01", periods=40)
    s = pd.Series(np.linspace(20, 30, 40), index=idx)
    lab = label_one(s, pd.Timestamp("2024-03-14"), 17.0)
    assert lab["first_close_date"] >= pd.Timestamp("2024-03-14")
    assert lab["first_day"] == pytest.approx(lab["first_close"] / 17.0 - 1)


def test_leakage_assertions_raise():
    with pytest.raises(LeakageError):
        assert_label_after_decision(pd.Timestamp("2024-03-14"), [pd.Timestamp("2024-03-13")])
    with pytest.raises(LeakageError):
        assert_chronological_splits(pd.Timestamp("2023-01-01"), pd.Timestamp("2022-06-01"), pd.Timestamp("2024-01-01"))
    with pytest.raises(LeakageError):
        assert_features_known_at(pd.DataFrame({"first_close": [1.0]}), set(FEATURES))


# ------------------------------------------------------------------ features

def test_priced_vs_range_codes():
    r = pd.Series({"price_low": 15.0, "price_high": 17.0, "offer_price": 18.0})
    assert priced_vs_range(r) == "above"
    assert priced_vs_range(pd.Series({"price_low": None, "price_high": None, "offer_price": 18.0})) == "unknown"


def test_regime_uses_only_strictly_earlier_listings():
    lab = pd.DataFrame({"first_close_date": pd.to_datetime(["2023-12-29", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]),
                        "first_day": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]})
    # decision date 2024-01-08: listings on the 8th and 9th must be excluded; five remain
    reg = regime(lab, pd.Series([pd.Timestamp("2024-01-08")]), window_days=30)
    assert reg["regime_n"].item() == 5
    assert reg["regime_first_day_median"].item() == pytest.approx(0.2)


def test_regime_is_nan_below_the_floor():
    lab = pd.DataFrame({"first_close_date": pd.to_datetime(["2024-01-02", "2024-01-03"]), "first_day": [0.1, 0.2]})
    reg = regime(lab, pd.Series([pd.Timestamp("2024-02-01")]))
    assert reg["regime_n"].item() == 2 and np.isnan(reg["regime_first_day_median"].item())


def test_build_declares_exactly_the_feature_columns():
    uni, lab = synthetic(120)
    X = build(uni, lab)
    assert list(X.columns) == FEATURES
    assert len(X) == 120 and not X.isna().any().any()


# ------------------------------------------------------------------ model

def test_conformal_coverage_holds_on_a_held_out_window():
    uni, lab = synthetic(900, seed=3)
    X = build(uni, lab)
    y = np.log(lab["first_close"].to_numpy() / uni["offer_price"].to_numpy())
    tr, ca, te = slice(0, 500), slice(500, 700), slice(700, 900)
    m = ConformalQuantileModel(alpha=0.2, n_estimators=100).fit(X.iloc[tr], y[tr], X.iloc[ca], y[ca])
    q = m.predict_log(X.iloc[te])
    cov = coverage(q["q_lo"].to_numpy(), q["q_hi"].to_numpy(), y[te])
    assert cov >= 0.72                                   # nominal 0.80, finite-sample slack
    ref = coverage(q["ref_lo"].to_numpy(), q["ref_hi"].to_numpy(), y[te])
    assert mean_width(q["q_lo"].to_numpy(), q["q_hi"].to_numpy()) < mean_width(q["ref_lo"].to_numpy(), q["ref_hi"].to_numpy()) or cov > ref


def test_prices_map_back_multiplicatively():
    uni, lab = synthetic(300)
    X = build(uni, lab)
    y = np.log(lab["first_close"].to_numpy() / uni["offer_price"].to_numpy())
    m = ConformalQuantileModel(alpha=0.2, n_estimators=50).fit(X.iloc[:200], y[:200], X.iloc[200:260], y[200:260])
    p = m.predict_price(X.iloc[260:], uni["offer_price"].iloc[260:])
    assert (p["lo"] <= p["hi"]).all() and (p["lo"] > 0).all()
    assert (p["ref_point"].to_numpy() == uni["offer_price"].iloc[260:].to_numpy()).all()


# ------------------------------------------------------------------ gate

def test_gate_reasons_in_declared_order():
    feats = pd.DataFrame({"has_range": [0.0, 1.0, 1.0, 1.0], "regime_n": [10, 2, 10, 10]})
    prices = pd.DataFrame({"lo": [8.0, 8.0, 5.0, 9.0], "hi": [12.0, 12.0, 15.0, 11.0]})
    offer = pd.Series([10.0, 10.0, 10.0, 10.0])
    g = gate_mod.apply(feats, prices, offer, gate_mod.GateConfig(min_regime_n=5, max_rel_width=0.6))
    assert list(g["reason"]) == ["no_range", "thin_regime", "too_wide", None]
    assert list(g["published"]) == [False, False, False, True]
    s = gate_mod.summary(g)
    assert s["published"] == 1 and s["withheld"] == {"no_range": 1, "thin_regime": 1, "too_wide": 1}


def test_gate_never_changes_the_interval():
    feats = pd.DataFrame({"has_range": [1.0], "regime_n": [10]})
    prices = pd.DataFrame({"lo": [5.0], "hi": [15.0]})
    before = prices.copy()
    gate_mod.apply(feats, prices, pd.Series([10.0]))
    pd.testing.assert_frame_equal(prices, before)
