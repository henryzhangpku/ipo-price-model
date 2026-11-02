"""The tracker's claims, executable: the frozen model is the published one, and the record only grows."""
import json

import pandas as pd
import pytest

from ipo_price_model import tracker as T

MODEL = T.FrozenModel.load()


def _regime(feats):
    return {"n": feats["regime_n"], "median": feats["regime_first_day_median"], "iqr": feats["regime_first_day_iqr"]}


def test_frozen_model_reproduces_the_fitted_model():
    for v in MODEL.spec["vectors"]:
        i = v["input"]
        p = MODEL.price(i["offer"], i["low"], i["high"], i["deal"], i["exchange"], _regime(v["features"]))
        for k in ("point", "lo", "hi"):
            assert p[k] == pytest.approx(v[k], rel=1e-6, abs=1e-3)
        assert p["reason"] == v["reason"]


def test_frozen_model_builds_the_same_features():
    v = MODEL.spec["vectors"][0]
    i = v["input"]
    f = MODEL.features(i["offer"], i["low"], i["high"], i["deal"], i["exchange"], _regime(v["features"]))
    assert f == pytest.approx(v["features"])


def test_parse_shares_reads_a_cover_and_not_a_blank_one():
    assert T.parse_shares("This is an initial public offering of 1,250,000,000 shares of Class A common stock") == 1_250_000_000
    assert T.parse_shares("This is an initial public offering of shares of Class A common stock. We are selling "
                          "1,000,000,000 shares and the selling stockholders are selling an additional "
                          "250,000,000 shares of Class A common stock.") == 1_250_000_000
    blank = ("This is an initial public offering of shares of Class A common stock of Anthropic, PBC. We are selling "
             "shares of Class A common stock and the selling stockholders are selling an additional shares. ")
    assert T.parse_shares(blank) is None
    # a share count elsewhere in the prospectus (here: a fund contribution) is never the offering
    assert T.parse_shares(blank + "x" * 2000 + " we contributed 846,840,960 shares of our Class N common stock") is None


def test_parse_offer_reads_the_final_prospectus():
    assert T.parse_offer("The initial public offering price is $44.00 per share.") == 44.0
    assert T.parse_offer("Price to Public $ 51.00 $ 1,000,000") == 51.0
    assert T.parse_offer("will be between $ and $ .") is None


def test_eligible_keeps_the_training_cohort_rule():
    assert T.eligible({"symbol": "OPAY", "name": "OPAY Ltd", "price": "19.00"})
    assert not T.eligible({"symbol": "BHAIU", "name": "BHAV Acquisition Corp II", "price": "10.00"})
    assert not T.eligible({"symbol": "ABCDW", "name": "Some Co", "price": "10.00"})
    assert not T.eligible({"symbol": "GBC", "name": "Georgia Banking Co", "price": None})


def test_regime_counts_only_closes_before_the_decision_and_known_today():
    rows = [{"first_close_date": f"2026-10-{d:02d}", "first_day": 0.1 * d} for d in range(1, 21)]
    decision = pd.Timestamp("2026-10-15")
    r = T.regime_stats(rows, decision, today=pd.Timestamp("2026-10-30"))
    assert r["n"] == 14                              # the 1st..14th, never the 15th or later
    r = T.regime_stats(rows, decision, today=pd.Timestamp("2026-10-05"))
    assert r["n"] == 5                               # nothing after today is visible yet


def test_tape_appends_only_when_something_changed(tmp_path):
    tape = tmp_path / "tape.jsonl"
    rec = {"as_of": "2026-11-02", "run_utc": "x", "stage": "filed_no_range", "code_commit": "a"}
    assert T.append_if_changed(dict(rec), tape)
    assert not T.append_if_changed({**rec, "run_utc": "y", "code_commit": "b"}, tape)   # a rerun is not news
    assert T.append_if_changed({**rec, "as_of": "2026-11-03"}, tape)
    lines = tape.read_text().splitlines()
    assert [json.loads(x)["seq"] for x in lines] == [1, 2]


def test_score_is_hit_or_miss_against_the_published_band():
    call = {"offer": 44.0, "point": 53.0, "lo": 46.0, "hi": 61.0}
    assert T.score(call, 50.0)["inside_band"]
    assert not T.score(call, 70.0)["inside_band"]
    assert T.score(call, 55.0)["first_day"] == pytest.approx(55 / 44 - 1)


DEAL = {"exchange": "NASDAQ Global Select", "placeholder_range": [40.0, 44.0],
        "assumed_deal_usd": 60e9, "assumed_deal_source": "press"}
REGIME = {"n": 12, "median": 0.02, "iqr": 0.27}


def test_call_uses_the_placeholder_until_a_range_is_filed():
    out = T.make_call(MODEL, DEAL, {"registration": {"low": None, "high": None, "shares": None}}, REGIME)
    assert out["range_source"] == "placeholder" and out["call"] is None
    assert next(s for s in out["scenarios"] if s["key"] == "no_range")["reason"] == "no_range"


def test_call_uses_the_filed_range_and_share_count():
    filing = {"registration": {"low": 30.0, "high": 33.0, "shares": 1_000_000_000}}
    out = T.make_call(MODEL, DEAL, filing, REGIME)
    top = next(s for s in out["scenarios"] if s["key"] == "top")
    assert out["range_source"] == "filed" and top["offer"] == 33.0
    assert top["deal_usd"] == pytest.approx(33e9) and top["deal_source"] == "filed shares x offer"


def test_stages_advance_with_the_filings():
    ft, today = pd.Timestamp("2026-11-16"), pd.Timestamp("2026-11-10")
    assert T.stage_of({"registration": {"low": None}}, ft, today, None) == "filed_no_range"
    assert T.stage_of({"registration": {"low": 30.0}}, ft, today, None) == "range_filed"
    assert T.stage_of({"registration": {"low": 30.0}, "prospectus": {"offer": 33.0}}, ft, today, None) == "priced"
    assert T.stage_of({"registration": {"low": 30.0}, "prospectus": {"offer": 33.0}}, ft, ft, None) == "trading"
    assert T.stage_of({}, ft, ft, {"inside_band": True}) == "scored"
