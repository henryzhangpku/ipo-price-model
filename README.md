# ipo-price-model

**Price a company the day before it trades — and know when not to.**

The day before a listing opens, the public record is thin: an offer price, a
filed range, a deal size, an exchange, and what the last few listings did.
This is a model that turns exactly that into a calibrated interval for the
first close, plus a gate that **withholds the number** when the evidence
cannot carry one. The refusals are the product.

```
$ ipo-price-model price --date 2026-10-14 --offer 21 --low 18 --high 20 --deal 450e6
{
  "offer": 21.0,
  "point": <fitted median>,
  "lo": <lower edge>,
  "hi": <upper edge>,
  "nominal_coverage": 0.8,
  "published": true,
  "reason": null,
  "regime_n": <prior listings in the trailing window>
}

$ ipo-price-model price --date 2026-10-14 --offer 21 --deal 450e6        # no filed range
{ ..., "published": false, "reason": "no_range" }
withheld: no_range — the interval above is shown for audit, not for use
```

The numbers come from a model fitted on the frozen calendar; the shape above
is what the command prints, and the values are whatever the data says on the
day. No figure in this README is quoted from a run that has not been made.

## Why this exists

Pricing something that does not trade yet is the same problem everywhere it
appears: few observations, stale ones, selection toward the deals that
happened, and a large gap between a transaction and a valuation. I built two
settlement-grade benchmarks for markets like that ([GPU compute](https://github.com/henryzhangpku/gpu-price-index),
[LLM tokens](https://github.com/henryzhangpku/token-price-index)), and the
part that mattered in both was not the estimator. It was the publication
gate: deciding when the evidence is too thin to print, and defending the
refusal.

An IPO the night before it opens is the cleanest public instance of that
problem, with a label that arrives the next afternoon. So it is a good place
to show the whole discipline in one small repo: T-1 features only, leakage
assertions that raise, a chronological fit, a distribution-free interval, a
reference the model has to beat, and a gate that says no.

## What it does

| step | what | where |
|---|---|---|
| **universe** | every priced US listing 2019–2025 from the public IPO calendar, SPACs and units tagged as cohorts | `universe.py` |
| **labels** | first close, first-day move, 30-session return — computed so T-1 can never see them | `labels.py` |
| **features** | deal facts + a trailing regime built only from listings that closed *strictly before* T-1 | `features.py` |
| **model** | gradient-boosted quantiles, then split-conformal calibration on a later window (CQR) | `model.py` |
| **reference** | offer price ± the calibration window's empirical quantiles; the bar to clear | `model.py` |
| **gate** | `no_range` → `thin_regime` → `too_wide`; declared thresholds, audited refusals | `gate.py` |
| **evaluate** | coverage and width, overall / by year / by deal size, published vs withheld, JSON either way | `evaluate.py` |

Full reasoning, including what is deliberately left out: [METHODOLOGY.md](METHODOLOGY.md).

## Run it

```bash
pip install -e ".[dev]"
export FINNHUB_API_KEY=...            # free tier is enough for the calendar
ipo-price-model universe --from 2019-01-01 --to 2025-12-31
ipo-price-model labels
ipo-price-model evaluate               # train ≤2022, calibrate 2023, test 2024→
ipo-price-model price --date 2026-10-14 --offer 21 --low 18 --high 20 --deal 450e6
pytest -q                              # 12 tests, no network, no keys
```

## What the tests pin down

- a feature column that is not on the declared T-1 list raises
- a regime that includes a listing on or after the decision date raises
- a label dated on or before the decision raises
- splits that are not train < calibrate < test raise
- conformal coverage holds on a held-out window of synthetic data
- the gate's reasons fire in the declared order and never touch the interval

## Honest limits

The calendar keeps the filed range only until pricing, so many historical
rows have the offer but not the range; those are withheld as `no_range`, and
the share withheld is reported, not hidden. No underwriter identity, no
sector, no S-1 text: each would be a new declared feature with its own
preregistration. The first-day move is not presented as an investable
return. Research tooling; nothing here is investment advice.

## Related

- [gpu-price-index](https://github.com/henryzhangpku/gpu-price-index) — a settlement-grade benchmark that declines to print
- [research2prod](https://github.com/henryzhangpku/research2prod) — write the signal once; it is already production code
- [autonomous-quant-researcher](https://github.com/henryzhangpku/autonomous-quant-researcher) — a research loop that keeps every refutation

MIT.
