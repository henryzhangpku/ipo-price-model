# ipo-price-model

**Price a company the day before it trades — and know when not to.**

The day before a listing opens, the public record is thin: an offer price, a
filed range, a deal size, an exchange, and what the last few listings did.
This is a model that turns exactly that into a calibrated interval for the
first close, plus a gate that **withholds the number** when the evidence
cannot carry one. The refusals are the product.

```
$ ipo-price-model price --date 2025-11-14 --offer 21 --low 18 --high 20 --deal 450e6
{
  "offer": 21.0,
  "point": 28.39,
  "lo": 24.03,
  "hi": 31.27,
  "nominal_coverage": 0.8,
  "published": true,
  "reason": null,
  "regime_n": 48
}

$ ipo-price-model price --date 2025-11-14 --offer 21 --deal 450e6            # no filed range
{ ..., "published": false, "reason": "no_range" }

$ ipo-price-model price --date 2025-11-14 --offer 4 --low 4 --high 5 --deal 8e6 --exchange "NASDAQ Capital"
{ "point": 3.88, "lo": 3.04, "hi": 6.63, ..., "published": false, "reason": "too_wide" }
```

Real output from the model fitted on the frozen 2019-2025 data (point and
edges rounded to cents). A $450M deal that priced above its range gets a
band. The same deal with no filed range is withheld. An $8M micro-cap gets a
band too wide to be useful, and the gate says so instead of printing a
midpoint.

## Results

First full run, 30 September 2026. Train on listings to 2022, calibrate on
2023, test on 2024 onward. Nominal coverage 80%.

| test window, 2024 onward | listings | coverage | mean width, x offer |
|---|---|---|---|
| model, every listing | 332 | 0.877 | 1.67 |
| reference, every listing | 332 | 0.831 | 0.70 |
| **model, the 83 the gate publishes** | 83 | **0.843** | **0.51** |
| reference, the same 83 | 83 | 0.831 | 0.70 |

**The verdict, by the rule fixed before the run: the reference stands.**
Across every listing the fitted model over-covers with intervals more than
twice as wide, because micro-cap listings (under $50M, 58% of the test window)
have first days the deal facts cannot predict. The point estimate is no
better than the offer price either (mean absolute error 24.6% vs 24.8% of the
offer).

**What the gate does with that.** It withholds 249 of 332: 173 as
`too_wide`, almost all micro-caps, and 76 as `no_range`. On the 83 it
publishes, the model's band is about a quarter narrower than the reference's
at the same coverage. That comparison is made after seeing the results, so
it is reported as a finding to test next, not as a win: the next version
fixes "published listings only" as the scoring rule in advance and re-runs.

Three data facts behind those numbers, each found by running it:

- **Filed ranges come from EDGAR, not the calendar.** The public calendar
  keeps none. 1,083 of 1,383 core listings (78%) carry a range read from the
  last S-1/A or F-1/A cover filed before the first trade.
- **First closes are restored from split-adjusted history.** 292 listings
  have a later split; without the restoration a 1-for-10 reverse split shows
  a first close ten times too high.
- **951 listings have a usable first close.** The rest are delisted from the
  free price source, or fail the declared validity bounds (first print more
  than a week after listing, or a first close outside 0.1x-30x the offer,
  which in practice means a reused ticker).

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
| **ranges** | the filed range, read from the cover of the last S-1/A filed on EDGAR before the first trade (the calendar keeps none) | `filings.py` |
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
ipo-price-model ranges                 # EDGAR; set SEC_USER_AGENT to your own contact
ipo-price-model labels                 # throttled and resumable; re-run if rate-limited
ipo-price-model evaluate               # train ≤2022, calibrate 2023, test 2024→
ipo-price-model price --date 2026-10-14 --offer 21 --low 18 --high 20 --deal 450e6
pytest -q                              # 15 tests, no network, no keys
```

## What the tests pin down

- a feature column that is not on the declared T-1 list raises
- a regime that includes a listing on or after the decision date raises
- a label dated on or before the decision raises
- a filed range from an amendment dated on or after the first trade raises
- the range parser reads both cover phrasings and rejects fixed prices, inverted and implausible ranges
- splits that are not train < calibrate < test raise
- conformal coverage holds on a held-out window of synthetic data
- the gate's reasons fire in the declared order and never touch the interval

## Honest limits

The public calendar keeps no filed ranges, so ranges are read from EDGAR
registration covers filed before the first trade; deals whose cover states a
fixed price, and deals the parser cannot read, are withheld as `no_range`, and
the share withheld is reported, not hidden. Labels come from a free price
source, so listings that have since delisted can lack a first close; that
biases the sample toward survivors, and the count dropped is reported. No underwriter identity, no
sector, no S-1 text: each would be a new declared feature with its own
preregistration. The first-day move is not presented as an investable
return. Research tooling; nothing here is investment advice.

## Related

- [gpu-price-index](https://github.com/henryzhangpku/gpu-price-index) — a settlement-grade benchmark that declines to print
- [research2prod](https://github.com/henryzhangpku/research2prod) — write the signal once; it is already production code
- [autonomous-quant-researcher](https://github.com/henryzhangpku/autonomous-quant-researcher) — a research loop that keeps every refutation

MIT.
