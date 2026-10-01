# Methodology

**The question.** Given only what is public the day before a company first
trades, what will it close at, and how sure can anyone be? And — the part
that matters more — when is the honest answer *no number*?

Everything below was fixed before any result was looked at. The tests in
`tests/` are these claims, executable.

---

## 1. The decision point is T-1

A listing prices the evening before it trades. From that moment the offer
price, the filed range, the deal size and the exchange are public. Nothing
that happens after the open is available to the model: no first tick, no
first-day volume, no news about the debut. Every feature is declared in one
list (`features.FEATURES`), and a leakage assertion refuses any column that is
not on it.

## 2. Two kinds of evidence

**Deal facts**, from the calendar row: log offer price, log deal size, where
the offer priced against the filed range (below / within / at the top /
above), the width of that range, whether a range was found at all, and whether
the venue is a major exchange.

**Where the filed range comes from.** Not the calendar: a public IPO calendar
overwrites the range with the offer price once a deal prices, so a historical
pull carries no ranges at all (the first run of this repo found zero). The
range is read instead from the cover of the registration statement itself,
the S-1/A (F-1/A for foreign issuers, S-11/A for REITs) on SEC EDGAR:
"the initial public offering price will be between $15.00 and $17.00 per
share". Only amendments filed **strictly before the first trading day** are
read, newest first, and the first cover that states a range wins, so a
re-range filed before pricing counts and the final prospectus (424B4, filed
after pricing) never does. `leakage.assert_filing_before_listing` raises on a
violation. Fixed-price small-cap deals print no range on the cover; those stay
`no_range` and are withheld, which is the correct answer for them.

**Regime**: what the previous listings did. For each listing, the median and
interquartile range of the first-day move across listings whose first close
fell in the trailing 90 days *and strictly before this one's T-1*. A second
assertion checks the strict ordering row by row. Below five prior listings the
regime is recorded as unknown rather than estimated from three points.

## 3. The target and the model

The target is `y = log(first_close / offer)`. In log space the interval is
close to symmetric and maps back to price multiplicatively, so a lower bound
can never go below zero.

Three gradient-boosted quantile regressors (pinball loss) estimate the 10th,
50th and 90th percentiles of `y` from the features. That gives a shape but no
guarantee, so the interval is then **conformally calibrated**: on a later,
disjoint window the nonconformity score `max(q10 − y, y − q90)` is computed
for every listing, and its `(1 − α)` empirical quantile widens both edges by a
constant. Under exchangeability that yields marginal coverage of at least
`1 − α` in finite samples, with no assumption about the error distribution
(conformalised quantile regression, Romano, Patterson & Candès 2019). The
default `α = 0.20`: an 80% interval.

The three quantile models are fitted separately, so they can cross. They are
sorted row by row before calibration and before every prediction (monotone
rearrangement, Chernozhukov, Fernández-Val & Galichon 2010), and a negative
conformal shift never narrows an edge past the median. This was a bug in the
first version, found by the `price` command printing a point outside its own
band, and fixed before the results below were produced.

**The label, and what counts as one.** The price source adjusts history for
every split up to today; the offer price is not adjusted. The first close is
therefore restored to the price that printed: the split-adjusted close times
the product of every split ratio after that day. Two data-validity bounds were
fixed before any result and are applied to every listing: the first print
must land within seven days of the listing date, and the first close must lie
between 0.1x and 30x the offer. A listing outside them is a data error
(usually a ticker since reused by another company), not an outcome, and is
dropped and counted.

## 4. The reference the model has to beat

The offer price as the point estimate, and the calibration window's empirical
10th–90th percentile of `y` as a constant interval around it. This is what a
careful person does with a spreadsheet, and it has the same coverage guarantee
the model has. The model earns its place only if it is **narrower at the same
coverage**; a wider or equally wide interval with a fancier fit is reported as
"reference stands".

## 5. Chronology, never shuffled

Train, then calibrate, then test, in date order and disjoint. The defaults are
train ≤ 2022, calibrate 2023, test 2024 onward. A random split would let 2024
regimes teach a model that is then scored on 2024; the assertion in
`leakage.assert_chronological_splits` makes that impossible by accident.

## 6. The publication gate

A price is printed only when three declared conditions hold, checked in this
order, and the first failure is the recorded reason:

| reason | condition | why |
|---|---|---|
| `no_range` | the filed range is missing | the single most informative T-1 fact is absent; printing would be guessing |
| `thin_regime` | fewer than 5 comparable listings in the trailing window | the market has not spoken recently enough for the regime to mean anything |
| `too_wide` | the 80% interval spans more than 60% of the offer price | "between half and double" is honest and useless, and should say so |

The gate never narrows an interval. It decides whether to show one. Coverage
is reported on all listings, on the published subset and on the withheld
subset separately, so a reader can see what the refusals cost and what they
protected.

## 7. What is reported, either way

Coverage and mean relative width for the model and the reference; overall, by
year, by deal-size bucket; the gate's counts by reason; point-estimate error
for both. The JSON in `results/` is the record. A run where the reference
stands is written the same way as one where the model wins.

## 8. What this is not

Not a trading signal, not an allocation recommendation, and not a claim about
the first-day pop as an investable return (retail rarely gets allocation at
the offer). It is a pricing exercise on the thinnest public data a listed
company will ever have, built to be wrong out loud rather than confident in
private.

## 9. Not in this version

The full re-range path through the S-1/A sequence (only the last stated range
is used); underwriter identity;
sector; a 30-session horizon model (the label is recorded, the model is not
fitted). Each is a change to §2 and gets its own preregistration.
