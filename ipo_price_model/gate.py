"""gate.py — the publication gate. A price is printed only when the evidence supports one.

Three reasons to withhold, checked in order, and the first one that fires is
the reason recorded. None of them is a judgement call at print time: each is a
declared threshold, so a refusal can be audited the same way a print can.

* ``no_range``      — the calendar lost the filed range, so the single most
                      informative T-1 fact (where it priced against the range)
                      is missing. Printing would be guessing.
* ``thin_regime``   — fewer than ``min_regime_n`` comparable listings closed in
                      the trailing window. The market has not spoken recently
                      enough for a regime feature to mean anything.
* ``too_wide``      — the conformal interval, mapped to price, spans more than
                      ``max_rel_width`` of the offer price. An 80% interval that
                      says "somewhere between half and double" is honest, and
                      useless, and should say so rather than print a midpoint.

The gate never tightens an interval. It only decides whether to show it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

REASONS = ("no_range", "thin_regime", "too_wide")


@dataclass(frozen=True)
class GateConfig:
    min_regime_n: int = 5
    max_rel_width: float = 0.60     # (hi - lo) / offer


def apply(features: pd.DataFrame, prices: pd.DataFrame, offer: pd.Series, cfg: GateConfig = GateConfig()) -> pd.DataFrame:
    """Return a frame with ``published`` (bool) and ``reason`` (str or None), aligned to ``prices``."""
    o = offer.to_numpy(dtype=float)
    width = (prices["hi"].to_numpy() - prices["lo"].to_numpy()) / o
    reason = np.array([None] * len(prices), dtype=object)
    no_range = features["has_range"].to_numpy() < 0.5
    thin = features["regime_n"].to_numpy() < cfg.min_regime_n
    wide = width > cfg.max_rel_width
    reason[wide] = "too_wide"
    reason[thin] = "thin_regime"       # earlier reasons overwrite later ones
    reason[no_range] = "no_range"
    out = pd.DataFrame({"published": pd.isna(reason), "rel_width": width}, index=prices.index)
    out["reason"] = pd.Series(list(reason), index=prices.index, dtype="object")
    out["reason"] = out["reason"].where(out["reason"].notna(), None)
    return out[["published", "reason", "rel_width"]]


def summary(gated: pd.DataFrame) -> dict:
    counts = gated["reason"].value_counts(dropna=True).to_dict()
    return {"n": int(len(gated)), "published": int(gated["published"].sum()),
            "withheld": {r: int(counts.get(r, 0)) for r in REASONS}}
