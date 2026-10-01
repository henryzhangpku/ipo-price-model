"""leakage.py — hard assertions. They raise; they never warn.

The decision for a listing is made at T-1, the day before first trade. Every
feature must be knowable then; every label must come strictly after. Splits are
chronological and disjoint. A violation is a bug in the experiment, not a
data-quality note, so it stops the run.
"""
from __future__ import annotations

import pandas as pd


class LeakageError(AssertionError):
    """Raised when a label, feature or split would let the future into T-1."""


def assert_label_after_decision(decision_date: pd.Timestamp, label_dates: list[pd.Timestamp]) -> None:
    for d in label_dates:
        if pd.Timestamp(d) <= pd.Timestamp(decision_date):
            raise LeakageError(f"label date {pd.Timestamp(d).date()} is not after decision {pd.Timestamp(decision_date).date()}")


def assert_chronological_splits(train_end: pd.Timestamp, calib_end: pd.Timestamp, test_start: pd.Timestamp) -> None:
    if not (pd.Timestamp(train_end) < pd.Timestamp(calib_end) < pd.Timestamp(test_start)):
        raise LeakageError("splits must be train < calibration < test in time, and disjoint")


def assert_features_known_at(features: pd.DataFrame, allowed: set[str]) -> None:
    extra = set(features.columns) - allowed
    if extra:
        raise LeakageError(f"feature columns not declared as T-1 knowable: {sorted(extra)}")


def assert_regime_uses_only_past(regime_dates: pd.Series, decision_dates: pd.Series) -> None:
    """A regime feature for listing i may only aggregate listings that closed before its T-1."""
    bad = (pd.to_datetime(regime_dates) >= pd.to_datetime(decision_dates)).sum()
    if bad:
        raise LeakageError(f"{int(bad)} regime rows aggregate listings on or after the decision date")


def assert_filing_before_listing(filed: pd.Timestamp, first_trade: pd.Timestamp) -> None:
    """A filed range may only come from a registration amendment filed before the first trading day."""
    if pd.Timestamp(filed) >= pd.Timestamp(first_trade):
        raise LeakageError(f"range filed {pd.Timestamp(filed).date()} is not before first trade {pd.Timestamp(first_trade).date()}")
