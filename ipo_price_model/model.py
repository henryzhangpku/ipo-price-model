"""model.py — a conformal interval for the first close, from T-1 features.

The target is ``y = log(first_close / offer_price)``: the first-day move in log
space, so the interval is symmetric-ish and multiplicative once mapped back to
a price. Three parts:

1. **Quantile regressors** (gradient boosting, pinball loss) for the lower and
   upper quantile of ``y`` given the features, fitted on the training window.
2. **Split-conformal calibration**: on a later, disjoint calibration window the
   nonconformity score ``max(q_lo - y, y - q_hi)`` is computed, and its
   ``(1 - alpha)`` quantile widens both edges by a constant. That gives finite-
   sample marginal coverage of at least ``1 - alpha`` on exchangeable data,
   with no distributional assumption (CQR: Romano, Patterson & Candes, 2019).
3. **Mapping back**: ``price = offer * exp(q)``.

The point estimate is the fitted median. Everything is chronological: train,
then calibrate, then test, and never the other order.

A *reference* model is carried alongside: the offer price as the point
estimate, and the calibration window's empirical quantiles of ``y`` as a
constant interval. It is what a careful person would do with a spreadsheet,
and the fitted model has to beat it on interval width at the same coverage
or it has no reason to exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from .features import FEATURES


@dataclass
class ConformalQuantileModel:
    alpha: float = 0.20                 # 80% intervals by default
    n_estimators: int = 200
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    random_state: int = 7
    q_lo_: GradientBoostingRegressor | None = field(default=None, repr=False)
    q_hi_: GradientBoostingRegressor | None = field(default=None, repr=False)
    q_med_: GradientBoostingRegressor | None = field(default=None, repr=False)
    conformal_shift_: float | None = None
    ref_lo_: float | None = None
    ref_hi_: float | None = None
    n_train_: int = 0
    n_calib_: int = 0

    def _gbr(self, q: float) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=self.n_estimators,
                                         max_depth=self.max_depth, learning_rate=self.learning_rate,
                                         min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray, X_calib: pd.DataFrame, y_calib: np.ndarray) -> "ConformalQuantileModel":
        X_train, X_calib = X_train[FEATURES], X_calib[FEATURES]
        lo, hi = self.alpha / 2, 1 - self.alpha / 2
        self.q_lo_ = self._gbr(lo).fit(X_train, y_train)
        self.q_hi_ = self._gbr(hi).fit(X_train, y_train)
        self.q_med_ = self._gbr(0.5).fit(X_train, y_train)
        # split-conformal widening on the disjoint calibration window
        pl, _, ph = _rearrange(self.q_lo_.predict(X_calib), self.q_med_.predict(X_calib), self.q_hi_.predict(X_calib))
        scores = np.maximum(pl - y_calib, y_calib - ph)
        n = len(scores)
        k = int(np.ceil((n + 1) * (1 - self.alpha)))
        k = min(max(k, 1), n)
        self.conformal_shift_ = float(np.sort(scores)[k - 1])
        # the reference interval: calibration-window empirical quantiles of y around the offer
        self.ref_lo_, self.ref_hi_ = float(np.quantile(y_calib, lo)), float(np.quantile(y_calib, hi))
        self.n_train_, self.n_calib_ = int(len(y_train)), int(n)
        return self

    def predict_log(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[FEATURES]
        s = self.conformal_shift_ or 0.0
        lo, med, hi = _rearrange(self.q_lo_.predict(X), self.q_med_.predict(X), self.q_hi_.predict(X))
        # a negative conformal shift narrows the band; never past the median
        return pd.DataFrame({
            "q_lo": np.minimum(lo - s, med),
            "q_med": med,
            "q_hi": np.maximum(hi + s, med),
            "ref_lo": self.ref_lo_,
            "ref_hi": self.ref_hi_,
        }, index=X.index)

    def predict_price(self, X: pd.DataFrame, offer: pd.Series) -> pd.DataFrame:
        q = self.predict_log(X)
        o = offer.to_numpy(dtype=float)
        return pd.DataFrame({
            "point": o * np.exp(q["q_med"].to_numpy()),
            "lo": o * np.exp(q["q_lo"].to_numpy()),
            "hi": o * np.exp(q["q_hi"].to_numpy()),
            "ref_point": o,
            "ref_lo": o * np.exp(q["ref_lo"].to_numpy()),
            "ref_hi": o * np.exp(q["ref_hi"].to_numpy()),
        }, index=X.index)


def _rearrange(lo: np.ndarray, med: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sort the three fitted quantiles row by row so they never cross (monotone rearrangement,
    Chernozhukov, Fernandez-Val & Galichon, 2010). Separately fitted quantile models can cross."""
    q = np.sort(np.vstack([lo, med, hi]), axis=0)
    return q[0], q[1], q[2]


def coverage(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    ok = ~np.isnan(y)
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok]))) if ok.any() else float("nan")


def mean_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))
