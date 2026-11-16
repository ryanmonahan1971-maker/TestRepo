"""
PerformanceMetrics — a metric that carries its own BASIS, and refuses to be
compared against a metric computed on a different one.

WHY THIS EXISTS (control C3, MD/RESEARCH_INTEGRITY.md)
------------------------------------------------------
Most of the alarming results in this program were **comparison errors, not code
errors**. Three real examples from the 2026-07 audit:

  * frozen's SUE at a **standalone 4.0x** gross cap (15.7%) was compared against
    the same leg at its **25% blend weight** (1.5%) and read as a catastrophic
    alpha collapse. It was a 4x sizing difference.
  * **fixed-$ vs compounded** sizing was worth ~4.8pp of CAGR — the "honest"
    model looked far worse than frozen purely from the convention.
  * ~**3.1pp** of a headline CAGR was **cash carry**, not strategy alpha.

Each was caught eventually, by hand, after hours of confusion. This class makes
the basis a first-class part of the number so the mistake becomes mechanical to
catch: `.compare()` and `.bridge()` **raise** on a basis mismatch instead of
silently producing a misleading delta.

Usage
-----
    m = PerformanceMetrics.from_returns(daily, basis=Basis(
            weighting="standalone_4.0x", compounding=True, returns="raw",
            cash_carry=False, costs="net_tca", price_vintage=vintage_hash),
            label="SUE @ auction fills")
    print(m)                 # always prints WITH its basis
    m.compare(other)         # raises BasisMismatch unless bases are identical
    m.bridge(other)          # explicit, itemised difference table
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd

TRADING_DAYS = 252


class BasisMismatch(ValueError):
    """Raised when two metrics computed on different bases are compared."""


@dataclass(frozen=True)
class Basis:
    """Everything that must match before two numbers may be compared.

    Every field here has, at least once, silently caused a wrong conclusion in
    this repo — none of them are hypothetical.
    """
    weighting: str          # e.g. "standalone_4.0x", "in_blend_25pct"
    compounding: bool       # True = NAV-proportional sizing (returns compound)
    returns: str            # "raw" | "excess"  (excess = benchmark-relative)
    cash_carry: bool        # is cash interest INSIDE the return?
    costs: str              # "gross" | "net_tca" | "net_tca_borrow"
    price_vintage: str = "" # prices_store.vintage()["hash"] — reproducibility
    hedge: str = "none"     # "none" | "iwm" | "demeaned"
    notes: str = ""

    def describe(self) -> str:
        return (f"weighting={self.weighting} · "
                f"{'compounded' if self.compounding else 'fixed-$'} · "
                f"{self.returns} · hedge={self.hedge} · "
                f"cash_carry={'incl' if self.cash_carry else 'excl'} · "
                f"costs={self.costs}"
                + (f" · vintage={self.price_vintage}" if self.price_vintage else ""))

    def diff(self, other: "Basis") -> dict:
        out = {}
        for f in ("weighting", "compounding", "returns", "cash_carry",
                  "costs", "price_vintage", "hedge"):
            a, b = getattr(self, f), getattr(other, f)
            if a != b:
                out[f] = (a, b)
        return out


class PerformanceMetrics:
    def __init__(self, equity: pd.Series, basis: Basis, label: str = ""):
        if not isinstance(equity, pd.Series) or len(equity) < 3:
            raise ValueError("equity must be a Series with >=3 points")
        self.equity = equity.astype(float)
        self.basis = basis
        self.label = label

    # ── constructors ────────────────────────────────────────────────────
    @classmethod
    def from_returns(cls, returns: Iterable[float] | pd.Series, basis: Basis,
                     label: str = "", start_equity: float = 1.0) -> "PerformanceMetrics":
        r0 = pd.Series(returns).astype(float)
        # ⚠️ A MISSING RETURN IS NOT A FLAT DAY (audit 2026-08-01).
        # `.fillna(0.0)` silently reads every gap as "the book was flat that
        # period". That is not neutral. It shifts BOTH the mean and the standard
        # deviation, and the direction depends on the series — a zero pulls the
        # mean toward zero while also adding a low-dispersion observation, so
        # Sharpe can move either way (measured on a synthetic series: 11.80 ->
        # 11.60 with 3 of 250 imputed). What is invariant is that the number
        # changes and the data gap that caused it disappears. This class computes
        # the headline Sharpe for the locks, so a filled gap is a published number.
        #
        # The fill is retained deliberately -- changing it would move locked
        # numbers, which is a v6 decision, not an engineering one (see
        # MD/ECS/V6_OPTIMIZATION_PLAN.md §9). What ends here is the SILENCE:
        # the count is warned about and carried on the object, so a caller or a
        # lock can state how much of its series was imputed.
        n_missing = int(r0.isna().sum())
        r = r0.fillna(0.0)
        if n_missing:
            warnings.warn(
                f"PerformanceMetrics.from_returns({label!r}): {n_missing} of "
                f"{len(r0)} returns were NaN and have been treated as 0.0% "
                f"(flat). This moves the mean and the volatility, so every "
                f"metric on this object is computed on an imputed series. Check "
                f"the input for gaps before quoting it.",
                RuntimeWarning, stacklevel=2)
        eq = start_equity * (1.0 + r).cumprod()
        m = cls(eq, basis, label)
        m._returns = r
        m.n_imputed_returns = n_missing
        return m

    @classmethod
    def from_equity(cls, equity: pd.Series, basis: Basis, label: str = "") -> "PerformanceMetrics":
        return cls(equity, basis, label)

    # ── metrics ─────────────────────────────────────────────────────────
    @property
    def returns(self) -> pd.Series:
        if hasattr(self, "_returns"):
            return self._returns
        return self.equity.pct_change().dropna()

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def years(self) -> float:
        idx = self.equity.index
        if isinstance(idx, pd.DatetimeIndex):
            return max((idx[-1] - idx[0]).days / 365.25, 1e-9)
        return max(len(self.equity) / TRADING_DAYS, 1e-9)

    @property
    def cagr(self) -> float:
        return (1.0 + self.total_return) ** (1.0 / self.years) - 1.0

    @property
    def sharpe(self) -> float:
        """Daily-return Sharpe, annualised. Use `sharpe_monthly` for the
        paper-comparable figure the ECS locks quote."""
        r = self.returns
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else float("nan")

    @property
    def sharpe_monthly(self) -> float:
        """Monthly-resampled Sharpe — the convention the frozen ECS locks use."""
        if not isinstance(self.equity.index, pd.DatetimeIndex):
            return float("nan")
        m = self.equity.resample("ME").last().pct_change().dropna()
        return float(m.mean() / m.std() * np.sqrt(12)) if m.std() > 0 else float("nan")

    @property
    def max_dd(self) -> float:
        return float((self.equity / self.equity.cummax() - 1.0).min())

    @property
    def calmar(self) -> float:
        dd = self.max_dd
        return self.cagr / abs(dd) if dd < 0 else float("nan")

    def as_dict(self) -> dict:
        return {"label": self.label, "CAGR_%": round(self.cagr * 100, 2),
                "Sharpe": round(self.sharpe, 2),
                "Sharpe_monthly": round(self.sharpe_monthly, 2),
                "maxDD_%": round(self.max_dd * 100, 2),
                "Calmar": round(self.calmar, 2), "basis": self.basis.describe()}

    def __repr__(self) -> str:
        return (f"{self.label or 'metrics'}: CAGR {self.cagr*100:.2f}% · "
                f"Sharpe {self.sharpe:.2f} · maxDD {self.max_dd*100:.2f}% "
                f"[{self.basis.describe()}]")

    # ── the guard rails ─────────────────────────────────────────────────
    def compare(self, other: "PerformanceMetrics") -> dict:
        """Delta vs another metric. RAISES unless the bases are identical —
        this is the whole point of the class."""
        d = self.basis.diff(other.basis)
        if d:
            raise BasisMismatch(
                "Refusing to compare metrics on different bases: "
                + "; ".join(f"{k}: {a!r} vs {b!r}" for k, (a, b) in d.items())
                + ". Use .bridge() to lay the difference out explicitly.")
        return {"cagr_delta_pp": round((self.cagr - other.cagr) * 100, 2),
                "sharpe_delta": round(self.sharpe - other.sharpe, 2),
                "maxdd_delta_pp": round((self.max_dd - other.max_dd) * 100, 2)}

    def bridge(self, other: "PerformanceMetrics") -> pd.DataFrame:
        """Explicit, itemised comparison when the bases DO differ — the only
        legitimate way to put two different-basis numbers side by side."""
        # ⚠️ **`delta_pp` LABELLED A SHARPE DIFFERENCE AS PERCENTAGE POINTS**
        # (`O197`, fixed 2026-09-05). CAGR and maxDD deltas ARE pp; a Sharpe
        # delta is a bare ratio difference and has no pp unit. One column
        # header over three different units is the C3 shape in miniature — and
        # this is the module whose whole purpose is refusing to put
        # different-basis numbers side by side without saying so.
        # 📌 The column is `delta` and each row now carries its own `unit`, so
        # the header cannot imply a unit the row does not have.
        d = self.basis.diff(other.basis)
        rows = [{"item": "CAGR_%", "this": round(self.cagr * 100, 2),
                 "other": round(other.cagr * 100, 2), "unit": "pp",
                 "delta": round((self.cagr - other.cagr) * 100, 2)},
                {"item": "Sharpe", "this": round(self.sharpe, 2),
                 "other": round(other.sharpe, 2), "unit": "ratio",
                 "delta": round(self.sharpe - other.sharpe, 2)},
                {"item": "maxDD_%", "this": round(self.max_dd * 100, 2),
                 "other": round(other.max_dd * 100, 2), "unit": "pp",
                 "delta": round((self.max_dd - other.max_dd) * 100, 2)}]
        for k, (a, b) in d.items():
            rows.append({"item": f"BASIS DIFFERS: {k}", "this": a, "other": b,
                         "unit": None, "delta": None})
        return pd.DataFrame(rows)

    def rebased(self, equity: pd.Series, **changes) -> "PerformanceMetrics":
        """Copy with an amended basis AND the series that basis describes.

        🔴 **`equity` BECAME REQUIRED 2026-09-05 (`O197`), BECAUSE THE OLD
        SIGNATURE COULD ONLY DO THE THING THIS DOCSTRING FORBIDS.** It read
        `rebased(self, **changes)` and reused `self.equity`, so every call
        relabelled a STALE series with a new basis — *"never to relabel a stale
        number"* was the one behaviour available. A caller who HAD recomputed
        had no way to hand the new series in.

        ⚠️ **The docstring was the control, and a docstring is not a control.**
        The signature now enforces it: you cannot rebase without supplying the
        series the new basis describes. Zero callers existed, so nothing
        breaks — which is exactly why it was cheap to fix and why it had
        survived.
        """
        return PerformanceMetrics(equity, replace(self.basis, **changes),
                                  self.label)
