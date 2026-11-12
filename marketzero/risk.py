"""Risk governor: survival rules for personal trading.

The governor can (a) kill an episode when drawdown or single-day-loss limits
are breached -- the env is forced flat and stays flat until manually re-armed
-- and (b) shrink exposure toward a volatility target so a 60%-vol asset
doesn't get full notional. Optimization can wait; survival cannot.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GovernorState:
    halted: bool = False
    reasons: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []


class RiskGovernor:
    def __init__(self, cfg):
        self.max_dd = cfg.risk_max_drawdown
        self.max_daily = cfg.risk_max_daily_loss
        self.vol_target = cfg.risk_vol_target
        self.state = GovernorState()
        self._vol_ema = 0.02 ** 2          # daily variance EMA (start ~32% ann.)

    # ------------------------------------------------------------------ checks
    def check(self, info: dict) -> tuple[bool, list[str]]:
        """Call each env step with the info dict. Returns (ok, reasons)."""
        if self.state.halted:
            return False, self.state.reasons
        reasons = []
        if info.get("drawdown", 0.0) >= self.max_dd:
            reasons.append(f"drawdown {info['drawdown']:.1%} >= limit {self.max_dd:.0%}")
        if info.get("daily_loss", 0.0) >= self.max_daily:
            reasons.append(f"daily loss {info['daily_loss']:.1%} >= limit {self.max_daily:.0%}")
        if reasons:
            self.state.halted = True
            self.state.reasons = reasons
            return False, reasons
        return True, []

    def observe_return(self, r: float) -> None:
        """Feed realized strategy return to keep the volatility estimate warm."""
        self._vol_ema = 0.97 * self._vol_ema + 0.03 * (r * r)

    def position_scale(self, position: int) -> float:
        """Vol-targeted size multiplier in [0, 1]."""
        if self.vol_target is None:
            return 1.0
        ann_vol = (self._vol_ema ** 0.5) * (365 ** 0.5)
        if ann_vol <= 0:
            return 1.0
        return float(min(1.0, self.vol_target / ann_vol))

    def re_arm(self) -> None:
        """Manual re-enable after a halt. Never automatic by design."""
        self.state = GovernorState()
