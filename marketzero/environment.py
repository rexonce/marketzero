"""Trading environment: a market slice the agent replays.

Alignment contract (see data.py): the agent observes features of day i
(including its current exposure), chooses a target position at the close of
day i, and that position earns mu[i+1] -- the log-return from close i to
close i+1 -- minus proportional fees on turnover.

Reward is the daily log-equity growth:
    r_t = w_t * mu[i+1] - fee * |w_new - w_t|
Maximizing the sum of rewards maximizes final log-wealth (Kelly growth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def action_to_position(action: int, n_actions: int) -> float:
    """Map action index to target exposure.

    3 actions: 0/1/2 -> -1/0/+1; 5 actions: -1/-0.5/0/+0.5/+1; 2 actions: 0/+1.
    """
    if n_actions == 5:
        return (-1.0, -0.5, 0.0, 0.5, 1.0)[action]
    if n_actions == 3:
        return (-1, 0, 1)[action]
    if n_actions == 2:
        return (0, 1)[action]
    raise ValueError(f"unsupported n_actions={n_actions}")


def flat_action(n_actions: int) -> int:
    return {5: 2, 3: 1, 2: 0}[n_actions]


@dataclass
class EpisodeStats:
    equity: float = 1.0
    peak: float = 1.0
    max_drawdown: float = 0.0
    rewards: list[float] = field(default_factory=list)


class TradingEnv:
    """Replays a fixed slice of (scaled features, log-returns)."""

    def __init__(self, features: np.ndarray, log_returns: np.ndarray,
                 dates=None, fee_rate: float = 0.001, n_actions: int = 3):
        assert len(features) == len(log_returns), "feature/return length mismatch"
        self.feat = np.asarray(features, dtype=np.float32)
        self.mu = np.asarray(log_returns, dtype=np.float64)
        self.dates = dates
        self.fee_rate = fee_rate
        self.n_actions = n_actions
        self.n_features = self.feat.shape[1] + 1  # + position column
        self.T = len(self.mu)
        self.i = 0
        self._end = self.T - 1
        self.w = 0.0  # current effective exposure (float; discrete targets x size scale)
        self.stats = EpisodeStats()
        self._forced_flat = False  # set by the risk governor's kill-switch

    # ------------------------------------------------------------------ API
    def reset(self, start: int | None = None, rng: np.random.Generator | None = None,
              episode_len: int | None = None) -> np.ndarray:
        length = episode_len or (self.T - 1)
        if start is None:
            hi = max(1, self.T - length - 1)
            start = int(rng.integers(1, hi)) if hi > 1 else 1
        self.i = int(start)
        self._end = min(self.i + length, self.T - 1)   # last valid index
        self.w = 0
        self.stats = EpisodeStats()
        self._forced_flat = False
        return self.obs()

    def force_flat(self) -> None:
        """Risk governor kill-switch: liquidate and stay flat."""
        self._forced_flat = True

    def obs(self) -> np.ndarray:
        return np.append(self.feat[self.i], float(self.w)).astype(np.float32)

    def step(self, action: int, size_scale: float = 1.0) -> tuple[np.ndarray, float, bool, dict]:
        """Trade to the target position at the close of day i; that position
        earns mu[i+1] (the return from close i to close i+1), net of fees on
        turnover.

        `size_scale` shrinks the effective exposure toward a risk target
        (e.g. the governor's volatility targeting): w_eff = target * scale,
        with fees charged on the effective turnover."""
        assert 0 <= action < self.n_actions
        assert 0.0 <= size_scale <= 1.0
        if self._forced_flat:
            action = flat_action(self.n_actions)
            size_scale = 1.0
        w_new = action_to_position(action, self.n_actions) * size_scale
        j = self.i + 1
        if j >= self.T:
            raise IndexError("step() past end of slice")
        mu = self.mu[j]
        fee = self.fee_rate * abs(w_new - self.w)
        reward = w_new * mu - fee          # newly chosen position earns day j
        self.w = w_new
        self.i = j
        # bookkeeping
        st = self.stats
        st.rewards.append(reward)
        st.equity *= float(np.exp(reward))
        st.peak = max(st.peak, st.equity)
        st.max_drawdown = max(st.max_drawdown, 1.0 - st.equity / st.peak)
        done = self.i >= self._end
        info = {
            "index": self.i,
            "date": self.dates[self.i] if self.dates is not None else self.i,
            "position": self.w,
            "mu": mu,
            "fee": fee,
            "reward": reward,
            "equity": st.equity,
            "daily_loss": -reward,
            "drawdown": 1.0 - st.equity / st.peak,
            "halted": self._forced_flat,
        }
        return self.obs(), reward, done, info
