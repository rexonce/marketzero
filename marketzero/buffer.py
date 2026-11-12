"""Replay buffer of self-play games and MuZero-style window sampling."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Game:
    obs: list[np.ndarray] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    policies: list[np.ndarray] = field(default_factory=list)   # improved (visit) policies
    root_values: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)         # discounted returns-to-go

    def compute_returns(self, gamma: float) -> None:
        """z_t = r_t + gamma * z_{t+1}; terminal z = 0."""
        T = len(self.rewards)
        self.returns = [0.0] * T
        acc = 0.0
        for t in reversed(range(T)):
            acc = self.rewards[t] + gamma * acc
            self.returns[t] = acc

    def __len__(self):
        return len(self.rewards)


class ReplayBuffer:
    """Uniform or prioritized replay (canonical MuZero priorities).

    Priority p_t = |z_t/s - v_t| + eps: the gap between the realized
    (scaled) return and the MCTS root value at that position -- i.e. the
    model's surprise. Sampled with p^alpha; importance-sampling weights
    (N·P_i)^-beta (normalized to max 1) correct the bias. alpha=0 reduces
    exactly to uniform replay with weights 1."""

    def __init__(self, capacity: int = 512, seed: int = 0,
                 alpha: float = 0.0, is_beta: float = 0.5):
        self.capacity = capacity
        self.games: list[Game] = []
        self.rng = np.random.default_rng(seed)
        self.alpha = alpha
        self.is_beta = is_beta

    @staticmethod
    def _priorities(game: Game, value_scale: float) -> np.ndarray:
        z = np.asarray(game.returns, dtype=np.float64) / max(value_scale, 1e-6)
        v = np.asarray(game.root_values, dtype=np.float64)
        return np.abs(z - v) + 1e-6

    def append(self, game: Game, value_scale: float = 1.0) -> None:
        game.priorities = self._priorities(game, value_scale)  # type: ignore[attr-defined]
        self.games.append(game)
        if len(self.games) > self.capacity:
            self.games.pop(0)

    def refresh_priorities(self, game: Game, value_scale: float) -> None:
        """Recompute after reanalysis updates root_values."""
        game.priorities = self._priorities(game, value_scale)  # type: ignore[attr-defined]

    def total_steps(self) -> int:
        return sum(len(g) for g in self.games)

    def _position_weights(self, game: Game, n_valid: int) -> np.ndarray:
        pr = getattr(game, "priorities", None)
        if pr is None or self.alpha == 0.0:
            return np.full(n_valid, 1.0 / n_valid)
        w = np.asarray(pr[:n_valid], dtype=np.float64) ** self.alpha
        return w / w.sum()

    def sample_position(self, unroll: int) -> tuple[Game, int, float]:
        """Sample (game, t, is_weight): game ~ sum p^a, position ~ p^a."""
        a = self.alpha
        for _ in range(64):
            if a > 0.0:
                gw = np.array([
                    float(np.asarray(getattr(g, "priorities", np.ones(len(g)))
                                     [:len(g)], dtype=np.float64).sum() ** a)
                    for g in self.games])
                gidx = int(self.rng.choice(len(self.games), p=gw / gw.sum()))
            else:
                gidx = int(self.rng.integers(0, len(self.games)))
            game = self.games[gidx]
            n_valid = len(game) - unroll - 1
            if n_valid > 0:
                pw = self._position_weights(game, n_valid)
                t = int(self.rng.choice(n_valid, p=pw))
                w = float((n_valid * pw[t]) ** -self.is_beta) if a > 0.0 else 1.0
                return game, t, w
        game = max(self.games, key=len)
        return game, max(0, len(game) - unroll - 2), 1.0

    def sample_batch(self, batch_size: int, unroll: int) -> list[tuple[Game, int, float]]:
        return [self.sample_position(unroll) for _ in range(batch_size)]
