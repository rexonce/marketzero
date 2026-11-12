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
    def __init__(self, capacity: int = 512, seed: int = 0):
        self.capacity = capacity
        self.games: list[Game] = []
        self.rng = np.random.default_rng(seed)

    def append(self, game: Game) -> None:
        self.games.append(game)
        if len(self.games) > self.capacity:
            self.games.pop(0)

    def total_steps(self) -> int:
        return sum(len(g) for g in self.games)

    def sample_position(self, unroll: int) -> tuple[Game, int]:
        """Sample a (game, t) with enough room for a K-step unroll."""
        for _ in range(64):
            game = self.games[int(self.rng.integers(0, len(self.games)))]
            if len(game) > unroll + 1:
                t = int(self.rng.integers(0, len(game) - unroll - 1))
                return game, t
        game = max(self.games, key=len)
        return game, max(0, len(game) - unroll - 2)

    def sample_batch(self, batch_size: int, unroll: int) -> list[tuple[Game, int]]:
        return [self.sample_position(unroll) for _ in range(batch_size)]
