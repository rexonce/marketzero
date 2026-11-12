import numpy as np
import pytest

from marketzero.config import Config
from marketzero.buffer import ReplayBuffer
from marketzero.trainer import MuZeroTrainer


class FakeGame:
    """Minimal game stand-in with the fields the trainer samples."""
    def __init__(self, obs_dim, n_actions, T=24, seed=0):
        rng = np.random.default_rng(seed)
        self.obs = [rng.normal(size=obs_dim).astype(np.float32) for _ in range(T)]
        self.actions = rng.integers(0, n_actions, T).tolist()
        self.rewards = rng.normal(0.001, 0.02, T).tolist()
        pols = rng.dirichlet(np.ones(n_actions), size=T)
        self.policies = [pols[i] for i in range(T)]
        self.returns = list(np.cumsum(self.rewards[::-1])[::-1])

    def __len__(self):
        return len(self.rewards)


def test_losses_finite_and_decreasing_setup():
    from marketzero.models import MuZeroNet
    cfg = Config()
    cfg.train_batch_size = 16
    cfg.train_unroll = 3
    net = MuZeroNet(obs_dim=6, n_actions=3, hidden=32, latent=16, support=10)
    buf = ReplayBuffer(seed=0)
    for i in range(3):
        buf.append(FakeGame(6, 3, seed=i))  # type: ignore[arg-type]
    trainer = MuZeroTrainer(net, cfg)
    l0 = trainer.train_step(buf)
    assert all(np.isfinite(v) for v in l0.values())
    for _ in range(4):
        l = trainer.train_step(buf)
    assert l["total"] < l0["total"] + 1.0     # can memorize synthetic targets
    assert trainer.value_scale > 0


def test_sample_position_bounds():
    buf = ReplayBuffer(seed=1)
    for i in range(5):
        buf.append(FakeGame(4, 2, T=30, seed=i))  # type: ignore[arg-type]
    for _ in range(50):
        game, t = buf.sample_position(5)
        assert 0 <= t < len(game) - 5


def test_reward_targets_align_with_consumed_action():
    """Unroll step k consumes actions[t+k-1] -> its reward target must be
    rewards[t+k-1] (the reward INTO position t+k), not rewards[t+k]."""
    from marketzero.models import MuZeroNet
    cfg = Config()
    cfg.train_unroll = 4
    net = MuZeroNet(obs_dim=6, n_actions=3, hidden=32, latent=16, support=10)
    trainer = MuZeroTrainer(net, cfg)
    g = FakeGame(6, 3, T=40, seed=0)
    g.rewards = [0.001 * (i + 1) for i in range(len(g))]   # distinct, in-range
    batch = [(g, 10)]
    _, _, T_rew, T_mask = trainer._build_targets(batch, cfg.train_unroll)
    for k in range(1, cfg.train_unroll + 1):
        assert T_rew[k][0].item() == pytest.approx(g.rewards[10 + k - 1])
        # policy/value sit at position t+k, reward one index earlier
        assert T_mask[k][0].item() == 1.0
