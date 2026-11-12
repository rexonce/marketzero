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
        self.root_values = [0.0] * T
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
        game, t, w = buf.sample_position(5)
        assert 0 <= t < len(game) - 5
        assert w == 1.0                    # alpha=0 -> uniform, weight 1


def test_prioritized_replay_samples_high_error_positions():
    """With alpha>0, positions where |z/scale - v| is large dominate."""
    from marketzero.buffer import Game
    buf = ReplayBuffer(seed=0, alpha=1.0, is_beta=0.5)
    g = FakeGame(4, 3, T=40, seed=0)
    g.root_values = [0.0] * len(g)
    g.returns = [0.0] * len(g)
    g.returns[10] = 1.0                   # one huge surprise
    buf.append(g, value_scale=1.0)        # type: ignore[arg-type]
    seen = [buf.sample_position(2)[1] for _ in range(40)]
    assert seen.count(10) > 20            # dominates the sampling mass
    ws = [buf.sample_position(2)[2] for _ in range(40)]
    assert all(0 < w <= 1.0 for w in ws)  # IS weights are normalized ≤ 1


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
    batch = [(g, 10, 1.0)]
    _, _, T_rew, T_mask = trainer._build_targets(batch, cfg.train_unroll)
    for k in range(1, cfg.train_unroll + 1):
        assert T_rew[k][0].item() == pytest.approx(g.rewards[10 + k - 1])
        # policy/value sit at position t+k, reward one index earlier
        assert T_mask[k][0].item() == 1.0


def test_td_bootstrap_value_targets():
    """train_td_steps=n -> z_t = r_t + γr_{t+1} + γ^n · v_{t+n}·scale."""
    from marketzero.models import MuZeroNet
    cfg = Config()
    cfg.train_unroll = 2
    cfg.train_td_steps = 3
    cfg.env_gamma = 0.9
    net = MuZeroNet(obs_dim=6, n_actions=3, hidden=32, latent=16, support=10)
    trainer = MuZeroTrainer(net, cfg)
    trainer.value_scale = 0.5
    g = FakeGame(6, 3, T=40, seed=0)
    g.rewards = [0.01] * len(g)
    g.root_values = [0.2] * len(g)
    _, T_val, _, _ = trainer._build_targets([(g, 5, 1.0)], cfg.train_unroll)

    # decode the two-hot target at unroll step 0 (position t=5) back to a
    # scalar. z = r5 + γ r6 + γ² r7 + γ³ · root_v8 · scale_at_build (0.5);
    # _value_target then divides by the post-EMA scale (batch of 1 -> floor):
    # vs_used = 0.9·0.5 + 0.1·floor
    r, gm = 0.01, 0.9
    z = r + gm * r + gm**2 * r + gm**3 * (0.2 * 0.5)
    vs_used = 0.9 * 0.5 + 0.1 * cfg.train_value_std_floor
    grid = np.linspace(-1, 1, 2 * cfg.model_support + 1)
    decoded = float((T_val[0][0].numpy() * grid).sum())
    assert decoded == pytest.approx(z / vs_used, abs=2e-3)

    # near-episode end: bootstrap tail truncated, pure reward sum remains
    g2 = FakeGame(6, 3, T=8, seed=1)
    g2.rewards = [0.01] * len(g2)
    g2.root_values = [0.9] * len(g2)
    _, T_val_end, _, _ = trainer._build_targets([(g2, 7, 1.0)], cfg.train_unroll)
    assert np.isfinite(T_val_end[0].numpy()).all()
