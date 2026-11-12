import numpy as np
import pytest

from marketzero.config import Config
from marketzero.data import synthetic_daily
from marketzero.environment import TradingEnv, action_to_position, flat_action


def make_env(n=50, fee=0.001, n_actions=3):
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(n, 3)).astype(np.float32)
    mu = rng.normal(0.001, 0.02, size=n)
    return TradingEnv(feats, mu, dates=list(range(n)), fee_rate=fee, n_actions=n_actions)


def test_position_mapping():
    assert [action_to_position(a, 3) for a in range(3)] == [-1, 0, 1]
    assert [action_to_position(a, 2) for a in range(2)] == [0, 1]
    assert [action_to_position(a, 5) for a in range(5)] == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert flat_action(5) == 2


def test_size_scale_shrinks_effective_position():
    mu = np.array([0.0, 0.0, 0.10])
    env = TradingEnv(np.zeros((3, 2)), mu, fee_rate=0.001, n_actions=3)
    env.reset(start=1, episode_len=1)
    _, r, _, info = env.step(2, size_scale=0.5)   # long at half size into +10%
    assert info["position"] == pytest.approx(0.5)
    # half exposure earns half the market return; fee charged on effective size
    assert r == pytest.approx(0.5 * 0.10 - 0.001 * 0.5)


def test_fee_and_reward_math():
    # decide at close of day i; the newly chosen position earns mu[i+1]
    mu = np.array([0.0, 0.0, 0.10, -0.05])
    env = TradingEnv(np.zeros((4, 2)), mu, fee_rate=0.001, n_actions=3)
    env.reset(start=1, episode_len=2)
    _, r1, _, i1 = env.step(2)            # flat -> long into a +10% day
    assert r1 == pytest.approx(0.10 - 0.001)
    assert i1["position"] == 1
    _, r2, _, i2 = env.step(1)            # long -> flat into a -5% day
    assert r2 == pytest.approx(-0.001)     # flat: no market P&L, exit fee only
    assert i2["position"] == 0


def test_short_exposure():
    mu = np.array([0.0, 0.0, 0.10])
    env = TradingEnv(np.zeros((3, 2)), mu, fee_rate=0.0, n_actions=3)
    env.reset(start=1, episode_len=1)
    _, r, _, info = env.step(0)            # short into a +10% day
    assert r == pytest.approx(-0.10)
    assert info["position"] == -1


def test_observation_contains_position():
    env = make_env()
    obs = env.reset(start=5, episode_len=10)
    assert obs[-1] == 0.0
    env.step(2)
    assert env.obs()[-1] == 1.0


def test_equity_and_drawdown_tracking():
    mu = np.array([0.0, 0.0, 0.0, 0.10, -0.30])
    env = TradingEnv(np.zeros((5, 2)), mu, fee_rate=0.0, n_actions=3)
    env.reset(start=2, episode_len=2)
    env.step(2)                            # long: +10%
    env.step(2)                            # long: -30%
    assert env.stats.equity == pytest.approx(np.exp(0.10 - 0.30))
    # peak = e^{0.10}, trough = e^{-0.20} -> dd = 1 - e^{-0.30}
    assert env.stats.max_drawdown == pytest.approx(1 - np.exp(-0.30))


def test_force_flat_locks_position():
    env = make_env()
    env.reset(start=2, episode_len=10)
    env.step(2)                             # go long
    env.force_flat()
    _, _, _, info = env.step(2)             # tries to go long again
    assert info["position"] == 0
    assert info["halted"] is True


def test_synthetic_data_shape():
    df = synthetic_daily(500, seed=3)
    assert len(df) == 500
    assert set(["open", "high", "low", "close", "volume"]) <= set(df.columns)
    assert (df["high"] >= df["low"]).all()
