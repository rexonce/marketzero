import numpy as np
import pytest
import torch

from marketzero.config import Config
from marketzero.models import MuZeroNet, logits_to_scalar, scalar_to_two_hot
from marketzero.mcts import run_mcts, select_action


@pytest.fixture(scope="module")
def net():
    torch.manual_seed(0)
    return MuZeroNet(obs_dim=6, n_actions=3, hidden=32, latent=16, support=10)


@pytest.fixture(scope="module")
def cfg():
    return Config()


def test_two_hot_roundtrip(net):
    for x in (-0.95, -0.3, 0.0, 0.25, 0.87):
        t = scalar_to_two_hot(torch.tensor([x]), 10)
        assert t.shape == (1, 21)
        assert t.sum() == pytest.approx(1.0)
        logits = torch.log(t.clamp_min(1e-9))
        assert logits_to_scalar(logits, 10).item() == pytest.approx(x, abs=1e-5)


def test_two_hot_clips_out_of_range():
    t = scalar_to_two_hot(torch.tensor([42.0]), 10)
    assert t.sum() == pytest.approx(1.0)          # all mass on the top bin


def test_initial_and_recurrent_shapes(net):
    obs = torch.randn(4, 6)
    p, v, r, s = net.initial_inference(obs)
    assert p.shape == (4, 3) and v.shape == (4,) and s.shape == (4, 16)
    p2, v2, r2, s2 = net.recurrent_inference(s[0], 1)     # single latent state
    assert p2.shape == (3,) and v2.dim() == 0 and s2.shape == (16,)


def test_mcts_visit_invariants(net, cfg):
    obs = np.zeros(6, dtype=np.float32)
    res = run_mcts(net, obs, cfg, training=True, sims=20)
    assert res["visits"].sum() == 20
    assert res["policy"].sum() == pytest.approx(1.0)
    assert 0 <= res["action"] < 3
    assert -1.0 <= res["root_value"] <= 1.0


def test_mcts_eval_deterministic(net, cfg):
    obs = np.ones(6, dtype=np.float32) * 0.3
    a1 = run_mcts(net, obs, cfg, training=False)["action"]
    a2 = run_mcts(net, obs, cfg, training=False)["action"]
    assert a1 == a2


def test_temperature_sampling():
    rng = np.random.default_rng(0)
    policy = np.array([0.2, 0.5, 0.3])
    greedy = select_action(policy, 0.0, rng)
    assert greedy == 1
    counts = {select_action(policy, 1.0, rng) for _ in range(200)}
    assert len(counts) > 1                     # explores under tau=1
