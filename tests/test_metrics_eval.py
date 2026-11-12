import numpy as np
import pytest

from marketzero import metrics as M
from marketzero.config import Config
from marketzero.evaluate import walk_forward_windows
from marketzero.kronos import BootstrapScenarioSampler


def test_metrics_on_known_sequences():
    # rewards are log-returns: +1% log daily for 100 days -> equity = e^1
    r = np.full(100, 0.01)
    assert M.total_return(r) == pytest.approx(np.e - 1.0, rel=1e-9)
    assert M.max_drawdown(r) == pytest.approx(0.0)
    # constant positive stream has zero std -> sharpe guard returns 0
    assert M.sharpe(r) == 0.0
    # drawdown: +50% then -50% in log terms
    r2 = np.array([np.log(1.5), np.log(0.5)])
    assert M.max_drawdown(r2) == pytest.approx(0.5)


def test_summary_keys():
    s = M.summary(np.random.default_rng(0).normal(0.001, 0.02, 300))
    assert {"total_return", "cagr", "sharpe", "max_drawdown", "hit_rate"} <= set(s)


def test_walk_forward_windows_respect_purge():
    cfg = Config()
    cfg.eval_train_window = 730
    cfg.eval_test_window = 90
    cfg.eval_purge = 10
    wins = list(walk_forward_windows(2000, cfg))
    assert len(wins) >= 2
    for ts_, te_, vs, ve in wins:
        assert vs - te_ == cfg.eval_purge          # embargo gap before test
        assert ve - vs == cfg.eval_test_window     # exact test width
        assert te_ - ts_ >= 252                    # at least a year to fit


def test_bootstrap_scenarios():
    mu = np.random.default_rng(2).normal(0.001, 0.02, 400)
    s = BootstrapScenarioSampler(block_len=20, seed=5)
    paths = s.sample(mu, n_scenarios=32, horizon=15)
    assert paths.shape == (32, 15)
    assert np.isfinite(paths).all()
    # block structure means consecutive draws can repeat history verbatim
    assert not np.allclose(paths[0], paths[1])
