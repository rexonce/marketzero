"""P0 gate: end-to-end wiring test on real data, with honest sanity checks.

The central assertion: an UNTRAINED agent must lose to buy & hold over the
full history. An untrained net that "wins" is proof of a bug -- lookahead
leakage being the usual suspect.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketzero import metrics as M
from marketzero.actor import play_episode
from marketzero.buffer import ReplayBuffer
from marketzero.config import Config, REPO_ROOT
from marketzero.data import FeatureScaler, compute_features, load_daily
from marketzero.environment import TradingEnv
from marketzero.models import MuZeroNet
from marketzero.mcts import run_mcts
from marketzero.risk import RiskGovernor
from marketzero.trainer import MuZeroTrainer, load_checkpoint, save_checkpoint
from marketzero.evaluate import run_policy


def main() -> int:
    cfg = Config()
    ok = lambda name: print(f"  PASS  {name}")  # noqa: E731

    print("[1/7] data")
    df, source = load_daily(cfg.data_symbol, cfg.data_start,
                            cfg.resolved_path(cfg.data_cache))
    feats, logret = compute_features(df, cfg.feat_window)
    X = feats.to_numpy()
    mu = logret.to_numpy()
    assert len(X) > 1000, "need >1000 bars"
    assert not np.isnan(X).any(), "NaNs in features"
    print(f"  loaded {len(X)} bars from {source} ({df.index[0].date()} .. {df.index[-1].date()})")

    print("[2/7] scaler (fit on first 60% = training discipline)")
    cut = int(len(X) * 0.6)
    scaler = FeatureScaler().fit(X[:cut])
    Xs = scaler.transform(X)
    ok(f"feature matrix {Xs.shape}, mean|val| {np.abs(Xs).mean():.3f}")

    print("[3/7] environment fee math (hand-checked)")
    mu_small = np.array([0.0, 0.0, 0.10, -0.05])
    f_small = np.zeros((4, 2))
    env = TradingEnv(f_small, mu_small, fee_rate=0.001, n_actions=3)
    env.reset(start=1, episode_len=2)
    # close of day1: go LONG -> position earns day2's +10%, pays entry fee
    _, r1, _, _ = env.step(2)
    assert abs(r1 - (0.10 - 0.001)) < 1e-12, r1
    # close of day2: go FLAT -> no exposure into day3, pays exit fee
    _, r2, _, _ = env.step(1)
    assert abs(r2 - (-0.001)) < 1e-12, r2
    ok("fee accounting exact")

    print("[4/7] MuZero networks + MCTS invariants")
    net = MuZeroNet(obs_dim=Xs.shape[1] + 1, n_actions=cfg.env_n_actions,
                    hidden=cfg.model_hidden, latent=cfg.model_latent,
                    support=cfg.model_support)
    res = run_mcts(net, np.append(Xs[300], 0.0), cfg, training=True)
    assert res["visits"].sum() == cfg.mcts_sims, res["visits"]
    assert abs(res["policy"].sum() - 1.0) < 1e-9
    res2 = run_mcts(net, np.append(Xs[300], 0.0), cfg, training=False)
    assert res2["action"] == run_mcts(net, np.append(Xs[300], 0.0), cfg,
                                      training=False)["action"], "eval nondeterminism"
    ok(f"visits==sims, policy sums to 1, eval deterministic ({net.param_count():,} params)")

    print("[5/7] self-play + training step (losses finite)")
    rng = np.random.default_rng(cfg.train_seed)
    env_t = TradingEnv(Xs[:cut], mu[:cut], fee_rate=cfg.env_fee_rate,
                       n_actions=cfg.env_n_actions)
    buf = ReplayBuffer(capacity=8, seed=cfg.train_seed)
    trainer = MuZeroTrainer(net, cfg)
    for _ in range(2):
        buf.append(play_episode(env_t, net, cfg, rng, training=True,
                                episode_len=64))
    losses = trainer.train_epochs(buf, epochs=2)
    assert all(np.isfinite(v) for v in losses.values()), losses
    ok(f"losses finite: {({k: round(v, 4) for k, v in losses.items()})}")

    print("[6/7] checkpoint roundtrip")
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "smoke.pt"
        save_checkpoint(ckpt, net, cfg, scaler)
        net2, cfg2, payload = load_checkpoint(ckpt)
        o = np.append(Xs[500], 0.0)
        a1 = run_mcts(net, o, cfg, training=False)["action"]
        a2 = run_mcts(net2, o, cfg2, training=False)["action"]
        assert a1 == a2, (a1, a2)
    ok("save/load preserves decisions")

    print("[7/7] honest gate: untrained agent vs buy & hold (full history, net of fees)")
    dfn = run_policy(net, Xs, mu, list(df.index), 1, len(X) - 1, cfg, use_risk=True)
    agent = M.summary(dfn["reward"].to_numpy())
    bh = M.summary(mu[2:])
    print(f"  untrained agent : {agent}")
    print(f"  buy & hold      : {bh}")
    assert agent["total_return"] < bh["total_return"], "untrained agent beat B&H -- suspect lookahead bug!"
    ok("untrained agent loses to B&H (correct wiring behavior)")

    print("[extra] risk kill-switch + scenario sampler")
    gov = RiskGovernor(cfg)
    ok_flag, reasons = gov.check({"drawdown": 0.25, "daily_loss": 0.0})
    assert not ok_flag, "kill-switch failed to trip at 25% DD"
    ok_flag, _ = gov.check({"drawdown": 0.0, "daily_loss": 0.0})
    assert not ok_flag, "kill-switch must stay halted until re-armed"
    from marketzero.kronos import BootstrapScenarioSampler
    scen = BootstrapScenarioSampler().sample(mu[:500], n_scenarios=16, horizon=10)
    assert scen.shape == (16, 10) and np.isfinite(scen).all()
    ok(f"kill-switch trips ({'; '.join(reasons)}), stays sticky; scenarios sample correctly")

    print("\nSMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
