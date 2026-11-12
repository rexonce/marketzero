"""Policy evaluation and walk-forward backtesting with purge/embargo.

The iron rule (docs/DEEP_DIVE.md §6): a result that hasn't survived
out-of-sample walk-forward with purge+embargo is a hypothesis, not a result.
Every run appends to artifacts/experiments.jsonl so the number of tried
configurations is always on record (multiple-testing honesty).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics as M
from .actor import play_episode
from .buffer import ReplayBuffer
from .config import Config
from .environment import TradingEnv, flat_action
from .models import MuZeroNet
from .mcts import run_mcts
from .risk import RiskGovernor
from .trainer import MuZeroTrainer, save_checkpoint


def build_net(obs_dim: int, cfg: Config, device: str, seed: int) -> MuZeroNet:
    torch_seed(obs_dim, seed)
    net = MuZeroNet(obs_dim, cfg.env_n_actions, cfg.model_hidden,
                    cfg.model_latent, cfg.model_support)
    return net.to(device)


def torch_seed(_obs_dim: int, seed: int) -> None:
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_policy(net: MuZeroNet, features: np.ndarray, mu: np.ndarray, dates,
               start: int, end: int, cfg: Config, use_risk: bool = True,
               verbose: bool = False) -> pd.DataFrame:
    """Greedy (tau=0) replay of [start, end) with the risk governor active."""
    env = TradingEnv(features[start:end], mu[start:end],
                     dates=dates[start:end] if dates is not None else None,
                     fee_rate=cfg.env_fee_rate, n_actions=cfg.env_n_actions)
    gov = RiskGovernor(cfg) if use_risk else None
    sampler = None
    if cfg.mcts_chance_scenarios > 0:
        from .kronos import BootstrapScenarioSampler
        sampler = BootstrapScenarioSampler()
    obs = env.reset(start=0)   # run to the end of the slice
    rows = []
    done = False
    while not done:
        res = run_mcts(net, obs, cfg, training=False)
        action = res["action"]
        if sampler is not None:
            # Phase-2 hook (expectimax-lite): re-score root actions against
            # sampled market paths and let the scenarios shift visit counts.
            action = _chance_rerank(net, obs, res, sampler, mu[:start + env.i], cfg)
        scale = 1.0
        if gov is not None and cfg.risk_apply_vol_target:
            scale = gov.position_scale(env.w)
        obs, reward, done, info = env.step(action, size_scale=scale)
        if gov is not None:
            gov.observe_return(reward)
            ok, reasons = gov.check(info)
            if not ok:
                env.force_flat()
                if verbose:
                    print(f"  [kill-switch] {info['date']}: {'; '.join(reasons)} -> flat")
        rows.append({
            "date": info["date"], "action": action, "position": info["position"],
            "size_scale": scale,
            "reward": reward, "mu": info["mu"], "fee": info["fee"],
            "equity": info["equity"], "drawdown": info["drawdown"],
        })
    return pd.DataFrame(rows)


def _chance_rerank(net: MuZeroNet, obs: np.ndarray, mcts_res: dict, sampler,
                   mu_history: np.ndarray, cfg: Config) -> int:
    """Re-score the root's visit counts with scenario-path Q values.

    Each action's scenario Q (averaged discounted reward over sampled market
    paths) is centered and added to its visit count at `mcts_chance_weight`
    of the simulation budget -- bounded, so scenarios can veto a weak visit
    leader but not override a strong one."""
    from .kronos import scenario_q
    q = scenario_q(net, obs, list(range(net.n_actions)), sampler,
                   mu_history, horizon=cfg.mcts_chance_horizon,
                   n_scenarios=cfg.mcts_chance_scenarios, cfg=cfg)
    visits = mcts_res["visits"].astype(np.float64)
    qv = np.array([q[a] for a in range(net.n_actions)])
    q_centered = qv - qv.mean()
    q_norm = q_centered / (np.abs(q_centered).max() + 1e-9)   # in [-1, 1]
    budget = cfg.mcts_chance_weight * visits.sum()
    return int((visits + budget * q_norm).argmax())


def buy_hold(mu: np.ndarray, start: int, end: int) -> np.ndarray:
    return mu[start + 1:end]     # hold 1 unit through the window


# ------------------------------------------------------------------- windows
def walk_forward_windows(T: int, cfg: Config):
    """Yield (train_start, train_end, test_start, test_end) with purge/embargo."""
    tw, te_w, purge, test = (cfg.eval_train_window, cfg.eval_test_window,
                             cfg.eval_purge, cfg.eval_test_window)
    t0 = tw + purge                       # first test needs one train window behind it
    while t0 + test <= T:
        train_end = t0 - purge            # embargo before test
        train_start = 0 if cfg.eval_anchored else max(0, train_end - tw)
        if train_end - train_start >= 252:  # need >= 1y to fit features honestly
            yield train_start, train_end, t0, min(t0 + test, T)
        t0 += test


def fit_on_window(net: MuZeroNet, features, mu, dates, cfg: Config,
                  train_start: int, train_end: int, seed: int) -> dict:
    """Self-play + training inside a training window (fresh buffer)."""
    rng = np.random.default_rng(seed)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity, seed=seed)
    env = TradingEnv(features[train_start:train_end], mu[train_start:train_end],
                     dates=None, fee_rate=cfg.env_fee_rate,
                     n_actions=cfg.env_n_actions)
    trainer = MuZeroTrainer(net, cfg)
    losses = {}
    for g in range(cfg.eval_wf_games):
        game = play_episode(env, net, cfg, rng, training=True)
        buffer.append(game)
        losses = trainer.train_epochs(buffer, cfg.eval_wf_epochs)
    return losses


def walk_forward(features: np.ndarray, mu: np.ndarray, dates, cfg: Config,
                 out_dir: Path, tag: str = "wf") -> dict:
    """Full walk-forward: fit per window, test out-of-sample, stitch results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    T = len(mu)
    windows = list(walk_forward_windows(T, cfg))
    stitched: list[np.ndarray] = []
    per_window = []
    for w_i, (ts_, te_, vs, ve) in enumerate(windows):
        net = build_net(features.shape[1] + 1, cfg, "cpu", cfg.train_seed + w_i)
        fit_on_window(net, features, mu, dates, cfg, ts_, te_, seed=cfg.train_seed + w_i)
        df = run_policy(net, features, mu, dates, vs, ve, cfg)
        stitched.append(df["reward"].to_numpy())
        stats = M.summary(df["reward"].to_numpy())
        bh = M.summary(buy_hold(mu, vs, ve))
        per_window.append({
            "window": w_i,
            "test_range": f"{df['date'].iloc[0]} .. {df['date'].iloc[-1]}",
            "agent": stats, "buy_hold": bh,
        })
        print(f"  W{w_i:02d} {df['date'].iloc[0]}..{df['date'].iloc[-1]} "
              f"agent SR {stats['sharpe']:+.2f} ret {stats['total_return']:+.1%} | "
              f"B&H SR {bh['sharpe']:+.2f} ret {bh['total_return']:+.1%}")
    if not stitched:
        raise RuntimeError("no walk-forward windows fit in the data")
    agent_ret = np.concatenate(stitched)
    bh_ret = np.concatenate([buy_hold(mu, vs, ve) for (_a, _b, vs, ve) in windows])
    result = {
        "tag": tag,
        "agent": M.summary(agent_ret),
        "buy_hold": M.summary(bh_ret),
        "n_windows": len(windows),
        "per_window": per_window,
    }
    # artifacts: equity curve + plot + registry
    eq = pd.DataFrame({
        "agent_equity": M.equity_curve(agent_ret),
        "bh_equity": M.equity_curve(bh_ret),
    })
    eq.to_csv(out_dir / f"{tag}_equity.csv", index=False)
    _plot_equity(out_dir / f"{tag}_equity.png", eq)
    _register(out_dir, cfg, result)
    (out_dir / f"{tag}_summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def _plot_equity(path: Path, eq: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.plot(eq["agent_equity"], label="MarketZero agent (OOS)", lw=1.4)
    ax.plot(eq["bh_equity"], label="Buy & hold", lw=1.4, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("out-of-sample day")
    ax.set_ylabel("equity (log)")
    ax.set_title("Walk-forward out-of-sample equity — net of fees, kill-switch active")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _register(out_dir: Path, cfg: Config, result: dict) -> None:
    """Append to the experiment registry: multiple-testing honesty ledger."""
    cfg_hash = hashlib.md5(json.dumps(cfg.to_dict(), sort_keys=True).encode()).hexdigest()[:10]
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cfg_hash": cfg_hash,
        "agent": result["agent"], "buy_hold": result["buy_hold"],
        "n_windows": result["n_windows"],
    }
    reg = out_dir / "experiments.jsonl"
    n_prev = 0
    if reg.exists():
        n_prev = sum(1 for _ in reg.open(encoding="utf-8"))
    with reg.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    result["prior_registered_runs"] = n_prev
