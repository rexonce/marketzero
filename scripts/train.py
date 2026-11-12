"""Train the MuZero agent by self-play on a slice of history (P1)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketzero.actor import play_episode
from marketzero.buffer import ReplayBuffer
from marketzero.config import Config
from marketzero.data import FeatureScaler, compute_features, load_daily
from marketzero.environment import TradingEnv
from marketzero.models import MuZeroNet
from marketzero.trainer import MuZeroTrainer, save_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--iters", type=int, default=20, help="self-play/training iterations")
    ap.add_argument("--out", default="artifacts/marketzero.pt")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="fit only on the first fraction of history (rest is OOS)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    np.random.seed(cfg.train_seed)
    import torch
    torch.manual_seed(cfg.train_seed)
    device = "cpu" if cfg.train_device == "auto" else cfg.train_device

    df, source = load_daily(cfg.data_symbol, cfg.data_start,
                            cfg.resolved_path(cfg.data_cache))
    feats, logret = compute_features(df, cfg.feat_window)
    cut = int(len(df) * args.train_frac)
    scaler = FeatureScaler().fit(feats.to_numpy()[:cut])
    Xs = scaler.transform(feats.to_numpy())
    mu = logret.to_numpy()

    env = TradingEnv(Xs[:cut], mu[:cut], dates=list(df.index[:cut]),
                     fee_rate=cfg.env_fee_rate, n_actions=cfg.env_n_actions)
    net = MuZeroNet(Xs.shape[1] + 1, cfg.env_n_actions, cfg.model_hidden,
                    cfg.model_latent, cfg.model_support).to(device)
    trainer = MuZeroTrainer(net, cfg)
    buffer = ReplayBuffer(capacity=cfg.buffer_capacity, seed=cfg.train_seed)
    rng = np.random.default_rng(cfg.train_seed)

    out_path = cfg.resolved_path(args.out)
    log_path = out_path.with_suffix(".log.jsonl")
    print(f"data={source} bars={len(df)} train=[0..{cut}) params={net.param_count():,} device={device}")
    t0 = time.time()
    for it in range(1, args.iters + 1):
        ep_ret, ep_dd = [], []
        for _ in range(cfg.train_games_per_iter):
            game = play_episode(env, net, cfg, rng, training=True)
            buffer.append(game)
            ep_ret.append(float(np.exp(sum(game.rewards)) - 1.0))
            eq = np.exp(np.cumsum(game.rewards))
            ep_dd.append(float((1 - eq / np.maximum.accumulate(eq)).max()))
        losses = trainer.train_epochs(buffer, cfg.train_epochs_per_iter)
        rec = {"iter": it, "games": cfg.train_games_per_iter,
               "mean_episode_return": round(float(np.mean(ep_ret)), 4),
               "mean_episode_maxdd": round(float(np.mean(ep_dd)), 4),
               "buffer_steps": buffer.total_steps(), "elapsed_s": round(time.time() - t0, 1),
               **{f"loss_{k}": round(v, 5) for k, v in losses.items()},
               "ts": datetime.now(timezone.utc).isoformat()}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  it{it:03d} ep_ret {rec['mean_episode_return']:+.2%} "
              f"ep_maxDD {rec['mean_episode_maxdd']:.1%} "
              f"loss p/v/r {rec['loss_policy']:.3f}/{rec['loss_value']:.3f}/{rec['loss_reward']:.3f} "
              f"[{rec['elapsed_s']}s]")
        if it % 5 == 0 or it == args.iters:
            save_checkpoint(out_path, net, cfg, scaler,
                            {"iterations": it, "source": source})
    print(f"checkpoint -> {out_path}\nlog        -> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
