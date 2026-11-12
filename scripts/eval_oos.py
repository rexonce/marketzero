"""Out-of-sample evaluation of a trained checkpoint on the held-out slice.

train.py fits on --train-frac (default 0.7) of history; this replays the
agent over the untouched remainder, net of fees with the risk governor on,
and compares against buy & hold over the same days. One window, not the
full walk-forward gate (scripts/backtest.py) -- a fast sanity read.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketzero import metrics as M
from marketzero.config import Config
from marketzero.data import FeatureScaler, compute_features, load_daily
from marketzero.evaluate import buy_hold, run_policy
from marketzero.trainer import load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="artifacts/marketzero.pt")
    ap.add_argument("--train-frac", type=float, default=0.7,
                    help="fraction the checkpoint trained on (OOS = remainder)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="optional CSV of daily rows")
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, ckpt_cfg, payload = load_checkpoint(
        Config().resolved_path(args.checkpoint), device)
    cfg = Config.from_yaml(args.config) if args.config else ckpt_cfg

    df, source = load_daily(cfg.data_symbol, cfg.data_start,
                            cfg.resolved_path(cfg.data_cache))
    feats, logret = compute_features(df, cfg.feat_window)
    scaler = (FeatureScaler.from_state(payload["scaler"])
              if payload.get("scaler") else None)
    Xs = scaler.transform(feats.to_numpy()) if scaler else feats.to_numpy()
    mu = logret.to_numpy()

    start = int(len(df) * args.train_frac)
    out = run_policy(net, Xs, mu, list(df.index), start, len(df) - 1, cfg,
                     ohlcv=df)
    agent = M.summary(out["reward"].to_numpy())
    bh = M.summary(buy_hold(mu, start, len(df) - 1))
    n_long = float((out.position > 0).mean())
    n_short = float((out.position < 0).mean())
    turnover = float(out.position.diff().abs().sum())

    print(f"checkpoint {args.checkpoint} (iters={payload.get('iterations')}) "
          f"| OOS {out['date'].iloc[0].date()}..{out['date'].iloc[-1].date()} "
          f"({len(out)} days) | device={device}")
    print(f"  agent    : {agent}")
    print(f"  buy&hold : {bh}")
    print(f"  exposure : long {n_long:.0%} | short {n_short:.0%} | "
          f"flat {1 - n_long - n_short:.0%} | turnover {turnover:.1f}x")
    print(f"  fees paid: {out.fee.sum():.4f} log-pts | "
          f"mean |pos|: {out.position.abs().mean():.2f}")
    if args.out:
        out.to_csv(Config().resolved_path(args.out), index=False)
        print(f"  rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
