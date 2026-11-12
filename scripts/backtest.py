"""Walk-forward backtest with purge/embargo — the only results that count (P2 gate).

Fits a fresh agent per window (train on the past, test on the untouched next
window), stitches out-of-sample segments, and reports agent vs buy & hold
net of fees with the risk kill-switch active. Appends to the experiment
registry for multiple-testing honesty.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketzero.config import Config
from marketzero.data import FeatureScaler, compute_features, load_daily
from marketzero.evaluate import walk_forward, walk_forward_windows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tag", default="wf")
    ap.add_argument("--out-dir", default="artifacts")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    np.random.seed(cfg.train_seed)

    df, source = load_daily(cfg.data_symbol, cfg.data_start,
                            cfg.resolved_path(cfg.data_cache))
    feats, logret = compute_features(df, cfg.feat_window)
    # scaler fitted on the first window only (honest: test data unseen)
    first_train_end = next(iter(walk_forward_windows(len(df), cfg)),
                           (0, 730, 740, 830))[1]
    scaler = FeatureScaler().fit(feats.to_numpy()[:first_train_end])
    Xs = scaler.transform(feats.to_numpy())

    print(f"data={source} bars={len(df)}")
    print(f"windows: train {cfg.eval_train_window}d, test {cfg.eval_test_window}d, "
          f"purge/embargo {cfg.eval_purge}d, anchored={cfg.eval_anchored}")
    print(f"budget per window: {cfg.eval_wf_games} games x {cfg.eval_wf_epochs} epochs "
          f"(demo budget -- raise substantially for research)")
    result = walk_forward(Xs, logret.to_numpy(), list(df.index), cfg,
                          cfg.resolved_path(args.out_dir), tag=args.tag)

    print("\n=== stitched out-of-sample summary ===")
    print(f"  agent    : {result['agent']}")
    print(f"  buy&hold : {result['buy_hold']}")
    print(f"  windows  : {result['n_windows']}  (prior registered runs: "
          f"{result.get('prior_registered_runs', 0)})")
    print("  artifacts: equity curve CSV/PNG, summary JSON, experiments.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
