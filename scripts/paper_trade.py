"""Paper trading replay: run the trained agent over recent bars, log decisions.

Refuses live mode by default (and live execution is not implemented at all).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marketzero.config import Config
from marketzero.data import compute_features, load_daily
from marketzero.paper import PaperTrader


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="artifacts/marketzero.pt")
    ap.add_argument("--bars", type=int, default=30)
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else None
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trader = PaperTrader(Config().resolved_path(args.checkpoint), cfg,
                         device=device)
    trader.check_live_gates(args.live, os.environ.get("I_UNDERSTAND_THE_RISKS"))
    cfg = trader.cfg  # checkpoint config when --config not given

    df, source = load_daily(cfg.data_symbol, cfg.data_start,
                            cfg.resolved_path(cfg.data_cache))
    feats, logret = compute_features(df, cfg.feat_window)
    print(f"paper replay | data={source} | last {args.bars} bars of {cfg.data_symbol}\n")
    out = trader.run_replay(df, feats, logret, last_n=args.bars,
                            out_csv=cfg.resolved_path("artifacts/paper_decisions.csv"))
    print(out.to_string(index=False))
    print(f"\nrows saved -> artifacts/paper_decisions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
