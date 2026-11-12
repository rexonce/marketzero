"""Fetch daily OHLCV from Yahoo Finance into the CSV cache (real data pipeline)."""
from __future__ import annotations

import argparse
from pathlib import Path

from marketzero import metrics as M
from marketzero.config import Config
from marketzero.data import compute_features, load_daily


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--start", default=None)
    ap.add_argument("--force", action="store_true", help="bypass cache")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.symbol:
        cfg.data_symbol = args.symbol
    if args.start:
        cfg.data_start = args.start

    cache = cfg.resolved_path(cfg.data_cache)
    if args.force:
        # true bypass: fetch fresh and rewrite the cache
        from marketzero.data import fetch_yahoo_daily
        try:
            df = fetch_yahoo_daily(cfg.data_symbol, cfg.data_start)
            cache.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache)
            source = "yahoo (forced refresh)"
        except ConnectionError as exc:
            print(f"fetch failed ({exc}); falling back to cache")
            df, source = load_daily(cfg.data_symbol, cfg.data_start, cache)
    else:
        df, source = load_daily(cfg.data_symbol, cfg.data_start, cache,
                                force_synthetic=False)
    df = df[df.index >= cfg.data_start]
    feats, logret = compute_features(df, cfg.feat_window)
    ret = logret.to_numpy()[1:]
    print(f"source      : {source}")
    print(f"symbol      : {cfg.data_symbol}")
    print(f"bars        : {len(df)}  ({df.index[0].date()} .. {df.index[-1].date()})")
    print(f"features    : {feats.shape[1]} columns")
    print(f"close stats : first {df['close'].iloc[0]:.2f}, last {df['close'].iloc[-1]:.2f}")
    print(f"B&H summary : {M.summary(ret)}")
    print(f"worst day   : {ret.min():.2%}   best day: {ret.max():+.2%}")


if __name__ == "__main__":
    main()
