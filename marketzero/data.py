"""Data pipeline: fetch, cache, synthesize, feature-engineer, scale.

Primary source is the public Yahoo Finance chart API (works where Binance is
geo-blocked); a regime-switching synthetic generator is the offline fallback.

Causality contract: every feature at row i uses bars with index <= i. The
agent decides at the close of day i and its position earns mu[i+1], the
log-return from close i to close i+1. No feature ever looks ahead.
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume"]


# --------------------------------------------------------------------- fetch
def fetch_yahoo_daily(symbol: str, start: str, end: str | None = None,
                      retries: int = 3, timeout: int = 20) -> pd.DataFrame:
    """Daily OHLCV for `symbol` from the public Yahoo chart API."""
    t1 = int(pd.Timestamp(start).timestamp())
    t2 = int(pd.Timestamp(end).timestamp() if end else pd.Timestamp.now().timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"period1": t1, "period2": t2, "interval": "1d"}
    headers = {"User-Agent": "Mozilla/5.0 (marketzero research)"}
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            payload = r.json()["chart"]["result"][0]
            ts = payload["timestamp"]
            quote = payload["indicators"]["quote"][0]
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None),
                    "open": quote["open"],
                    "high": quote["high"],
                    "low": quote["low"],
                    "close": quote["close"],
                    "volume": quote["volume"],
                }
            ).dropna(subset=["close"])
            return df.set_index("date").astype(float).sort_index()
        except Exception as exc:  # noqa: BLE001 - retry any network hiccup
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise ConnectionError(f"Yahoo fetch failed for {symbol}: {last_err}")


def synthetic_daily(n_days: int, seed: int = 7) -> pd.DataFrame:
    """Regime-switching fat-tailed synthetic OHLCV (offline fallback).

    Bull/bear drift regimes via a Markov chain, volatility clustering via an
    AR(1) log-vol process, plus rare jump days. Captures the stylized facts
    that matter for a trading env (fat tails, vol clustering, regime shifts);
    it is NOT real market data and results on it prove wiring only.
    """
    rng = np.random.default_rng(seed)
    n = n_days
    # drift regime chain: 0=bull, 1=bear
    mu_bull, mu_bear = 0.0022, -0.0026
    p_stay = 0.985
    logv = np.zeros(n)          # log volatility state
    mu = np.zeros(n)
    regime = 0
    for t in range(n):
        mu[t] = mu_bull if regime == 0 else mu_bear
        if rng.random() > p_stay:
            regime = 1 - regime
        logv[t] = 0.90 * logv[t - 1] + 0.10 * np.log(0.025) + 0.12 * rng.standard_normal()
    ret = mu + np.exp(logv) * rng.standard_normal(n)
    jumps = rng.random(n) < 0.008                       # rare fat-tail days
    ret += jumps * rng.normal(0.0, 0.09, n)
    close = 100.0 * np.exp(np.cumsum(ret))
    intraday = np.abs(np.random.default_rng(seed + 1).normal(0, 0.012, n))
    df = pd.DataFrame(
        {
            "open": close * np.exp(-intraday * 0.5),
            "high": close * np.exp(np.abs(intraday)),
            "low": close * np.exp(-np.abs(intraday)),
            "close": close,
            "volume": rng.integers(1e4, 1e6, n).astype(float),
        },
        index=pd.date_range("2015-01-01", periods=n, freq="D", name="date"),
    )
    return df


def load_daily(symbol: str, start: str, cache_path, force_synthetic: bool = False,
               synthetic_days: int = 3000, synthetic_seed: int = 7) -> tuple[pd.DataFrame, str]:
    """Load OHLCV from cache or network; returns (df, source_tag)."""
    if not force_synthetic and cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if len(df) > 200:
            return df, "cache"
    if force_synthetic:
        df = synthetic_daily(synthetic_days, synthetic_seed)
        return df, "synthetic"
    try:
        df = fetch_yahoo_daily(symbol, start)
        buf = io.StringIO()
        df.to_csv(buf)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(buf.getvalue(), encoding="utf-8")
        return df, "yahoo"
    except ConnectionError:
        df = synthetic_daily(synthetic_days, synthetic_seed)
        return df, "synthetic (network unavailable)"


# ------------------------------------------------------------------ features
def compute_features(df: pd.DataFrame, feat_window: int = 8) -> pd.DataFrame:
    """Causal feature frame; row i sees bars <= i only."""
    c = df["close"]
    logret = np.log(c / c.shift(1))
    feats = pd.DataFrame(index=df.index)
    # recent tone: last `feat_window` daily log-returns
    for k in range(1, feat_window + 1):
        feats[f"ret_lag{k}"] = logret.shift(k)
    # trend: price vs moving averages
    for w in (20, 50, 200):
        sma = c.rolling(w).mean()
        feats[f"px_vs_sma{w}"] = c / sma - 1.0
    # volatility
    feats["vol20"] = logret.rolling(20).std()
    prev_close = c.shift(1)
    tr = pd.concat(
        [df["high"] - df["low"],
         (df["high"] - prev_close).abs(),
         (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    feats["atr14"] = tr.rolling(14).mean() / c
    # momentum oscillators
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    feats["rsi14"] = (gain / (gain + loss + 1e-12)) - 0.5
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feats["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / (c * feats["vol20"] + 1e-9)
    # range structure
    lo, hi = c.rolling(20).min(), c.rolling(20).max()
    feats["range20_pos"] = (c - lo) / (hi - lo + 1e-9) - 0.5
    feats = feats.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return feats, logret.fillna(0.0)


@dataclass
class FeatureScaler:
    """Standardization fitted on the training window only (leakage discipline)."""
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, feats: np.ndarray) -> "FeatureScaler":
        self.mean = feats.mean(axis=0)
        self.std = np.maximum(feats.std(axis=0), 1e-8)
        return self

    def transform(self, feats: np.ndarray) -> np.ndarray:
        return ((feats - self.mean) / self.std).astype(np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, d: dict) -> "FeatureScaler":
        return cls(np.array(d["mean"], dtype=np.float64), np.array(d["std"], dtype=np.float64))
