"""Performance metrics. Daily crypto trades 365 days/year, so periods=365."""
from __future__ import annotations

import numpy as np


def equity_curve(returns: np.ndarray) -> np.ndarray:
    return np.exp(np.cumsum(np.asarray(returns, dtype=np.float64)))


def total_return(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    return float(np.exp(r.sum()) - 1.0)


def sharpe(returns: np.ndarray, periods: int = 365, rf: float = 0.0) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-12:                       # degenerate (constant) stream
        return 0.0
    excess = r - rf / periods
    return float(np.sqrt(periods) * excess.mean() / excess.std(ddof=1))


def max_drawdown(returns: np.ndarray) -> float:
    eq = equity_curve(returns)
    peak = np.maximum.accumulate(eq)
    return float((1.0 - eq / peak).max())


def cagr(returns: np.ndarray, periods: int = 365) -> float:
    r = np.asarray(returns, dtype=np.float64)
    years = len(r) / periods
    if years <= 0:
        return 0.0
    return float(np.exp(r.sum() / years) - 1.0)


def hit_rate(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=np.float64)
    return float((r > 0).mean()) if len(r) else 0.0


def summary(returns: np.ndarray) -> dict:
    return {
        "total_return": round(total_return(returns), 4),
        "cagr": round(cagr(returns), 4),
        "sharpe": round(sharpe(returns), 3),
        "max_drawdown": round(max_drawdown(returns), 4),
        "hit_rate": round(hit_rate(returns), 3),
        "n_days": int(len(returns)),
    }
