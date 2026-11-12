"""Scenario sampling: Kronos (foundation model) or block-bootstrap fallback.

Why scenarios at all? A deterministic dynamics net g predicts one future;
markets are a distribution of futures. Chance-node planning (Phase 2) scores
root actions against *many* sampled market paths, so fat tails and regime
variety enter the search before they enter the P&L.

- `BootstrapScenarioSampler`: stationary block bootstrap over historical
  log-return blocks. Needs no extra dependency; preserves fat tails and
  volatility clustering (not nonlinear structure).
- `KronosScenarioSampler`: calls the open-source Kronos K-line foundation
  model (arXiv 2508.02739) to sample plausible future OHLCV paths when the
  `kronos` package + weights are installed (pip install kronos; weights
  `NeoQuasar/Kronos-base` / `-mini` from HuggingFace).

Both expose `sample(mu_history, n_scenarios, horizon) -> (n, horizon)` array
of daily log-return paths, so downstream code is sampler-agnostic.
"""
from __future__ import annotations

import numpy as np

from .environment import action_to_position


class BootstrapScenarioSampler:
    """Stationary block bootstrap over historical return blocks."""

    def __init__(self, block_len: int = 20, seed: int = 11):
        self.block_len = block_len
        self.rng = np.random.default_rng(seed)

    def sample(self, mu_history: np.ndarray, n_scenarios: int, horizon: int) -> np.ndarray:
        mu = np.asarray(mu_history, dtype=np.float64)
        L = min(self.block_len, max(2, len(mu) // 4))
        n_blocks = int(np.ceil(horizon / L))
        starts = self.rng.integers(0, len(mu) - L + 1, size=(n_scenarios, n_blocks))
        idx = starts[:, :, None] + np.arange(L)[None, None, :]
        paths = mu[idx].reshape(n_scenarios, n_blocks * L)[:, :horizon]
        return paths


class KronosScenarioSampler:
    """Kronos-based scenario sampler (optional heavy dependency).

    Uses the vendored upstream code in ``marketzero/vendor/kronos_model``
    (falling back to an installed ``kronos`` package). Weights are pulled
    from HuggingFace (NeoQuasar/Kronos-*) on first use. Each scenario is one
    stochastic ``predict`` call; sampled close paths become log-return paths.
    """

    def __init__(self, model_name: str = "NeoQuasar/Kronos-mini",
                 tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-2k",
                 pred_len: int = 8, temperature: float = 1.0,
                 top_p: float = 0.9, device: str = "cpu",
                 max_context: int = 512):
        try:
            from .vendor.kronos_model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError:
            try:
                from kronos import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "Kronos not available: the vendored copy in "
                    "marketzero/vendor/kronos_model could not be imported "
                    "(needs torch + huggingface_hub). Install the 'kronos' "
                    "package, or fall back to BootstrapScenarioSampler."
                ) from exc
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
        model = Kronos.from_pretrained(model_name)
        self.predictor = KronosPredictor(model, tokenizer, device=device,
                                         max_context=max_context)
        self.pred_len = pred_len
        self.temperature = temperature
        self.top_p = top_p

    def sample(self, df, n_scenarios: int, horizon: int, x_timestamp=None,
               y_timestamp=None) -> np.ndarray:
        """df: OHLCV DataFrame with columns open/high/low/close/volume and a
        DatetimeIndex. Returns (n_scenarios, horizon) daily log-return paths."""
        import pandas as pd
        ctx = min(len(df), self.predictor.max_context)
        df_ctx = df.iloc[-ctx:]
        x_ts = x_timestamp if x_timestamp is not None else df_ctx.index
        if y_timestamp is None:
            last = pd.Timestamp(df_ctx.index[-1])
            y_timestamp = pd.date_range(last + pd.Timedelta(days=1),
                                        periods=self.pred_len, freq="D")
        paths = []
        for _ in range(n_scenarios):
            pred_df = self.predictor.predict(
                df_ctx, x_timestamp=x_ts, y_timestamp=y_timestamp,
                pred_len=self.pred_len, T=self.temperature,
                top_p=self.top_p, sample_count=1, verbose=False,
            )
            closes = np.asarray(pred_df["close"], dtype=np.float64)
            # include the last real close -> first predicted close return
            prev = np.concatenate([[df_ctx["close"].iloc[-1]], closes[:-1]])
            paths.append(np.log(closes / prev))
        logpaths = np.stack(paths)                       # (n, pred_len)
        reps = int(np.ceil(horizon / max(1, logpaths.shape[1])))
        return np.tile(logpaths, (1, reps))[:, :horizon]


def get_sampler(kind: str = "bootstrap", **kwargs):
    if kind == "bootstrap":
        return BootstrapScenarioSampler(**kwargs)
    if kind == "kronos":
        return KronosScenarioSampler(**kwargs)
    raise ValueError(f"unknown sampler kind: {kind}")


def scenario_q(net, obs: np.ndarray, actions: list[int], sampler, mu_history: np.ndarray,
               horizon: int, n_scenarios: int, cfg) -> dict[int, float]:
    """Chance-aware Q at the root's first ply (Phase-2 experimental hook).

    For each root action and each sampled market path, unroll the dynamics
    greedily while crediting the *scenario's* returns under the position
    schedule chosen; the discounted sum gives a per-scenario value, averaged
    across scenarios. Blend with MCTS's own Q downstream (search stays the
    primary teacher). Honest label: experimental; full chance-node MCTS is a
    roadmap item, not shipping code.
    """
    import torch

    device = next(net.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    scen = sampler.sample(mu_history, n_scenarios, horizon)
    fees = cfg.env_fee_rate
    q: dict[int, float] = {}
    with torch.no_grad():
        s0 = net.rep(obs_t)
        for a in actions:
            vals = np.zeros(n_scenarios)
            for i in range(n_scenarios):
                s = s0
                w = float(obs[-1])  # current exposure from the observation
                acc, disc = 0.0, 1.0
                for h in range(horizon):
                    a_h = a if h == 0 else int(torch.argmax(net.pred(s)[0]).item())
                    w_new = action_to_position(a_h, net.n_actions)
                    # env semantics: the position chosen now earns this bar's
                    # return and pays fees on the turnover into it
                    acc += disc * (w_new * scen[i, h] - fees * abs(w_new - w))
                    w = w_new
                    disc *= cfg.env_gamma
                    onehot = torch.zeros(1, net.n_actions, device=device)
                    onehot[0, a_h] = 1.0
                    s = net.dyn(s, onehot)[0]
                vals[i] = acc
            q[a] = float(vals.mean())
    return q
