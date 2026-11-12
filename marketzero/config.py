"""MarketZero configuration.

A single flat dataclass with prefixed fields (data_, env_, model_, mcts_,
train_, eval_, risk_) loaded from YAML. Flat keeps YAML, CLI overrides and
checkpoints dead simple.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    # --- data ---------------------------------------------------------------
    data_symbol: str = "BTC-USD"
    data_start: str = "2014-01-01"          # first daily bar to request
    data_cache: str = "data/btc_usd_daily.csv"  # relative to repo root
    data_force_synthetic: bool = False      # use synthetic generator instead
    data_synthetic_days: int = 3000
    data_synthetic_seed: int = 7

    # --- features / environment ----------------------------------------------
    feat_window: int = 8                    # lagged daily log-returns in obs
    obs_window: int = 64                    # bars of history per observation
                                          # (<=1 = flat feature vector mode)
    env_fee_rate: float = 0.001             # proportional fee per unit turnover
    env_episode_len: int = 252              # trading days per self-play episode
    env_n_actions: int = 3                  # 3: {-1,0,+1}, 5: {-.5 steps}, 2: {0,+1}
    env_gamma: float = 0.997                # daily discount (~1y horizon)

    # --- MuZero networks ------------------------------------------------------
    model_hidden: int = 128
    model_latent: int = 64
    model_support: int = 10                 # value/reward bins = 2*10+1 on [-1,1]
    model_tx_layers: int = 2                # transformer encoder layers in h
                                          # (windowed-obs mode only)

    # --- MCTS -----------------------------------------------------------------
    mcts_sims: int = 32                     # simulations per move (training)
    mcts_eval_sims: int = 32                # simulations per move (evaluation)
    mcts_c_puct: float = 1.25
    mcts_dirichlet_alpha: float = 0.3
    mcts_dirichlet_epsilon: float = 0.25
    mcts_temperature: float = 1.0           # visit-count sampling temp (training)
    mcts_temp_explore_steps: int = 60       # episode steps at tau=temperature
    mcts_temp_late: float = 0.25            # tau after explore steps (0 = argmax)
    mcts_max_depth: int = 4                 # tree depth cap (0 = unbounded)
    mcts_chance_scenarios: int = 0          # >0: Phase-2 scenario rerank at eval
    mcts_chance_horizon: int = 4            # scenario planning horizon (bars)
    mcts_chance_weight: float = 0.25        # fraction of sims the rerank can shift
    mcts_chance_sampler: str = "bootstrap"  # bootstrap | kronos
    kronos_model: str = "NeoQuasar/Kronos-mini"  # HF repo id (mini=4M, base=102M)
    mcts_leaf_batch: int = 8                # leaf evals batched per wave (1=serial)

    # --- training -------------------------------------------------------------
    train_games_per_iter: int = 8
    train_epochs_per_iter: int = 10
    train_batch_size: int = 128
    train_unroll: int = 5                   # dynamics unroll steps K
    train_lr: float = 1e-3
    train_weight_decay: float = 1e-4
    train_value_std_floor: float = 0.05
    train_w_value: float = 1.0              # loss weights (blueprint v:p:r = 1:1:0.5)
    train_w_policy: float = 1.0
    train_w_reward: float = 0.5
    train_seed: int = 42
    train_device: str = "auto"              # auto = cuda when available
    train_reanalyze_games: int = 1          # buffer games re-searched per iter
    train_td_steps: int = 0                 # 0 = full MC returns; >0 = n-step
                                          # bootstrap with MCTS root values
    buffer_capacity: int = 512              # replay buffer capacity (games)
    buffer_priority_alpha: float = 0.0      # 0 = uniform replay; ~1 canonical
    buffer_is_beta: float = 0.5             # IS-correction exponent (0 = off)

    # --- walk-forward evaluation ----------------------------------------------
    eval_train_window: int = 730            # days used to fit per window
    eval_test_window: int = 90              # untouched out-of-sample days
    eval_purge: int = 10                    # purge+embargo days around boundary
    eval_anchored: bool = False             # expanding (True) vs rolling train
    eval_wf_games: int = 6                  # self-play games per window fit
    eval_wf_epochs: int = 8                 # training epochs per window fit
    eval_warm_start: bool = False           # carry net+buffer across windows
                                          # (production-style periodic retrain;
                                          # still strictly causal per window)

    # --- risk -------------------------------------------------------------------
    risk_max_drawdown: float = 0.20         # kill-switch: episode DD from peak
    risk_max_daily_loss: float = 0.08       # kill-switch: single-day loss
    risk_vol_target: float | None = 0.15    # annualized vol target (None=off)
    risk_apply_vol_target: bool = True      # actually scale positions in eval/paper

    # --- live gating ------------------------------------------------------------
    allow_live: bool = False                # paper trader refuses live unless True

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Config":
        known = {f.name for f in fields(cls)}
        picks = {k: v for k, v in raw.items() if k in known}
        return cls(**picks)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p
