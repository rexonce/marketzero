"""Paper trader: replay recent bars with the trained agent, log decisions.

Live mode is refused by a triple gate AND not implemented at all — this repo
executes no orders, by design. Real capital requires the roadmap gates in
docs/DEEP_DIVE.md §10 (including 6 months of profitable paper trading), an
exchange integration this repo deliberately omits, and your explicit signoff.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import FeatureScaler
from .environment import TradingEnv, flat_action
from .mcts import run_mcts
from .risk import RiskGovernor
from .trainer import load_checkpoint


class PaperTrader:
    def __init__(self, checkpoint: Path, cfg: Config | None = None, device: str = "cpu"):
        self.net, ckpt_cfg, payload = load_checkpoint(checkpoint, device)
        self.cfg = cfg or ckpt_cfg
        for f in ("env_n_actions", "feat_window", "model_support", "obs_window"):
            if getattr(self.cfg, f) != getattr(ckpt_cfg, f):
                raise ValueError(
                    f"config/checkpoint mismatch on {f}: "
                    f"{getattr(self.cfg, f)} != {getattr(ckpt_cfg, f)} -- "
                    "run without --config to use the checkpoint's config")
        self.scaler = FeatureScaler.from_state(payload["scaler"]) if payload.get("scaler") else None
        self.gov = RiskGovernor(self.cfg)

    # ------------------------------------------------------------------ gates
    def check_live_gates(self, requested_live: bool, env_var: str | None) -> None:
        problems = []
        if requested_live:
            problems.append("--live was passed")
        if self.cfg.allow_live:
            problems.append("config allow_live=true")
        if env_var == "1":
            problems.append("I_UNDERSTAND_THE_RISKS=1 is set")
        enabled = [p for p in problems if p]
        if enabled:
            raise SystemExit(
                "REFUSING LIVE MODE. Live gates triggered: "
                + "; ".join(enabled)
                + ". This build cannot trade: order execution is not implemented. "
                "Paper mode only (run without --live)."
            )

    # ------------------------------------------------------------------- main
    def run_replay(self, df: pd.DataFrame, features: pd.DataFrame, logrets: pd.Series,
                   last_n: int = 30, out_csv: Path | None = None) -> pd.DataFrame:
        feats = features.to_numpy()
        if self.scaler:
            feats = self.scaler.transform(feats)
        mu = logrets.to_numpy()
        start = max(1, len(df) - last_n)
        env = TradingEnv(feats, mu, dates=list(df.index), fee_rate=self.cfg.env_fee_rate,
                         n_actions=self.cfg.env_n_actions,
                         obs_window=self.cfg.obs_window)
        obs = env.reset(start=start, episode_len=len(df) - start - 1)
        rows, done = [], False
        while not done:
            res = run_mcts(self.net, obs, self.cfg, training=False)
            action = res["action"]
            scale = (self.gov.position_scale(env.w)
                     if self.cfg.risk_apply_vol_target else 1.0)
            obs, reward, done, info = env.step(action, size_scale=scale)
            self.gov.observe_return(reward)
            ok, reasons = self.gov.check(info)
            if not ok:
                env.force_flat()
            pos = info["position"]
            rows.append({
                "date": info["date"],
                "decision": ("SHORT" if pos < 0 else "LONG" if pos > 0 else "FLAT"),
                "size_scale": round(scale, 2),
                "confidence": round(float(res["policy"].max()), 3),
                "daily_pnl": round(reward, 5),
                "equity": round(info["equity"], 4),
                "kill_switch": "" if ok else "HALT: " + "; ".join(reasons),
            })
        out = pd.DataFrame(rows)
        if out_csv:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(out_csv, index=False)
        return out
