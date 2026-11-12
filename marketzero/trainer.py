"""MuZero trainer: unroll the learned dynamics K steps, match predictions
against stored search policies, observed rewards and discounted returns."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .buffer import ReplayBuffer
from .config import Config
from .models import MuZeroNet, scalar_to_two_hot


class MuZeroTrainer:
    def __init__(self, net: MuZeroNet, cfg: Config):
        self.net = net
        self.cfg = cfg
        self.opt = torch.optim.AdamW(net.parameters(), lr=cfg.train_lr,
                                     weight_decay=cfg.train_weight_decay)
        self.value_scale = 1.0      # running EMA of std of returns-to-go
        self.step_count = 0

    # ------------------------------------------------------------------ core
    def _value_target(self, z: np.ndarray) -> torch.Tensor:
        """Scale discounted returns by a running std, clip into the support."""
        zs = np.std(z) if len(z) > 1 else self.cfg.train_value_std_floor
        zs = max(zs, self.cfg.train_value_std_floor)
        self.value_scale = 0.9 * self.value_scale + 0.1 * zs
        scaled = np.clip(z / self.value_scale, -0.999, 0.999)
        return scalar_to_two_hot(torch.as_tensor(scaled, dtype=torch.float32),
                                 self.cfg.model_support)

    def _build_targets(self, batch: list, unroll: int):
        """Targets aligned to a K-step unroll (canonical MuZero conventions).

        Our game arrays are departure-indexed: rewards[i] is the reward
        *produced by* actions[i] (arriving at position i+1). Unroll step k
        consumes actions[t+k-1] and arrives at position t+k, so:
        - policy target = MCTS visits at position t+k
        - value target  = return-to-go at position t+k
        - reward target = rewards[t+k-1]  (the reward INTO position t+k)
        """
        T_pol, T_val, T_rew, T_mask = [], [], [], []
        for k in range(unroll + 1):
            pols, zs, rews, mask = [], [], [], []
            for g, t in batch:
                idx = t + k
                inb = idx < len(g)
                mask.append(1.0 if inb else 0.0)
                pols.append(g.policies[idx] if inb else g.policies[-1])
                zs.append(g.returns[idx] if inb else g.returns[-1])
                ridx = idx - 1                      # reward INTO position idx
                rews.append(g.rewards[ridx] if 0 <= ridx < len(g) else 0.0)
            T_pol.append(torch.as_tensor(np.stack(pols), dtype=torch.float32))
            T_val.append(self._value_target(np.array(zs)))
            T_rew.append(torch.as_tensor(np.clip(np.array(rews), -0.999, 0.999),
                                         dtype=torch.float32))
            T_mask.append(torch.as_tensor(np.array(mask), dtype=torch.float32))
        return T_pol, T_val, T_rew, T_mask

    def train_step(self, buffer: ReplayBuffer) -> dict[str, float]:
        self.net.train()
        cfg = self.cfg
        device = next(self.net.parameters()).device
        batch = buffer.sample_batch(cfg.train_batch_size, cfg.train_unroll)
        obs0 = np.stack([g.obs[t] for g, t in batch])
        obs0 = torch.as_tensor(obs0, dtype=torch.float32, device=device)

        T_pol, T_val, T_rew, T_mask = self._build_targets(batch, cfg.train_unroll)
        T_pol = [t.to(device) for t in T_pol]
        T_val = [t.to(device) for t in T_val]
        T_rew = [t.to(device) for t in T_rew]
        T_mask = [t.to(device) for t in T_mask]

        # canonical weighting: step 0 counts 1.0; each unroll step counts 1/K
        step_w = [1.0] + [1.0 / cfg.train_unroll] * cfg.train_unroll
        total = torch.zeros((), device=device)
        agg = {"policy": 0.0, "value": 0.0, "reward": 0.0}
        s = self.net.rep(obs0)
        for k in range(cfg.train_unroll + 1):
            if k > 0:
                # dynamics step with the action stored at t+k-1
                acts = torch.as_tensor([g.actions[t + k - 1] if t + k - 1 < len(g)
                                        else g.actions[-1] for g, t in batch],
                                       device=device)
                onehot = F.one_hot(acts, self.net.n_actions).float()
                s, r_logits = self.net.dyn(s, onehot)
                s.register_hook(lambda grad: grad * 0.5)  # MuZero appendix
            p_logits, v_logits = self.net.pred(s)
            m = T_mask[k]
            msum = m.sum().clamp_min(1.0)
            lp = (-(T_pol[k] * F.log_softmax(p_logits, dim=-1)).sum(-1) * m).sum() / msum
            lv = (-(T_val[k] * F.log_softmax(v_logits, dim=-1)).sum(-1) * m).sum() / msum
            total = total + step_w[k] * (cfg.train_w_policy * lp
                                         + cfg.train_w_value * lv)
            agg["policy"] += float(lp.detach()) / (cfg.train_unroll + 1)
            agg["value"] += float(lv.detach()) / (cfg.train_unroll + 1)
            if k > 0:
                r_target = scalar_to_two_hot(T_rew[k], cfg.model_support)
                lr_ = (-(r_target * F.log_softmax(r_logits, dim=-1)).sum(-1)
                       * m).sum() / msum
                total = total + step_w[k] * cfg.train_w_reward * lr_
                agg["reward"] += float(lr_.detach()) / cfg.train_unroll

        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
        self.opt.step()
        self.step_count += 1
        return {
            "policy": agg["policy"], "value": agg["value"],
            "reward": agg["reward"], "total": float(total.item()),
        }

    def train_epochs(self, buffer: ReplayBuffer, epochs: int) -> dict[str, float]:
        agg: dict[str, float] = {}
        for _ in range(epochs):
            losses = self.train_step(buffer)
            for k, v in losses.items():
                agg[k] = agg.get(k, 0.0) + v / epochs
        return agg


# --------------------------------------------------------------- checkpoints
def save_checkpoint(path: Path, net: MuZeroNet, cfg: Config, scaler, extra: dict | None = None):
    payload = {
        "state_dict": net.state_dict(),
        "config": cfg.to_dict(),
        "scaler": scaler.state() if scaler else None,
        "obs_dim": net.rep.net[0].in_features,
        "n_actions": net.n_actions,
        "param_count": net.param_count(),
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[MuZeroNet, Config, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = Config.from_dict(payload["config"])
    net = MuZeroNet(payload["obs_dim"], payload["n_actions"],
                    cfg.model_hidden, cfg.model_latent, cfg.model_support)
    net.load_state_dict(payload["state_dict"])
    net.to(device).eval()
    return net, cfg, payload
