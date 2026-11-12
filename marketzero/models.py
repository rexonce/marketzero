"""MuZero networks: representation h, dynamics g, prediction f.

Values and rewards are predicted as categorical distributions over a
discretized support of 2*support+1 bins spanning [-1, 1]; scalars are encoded
as two-hot linear interpolations of adjacent bins and decoded as the softmax
expectation. Cross-entropy on a bounded support is far more stable than MSE
on raw scalars and handles multi-modal targets (see docs/DEEP_DIVE.md §3.2).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ support
def support_grid(support: int, device=None) -> torch.Tensor:
    return torch.linspace(-support, support, 2 * support + 1, device=device) / support


def scalar_to_two_hot(x: torch.Tensor, support: int) -> torch.Tensor:
    """Encode scalars in [-1, 1] as two-hot distributions over the grid."""
    x = x.clamp(-0.999, 0.999)
    n = 2 * support + 1
    pos = (x + 1.0) * 0.5 * (n - 1)          # fractional bin index
    lo = pos.floor().long()
    frac = pos - lo.float()
    out = torch.zeros(*x.shape, n, device=x.device)
    out.scatter_(-1, lo.unsqueeze(-1), (1.0 - frac).unsqueeze(-1))
    out.scatter_add_(-1, (lo + 1).clamp(max=n - 1).unsqueeze(-1), frac.unsqueeze(-1))
    return out


def logits_to_scalar(logits: torch.Tensor, support: int) -> torch.Tensor:
    grid = support_grid(support, device=logits.device)
    return (F.softmax(logits, dim=-1) * grid).sum(dim=-1)


def two_hot_numpy(x: float, support: int) -> np.ndarray:
    t = scalar_to_two_hot(torch.tensor([float(x)]), support)
    return t[0].numpy()


# ------------------------------------------------------------------ networks
class RepresentationNet(nn.Module):
    """h(obs) -> latent state (flat observation vector)."""

    def __init__(self, obs_dim: int, hidden: int, latent: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )

    def forward(self, obs):
        return self.net(obs)


class TransformerRepresentation(nn.Module):
    """h(obs) -> latent state for windowed observations.

    The observation is a W x F window of per-bar features plus the current
    exposure, flattened to (W*F + 1). Each bar becomes a token; the account
    state is injected as a broadcast embedding (blueprint §7), and attention
    over bars lets the model locate regime structure without hand-coding."""

    def __init__(self, n_feat: int, window: int, hidden: int, latent: int,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.n_feat, self.window = n_feat, window
        self.token = nn.Linear(n_feat, hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, window, hidden))
        self.acct = nn.Linear(1, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
            batch_first=True, norm_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, latent))
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, obs):
        B = obs.shape[0]
        x = obs[:, : self.window * self.n_feat].reshape(B, self.window, self.n_feat)
        w = obs[:, self.window * self.n_feat:]
        t = self.token(x) + self.pos_emb + self.acct(w).unsqueeze(1)
        h = self.encoder(t)
        return self.head(h.mean(dim=1))


class DynamicsNet(nn.Module):
    """g(latent, action_onehot) -> (next latent, reward logits)."""

    def __init__(self, latent: int, n_actions: int, hidden: int, support: int):
        super().__init__()
        self.latent, self.n_actions = latent, n_actions
        self.body = nn.Sequential(
            nn.Linear(latent + n_actions, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_state = nn.Linear(hidden, latent)
        self.head_reward = nn.Linear(hidden, 2 * support + 1)

    def forward(self, state, action_onehot):
        h = self.body(torch.cat([state, action_onehot], dim=-1))
        return self.head_state(h), self.head_reward(h)


class PredictionNet(nn.Module):
    """f(latent) -> (policy logits, value logits)."""

    def __init__(self, latent: int, n_actions: int, hidden: int, support: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head_policy = nn.Linear(hidden, n_actions)
        self.head_value = nn.Linear(hidden, 2 * support + 1)

    def forward(self, state):
        h = self.body(state)
        return self.head_policy(h), self.head_value(h)


class MuZeroNet(nn.Module):
    """The complete MuZero model with batched inference helpers.

    Windowed mode (window > 1): obs is a flattened (window * n_feat + 1)
    vector and h is a transformer encoder over bar tokens.
    Flat mode (window <= 1): obs is a (n_feat + 1) vector and h is an MLP."""

    def __init__(self, obs_dim: int, n_actions: int = 3,
                 hidden: int = 128, latent: int = 64, support: int = 10,
                 window: int = 0, n_feat: int = 0, tx_layers: int = 2):
        super().__init__()
        self.n_actions = n_actions
        self.support = support
        self.window = window
        self.obs_dim = obs_dim
        self.n_feat = n_feat if n_feat else obs_dim - 1
        if window > 1:
            self.rep = TransformerRepresentation(n_feat, window, hidden, latent,
                                                 n_layers=tx_layers)
        else:
            self.rep = RepresentationNet(obs_dim, hidden, latent)
        self.dyn = DynamicsNet(latent, n_actions, hidden, support)
        self.pred = PredictionNet(latent, n_actions, hidden, support)

    # -- inference ------------------------------------------------------------
    def initial_inference(self, obs: torch.Tensor):
        """Root inference: returns (policy_logits, value, reward=0, latent)."""
        s = self.rep(obs)
        p_logits, v_logits = self.pred(s)
        value = logits_to_scalar(v_logits, self.support)
        return p_logits, value, torch.zeros_like(value), s

    def recurrent_inference(self, state: torch.Tensor, action):
        """One g+f step from a latent state. `action` is an int for a single
        transition or a 1-D tensor of ints for a batch of transitions."""
        single = state.dim() == 1
        if single:
            state = state.unsqueeze(0)
        if isinstance(action, int):
            action = torch.full((state.shape[0],), action,
                                dtype=torch.long, device=state.device)
        onehot = F.one_hot(action.to(state.device), self.n_actions).float()
        s_next, r_logits = self.dyn(state, onehot)
        p_logits, v_logits = self.pred(s_next)
        value = logits_to_scalar(v_logits, self.support)
        reward = logits_to_scalar(r_logits, self.support)
        if single:
            return p_logits[0], value[0], reward[0], s_next[0]
        return p_logits, value, reward, s_next

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
