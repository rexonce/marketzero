"""PUCT Monte-Carlo Tree Search over the learned model.

Each simulation: select children maximizing
    Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
expand the first-reached leaf via g, evaluate it with f, and back up
(reward + discounted value) along the path. Root priors get Dirichlet
noise during training for exploration. The improved policy is the root
visit distribution -- search output is the network's teacher.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch

from .models import MuZeroNet


class Node:
    __slots__ = ("prior", "visit_count", "value_sum", "reward", "children",
                 "hidden", "expanded", "depth")

    def __init__(self, prior: float = 0.0):
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.reward = 0.0            # predicted reward on the edge INTO this node
        self.children: dict[int, Node] = {}
        self.hidden: Optional[torch.Tensor] = None
        self.expanded = False
        self.depth = 0               # plies below the root

    def q(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expand(self, priors: np.ndarray):
        self.expanded = True
        self.children = {a: Node(float(priors[a])) for a in range(len(priors))}
        for ch in self.children.values():
            ch.depth = self.depth + 1


class MinMaxStats:
    """Tracks the min/max of backed-up values across the tree so PUCT compares
    Q values on a normalized [0,1] scale (MuZero paper, appendix)."""

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value: float) -> None:
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def _puct(child: Node, parent_visits: int, c_puct: float, mm: MinMaxStats) -> float:
    u = c_puct * child.prior * math.sqrt(parent_visits) / (1.0 + child.visit_count)
    return mm.normalize(child.q()) + u


def backpropagate(path: list[Node], value: float, gamma: float,
                  mm: MinMaxStats | None = None) -> None:
    """Each node accumulates the discounted rewards below it plus leaf value."""
    for node in reversed(path):
        node.value_sum += value
        node.visit_count += 1
        if mm is not None:
            mm.update(node.q())
        value = node.reward + gamma * value


def run_mcts(net: MuZeroNet, obs: np.ndarray, cfg, training: bool = True,
             sims: int | None = None) -> dict:
    """Run a search from a raw observation. Returns action, improved policy,
    root value and the root node (for analysis / acting)."""
    net.eval()
    device = next(net.parameters()).device
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    n_sims = sims or (cfg.mcts_sims if training else cfg.mcts_eval_sims)

    with torch.no_grad():
        p_logits, root_value, _, root_hidden = net.initial_inference(obs_t)
        root = Node()
        root.hidden = root_hidden[0]
        root.expand(torch.softmax(p_logits, dim=-1)[0].cpu().numpy())
        if training and cfg.mcts_dirichlet_epsilon > 0:
            noise = np.random.dirichlet([cfg.mcts_dirichlet_alpha] * net.n_actions)
            eps = cfg.mcts_dirichlet_epsilon
            for a, node in root.children.items():
                node.prior = (1 - eps) * node.prior + eps * noise[a]

        mm = MinMaxStats()
        max_depth = cfg.mcts_max_depth or 0
        for _ in range(n_sims):
            node, path, new_node = root, [root], None
            # selection down to first unexpanded child, honoring the depth cap
            while node.expanded and (max_depth <= 0 or node.depth < max_depth):
                a = max(node.children, key=lambda k: _puct(node.children[k], node.visit_count, cfg.mcts_c_puct, mm))
                child = node.children[a]
                path.append(child)
                if not child.expanded:
                    new_node = child
                    break
                node = child
            if new_node is not None:
                # expand the freshly reached leaf: g from parent, then f for priors
                parent = path[-2]
                a = [k for k, v in parent.children.items() if v is new_node][0]
                p_l, v, r, s_next = net.recurrent_inference(parent.hidden, a)
                new_node.hidden = s_next          # 1-D latent, batch rank handled in net
                new_node.reward = float(r)
                new_node.expand(torch.softmax(p_l, dim=-1).cpu().numpy())
                leaf_value = float(v)
            else:
                # depth-capped node (or fully expanded leaf): bootstrap its Q
                leaf_value = path[-1].q()
            backpropagate(path, leaf_value, cfg.env_gamma, mm)

    visits = np.array([root.children[a].visit_count for a in range(net.n_actions)], dtype=np.float64)
    policy = visits / visits.sum()
    return {
        "action": int(visits.argmax()),
        "policy": policy,
        "root_value": float(root.q()),
        "visits": visits,
        "root": root,
    }


def select_action(policy: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    """Temperature sampling from the visit distribution; tau->0 is greedy."""
    if temperature <= 1e-3:
        return int(policy.argmax())
    probs = policy ** (1.0 / temperature)
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))
