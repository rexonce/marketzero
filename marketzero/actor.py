"""Self-play actor: replays random windows of history with search-guided play."""
from __future__ import annotations

import numpy as np

from .buffer import Game
from .environment import TradingEnv
from .models import MuZeroNet
from .mcts import run_mcts, select_action


def play_episode(env: TradingEnv, net: MuZeroNet, cfg, rng: np.random.Generator,
                 training: bool = True, episode_len: int | None = None) -> Game:
    """One episode = one trajectory through a fixed market window.

    The market's moves are history (the opponent's play); ours depend on the
    agent's own decisions -- that dependence is what the agent must learn.
    """
    length = episode_len or cfg.env_episode_len
    obs = env.reset(rng=rng, episode_len=length)
    game = Game()
    done = False
    while not done:
        result = run_mcts(net, obs, cfg, training=training)
        if training:
            # temperature schedule: explore early in the episode, then exploit
            tau = (cfg.mcts_temperature if len(game) < cfg.mcts_temp_explore_steps
                   else cfg.mcts_temp_late)
            action = select_action(result["policy"], tau, rng)
        else:
            action = result["action"]
        game.obs.append(obs.copy())
        game.policies.append(result["policy"].copy())
        game.root_values.append(result["root_value"])
        game.actions.append(action)
        obs, reward, done, _info = env.step(action)
        game.rewards.append(reward)
    game.compute_returns(cfg.env_gamma)
    return game
