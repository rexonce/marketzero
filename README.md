# MarketZero

**A MuZero-style self-play trading research system.** You can't build literal
AlphaZero for markets (no known rules, no simulator, non-stationary, fat-tailed,
costly) — but you can build **MuZero**: AlphaZero's successor, which *learns*
the rules (a world model) and plans inside it with PUCT Monte-Carlo tree
search. Start with **[docs/DEEP_DIVE.md](docs/DEEP_DIVE.md)** — it explains
every concept from first principles, with sources.

> ⚠️ **Research code for personal trading study. Not investment advice. No
> live execution exists in this repo — by design.** The roadmap gates (6
> months of profitable paper trading included) exist because an agent that
> hasn't survived out-of-sample evaluation is a hypothesis, not a strategy.

## Quickstart

```bash
pip install -r requirements.txt          # torch CPU wheel is enough
python scripts/fetch_data.py             # real BTC-USD daily bars (Yahoo), cached to data/
python scripts/smoke_test.py             # P0 gate: wiring + honesty checks, ~1 min on CPU
python scripts/train.py --iters 20       # P1: self-play training -> artifacts/marketzero.pt
python scripts/backtest.py               # P2 gate: walk-forward, purge/embargo, net of fees
python scripts/paper_trade.py            # paper replay of recent bars (refuses live mode)
pytest                                   # unit tests
```

To customize: `python scripts/train.py --config configs/default.yaml`.
With a CUDA GPU: `--config configs/gpu.yaml` (transformer h, 5 actions,
96 sims, prioritized replay, n-step bootstrap — `train_device: auto`).

## Architecture

```
                    ┌──────────────────────── data.py ────────────────────────┐
                    │ Yahoo/synthetic OHLCV -> causal features -> scaler      │
                    └────────────────────────────┬────────────────────────────┘
                                                 │
             ┌───────────────────────────────────▼───────────────────────────────────┐
             │                    environment.py  (market replay)                     │
             │  obs = features + position | actions = SHORT/FLAT/LONG targets         │
             │  reward = w·μ − fee·|Δw|   (daily log-equity growth, net of costs)     │
             └───────────────────────────────────┬───────────────────────────────────┘
                                                 │
   self-play (actor.py)                          │            training (trainer.py)
   ┌─────────────────────────────────────────────▼──────────────────────────────┐
   │  models.py:  h(obs)→s   g(s,a)→(s',r̂)   f(s)→(policy,value)                │
   │  mcts.py:    PUCT search over g's tree, Dirichlet noise at root             │
   │  buffer.py:  games -> (obs, action, reward, search-policy, return) samples  │
   │  losses:     CE(policy) + CE(value two-hot) + CE(reward two-hot) + L2       │
   └────────────────────────────────────────────────────────────────────────────┘
                                                 │
      ┌──────────────────────────────────────────┼─────────────────────────────────┐
      │                                          │                                 │
 evaluate.py                            risk.py                                   kronos.py
 walk-forward + purge/embargo           kill-switch, vol targeting               scenario sampling
 experiment registry (JSONL)           re-arm required after halt                (bootstrap or Kronos
 agent vs buy&hold, net of fees                                                    foundation model)
```

**How a decision happens:** the observation (a 64-bar window of 16 causal
per-bar features plus the current exposure) enters `h` — a transformer
encoder over bar tokens — producing a latent state; MCTS runs PUCT
simulations through `g`'s predicted futures guided by `f`'s priors and
values (MinMaxStats-normalized, wave-batched leaf evals, depth-capped); the
root visit distribution is the acting policy. Training replays stored games
(optionally prioritized by |z − v̂| surprise with IS correction), unrolls `g`
five steps with canonical 1/K step weighting and 0.5× hidden-state gradient
scaling, and pushes `f`'s predictions toward the search's improved policies
and n-step-bootstrapped returns — search output is the teacher (that's the
AlphaZero trick, translated by MuZero to unknown rules). Reanalysis
refreshes stored games with the latest network each iteration. At eval, a
Phase-2 hook can re-score root actions against sampled market paths —
block-bootstrap or the Kronos foundation model (`mcts_chance_sampler`).
(Set `obs_window: 1` for the original flat 17-float vector + MLP encoder.)

## What the honest checks assert

- `scripts/smoke_test.py` fails if an **untrained** agent beats buy & hold on
  real data — that would indicate lookahead leakage, not skill.
- The backtest is **net of fees** with the risk kill-switch active, evaluated
  **out-of-sample only** on stitched walk-forward windows separated from
  training by purge + embargo.
- Every backtest appends to `artifacts/experiments.jsonl` — the count of tried
  configurations is your multiple-testing deflator.

## Repo map

| Path | Role |
|---|---|
| `marketzero/config.py` | all hyperparameters (flat dataclass, YAML-loadable) |
| `marketzero/data.py` | Yahoo fetch + cache, synthetic fallback, features, scaler |
| `marketzero/environment.py` | market replay env, fee math, log-equity reward, fractional sizing |
| `marketzero/models.py` | h/g/f networks (transformer or MLP h), two-hot support |
| `marketzero/mcts.py` | PUCT search, MinMaxStats, Dirichlet noise, wave-batched leaves, depth cap |
| `marketzero/buffer.py` | replay buffer, uniform or prioritized sampling + IS weights |
| `marketzero/actor.py` | self-play episode runner, reanalysis |
| `marketzero/trainer.py` | K-step unroll losses, n-step bootstrap, checkpoints |
| `marketzero/evaluate.py` | policy replay, walk-forward windows, chance-node rerank, registry |
| `marketzero/risk.py` | kill-switch governor, vol-targeted sizing |
| `marketzero/kronos.py` | bootstrap/Kronos scenario sampling (Phase 2 hook) |
| `marketzero/paper.py` | paper trader, live-mode refusal gates |
| `configs/gpu.yaml` | blueprint-aligned research preset (5 actions, 96 sims, transformer h) |
| `docs/DEEP_DIVE.md` | the full conceptual explanation, with sources |

## Roadmap

P0 wiring ✅ → P1 full training → **P2 chance-node planning (scenarios) +
out-of-sample edge → gate to anything real** → P3 Kronos backbone → P4 CPCV +
deflated Sharpe → P5 risk hardening + 6-month paper gate → P6 small capital.
See DEEP_DIVE.md §10 for the gates between phases.

## License

MIT.
