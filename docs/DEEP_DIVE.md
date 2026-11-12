# MarketZero — Deep Dive

*Everything you need to understand the system, from first principles to implementation, with sources.*

---

## 1. The seductive idea — and why it's literally wrong

In 2017 AlphaZero taught itself chess and Go from pure self-play, starting from random moves, and reached superhuman strength in hours. The natural question: *can we do that to the market and get rich?*

No — and understanding **why not** is the foundation of every design decision in MarketZero.

| Property | Chess / Go | Markets |
|---|---|---|
| **Rules** | Perfect, known, deterministic | Unknown; the "rules" are the aggregate behavior of millions of participants |
| **Simulator** | Exact — you can branch any position for free | Reality runs once; you cannot re-run 2022 with a different action |
| **Stationarity** | Fixed forever | Regimes shift; yesterday's edge decays; other agents adapt to you |
| **Opponent** | Bounded rationality, no memory of *you* | Reflexive — strategies that work get arbitraged away |
| **Observability** | Fully observable state | Partially observable; price is a noisy aggregate of hidden information flow |
| **Cost of actions** | Zero | Fees, slippage, spread, borrowing, taxes |
| **Return distribution** | Win/draw/loss | Fat-tailed — BTC has printed −50% *single days*; single-stock crashes of −80%+ exist |
| **Signal-to-noise** | Enormous — a pawn up usually wins | Tiny — annualized Sharpe > 2 is world-class for a single strategy |

The deepest problem is the **simulator gap**. AlphaZero's power comes from planning inside an exact simulator (MCTS branches the true rules millions of times). In markets you don't get to branch reality. You can only branch a *model* of reality.

That single observation points directly at the fix.

## 2. The lineage: AlphaGo → AlphaGo Zero → AlphaZero → MuZero

- **AlphaGo (2016)**: policy/value networks bootstrapped from human master games + reinforcement learning + MCTS. Needed human data.
- **AlphaGo Zero (2017)**: no human data at all. Pure self-play, neural MCTS. Beat AlphaGo 100–0.
- **AlphaZero (2017)**: same algorithm generalized to chess and shogi with no game-specific tuning.
- **MuZero (2019, Schrittwieser et al.)**: the missing piece. AlphaZero needs the *rules* to run search. **MuZero learns the rules.** It plans inside a *learned model* — three neural networks that together emulate the environment well enough to search. It matched AlphaZero on Go/chess/shogi **and** superhuman performance on 57 Atari games it had never been given the dynamics for.
- **EfficientZero (2021)**: sample-efficient upgrades — value prefixing, replaying with reanalyzed targets, self-supervised consistency loss. Important when data is expensive. (Markets: data is cheap, but *regimes* are scarce — EfficientZero-style ideas matter less for daily bars, more for intraday.)

**The key sentence of this whole document:** *you can't build literal AlphaZero for markets — but you can build MuZero, which is exactly AlphaZero's successor designed for environments without known rules.* The learned world model **is** the simulator substitute.

## 3. MuZero internals — the three networks

MuZero learns a latent space in which it can plan. Given observations `o` and actions `a`:

```
                ┌────────────────────────────────────────────┐
   o₁…oₜ        │  h : representation      oₜ ──────────▶ sₜ │
   (features)   │  g : dynamics       (sₜ, aₜ) ──▶ sₜ₊₁, rₜ │
                │  f : prediction       sₜ ──▶ pₜ (policy)   │
                │                            └─▶ vₜ (value)  │
                └────────────────────────────────────────────┘
```

- **h(oₜ) → sₜ** — *representation*: compresses the observation (here: our feature window) into a latent state.
- **g(sₜ, aₜ) → (sₜ₊₁, r̂ₜ)** — *dynamics*: the learned rules. Given a latent state and an action, predicts the next latent state and the reward. This replaces the chess move generator.
- **f(sₜ) → (pₜ, vₜ)** — *prediction*: given a latent state, outputs a policy prior over actions (to guide search) and a value estimate (to guide backup).

Crucially, MuZero never requires the latent state to *look like* the true state — only to contain enough information about future rewards/values. The model is trained to be *useful for planning*, not to reconstruct pixels.

### 3.1 The three losses

For a sampled timestep `t` from a stored self-play game, MuZero unrolls the dynamics `K` steps:

1. **Value loss**: cross-entropy between `f(h(oₜ)).value` and the (discounted) return actually observed `zₜ`.
2. **Policy loss**: cross-entropy between `f(s).policy` and the improved search policy πₜ (the visit distribution MCTS computed at that step — search output is the teacher).
3. **Reward loss**: cross-entropy between `g`'s predicted reward at each unroll step and the observed reward.

Plus L2 regularization. Total: `ℒ = Σₖ (ℒ^V + ℒ^p + ℒ^r) + c‖θ‖²`.

### 3.2 Value & reward *support* (the detail everyone skips)

Values and rewards are **not** scalars. They're predicted as categorical distributions over a discretized support — e.g. 21 bins spanning [−1, +1] — and a scalar is recovered as the expectation. A target scalar is encoded as a **two-hot** linear interpolation of its two neighboring bins (e.g. 0.034 → 70% mass on bin 0.03, 30% on 0.04). Why: cross-entropy on a bounded categorical distribution is dramatically more stable than MSE on unbounded scalars, and it handles multi-modality (a position that either doubles or dies). DeepMind used a 601-bin support over [−300, 300] scaled by the reward magnitudes; we use a small [−1, 1] support because our rewards are *log-returns* (bounded per step, see §5.3). The general recipe: pick the support so that ≥99.9% of targets fall inside; clip the rest.

### 3.3 Reward scaling & normalization

MuZero normalizes each game's returns by a running scale so value targets live in a stable range. MarketZero keeps a running standard deviation of observed episode returns and scales targets with it.

## 4. Search: PUCT Monte-Carlo Tree Search

The policy that acts is not the network — it's the network *plus* search. Each decision:

1. **h(oₜ)** creates the root latent state; **f** gives root prior `P` and value.
2. Add **Dirichlet noise** to root priors for exploration: `P'(a) = (1−ε)·P(a) + ε·η`, `η ~ Dir(α)` (DeepMind: ε=0.25; α=0.3 chess, 0.03 Go; MarketZero uses 0.3 for 3 actions).
3. Run `N_sim` simulations. Each simulation:
   - **Select** from root, repeatedly picking the child maximizing **PUCT**:

     ```
     a* = argmax_a [ Q(s,a) + c_puct · P(s,a) · √(Σ_b N(s,b)) / (1 + N(s,a)) ]
     ```

     First term: exploit what's good. Second: the prior favors promising actions; the √ΣN/(1+N) term decays as an action is explored, forcing alternatives to be tried. `c_puct` balances the two (MarketZero: 1.25, configurable — it mattered in the original build, which is why it's a config knob).
   - **Expand** a leaf: `g(s,a)` → child latent + predicted reward; `f(child)` → prior + value.
   - **Backup** the value (plus predicted rewards along the path) to all visited nodes.
4. **Act**: sample from the root visit distribution (temperature τ=1 early for exploration; τ→0 [greedy] later / in evaluation).

In AlphaZero the tree children are generated by the *true* rules. In MuZero they're generated by **g** — search works only as well as the learned model, which is why model quality is monitored (reward/value loss curves) and why the training loop interleaves self-play and learning continuously.

## 5. Markets as a (PO)MDP — the honest translation

### 5.1 State (observation)

We cannot observe the market state; we observe *features computed from history*. MarketZero's observation vector at day `t` (all computed **causally**, no lookahead):

- Last `L` daily log-returns (default L=8) — recent momentum/tone
- Position-in-trade features: normalized price vs 20/50/200-day MAs
- Volatility: 20-day rolling std of returns, ATR(14)/price
- Trend/momentum: RSI(14), MACD histogram normalized by price std
- Range structure: 20-day high/low position of current price
- Current position ∈ {−1, 0, +1} (the agent must *know* its own exposure — without this the process isn't Markov)

~20 floats, standardized with statistics computed on the **training window only** (leakage discipline).

### 5.2 Actions

Three discrete actions, interpreted as **target position**: `{0: short/exit-to-flat… }` — concretely `{0: SHORT, 1: FLAT, 2: LONG}` (target exposure −1/0/+1). Spot-only traders can restrict to `{FLAT, LONG}` in config (`n_actions=2`). Choosing target-position semantics (rather than buy/sell/hold deltas) keeps the policy interpretable and makes cost accounting simple.

### 5.3 Reward — log equity growth minus costs

```
rₜ = wₜ₋₁ · μₜ  −  fee_rate · |wₜ − wₜ₋₁| · multiplier
```

where `μₜ = ln(Pₜ/Pₜ₋₁)` is the market's daily log-return, `w` the chosen exposure, and the second term charges proportional fees on **turnover** (both entry and exit). Why log-wealth: it's additive across time (so maximizing Σr = maximizing final log equity, i.e. Kelly-style growth), it naturally penalizes variance, and single fat losses compound correctly. Per-step rewards are tiny (|μ| ~ 2–5% on BTC) — which is why the value support can be small.

**The cost term is not decoration.** With a 0.1% fee per side and daily rebalancing, a random policy churns positions ~2/3 of days and bleeds on the order of 30–50%/year in fees alone. Reward shaping that ignores costs produces a beautiful backtest and a broke trader.

### 5.4 Discounting

γ = 0.997 daily (config): a ~1-year horizon. Too small an effective horizon → the agent can't see through drawdowns; too large → the distant, poorly-modeled future dominates targets. The original build found mid-range γ best for daily bars.

### 5.5 Episode = a walk through history

An episode is a contiguous random window of the training data (e.g. 252 trading days). The agent replays it with its current policy + search. This is "self-play against the market": the market's moves are fixed history, but the agent's *trajectory through it* (and hence its P&L, its costs, its drawdowns) depends on its own choices — that dependence is exactly what the agent must learn. History is the opponent's opening book; our trades are our replies.

**And its central danger:** a single history path means a single deterministic opponent — perfect conditions for memorization. Countermeasures (§6.2) exist precisely for this.

## 6. Evaluation — where trading RL lives or dies

### 6.1 Walk-forward (anchored & rolling)

```
|──── train W1 ────|── test ──|
        |──── train W2 ────|── test ──|          (rolling)
|── expanding train ──────|── test ──|           (anchored)
```

Train on a window, freeze the agent, evaluate on the *next* untouched window, roll forward, stitch out-of-sample results. Anything the agent saw in training is in-sample forever; only stitched test segments count as performance evidence.

### 6.2 The overfitting arsenal (why plain walk-forward isn't enough)

- **Purging**: drop training samples whose label window overlaps the test window (prevents leakage at the boundary).
- **Embargo**: additionally blank out a buffer *after* each test window (serial correlation makes adjacent samples near-duplicates).
- **Combinatorial Purged CV (CPCV)** (López de Prado): instead of one path, test on many combinations of blocks — yields a *distribution* of out-of-sample results instead of a single lucky/unlucky path.
- **Deflated Sharpe Ratio**: correct the Sharpe for how many configurations you tried (if you backtested 500 variants, your best one's Sharpe is inflated by selection bias alone).
- MarketZero implements anchored/rolling walk-forward + purge/embargo, and reports the *number of tried configurations* next to every headline number so you can deflate it yourself. CPCV/DSR are Phase-5 items on the roadmap.

**The iron rule we adopt: a result that hasn't survived out-of-sample walk-forward with purge+embargo is a hypothesis, not a result.**

## 7. Kronos — the foundation-model wildcard

**Kronos** (arXiv 2508.02739, AAAI 2026; Yu Shi et al., open weights `NeoQuasar/Kronos-base/mini` on HuggingFace, code at github.com/shiyu-coder/Kronos) is the first open-source *financial K-line foundation model*: a T5-family seq2seq trained on OHLCV bars from ~45 exchanges, covering minute→monthly frequencies and thousands of symbols.

Why it matters for us — it does two things zero-shot:

1. **Conditional K-line generation.** Given recent history, Kronos *samples plausible future OHLCV paths* — a learned, regime-aware market simulator you can branch. On generative fidelity benchmarks it beats general-purpose TS foundation models by ~22%, and on forecasting RankIC by ~93%. That is exactly the "simulator gap" of §1, addressed by a model trained on far more market data than any of us will ever download.
2. **Representations.** Its encoder can be fine-tuned as a backbone for our representation network h — 45 exchanges of pretraining is a lot of prior for our small MLP to rediscover.

MarketZero's integration plan (matching the original blueprint's phases):

- **Phase 2 — chance nodes.** Extend MCTS with chance nodes: at the root, sample `S` scenarios (Kronos `predict(...)`, or the built-in block-bootstrap fallback); run one sub-tree per scenario; average Q across scenarios before backup. The tree now plans against *distributions* of futures, not the single learned deterministic dynamics — this directly attacks the fat-tail/reflexivity critique of §1.
- **Phase 3 — backbone.** Replace/augment h with fine-tuned Kronos encoder embeddings.

The adapter (`marketzero/kronos.py`) ships with a **bootstrap scenario sampler** that needs no extra dependency (stationary block bootstrap over historical return blocks — preserves fat tails and volatility clustering, if not nonlinear structure), plus a thin wrapper that calls real Kronos when the `kronos` package + weights are installed. All downstream code is sampler-agnostic.

## 8. Risk framework — first-class, not an afterthought

For **personal trading**, survival dominates optimization:

1. **Kill-switch (tested)**: `RiskGovernor` halts trading when (a) episode drawdown from peak equity exceeds `max_drawdown` (default 20%), or (b) single-day loss exceeds `max_daily_loss` (default 8%). Post-halt it requires manual re-arming — the point is to stop a broken agent from compounding a bad regime.
2. **Vol-targeted sizing**: exposure `× min(1, target_vol / realized_vol)` — a 15%-annualized-vol agent on a 60%-vol asset holds ~¼ notional. Drawdowns scale with vol, not conviction.
3. **Turnover penalty in reward** (already in §5.3) — the agent is *trained* to feel costs.
4. **Paper-trading gate**: `PaperTrader` refuses live mode unless `--i-accept-live-risk` is passed **and** config `allow_live: true` **and** `I_UNDERSTAND_THE_RISKS=1` env var is set **and** the governor passes. Even then it starts in shadow mode. The blueprint's go/no-go gates require ≥6 months of profitable paper trading before capital.
5. **Multiple-testing honesty**: every experiment logs to a registry (JSON lines) — you always know how many backtests produced "the best" one.

## 9. Honest expectations

Realities the system is built to surface, not hide:

- **Buy & hold BTC (2014→, daily) Sharpe ≈ 0.4–0.6** with max drawdown ≈ −80%+. Any strategy must be judged *net of costs, out-of-sample, after multiple-testing correction*, against this.
- **A random policy loses ~30–50%/yr to fees** at daily frequency with retail fees. The first thing a cost-aware agent must learn is *not trading*.
- **The untrained agent must lose to buy & hold.** Our smoke test asserts exactly this — an untrained net that "wins" is proof of a bug (lookahead leakage being the usual suspect).
- **Capacity**: daily-bar strategies on BTC have tiny alpha vs. the noise; expect modest, fragile edges. The realistic win conditions for a personal system are (a) disciplined risk, (b) cost control, (c) avoiding the big left tails — not 10×.
- **Regime decay**: a 2020-fitted agent had never seen a rate-hike cycle. Anchored (expanding) training and periodic re-fitting are the pragmatic answer; CPCV quantifies the variance across paths.

## 10. Roadmap P0 → P6

| Phase | Content | Gate to next |
|---|---|---|
| **P0** | Wiring: data → env → agent → backtest on real data; tests green; untrained agent < buy&hold (sanity) | Smoke test passes |
| **P1** | Full self-play training on GPU/CPU-days; loss curves sane; in-sample Sharpe > buy&hold | In-sample edge appears |
| **P2** | Chance-node MCTS with bootstrap/Kronos scenarios | Out-of-sample walk-forward beats buy&hold net of costs |
| **P3** | Kronos encoder backbone (fine-tune) | Same, with margin |
| **P4** | CPCV + Deflated Sharpe evaluation harness | DSR > 0 |
| **P5** | Risk hardening: kill-switch drill, vol targeting stress tests | Paper trading profitable |
| **P6** | Small live capital, hard caps, continuous monitoring | — (never "done") |

**P2's gate is the first one that means anything about the real world.** Everything before it is plumbing and in-sample craft.

## 11. Glossary

| Term | Meaning |
|---|---|
| MDP / POMDP | (Partially observable) Markov decision process — the formal frame for sequential decisions |
| World model | Learned network(s) emulating environment dynamics |
| h / g / f | MuZero's representation / dynamics / prediction networks |
| Latent state | h's compressed internal state — never needs to equal the true state |
| PUCT | Predictor + Upper Confidence bound for Trees — the bandit-style selection rule inside MCTS |
| Dirichlet noise | Distribution over a simplex; mixed into root priors to force exploration |
| Two-hot support | Categorical target encoding: scalar → adjacent-bin probability mass |
| Walk-forward | Train-on-past / test-on-next-window rolling evaluation |
| Purge / embargo | Removing boundary and post-boundary samples to prevent leakage |
| CPCV | Combinatorial purged cross-validation (many OOS paths, not one) |
| DSR | Deflated Sharpe Ratio — Sharpe corrected for number of trials |
| Kelly / log-growth | Optimal capital growth = maximizing E[log wealth]; log-return reward encodes it |
| Chance node | MCTS node representing environment randomness (vs decision nodes = our choices) |
| Block bootstrap | Resampling *contiguous blocks* of history to keep serial structure (vol clustering) |
| Kronos | Open-source financial K-line foundation model (seq2seq, OHLCV-native) |

## 12. Sources

**MuZero / AlphaZero**
- Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero) — [arXiv:1911.08265](https://arxiv.org/abs/1911.08265) · [DeepMind blog](https://deepmind.google/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules) · [MuZero intuition (Schrittwieser)](https://www.julian.ac/blog/2020/12/22/muzero-intuition) · [Wikipedia](https://en.wikipedia.org/wiki/MuZero) · [Demystifying MuZero Planning](https://arxiv.org/html/2411.04580v2) · [Value function in MuZero](https://medium.com/@kaige.yang0110/how-to-learn-value-function-in-muzero-c443d3391a32) · [Remaking EfficientZero](https://www.alignmentforum.org/posts/bPa6AzRgGZGmxbq6n/remaking-efficientzero-as-best-i-can)
- Silver et al., *A general reinforcement learning algorithm that masters chess, shogi and Go through self-play* (AlphaZero) — [arXiv:1712.01815](https://arxiv.org/abs/1712.01815)
- PUCT formula & Dirichlet noise — [Matthew Landers, AlphaZero](https://mattlanders.net/alphazero.html) · [Charles University slides](https://ufal.mff.cuni.cz/~straka/courses/npfl139/2425/slides.pdf/npfl139-2425-11.pdf) · [Dirichlet noise purpose (Stats.SE)](https://stats.stackexchange.com/questions/322831/purpose-of-dirichlet-noise-in-the-alphazero-paper) · [Lessons from AlphaZero: parameter tweaking](https://medium.com/oracledevs/lessons-from-alphazero-part-3-parameter-tweaking-4dceb78ed1e5) · [UCT, Chess Programming Wiki](https://chessprogramming.org/UCT)

**Kronos**
- Shi et al., *Kronos: A Foundation Model for the Language of Financial Markets* (AAAI 2026) — [arXiv:2508.02739](https://arxiv.org/abs/2508.02739) · [GitHub (shiyu-coder/Kronos)](https://github.com/shiyu-coder/Kronos) · [NeoQuasar/Kronos-base on HuggingFace](https://huggingface.co/NeoQuasar/Kronos-base) · [TSFM.ai analysis](https://tsfm.ai/blog/kronos-financial-foundation-model) · [Independent review (Frenzel)](https://tim-frenzel.com/insights/kronos-foundation-model-for-financial-candlesticks)

**Evaluation & overfitting**
- López de Prado, *Advances in Financial Machine Learning* (purged CV, embargo, CPCV, DSR) — [QuantInsti tutorial](https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial) · [CPCV with code (QuantBeckman)](https://www.quantbeckman.com/p/with-code-combinatorial-purged-cross) · [Backtest overfitting in the ML era (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) · [Purged CV repo](https://github.com/eslazarev/purged-cross-validation/blob/main/paper/paper.md) · [Rigorous walk-forward framework (arXiv)](https://arxiv.org/html/2512.12924v1)
