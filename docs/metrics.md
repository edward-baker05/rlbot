# Metrics guide

A living document. For each metric family: what it measures, what healthy
looks like (trends and comparisons, never absolute thresholds), and which
decision it feeds. Metrics land in wandb and in `bot/build/metrics/<run>.csv`.

**The reward curve is never evidence.** It rises in every run, including
broken ones. Everything below exists so you never have to argue from it.

| Metric | Measures | Healthy | Feeds |
|---|---|---|---|
| `RewardShare/*` | Fraction of realized \|weighted reward\| per term (one sampled player per arena; zero-sum terms measured pre-zero-sum) | Outcome terms (Goal, StrongTouch) growing over a phase; no shaping term dominating late | Phase-gate decisions; farming detection |
| `Scenario/*/Share` | Fraction of arenas running each curriculum scenario | Matches configured curriculum weights | Verifying setters actually run |
| `Scenario/*/EndedInGoal` | Episode outcome per scenario | Rising within a phase | Whether a scenario is being learned |
| `Player/Ball Touch Ratio` | Touch frequency | Stable or rising; a collapse after a reward change = new degenerate behavior | Reward-change rollback |
| `Player/Touch Height` | Air game development | Rising from P2 onward without touch ratio falling | P2/P3 gates |
| `Phase/*` | Time share per play phase (metric labels only, not curriculum) | Neutral dominant; shifts tracking curriculum changes | Curriculum tuning |
| `Rating/*` (skill tracker) | Elo vs. past versions | Monotonic rise; falling while reward rises = farming | The primary gate signal |
| `Game/Goal Speed` | Shot power at goals | Rising through P1–P2 | Striking quality |

## How to watch a run

1. **`RewardShare/*` first.** If a shaping term (VelPlayerToBall, FaceBall)
   is absorbing the reward mass late in a phase, the policy is farming it.
2. **Then `Rating/*`.** Reward up + rating down is the farming signature.
3. **Then RocketSimVis** (`scripts/vis.sh` + `scripts/watch.sh`) for a few
   minutes — always within the first hour after any reward or curriculum
   change, which is when new degenerate behavior appears. A GC1 eye catches
   what no metric does (it caught backwards-driving once).

Comparisons are only valid between labeled runs recorded in `runs/RUNLOG.md`.
