# Session narrative — Phase 0 execution + the PPO learning bug (2026-08-17)

Follows `.claude/handoffs/2026-08-17-phase0-planning-session.md` (the planning
session). This session **executed** the Phase 0 plan and then went past it into
reward/hyperparameter derivation.

## What this session did

1. Executed all 13 tasks of
   `docs/superpowers/plans/2026-08-17-phase0-single-policy-rebuild.md`, inline
   on `master`, one commit per task (`6cf5626`..`b4aaac9`).
2. Ran the first-ever live RLBot v5 match with the user present.
3. Found and fixed a PPO configuration bug that had made *all* prior training
   runs incapable of learning (`03cac2e`).

Per-commit detail is in git log; per-run detail is in `runs/RUNLOG.md`. Neither
is repeated here.

## The main finding: training was never actually learning

Worth understanding before touching rewards again, because it invalidates the
intuition built from every run before `p1probe-f`.

**Symptom:** the bot flip-spams, lands upside down, and almost never touches
the ball. `Player/In Air Ratio` ~0.90 and `Player/Ball Touch Ratio` ~0.001,
both dead flat from the first iteration to the last, in every run.

**What it looked like:** a farming reward function. The first `RewardShare/*`
telemetry supported that — 67% of reward mass went to `VelPlayerToBall`, ~5% to
outcomes. Three probes (a, b, c) rebalanced weights and gated approach shaping
to on-wheels. **Every one moved the shares exactly as computed and changed
behavior not at all.** Probe c ran 42M steps to rule out "30M is too short":
still flat.

**Actual root cause:** the policy was never moving. PPO health metrics (which
nothing had been looking at) showed:

- `Policy Entropy` pinned at ~0.78 first iteration to last — GigaLearn
  normalizes entropy to [0,1], so this is *near-uniform*, i.e. an untrained
  policy;
- `Mean KL Divergence` and `SB3 Clip Fraction` both ~0.0000 — successive
  policies were nearly identical;
- `Policy Relative Entropy Loss` reaching **-22** — the entropy bonus was up to
  22x the policy-gradient term.

Three inherited settings combined to cause it: `entropyScale = 0.035` against a
*normalized* entropy (upstream default 0.018), `epochs = 1` (upstream 2), and
`policyLR/criticLR = 1.5e-4` (upstream 3e-4). The entropy term dominated the
objective while the optimizer got too few, too small updates to fight it.

**Fix, derived by probe (d–g), not guessed:** `entropyScale = 0.002`,
`epochs = 2`, LR `3e-4`. Probe f is the first run in the project's history where
entropy actually falls (0.77 → 0.65), KL rises an order of magnitude, and
behavioral metrics move. Probe g re-tested upstream's 0.018 with the fixed
LR/epochs and it pins again — 0.002 is load-bearing, not cargo-culted.

**Corroborating detail:** 18 of `DefaultAction`'s 90 actions press jump (20%),
pinned by `bot/tests/test_actionspace.cpp`. A uniform policy therefore jumps
~3x/second, which is precisely the observed ~90% air time. The "flip-spam" was
never a learned farming strategy — it is what a uniform distribution looks like
when rendered in Rocket League.

**Method lesson worth keeping:** `RewardShare/*` tells you where reward mass
goes, but says nothing about whether the policy can act on it. Check PPO health
(entropy, KL, clip fraction) *first*; reward-share analysis is only meaningful
once the policy is demonstrably moving. Three probes were spent tuning rewards
against a frozen policy.

## Live deployment verification (plan Task 13)

The RLBot v5 client connected to a real match for the first time and **passed
its parity check**: model loaded, no boost-pad mapping warnings, no console
errors, and in-game behavior matched training telemetry exactly (the same
flip-spam, faithfully reproduced). Nexto ran as an opponent, 2/2 agents spawned.

Three environment facts that cost time and are not in any doc upstream:

- **There is no `rlbot` CLI.** The v5 PyPI package (`rlbot 2.0.0b54`, install
  with `pip install --user --pre rlbot`) is a library. Matches start via
  `scripts/run_match.py`, which drives `rlbot.managers.MatchManager` plus the
  `RLBotServer` binary in `libs/rlbot/` (gitignored; download URL is in that
  script's docstring).
- **RLBotServer cannot auto-launch Rocket League under GE-Proton** — it throws
  `Could not find Proton installation`. The user launches the game manually with
  Steam launch options:
  `%command% -rlbot RLBot_ControllerURL=127.0.0.1:23233 RLBot_PacketSendRate=120 -nomovie`
  (these should be removed when not botting). Start the game *first*, then the
  match script.
- **`HIVE_MODEL` must be an absolute path** — RLBot runs `run.sh` from its own
  cwd. Also, checkpoint globs need `[0-9]*`: `policy_versions` sits beside the
  numbered step folders and sorts last.
- **Nexto needs no v4 bridge.** The official v5 port is cloned at
  `libs/opponents/NectoFamily` with a Python 3.12 uv venv (CPU torch) and a
  `run_command_linux` line added to `nexto/bot.toml`.

## State at end of session

- Working tree clean, all work committed on `master`, HEAD `03cac2e`.
- 33 tests pass (`cd bot/build && ./HiveTests`).
- **One loose end:** the `p1-validate` run (150M steps at the fixed config) was
  killed at ~2M steps when the session ended. Checkpoints and a metrics CSV
  survive under `bot/build/checkpoints/main-p1-validate/` and
  `bot/build/metrics/main-p1-validate.csv`. Rerunning with the same `--label`
  resumes from the checkpoint; deleting the folder starts clean. **Nothing has
  yet confirmed the fixed config keeps improving past 30M** — that is the open
  question.

## User context relevant to what comes next

The user is a GC1 player and stated plainly that reward design and telemetry-
driven weight tuning are *not* things they know how to do, and that they expect
this to be driven for them. Design decisions and their rationale should be
explained, not just applied; "which weight do you want" is not a useful question
to put to them. What they *can* judge better than any metric is how the bot
looks in RocketSimVis or a live match — use them for that.

Hardware/compute constraints from the planning session still hold (part-time
short runs until ~2026-09, then 24/7). A 30M-step probe is ~7 minutes at the
measured ~81k steps/s, which is what made the probe-driven method affordable.
