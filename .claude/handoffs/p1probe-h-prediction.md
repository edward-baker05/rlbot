# p1probe-h prediction, written before the run finished

Change: `velBallToGoal` 0.5 -> 0. One variable. Baseline is `p1-validate`
bucket-averaged over its first 30M steps (buckets 0-1 of 8).

## Mechanism under test

VelBallToGoal took 44.3% of all reward mass at 117M steps while being
(a) continuous, (b) zero-sum so mean ~0, (c) driven by ball motion the policy
did not cause at a 0.001 touch ratio. That is variance in the return with no
learnable content. If that is what is burying the gradient, removing it must
show up in the *return statistics*, not just the shares.

## Predictions (falsifiable)

| metric | p1-validate @ 0-30M | prediction if hypothesis holds | falsified if |
|---|---|---|---|
| `GAE/Returns STD` | 21.3 | drops materially, < ~15 | stays ~21 |
| `GAE/Avg Advantage` | 0.151 -> 0.107 (decaying) | stops decaying / flat or rising | still decays ~30% |
| `Policy Entropy` | 0.719 -> 0.692 | falls faster than baseline | tracks baseline or rises |
| `Policy Loss` magnitude | ~2e-3, sign-oscillating | larger, more consistent sign | unchanged |

Share reallocation is NOT evidence either way -- shares always sum to 1, so
they must move. Only the return statistics test the mechanism.

## What this probe cannot show

Touch ratio and In Air Ratio are NOT expected to move in 30M. p1-validate's
own interesting divergence sits between 40M and 80M, and `Rating/1v1` needs
10M steps per point. A flat touch ratio here does not falsify anything.

## Why this is necessary but probably not sufficient

Even with VelBallToGoal gone, nothing in the reward pays for keeping wheels on
the ground. `GroundedReward` makes approach shaping *absent* while airborne,
which is not a penalty. Meanwhile 42.9% of the 42 actions available to a
grounded car press jump, so the uniform prior is flip-spam. Expect probe-h to
improve the gradient-to-noise ratio without breaking the 0.91 In Air Ratio.
The grounding leg is a separate probe.
