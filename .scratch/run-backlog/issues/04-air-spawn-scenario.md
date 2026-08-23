# 04 - Air game: actually run the `aerial` spawn scenario

Status: open

## The gap

`CurriculumWeights` already defines `aerial = 10.f`, and
`TrainConfig::spawn` is `SpawnMode::Random`, so it has never run.

## Why this is the right lever, from this project's own reasoning

p13's conclusion about flips: "the incentive is not the problem and must not be
fixed. The payoff is conditional on connecting, and the exploration floor
samples dodges uniformly across states, so the good moments are a small
fraction of flip samples, the MEAN advantage is ~0, and PPO -- which sees the
mean -- suppresses it. **The lever is a state distribution concentrated on the
moments where flipping is right, not another reward term.**"

An aerial is the same shape of problem: correct in a narrow state distribution
that `RandomState` almost never produces, so its mean advantage under uniform
sampling is ~0 and PPO suppresses it. Pair with item 03 or run separately, but
do not run both plus a reward change at once.
