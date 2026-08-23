# 09 - Instruments for positioning and rotation

Status: open

## The blind spot, found the hard way

After p17 the operator reported "a few very tiny play style things that I liked
that I wasn't seeing before" while the telemetry said the largest behavioural
change in 300M steps was 1%.

Both can be true. The metric set covers shots, saves, touches, speeds, action
rates and phase shares. It covers **positioning, rotation, goal-side
discipline, approach angle and shot selection context: not at all.**

So "nothing moved" after p17 is a claim about what is measured, and it should
be written that way until this gap is closed. `docs/metrics.md` already says a
GC1 eye catches what no metric does -- it caught backwards-driving once. This
is the second time.

## Candidates, each cheap and each needing a null

- **Goal-side share**: is the bot between the ball and its own net, when the
  opponent has possession. Null: compute for a uniform position.
- **Approach angle at touch**: angle between car velocity and the ball-to-
  target-goal line at the contact rising edge. Distinguishes a shot lined up
  from one merely reached. Null: `E[cos]` for uniform direction.
- **Distance to own net when the opponent is in possession.**
- **Recovery time**: steps from a touch until the car is grounded, upright and
  moving ball-ward again.

Every one ships with its null in `docs/metrics.md` in the same commit, per the
standing rule. Note the p17 lesson: `Save/Converted`'s null was 0.789, not the
obvious 0.5, and only a frozen-policy probe could have told us.
