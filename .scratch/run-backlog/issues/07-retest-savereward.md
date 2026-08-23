# 07 - Re-test `SaveReward` once the policy can move

Status: open
Blocked by: p18entropy

## Standing verdict: UNFALSIFIED, not falsified

p17 shipped the term and 0 of 5 predictions passed -- but no kill criterion
fired and **the policy did not move on any axis**. The term carried 4.4% of
reward mass into a policy that was not responding to the other 96% either.

Nothing about the term's design was tested. It is still the best available
answer to "the bot throws possessions away on shots that get saved", and it is
already implemented, tested (7 unit cases) and instrumented.

## What it is, for whoever picks this up

Signed change in threat at the player's own net across a touch, through the
same `ProjectShot` and the same `MISS_SCALE` (892.755) `ShotOnTarget` uses:
`exp(-miss_before/S) - exp(-miss_after/S)`, on the rising edge only, wrapped
zero-sum. Pairs with `ShotOnTarget` to give: **an on-target shot is paid if and
only if it is not saved.**

Budget 16.5, solved at 28.05 for a 0.10 share and then capped by
`shotOnTarget * E[Shot/Strength]` = 32 * 0.5247 = 16.79, so that a saved
on-target shot is never net-negative for the shooter.

## Baselines to beat, from the p17cal frozen-policy probe

`Shot/Saved Share` 0.313, `Shot/Distance` 3932, `Shot/Time` 1.573,
`Save/Converted` **0.789** (measured null, NOT 0.5, not analytic),
`OwnHalf/Touch Rate` 0.0166.

At p17's end after 300M: 0.289 / 4009 / 1.593 / 0.793 / 0.017.
