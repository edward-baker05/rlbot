# 05 - `airTouch` breaks the finishing-block guard

Status: open

## The failing test

`bot/tests/test_rewards.cpp` - "an aerial pays for itself, with margin":

    airPerEp 64.46  <  finishPerEp 48.50   FAILS

At the live `airTouch` of 55, the air block pays **1.33x the entire finishing
block** (Goal + ShotOnTarget) per episode. The guard was written specifically
so "get it high" would never be the stack's loudest opinion again.

This is the only failing test in the suite (100 of 101 pass).

## Why it was not fixed in p17

Cutting `AirTouch` is a retarget, not a renormalisation, and it would have been
a second headline change in p17 -- aimed at a different behaviour, and one that
risks further damaging an air game the operator wants back.

## Note

The violation reads worse than it did before p17 because
`ASSUMED_SHOT_STRENGTH` was corrected from a guessed 0.25 to the measured
0.5247, which more than doubles the finishing block's shot contribution and
makes this guard strictly harder to pass. The test is now more accurate, not
more lenient.

## Decide, do not drift

Either cut `airTouch` to satisfy the guard, or amend the guard with a written
justification. Leaving a red test indefinitely is how the last AirTouch failure
hid: the guard "passed while the term was 3.9x BELOW break-even in play"
because it asserted at a ball height the bot no longer reached.
