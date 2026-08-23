# 02 - `touchGoalAccelOpponentScale` 0.8 -> 1.0

Status: open
Blocked by: p18entropy

## The arithmetic

At 1v1 a term wrapped at opponentScale k pays the actor `+S` and the opponent
`-kS`, so the POPULATION nets `S(1-k)` per event. Self-play samples both roles
from one policy, so any k < 1 pays the policy for the event OCCURRING, whoever
caused it.

At 0.8, `TouchGoalAccel` pays 0.2x for goalward ball acceleration happening at
all. p17 closed exactly this leak on `ShotOnTarget`, where it was subsidising
hopeless shots.

## Why it was left alone

It prices ball MOVEMENT rather than shot selection, which is not what the
operator complained about, and it has ~1B steps behind it in the best bot this
project has made. Changing it in p17 would have been a second variable aimed at
a different behaviour.

## The caveat that makes this non-obvious

See item 08. Exact zero-sum may remove the teaching signal along with the farm.
If 08 finds that k=1 is self-defeating in symmetric self-play, this item
inverts: 0.8 may be load-bearing precisely BECAUSE it is not zero-sum.

Do not action this before 08.
