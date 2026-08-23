# 10 - Rebound pressure: sustained shots that pull a defender out of position

Status: open
Blocked by: p18entropy, 06

## The pattern, from the operator

"Shooting a shot that gets saved but leaves the saver in an awkward position
that then makes the follow up shot much more lethal is a real pattern,
sometimes 4-5 saves in a row progressively leave the defender more awkward."

This is why p17's save term is capped rather than uncapped: if a saved shot is
net-negative for the shooter, the bot learns that a savable shot is not worth
taking, and this pattern is priced out.

## Why the cap is only half a fix

The cap stops the pattern being PUNISHED. It does nothing to make it REWARDED.

A 4-5 save sequence spans roughly 150-225 steps. At gamma 0.99 the eventual
goal is discounted to **0.105**, while the save penalty lands ~42 steps out at
**0.66** -- the penalty is about six times more visible than the payoff that
justifies it. The bot can see the cost of the sequence and effectively cannot
see its reward.

## Options

1. Item 06 (gamma 0.995) takes the sequence payoff from 0.105 to 0.325. Cheapest
   and helps items 01 and 07 too.
2. An explicit pressure term. Resist this until 06 has been tried -- a horizon
   problem should be fixed with the horizon, not with a term that pays for a
   proxy of what the horizon would have shown.
