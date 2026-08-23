# 01 - Possession term, keyed on the next touch

Status: open
Blocked by: p18entropy, 06

## The ask, in the operator's words

"A new reward that rewards being closer to the ball than the opponent."

## Why it was not built as asked

A dense "closer than the opponent" term does not address either failure that
motivated it. When the bot fires a hopeless long shot, it HAS possession and IS
closer -- it would collect the proximity reward in full right up to the moment
it throws the ball away. The term prices the state before the mistake and is
silent on the mistake. Worse, paying for proximity pays the bot to reach the
ball sooner, which manufactures the rushed contacts being complained about.

It is also a farm: collectable by parking near the ball and never touching it,
and it would trip p16's standing `Episode/Mean Steps > 500` kill criterion.

## The form to build instead

At each contact rising edge, pay **+1 if the previous touch was the bot's own,
-1 if it was the opponent's**. That is "closer to the ball than the opponent"
resolved as an OUTCOME rather than a position. Camping earns nothing; only
actually retaining the ball does.

- **Null: exactly 0.5** retained share at 1v1 by symmetry.
- Must not implicitly penalise scoring: a goal ends the episode with no "next
  touch", and that must not read as a lost possession.

## Why it needs item 06 first

`Touch/Edge Rate` 0.0121 means contact sequences are ~83 steps apart. At gamma
0.99 the next touch is discounted to 0.43; at 0.995 it is 0.66. Testing this at
0.99 risks a null result that means nothing.
