# 03 - Air game: double the jump actions

Status: open

## The diagnosis, which is narrower than "no air game"

The bot is airborne 44% of the time, and 71% of its touches are airborne -- but
`Touch/Had Flipped` is 0.566 and `Player/Air Time` is 0.772 s, about the length
of a flip. It is not aerialling; it is flipping into the ball at a mean touch
height of 226 with `Touch/Above 450` at 0.069.

What is missing is specifically **jump + boost + sustained climb**:
`Action/Boost When Airborne` 0.257 at `Player/Boost` 27.6, `Phase/AirDribble`
0.0074. The operator reports the bot could air dribble around 300M and cannot
now, and that not defending the air costs it roughly two ranks against humans.

**It is not a pricing problem.** `Action/Jump When Grounded Upright` is 0.047
against an unmasked null of 0.20 -- 4.3x BELOW chance -- while
`Critic/TD Delta Jump` is -0.011, i.e. the critic prices a jump as free. The
policy avoids an action the critic says is costless. `airTouch` has gone
2 -> 12 -> 20 -> 35 -> 55 across five runs against this.

## The remedy, and its provenance

Zealan, in `reference-guide/`: "add more jump actions... doubling the jump
actions seems to be enough to eliminate the need for air rewards." This is one
of the four trusted sources and the remedy has never been tried here. p14's own
kill criteria pre-committed to this branch.

## What it touches

- `Hive::MakeActionParser` is the only construction site.
- The 90-action table and every null derived from it: the figures in
  `docs/metrics.md` are asserted in `bot/tests/test_actionspace.cpp` and will
  all move. Recompute them in the same commit.
- **Parity trap**: `maskActions` must match at deployment via `HIVE_MASK_ACTIONS`;
  run `./HivemindBot verify <checkpoint>` afterwards.
