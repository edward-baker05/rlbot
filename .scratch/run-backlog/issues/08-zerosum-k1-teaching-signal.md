# 08 - Does zero-sum at exactly k=1 remove the teaching signal?

Status: open
Blocked by: p18entropy, 07

## The tension, unresolved

p17 set `saveOpponentScale` and `shotOnTargetOpponentScale` to exactly 1.0 on
this argument: at 1v1 a term at scale k pays the population `S(1-k)` per event,
and self-play samples both roles from one policy, so k < 1 pays for the event
OCCURRING and funds a farm. That arithmetic is correct.

**But it may prove too much.** At k=1 the term is purely differential, and
`trainAgainstOldChance` is 0.2 -- so 80% of the time the opponent is a current
copy of the same network, where the expected advantage in the symmetric
direction is exactly zero. Only the 20% against older versions supplies
asymmetry.

## The precedent that makes this worth a run

The standing lesson from p4pbrs, already in `runs/RUNLOG.md`:

> "The no-farm guarantee and the teaching signal are the same property viewed
> from opposite sides. Removing farmability without replacing the signal leaves
> only the outcome terms."

p4pbrs made a term unfarmable and thereby untelling. k=1 may be the same
mistake in a different costume.

## The discriminator

Once item 07 has a policy that can respond: run the identical save term at
k=1.0 and at k=0.5, everything else frozen, and compare whether `Save/Converted`
and `Shot/Saved Share` move. Cheap, and it settles a question that applies to
every zero-sum term in the stack, including item 02.

Do not action item 02 before this resolves -- it could invert its conclusion.
