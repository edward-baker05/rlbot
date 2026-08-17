# Architecture

## Two models, not nine

The bot runs exactly two policies:

- **Kickoff** — from the kickoff reset until the ball is first touched.
- **General** — everything else.

This started as a mixture-of-experts design with a policy per situation
(kickoff, aerial, flip reset, ground dribble, bump, …) and a router picking one
each step. That was cut, deliberately, and it is worth recording why so it does
not get re-added later.

### Why MoE fails here

A router-based MoE has three problems in a game with continuous dynamics:

1. **Seams.** Experts hand off mid-play at boundaries that are inherently
   fuzzy. Expert A exits in a state that expert B never trained on, and B has
   no idea how to recover. The failure is not in either expert — each looks
   fine in isolation — it is in the transition, which nothing optimises.

2. **The router is itself a hard learning problem.** A hand-written classifier
   ("is this an air dribble?") is brittle and flickers at threshold boundaries,
   swapping the controlling policy mid-manoeuvre. A learned router needs its
   own training signal, which you do not have.

3. **Compute divides.** Every expert needs its own run. Six experts on one 2060
   is six times the wall clock for policies that each see a sixth of the data.
   The single biggest driver of strength is total timesteps, and MoE spends
   them badly.

A monolithic policy has none of these. It sees every state, so there are no
off-distribution handoffs, and every timestep improves the one set of weights
that plays the whole game.

### Why kickoff is the exception

Kickoff survives all three objections, which is what makes it the one
defensible split:

| Property | Kickoff | A mid-play situation like "aerial" |
|---|---|---|
| Starts at | A hard reset, from a small set of fixed spawns | Ambiguous, gradual |
| Ends at | First ball touch — observable, exact | Ambiguous, gradual |
| Interleaves with general play? | Never | Constantly |
| Handover direction | Once, one way | Repeatedly, both ways |

There is exactly one transition, it happens at a moment both policies were
trained to expect, and the two regimes never interleave. The seam that kills
general MoE does not exist here.

The payoff is that kickoffs are a small set of near-deterministic openings
where a generalist wastes capacity — it has to spend weights covering positions
it sees for two seconds out of every thirty.

**If you are ever tempted to add a third model,** check it against all four rows
of that table first. Almost nothing else in Rocket League qualifies.

### The part of MoE that was worth keeping

Situation *labels* survive, in `bot/src/policy/Regime.h` as `PlayPhase`. They do
two useful jobs and neither involves routing:

- **Metrics.** `Phase/Aerial`, `Phase/Defend` etc. show what the policy actually
  spends its time doing. If you raise the aerial curriculum weight and the
  `Phase/Aerial` share does not move, your state setter is not doing what you
  think.
- **Curriculum.** State setters spawn episodes directly into each situation, so
  the *one* general policy practises rare skills far more often than natural
  play would provide.

That second point is where the real value of "think in situations" lives. It
just belongs in training, not in control flow.

---

## The observation

`RLGymCPP::DefaultObsPadded`, 165 floats wide, `maxPlayersPerTeam = 3`.

```
  9   ball position, velocity, angular velocity
  8   this car's previous action
 34   boost pad states
 19   this car
 38   teammates   (2 slots x 19, zero-padded)
 57   opponents   (3 slots x 19, zero-padded)
---
165
```

Three properties make one policy cover every team size:

**Fixed width.** Teammate and opponent slots are padded with zeros up to
`maxPlayersPerTeam`. The observation is the same size in 1s as in 3s, so the
network never resizes.

**Slot shuffling.** Teammates and opponents are shuffled every step, so the
policy cannot learn "slot 2 is the good one". This is what lets it treat an
unfamiliar occupant — a human, a different bot, a demoed car — as just another
slot.

**Team inversion.** Everything is mirrored for orange, so the policy always
plays towards +Y and one set of weights covers both teams.

An empty slot is all zeros, which is not a physically reachable car state (a
real car always has a non-zero rotation basis), so the network can distinguish
"empty" from "car at the origin" without an explicit occupancy flag.

> `maxPlayersPerTeam` is baked into the observation width and therefore into the
> network's input layer. Changing it invalidates every existing checkpoint.
> Treat it as permanent once a run starts.

### Humans

Humans need no code path anywhere. From the packet's point of view a human is a
car with a position and a velocity; the padded observation cannot tell whether a
slot holds a human, a bot, or nothing. A human joining, leaving, or switching
teams mid-match changes the packet contents and nothing else.

The one thing that *does* need attention is uneven teams, which happen briefly
whenever a human joins or leaves at the wrong moment. `TeamSizeMix` in
`Config.h` trains 10% of games asymmetric (1v2, 2v3, …) so the policy has seen
that before it happens in a real match.

---

## The hivemind

RLBot v5 provides this natively. With `hivemind = true` in `bot.toml`, every car
of the same `agent_id` **on the same team** is handed to one `Bot` instance,
with all their indices in `Bot::indices`. There is no coordination protocol to
write.

Beyond tidiness, it means all our cars go through **one batched forward pass**
rather than one pass each. Single-car inference is dominated by kernel launch
overhead, so a 3v3 hivemind costs barely more than a 1v1.

Grouping is per team — blue and orange run as separate processes even with the
same `agent_id`.

---

## Training / deployment parity

Three things must match between training and deployment or the bot silently
plays worse than it did in training. There is no crash and no warning.

| Setting | Where it is set (training) | Where it is set (deployment) |
|---|---|---|
| `maxPlayersPerTeam` | `Config.h` | `HIVE_MAX_PLAYERS_PER_TEAM` |
| `tickSkip` / `actionDelay` | `Config.h` | `HIVE_TICK_SKIP`, `HIVE_ACTION_DELAY` |
| Layer sizes, activation, layer norm | `Config.h` → `ModelShape` | `ModelShape` in `HivemindBot.h` |

The layer shapes are the nastiest: libtorch will happily load weights into a
network of the wrong shape if the sizes happen to line up, and the bot will
drive around playing badly with no error.

### The two riskiest conversions

`bot/src/rlbot/PacketConvert.cpp` translates the RLBot packet into the
GameState the policy trained on. Two things there are easy to get subtly wrong:

**Boost pad ordering.** RLGymCPP indexes 34 pads by its own hardcoded location
table; RLBot orders them by y then x. These agree today, but relying on that is
a silent-corruption bug waiting to happen. We build an explicit index map from
`FieldInfo` locations at connect time instead, and warn loudly if any pad fails
to match.

**Flip availability.** The observation includes `HasFlipOrJump()`, which
RocketSim derives from several internal timers. RLBot reports the same fact
directly via `dodge_timeout`. Rather than reconstruct RocketSim's internal
state, we set those fields to whatever makes RocketSim's derivation agree with
RLBot's ground truth.

Kickoff detection on the deployment side uses RLBot's `MatchPhase` enum
(`Countdown` / `Kickoff`) rather than a heuristic — the game tells us directly,
and it transitions to `Active` on first touch, exactly where we want to hand
over. The heuristic `KickoffTracker` in `Regime.h` exists for the training side,
where no such signal is available.

---

## Deployment status

The RLBot v5 client compiles and is written against the current v5 flatbuffer
schema, but **it has not yet been run against a live match.** Two things are
needed:

1. The RLBot v5 CLI — `pipx install rlbot`. It is not currently installed.
2. A trained checkpoint to load.

When you first run it, the things most likely to need attention, in order:

- Whether `run.sh` picks the right checkpoint folder (it globs for the
  highest-numbered one).
- Whether the boost pad mapping logs any warnings at startup.
- Whether the bot's action cadence looks right — if it stutters, check
  `HIVE_TICK_SKIP` and `HIVE_ACTION_DELAY` against what you trained with.

Note that Rocket League on Linux runs under Proton, and RLBot v5 supports this,
but it is a less-travelled path than Windows.
