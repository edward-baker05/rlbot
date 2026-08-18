# Reward Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the entire reward stack with eight terms built around car control, with goals as the reference unit and every weight declared as a budget in goal-units rather than a bare per-step float.

**Architecture:** Six new `RLGC::Reward` subclasses are added to `bot/src/env/Rewards.h` first, each independently tested while the old stack still runs and the build stays green. One atomic task then swaps the config surface (`RewardWeights` → `RewardBudget`), rewrites `GeneralRewardSpecs`, and deletes every dead term — these cannot be separated without breaking the build. Metrics and the run follow.

**Tech Stack:** C++20, RocketSim/RLGymCPP (`external/GigaLearnCPP-Leak/`), doctest, CMake. Build with `scripts/build.sh`; tests run as `cd bot/build && ./HiveTests`.

**Spec:** `docs/superpowers/specs/2026-08-18-reward-redesign-design.md`

## Global Constraints

- **Tabs, not spaces.** Matches the surrounding GigaLearn/RLGymCPP style.
- **`Hive::` namespace** for our code. `RLGC::` is RLGymCPP, `GGL::` is GigaLearn.
- **Comments explain *why*,** especially where a choice looks arbitrary or a bug would be silent. Every new class carries the reasoning from its spec decision — a term whose anti-farm argument is only in a design doc will be "simplified" away later.
- **Penalty sign convention:** penalty classes return **negative** values and carry **positive** weights, matching upstream `BumpedPenalty`/`DemoedPenalty`. A negative weight on a negative class value double-negates into a reward.
- **No bare per-step weights.** Every weight reaches `WeightedReward` through `RateWeight()`, `PerSecondWeight()`, or a pass-through event budget. This is the entire point of the redesign (D1).
- **Nothing except `GoalReward` is zero-summed** (D11).
- **Tests run from `bot/build`** so collision meshes resolve: `cd bot/build && ./HiveTests`.
- Derived constants, copied verbatim from the spec: `STEPS_PER_SECOND = 15`, `REFERENCE_EPISODE_STEPS = 150`, `LANDING_REF_IMPACT = 1100` uu/s, `THROTTLE_TOP_SPEED = 1410` uu/s, `HARSH_LOSS_THRESHOLD = 400` uu/s, `CAR_MAX_SPEED = 2300` uu/s.

---

### Task 1: Surface and landing rewards

The pair that makes air play net-positive: a flat penalty for any non-wheel contact, and a bonus for the landing that avoids it. Added but **not yet wired into the spec list** — the old stack still runs and the build stays green.

**Files:**
- Modify: `bot/src/env/Rewards.h` (append new classes before the `RewardSpec` struct, around line 230)
- Test: `bot/tests/test_rewards.cpp` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Hive::WrongSurfaceReward` (default-constructible), `Hive::CleanLandingReward(float refImpact = LANDING_REF_IMPACT)`, `Hive::LANDING_REF_IMPACT` (`constexpr float`, 1100). Task 3 constructs both with no arguments.

- [ ] **Step 1: Write the failing tests**

Append to `bot/tests/test_rewards.cpp`:

```cpp
// `Player p = {}` value-initializes through CarState's constructor, which sets
// isOnGround = true. Every test below sets it explicitly rather than relying on
// that, because the default is the opposite of what most of these cases want.
TEST_CASE("WrongSurfaceReward charges only non-wheel contact") {
	WrongSurfaceReward r;
	RLGC::Player p = {};
	RLGC::GameState s = {};

	// Driving normally: wheels down, chassis clear.
	p.isOnGround = true;
	p.worldContact.hasContact = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Airborne with nothing touching: free. Leaving the ground is not the
	// offence; landing wrong is.
	p.isOnGround = false;
	p.worldContact.hasContact = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// THE GATE. Chassis scraping while the wheels are still doing their job --
	// a wall-curve transition, a bottomed-out suspension -- is not a loss of
	// control and must not be charged.
	p.isOnGround = true;
	p.worldContact.hasContact = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// On the roof, the side, the nose: all the same, all fully charged.
	p.isOnGround = false;
	p.worldContact.hasContact = true;
	CHECK(r.GetReward(p, s, false) == -1.f);
}

TEST_CASE("CleanLandingReward pays the squared impact it absorbed") {
	CleanLandingReward r(LANDING_REF_IMPACT);
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;

	// No prev: the episode just reset and the velocity discontinuity is a
	// state-setter teleport, not a landing.
	RLGC::Player orphan = {};
	orphan.isOnGround = true;
	CHECK(r.GetReward(orphan, s, false) == 0.f);

	// Still airborne: no edge.
	p.isOnGround = false;
	prev.isOnGround = false;
	prev.vel = {0, 0, -1200};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Already grounded last step: no edge either.
	p.isOnGround = true;
	prev.isOnGround = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// A real landing edge, but the chassis is touching -- this is a crash, and
	// WrongSurface is already charging for it. Paying here too would let the
	// bot buy its way out of the penalty by crashing fast.
	p.isOnGround = true;
	prev.isOnGround = false;
	p.worldContact.hasContact = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Clean landing from a held single jump (~453 uu/s). The SQUARE is the
	// anti-farm mechanism: linear would make bunny-hopping competitive with a
	// real aerial on a per-second basis.
	p.worldContact.hasContact = false;
	prev.vel = {0, 0, -450};
	CHECK(r.GetReward(p, s, false)
	      == doctest::Approx((450.f / 1100.f) * (450.f / 1100.f)).epsilon(1e-4));

	// Saturates: falling faster than the reference pays 1, not more.
	prev.vel = {0, 0, -1100};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));
	prev.vel = {0, 0, -2300};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));

	// Rising into a landing is not a fall. Only downward speed counts, so a
	// wall landing scores zero -- a known and accepted limitation.
	prev.vel = {0, 0, 900};
	CHECK(r.GetReward(p, s, false) == 0.f);
	prev.vel = {2300, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("a bunny hop is worth far less per second than an aerial") {
	// The farm argument from the design doc, as an executable check. Hop:
	// ~450 uu/s off a held jump, ~1.4 s round trip. Aerial from ~1000 uu:
	// sqrt(2*650*1000) = 1140 uu/s, ~3.5 s round trip.
	CleanLandingReward r(LANDING_REF_IMPACT);
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;
	p.isOnGround = true;
	p.worldContact.hasContact = false;
	prev.isOnGround = false;

	prev.vel = {0, 0, -450};
	const float hopRate = r.GetReward(p, s, false) / 1.4f;

	prev.vel = {0, 0, -1140};
	const float aerialRate = r.GetReward(p, s, false) / 3.5f;

	// Under LINEAR scaling these are 0.24/s vs 0.27/s and hopping is a viable
	// farm. Squared, the aerial must dominate by a wide margin.
	CHECK(aerialRate > hopRate * 2.f);
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
scripts/build.sh 2>&1 | tail -20
```

Expected: FAIL at compile time — `'WrongSurfaceReward' was not declared in this scope`, `'CleanLandingReward' was not declared in this scope`, `'LANDING_REF_IMPACT' was not declared in this scope`.

- [ ] **Step 3: Write the implementation**

In `bot/src/env/Rewards.h`, insert before the `// Returns heap-allocated rewards;` comment block (currently around line 230):

```cpp
// Penalty for any part of the car that is not its wheels being against a
// surface.
//
// `worldContact.hasContact` is set only by
// Arena::_BtCallback_OnCarWorldCollision -- the chassis hitbox producing a
// Bullet manifold against world geometry. Wheels are raycast suspension and
// never generate a manifold, so this is exactly "something that is not a wheel
// is touching a surface", and it is correct on walls, the corner curve and the
// ceiling with no plane assumption anywhere.
//
// Gated on !isOnGround, which is defined as 3+ wheels in contact and is
// therefore the in-control discriminator: if the wheels were doing their job
// you would never be inside this penalty. That gate is why there is no
// orientation grading -- being on your side is as useless as being on your
// roof, so grading would only distinguish 45-degrees-wrong from
// 90-degrees-wrong.
//
// The recovery gradient grading would have bought is unnecessary:
// Car::_UpdateAutoFlip makes escaping your roof a single discrete input (jump,
// while chassis-contacting), and the epsilon-floor patch keeps that input
// sampled.
//
// Returns a NEGATIVE value and carries a POSITIVE weight (see the sign
// convention in GeneralRewardSpecs).
class WrongSurfaceReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return (player.worldContact.hasContact && !player.isOnGround) ? -1.f : 0.f;
	}
};

// Reference impact speed for a clean landing: a ~1000 uu aerial returns at
// sqrt(2 * 650 * 1000) = 1140 uu/s. A held single jump leaves the ground at
// ~453 uu/s (JUMP_IMMEDIATE_FORCE 875/3, plus JUMP_ACCEL 4375/3 held for
// JUMP_MAX_TIME 0.2 s, less GRAVITY_Z over the hold) and returns at the same.
inline constexpr float LANDING_REF_IMPACT = 1100.f;

// Pays for arriving back on the wheels, scaled by the fall that was absorbed.
//
// This is what makes going airborne net-POSITIVE rather than merely permitted.
// Without it the only term touching air play is a penalty, in a project that
// has extinguished the jump action three times.
//
// SQUARED, and that is the entire anti-farm argument. Under linear scaling a
// bunny hop pays 0.41 every ~1.6 s (0.24/s) against a real aerial's 1.0 every
// ~3.7 s (0.27/s) -- hopping is competitive, so it is a farm. Squared, the hop
// drops to 0.098/s and the aerial dominates 2.8x. The only way to raise this
// term is to go higher, which costs time and boost that could have gone at the
// ball.
//
// Measured as downward speed rather than |vel|: using speed would pay for
// horizontal pace, double-counting SpeedSquaredReward and biasing toward the
// wall. The accepted cost is that a wall landing has no vertical component and
// scores zero.
class CleanLandingReward : public RLGC::Reward {
public:
	float refImpact;

	explicit CleanLandingReward(float refImpact = LANDING_REF_IMPACT)
		: refImpact(refImpact) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		// EnvSet::ResetArena empties prevGameStates, so a null prev means the
		// episode just reset and any velocity change is a state-setter
		// teleport, not a landing.
		if (!player.prev)
			return 0.f;

		// The landing edge, and only a clean one. Chassis contact on the same
		// step means this was a crash, which WrongSurfaceReward is already
		// charging -- paying here too would let the bot buy its way out of that
		// penalty by crashing faster.
		if (!player.isOnGround || player.prev->isOnGround || player.worldContact.hasContact)
			return 0.f;

		const float impact = RS_MAX(0.f, -player.prev->vel.z);
		const float f = RS_MIN(1.f, impact / refImpact);
		return f * f;
	}
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
scripts/build.sh 2>&1 | tail -5 && cd bot/build && ./HiveTests -ts="*Surface*,*Landing*,*hop*"
```

Expected: PASS, 3 test cases.

- [ ] **Step 5: Run the full suite to confirm nothing regressed**

```bash
cd bot/build && ./HiveTests
```

Expected: all tests pass. The old reward stack is untouched and still wired in.

- [ ] **Step 6: Commit**

```bash
git add bot/src/env/Rewards.h bot/tests/test_rewards.cpp
git commit -m "Add WrongSurface and CleanLanding rewards

The pair that makes air play net-positive: a flat penalty for any
non-wheel contact, and a bonus for the landing that avoids it. Not yet
wired into the spec list.

CleanLanding scales with impact speed SQUARED, which is what makes it
non-farmable: linear scaling puts bunny-hopping at 0.24/s against a real
aerial's 0.27/s, and squaring drops the hop to 0.098/s.

WrongSurface needs no orientation grading because !isOnGround (3+ wheels)
is already the in-control discriminator.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Touch, facing and speed rewards

The four remaining terms. Still additive — the build stays green and the old stack still runs.

**Files:**
- Modify: `bot/src/env/Rewards.h` (append after `CleanLandingReward`)
- Test: `bot/tests/test_rewards.cpp` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `Hive::TouchEdgeReward`, `Hive::FaceBallAxisReward`, `Hive::SpeedSquaredReward` (all default-constructible), `Hive::HarshSpeedLossReward(float threshold = HARSH_LOSS_THRESHOLD)`, `Hive::THROTTLE_TOP_SPEED` (`constexpr float`, 1410), `Hive::HARSH_LOSS_THRESHOLD` (`constexpr float`, 400). Task 3 constructs all four with no arguments; Task 4 reads both constants for metrics.

- [ ] **Step 1: Write the failing tests**

Append to `bot/tests/test_rewards.cpp`:

```cpp
TEST_CASE("TouchEdgeReward pays once per contact, not once per step") {
	TouchEdgeReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;

	// Not touching.
	p.ballTouchedStep = false;
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// First contact of a sequence: paid.
	p.ballTouchedStep = true;
	prev.ballTouchedStep = false;
	CHECK(r.GetReward(p, s, false) == 1.f);

	// THE DRIBBLE GUARD. Still touching from last step pays nothing. A
	// per-step touch reward IS a dribble reward -- carrying the ball on the
	// nose would collect it ~180x an episode, which is the flick-bot local
	// optimum arriving through the back door.
	p.ballTouchedStep = true;
	prev.ballTouchedStep = true;
	CHECK(r.GetReward(p, s, false) == 0.f);

	// A touch on the first step after a reset is a genuine new contact.
	RLGC::Player fresh = {};
	fresh.ballTouchedStep = true;
	CHECK(r.GetReward(fresh, s, false) == 1.f);
}

TEST_CASE("FaceBallAxisReward is the unsigned lobe of facing") {
	FaceBallAxisReward r;
	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	p.pos = {0, 0, 17};

	// Nose at the ball.
	p.rotMat.forward = {0, 1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Nose directly AWAY pays exactly the same. This is the term's defining
	// property and the reason it is split out rather than hidden inside
	// rectified FaceBall weights.
	p.rotMat.forward = {0, -1, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Perpendicular pays nothing.
	p.rotMat.forward = {1, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f).epsilon(1e-4));

	// Sitting on the ball: no direction to face, and no divide by zero.
	p.pos = s.ball.pos;
	CHECK(r.GetReward(p, s, false) == 0.f);
}

TEST_CASE("FaceBall plus FaceBallAxis reconstruct the asymmetric form") {
	// The decomposition from the design doc, as an executable assertion:
	//
	//   w+ * max(0,c) + w- * min(0,c) == ws * c + wa * |c|
	//   with ws = (w+ + w-)/2 and wa = (w+ - w-)/2
	//
	// Facing away is sometimes correct, so the negative side is weaker. That
	// asymmetry is REAL but it is not free: the |c| half pays identically for
	// nose-at-ball and nose-directly-away, and it is an annuity. Shipping it as
	// two specs is what makes it visible in RewardShare.
	RLGC::FaceBallReward signedTerm;
	FaceBallAxisReward axisTerm;

	const float ws = 0.20f;          // FaceBall budget
	const float wa = 0.20f / 3.f;    // FaceBallAxis budget: exactly ws/3
	const float wPlus = ws + wa;
	const float wMinus = ws - wa;

	// The 2:1 ratio the design fixes.
	CHECK(wPlus / wMinus == doctest::Approx(2.f).epsilon(1e-5));

	RLGC::GameState s = {};
	s.ball.pos = {0, 1000, 93};
	RLGC::Player p = {};
	p.pos = {0, 0, 17};

	// Facing the ball: the pair must equal the positive rectified weight.
	p.rotMat.forward = {0, 1, 0};
	float combined = ws * signedTerm.GetReward(p, s, false)
	               + wa * axisTerm.GetReward(p, s, false);
	CHECK(combined == doctest::Approx(wPlus * 1.f).epsilon(1e-4));

	// Facing away: the pair must equal the WEAKER negative rectified weight.
	p.rotMat.forward = {0, -1, 0};
	combined = ws * signedTerm.GetReward(p, s, false)
	         + wa * axisTerm.GetReward(p, s, false);
	CHECK(combined == doctest::Approx(wMinus * -1.f).epsilon(1e-4));
	// And it must still be negative -- weaker, not inverted.
	CHECK(combined < 0.f);
}

TEST_CASE("SpeedSquaredReward discounts the free coasting floor") {
	SpeedSquaredReward r;
	RLGC::GameState s = {};
	RLGC::Player p = {};

	p.vel = {0, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.f));

	// Throttle-only top speed: holdable forever with no boost and no skill.
	// Linear scaling would hand this 0.613 of the term's maximum for free;
	// squaring cuts it to 0.375.
	p.vel = {THROTTLE_TOP_SPEED, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(0.375f).epsilon(0.01));

	// Supersonic.
	p.vel = {RLGC::CommonValues::CAR_MAX_SPEED, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f).epsilon(1e-4));

	// Boost/flip speed must beat coasting by 2.67x, not 1.63x. That ratio is
	// the reason for the square.
	p.vel = {THROTTLE_TOP_SPEED, 0, 0};
	const float coast = r.GetReward(p, s, false);
	p.vel = {RLGC::CommonValues::CAR_MAX_SPEED, 0, 0};
	const float fast = r.GetReward(p, s, false);
	CHECK(fast / coast == doctest::Approx(2.67f).epsilon(0.02));

	// Clamped, not extrapolated -- a bump or a ramp can exceed CAR_MAX_SPEED.
	p.vel = {5000, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(1.f));
}

TEST_CASE("HarshSpeedLossReward charges collisions, not braking or hits") {
	HarshSpeedLossReward r(HARSH_LOSS_THRESHOLD);
	RLGC::GameState s = {};
	RLGC::Player p = {}, prev = {};
	p.prev = &prev;

	// No prev: a reset, not a crash.
	RLGC::Player orphan = {};
	orphan.vel = {0, 0, 0};
	CHECK(r.GetReward(orphan, s, false) == 0.f);

	// Gaining speed.
	prev.vel = {500, 0, 0};
	p.vel = {900, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Hard braking: ~3500 uu/s^2 over a 1/15 s step is 233 uu/s, comfortably
	// under the 400 threshold. Deliberate deceleration must be free.
	prev.vel = {1400, 0, 0};
	p.vel = {1167, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// Exactly at the threshold is still free.
	prev.vel = {1400, 0, 0};
	p.vel = {1000, 0, 0};
	CHECK(r.GetReward(p, s, false) == 0.f);

	// A wall. Charged, and negative.
	prev.vel = {1400, 0, 0};
	p.vel = {0, 0, 0};
	const float crash = r.GetReward(p, s, false);
	CHECK(crash < 0.f);
	CHECK(crash == doctest::Approx(-(1400.f - 400.f) / (2300.f - 400.f)).epsilon(1e-4));

	// THE HIT EXEMPTION. A hard strike costs speed, and that is a good
	// outcome. Charging for it would penalise striking the ball -- the exact
	// mistake p4pbrs made from the other direction.
	p.ballTouchedStep = true;
	CHECK(r.GetReward(p, s, false) == 0.f);
	p.ballTouchedStep = false;

	// Saturates at a full-speed stop.
	prev.vel = {2300, 0, 0};
	p.vel = {0, 0, 0};
	CHECK(r.GetReward(p, s, false) == doctest::Approx(-1.f).epsilon(0.01));
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
scripts/build.sh 2>&1 | tail -20
```

Expected: FAIL at compile time — `'TouchEdgeReward' was not declared in this scope` and similar for the other three classes and two constants.

- [ ] **Step 3: Write the implementation**

In `bot/src/env/Rewards.h`, append after `CleanLandingReward`:

```cpp
// One payment per contact SEQUENCE, not per step of contact.
//
// A per-step touch reward IS a dribble reward: carrying the ball on the nose
// collects it every step, roughly 180 times in an episode. That is the
// flick-bot local optimum (roadmap spec D4) arriving through the back door.
// The rising edge makes carrying the ball worth exactly one touch, so the term
// pays for ARRIVING at the ball and nothing else.
class TouchEdgeReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep)
			return 0.f;

		// A null prev means the episode just reset, so a touch on this step is
		// a genuine new contact rather than the continuation of one.
		return (player.prev && player.prev->ballTouchedStep) ? 0.f : 1.f;
	}
};

// |forward . dirToBall|, shipped alongside upstream's signed FaceBallReward
// because together the two ARE the asymmetric form:
//
//   w+ * max(0,c) + w- * min(0,c)  ==  ((w+ + w-)/2) * c  +  ((w+ - w-)/2) * |c|
//
// Facing away from the ball is sometimes correct (shadow defence, retreating
// for a bounce), so the negative side should be weaker than the positive. But
// implementing that as rectified weights silently ships the second component,
// which pays IDENTICALLY for nose-at-ball and nose-directly-away and pays zero
// for perpendicular -- and which is an annuity, since for a policy with no
// facing preference c is uniform on [-1,1] and E|c| = 0.5.
//
// Split out so it gets its own RewardShare line and its own budget.
// RewardShare reports mean |r*w| and cannot tell a signed term from a
// rectified one, which is exactly how this would have gone unnoticed.
class FaceBallAxisReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const Vec toBall = state.ball.pos - player.pos;
		const float len = toBall.Length();
		if (len < 1e-4f)
			return 0.f;

		return std::fabs(player.rotMat.forward.Dot(toBall / len));
	}
};

// Throttle-only top speed: DRIVE_SPEED_TORQUE_FACTOR_CURVE reaches zero here,
// so any car can hold this indefinitely with no boost and no skill.
inline constexpr float THROTTLE_TOP_SPEED = 1410.f;

// (|v| / CAR_MAX_SPEED)^2.
//
// Squared, not linear, because linear leaves 1410/2300 = 0.613 of the term's
// maximum as a free annuity for holding throttle in a straight line. Squaring
// cuts that to 0.375 and raises the payoff for boost- and flip-derived speed
// over coasting from 1.63x to 2.67x, while keeping a rising gradient from zero
// so the term still bootstraps.
//
// GENERIC speed, not speed-toward-ball, on purpose. The ball-directed form is
// a PRODUCT of speed and alignment, and its cross term
// (d2R/d|v| dcos = 1/V, nonzero) charges a steering input on both factors at
// once: turning scrubs speed AND misaligns velocity. That is what drove
// Action/Steer Nonzero to 0.0006 on p3strike. SpeedSquared + FaceBall is the
// same intent factored additively, where a turn is charged once.
//
// The same factoring is why previous bots never flipped or boosted for speed:
// a flip's impulse is along the car's forward axis and costs ~1.25 s of
// steering authority, so under the product form its value is gated by
// alignment, while under |v| it is paid unconditionally.
class SpeedSquaredReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const float f = RS_MIN(1.f, player.vel.Length() / RLGC::CommonValues::CAR_MAX_SPEED);
		return f * f;
	}
};

// Speed that can be lost in one decision step without a collision. RL brakes at
// roughly 3500 uu/s^2, which over a 1/15 s step is 233 uu/s, so 400 is clear of
// any input-driven deceleration and only a collision reaches it.
//
// That 3500 is EMPIRICAL, not a RocketSim constant -- BRAKE_TORQUE_AMOUNT is a
// wheel torque and does not convert directly. The Speed/Max Step Decel metric
// exists to check this threshold against the real distribution. Do not treat
// 400 as settled.
inline constexpr float HARSH_LOSS_THRESHOLD = 400.f;

// Penalty for losing a lot of speed in one step: a wall, a bad recovery, a
// botched landing.
//
// Deliberately overlaps SpeedSquaredReward, which already makes losing speed
// cost future reward. What this adds is a sharp, single-step signal
// attributable to the collision itself, which is worth real money for credit
// assignment when gaeLambda puts the direct credit horizon around 1 second. It
// also fires alongside WrongSurfaceReward on the same events. Both overlaps are
// intentional and are recorded so the RewardShare numbers are not misread.
//
// Returns a NEGATIVE value and carries a POSITIVE weight.
class HarshSpeedLossReward : public RLGC::Reward {
public:
	float threshold;

	explicit HarshSpeedLossReward(float threshold = HARSH_LOSS_THRESHOLD)
		: threshold(threshold) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.prev)
			return 0.f;

		// A hard hit SHOULD cost speed: that is a good outcome, not a bad
		// recovery. Charging for it would penalise striking the ball.
		if (player.ballTouchedStep)
			return 0.f;

		const float lost = player.prev->vel.Length() - player.vel.Length();
		if (lost <= threshold)
			return 0.f;

		const float span = RLGC::CommonValues::CAR_MAX_SPEED - threshold;
		return -RS_MIN(1.f, (lost - threshold) / span);
	}
};
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
scripts/build.sh 2>&1 | tail -5 && cd bot/build && ./HiveTests -ts="*Touch*,*Face*,*Speed*"
```

Expected: PASS, 5 new test cases (plus the pre-existing touch tests, which still pass).

- [ ] **Step 5: Run the full suite**

```bash
cd bot/build && ./HiveTests
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add bot/src/env/Rewards.h bot/tests/test_rewards.cpp
git commit -m "Add touch-edge, facing-axis, speed and harsh-loss rewards

Touch fires on the RISING EDGE of ballTouchedStep, so carrying the ball
is worth one touch rather than ~180 -- a per-step touch reward is a
dribble reward and would reintroduce the flick-bot optimum.

FaceBallAxis is the |cos| half of the asymmetric facing form, split out
rather than hidden in rectified weights: it pays identically for facing
the ball and facing directly away, and it is an annuity. Splitting it
gives it a RewardShare line, which a rectified term would not have.

Speed is squared so the free coasting floor at 1410 uu/s falls from 61%
to 37.5% of the term's maximum. Generic rather than ball-directed: the
product form's cross term charges a turn on both speed and alignment,
which is what drove Steer Nonzero to 0.0006 on p3strike.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: The switch — budgets in, old stack out

One atomic change. The config surface, the spec list and the deletions cannot be separated while keeping the build green: `Train.cpp`'s `Pay/*` block calls `StrongTouchValue`/`AimMultiplier` and reads `g_RewardWeights.strongTouch`, so those die with the terms they measure. A reviewer cannot sensibly approve the new spec list while rejecting the config swap — they are the same change.

**Files:**
- Modify: `bot/src/Config.h` (delete `RewardPhase` enum ~line 51, replace `RewardWeights` ~lines 58-88, change `TrainConfig::rewardPhase` and `TrainConfig::rewards`)
- Modify: `bot/src/env/Rewards.h` (delete the old classes and free functions)
- Modify: `bot/src/env/Rewards.cpp` (rewrite `GeneralRewardSpecs`, delete `TouchHeightReward::GetReward`)
- Modify: `bot/src/main.cpp` (delete `--reward-phase` parsing ~lines 119-133 and its usage line ~46)
- Modify: `bot/src/train/Train.cpp` (delete `g_RewardWeights` ~line 57 and the `Pay/*` block ~lines 249-305)
- Test: `bot/tests/test_rewards.cpp` (delete tests for removed terms, add budget tests)

**Interfaces:**
- Consumes: all six classes and four constants from Tasks 1-2.
- Produces: `Hive::RateWeight(float)`, `Hive::PerSecondWeight(float)` (both `constexpr`), `Hive::RewardBudget` with fields `speed`, `faceBall`, `faceBallAxis`, `touch`, `cleanLanding`, `harshSpeedLoss`, `wrongSurface` and `static constexpr float GOAL`. `TrainConfig::rewards` becomes `RewardBudget`. Spec names, in order, become: `Goal`, `Touch`, `CleanLanding`, `WrongSurface`, `HarshSpeedLoss`, `Speed`, `FaceBall`, `FaceBallAxis` — Task 4's metrics rely on these exact strings.

- [ ] **Step 1: Write the failing tests**

Replace the **entire contents** of `bot/tests/test_rewards.cpp` below the includes with the Task 1 and Task 2 test cases (keep them verbatim) plus these, and **delete** the now-obsolete cases: `GroundedReward pays only on wheels`, `TouchHeightReward pays zero...`, `TouchHeightReward scales...`, `RewardPhase::Aerial swaps...`, `AimMultiplier smoothly prefers...`, `AimedStrongTouchReward pays only...`, `BallProgressReward telescopes...`, `BallGoalProgressReward rewards...`.

Keep `specs and built rewards agree in count and weight` and `spec names are unique` — they still apply. Add:

```cpp
TEST_CASE("budget conversion is the only route to a per-step weight") {
	// A rate budget is what holding the behaviour perfectly for one reference
	// episode earns. 150 steps = 10 s at 15 Hz.
	CHECK(REFERENCE_EPISODE_STEPS == doctest::Approx(150.f));
	CHECK(RateWeight(0.30f) == doctest::Approx(0.30f / 150.f));
	CHECK(RateWeight(0.30f) * REFERENCE_EPISODE_STEPS == doctest::Approx(0.30f));

	// A per-second budget is the cost of one second of the condition.
	CHECK(PerSecondWeight(0.10f) == doctest::Approx(0.10f / 15.f));
	CHECK(PerSecondWeight(0.10f) * STEPS_PER_SECOND == doctest::Approx(0.10f));
}

TEST_CASE("no shaping term can outearn a goal by accident") {
	// The p1air failure, as a regression test. `grounded = 0.05` integrated to
	// 9.0 goal-units per episode -- nine goals per episode for holding still on
	// the wheels -- and nobody noticed because nobody wrote down the integral.
	//
	// Every RATE term's whole-episode earnings must stay well under one goal.
	const RewardBudget b = {};
	CHECK(b.speed < RewardBudget::GOAL);
	CHECK(b.faceBall < RewardBudget::GOAL);
	CHECK(b.faceBallAxis < RewardBudget::GOAL);

	// And all of them together must not outweigh a goal either.
	CHECK(b.speed + b.faceBall + b.faceBallAxis < RewardBudget::GOAL);
}

TEST_CASE("FaceBallAxis is exactly one third of FaceBall") {
	// This ratio IS the 2:1 asymmetry: w+ = ws + wa, w- = ws - wa, and
	// wa = ws/3 gives w+/w- = 2 exactly. If the two budgets drift apart the
	// asymmetry silently becomes something else.
	const RewardBudget b = {};
	CHECK(b.faceBallAxis == doctest::Approx(b.faceBall / 3.f).epsilon(1e-5));

	const float wPlus = b.faceBall + b.faceBallAxis;
	const float wMinus = b.faceBall - b.faceBallAxis;
	CHECK(wPlus / wMinus == doctest::Approx(2.f).epsilon(1e-4));
}

TEST_CASE("the spec list is the eight designed terms, with positive weights") {
	auto specs = GeneralRewardSpecs(TrainConfig{});

	std::vector<std::string> names;
	for (auto& s : specs)
		names.push_back(s.name);

	const std::vector<std::string> expected = {
		"Goal", "Touch", "CleanLanding", "WrongSurface",
		"HarshSpeedLoss", "Speed", "FaceBall", "FaceBallAxis",
	};
	CHECK(names == expected);

	// THE SIGN CONVENTION. Penalty classes return negative values, so every
	// weight must be positive -- a negative weight on a negative class value
	// double-negates a penalty into a reward, and nothing else in the stack
	// would reveal it.
	for (auto& s : specs)
		CHECK(s.weight > 0.f);

	// Goal is the unit.
	CHECK(specs[0].weight == doctest::Approx(1.f));
}

TEST_CASE("no zero-weight placeholder specs remain") {
	// The old stack kept zero-weight specs so RewardShare indices stayed
	// aligned across reward phases. There are no phases now, so a zero-weight
	// spec would just be a term that silently does nothing.
	for (auto& s : GeneralRewardSpecs(TrainConfig{}))
		CHECK(s.weight != 0.f);
}
```

Add `#include <string>` and `#include <vector>` to the test file's includes if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
scripts/build.sh 2>&1 | tail -20
```

Expected: FAIL at compile time — `'RewardBudget' was not declared`, `'RateWeight' was not declared`, `'REFERENCE_EPISODE_STEPS' was not declared`.

- [ ] **Step 3a: Replace the config surface**

In `bot/src/Config.h`, **delete** the entire `enum class RewardPhase { ... };` block and the entire `struct RewardWeights { ... };` block, and insert in their place:

```cpp
// --- Reward budgets --------------------------------------------------------
//
// Every reward weight in this project is declared as a BUDGET in goal-units and
// converted to a per-step weight in exactly one place (Hive::GeneralRewardSpecs).
// A goal is 1.0 by definition; nothing else may be written as a bare per-step
// float.
//
// This exists because p1air's `grounded = 0.05` integrated to 9.0 goal-units
// per episode -- nine goals per episode for holding still on the wheels -- and
// nobody noticed, because nobody wrote down the integral. Declaring the
// integral makes that class of mistake unrepresentable.

// tickSkip 8 at RocketSim's 120 Hz.
inline constexpr float STEPS_PER_SECOND = 15.f;

// Working figure for one episode. noTouchTimeout caps a never-touching bot at
// 180 steps and goals end episodes early, so 10 s is a reasonable middle. The
// Episode/Mean Steps metric exists so this is re-derived from telemetry rather
// than staying a guess.
inline constexpr float REFERENCE_EPISODE_SECONDS = 10.f;
inline constexpr float REFERENCE_EPISODE_STEPS =
	STEPS_PER_SECOND * REFERENCE_EPISODE_SECONDS;

// Goal-units earned by holding a behaviour perfectly for one reference episode,
// converted to the per-step weight that earns it.
inline constexpr float RateWeight(float budgetPerEpisode) {
	return budgetPerEpisode / REFERENCE_EPISODE_STEPS;
}

// Goal-units of cost for one second of a condition, converted to per-step.
inline constexpr float PerSecondWeight(float budgetPerSecond) {
	return budgetPerSecond / STEPS_PER_SECOND;
}

// All values are goal-units. See docs/superpowers/specs/2026-08-18-reward-redesign-design.md
// for the derivation of each; none of them is a guess.
struct RewardBudget {
	// --- Rate: earned by holding the behaviour for one reference episode ---

	// Squared, so 0.375 of this is the free coasting floor at 1410 uu/s. The
	// whole-episode maximum equals exactly two ball touches, which is the cap
	// on how much a speed farm can ever be worth.
	float speed = 0.30f;

	float faceBall = 0.20f;

	// Exactly faceBall/3, which is the 2:1 asymmetry (w- = w+/2) expressed
	// through the decomposition. Changing one without the other changes the
	// asymmetry silently.
	float faceBallAxis = 0.20f / 3.f;

	// --- Event: earned per occurrence ---
	float touch = 0.15f;
	float cleanLanding = 0.10f;

	// Cost of a full-speed crash.
	float harshSpeedLoss = 0.10f;

	// --- Per second ---
	// Three seconds on your roof costs 0.30, which is 3x what the best possible
	// landing pays. Recovery is worth more as an avoided loss than as a bonus,
	// which is what stops the landing bonus becoming the objective.
	float wrongSurface = 0.10f;

	// The unit. Not adjustable, and scaling it could not help anyway: rewards
	// are standardized at GAE.cpp:52, and for a rare terminal payoff the
	// signal-to-noise sqrt(p/(1-p)) is independent of magnitude. The only lever
	// on the goal signal is how often goals happen.
	static constexpr float GOAL = 1.f;
};
```

Then in `struct TrainConfig`, **delete** the line `RewardPhase rewardPhase = RewardPhase::Foundations;` and change `RewardWeights rewards = {};` to:

```cpp
  RewardBudget rewards = {};
```

Also update the `CurriculumWeights` doc comment, which currently says "Matched to `RewardPhase::Foundations`" — replace that sentence with "Re-derive them from telemetry at each phase gate."

- [ ] **Step 3b: Rewrite the spec list**

In `bot/src/env/Rewards.h`, **delete** these entirely: `TOUCH_MIN_KPH`, `TOUCH_MAX_KPH`, `StrongTouchValue`, `AimMultiplier`, `TouchHeightReward`, `GroundedReward`, `AirRecoveryReward`, `GroundedBonusReward`, `AimedStrongTouchReward`, `BallProgressReward`, `BallGoalProgressReward`. Keep the `RewardSpec` struct and both builder declarations. The six classes from Tasks 1-2 stay.

Replace `bot/src/env/Rewards.cpp` entirely with:

```cpp
#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>

using namespace RLGC;

namespace Hive {

std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg) {
	const RewardBudget& b = cfg.rewards;

	// Budgets become per-step weights HERE and nowhere else. That single
	// conversion site is the whole point of the redesign: p1air's do-nothing
	// attractor was a per-step float whose episode integral nobody computed.
	//
	// SIGN CONVENTION: penalty classes return negative values and carry
	// POSITIVE weights, matching upstream BumpedPenalty/DemoedPenalty. A
	// negative weight here would double-negate a penalty into a reward, and
	// nothing downstream would reveal it.
	//
	// Nothing except Goal is ZeroSum-wrapped. These are car-control terms;
	// wrapping them would add opponent variance for no competitive meaning,
	// and GoalReward already carries the entire adversarial structure.
	return {
		// The unit. Already zero-sum: +1 scored, -1 conceded.
		{"Goal", RewardBudget::GOAL, [] { return new GoalReward(); }},

		{"Touch", b.touch, [] { return new TouchEdgeReward(); }},
		{"CleanLanding", b.cleanLanding, [] { return new CleanLandingReward(); }},
		{"WrongSurface", PerSecondWeight(b.wrongSurface),
		 [] { return new WrongSurfaceReward(); }},
		{"HarshSpeedLoss", b.harshSpeedLoss, [] { return new HarshSpeedLossReward(); }},

		{"Speed", RateWeight(b.speed), [] { return new SpeedSquaredReward(); }},
		// Upstream's FaceBallReward is already the full 3-D dot product; the
		// old stack gated it to grounded steps, which left the airborne policy
		// with no directional signal at all. Ungated here.
		{"FaceBall", RateWeight(b.faceBall), [] { return new FaceBallReward(); }},
		{"FaceBallAxis", RateWeight(b.faceBallAxis),
		 [] { return new FaceBallAxisReward(); }},
	};
}

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig& cfg) {
	std::vector<WeightedReward> out;
	for (auto& spec : GeneralRewardSpecs(cfg))
		out.push_back({spec.make(), spec.weight});
	return out;
}

} // namespace Hive
```

- [ ] **Step 3c: Delete the CLI flag**

In `bot/src/main.cpp`, delete the usage line `"  --reward-phase P     foundations (default) or aerial\n"` and the entire `else if (arg == "--reward-phase" && i + 1 < argc) { ... }` branch including its comment block.

- [ ] **Step 3d: Delete the Pay/\* metrics**

In `bot/src/train/Train.cpp`:
- Delete the `g_RewardWeights` declaration and its two-line comment.
- Delete `g_RewardWeights = cfg.rewards;` in `RunTraining()`.
- Delete the whole `// --- Which term pays for a flip? ---` block through the closing brace of its `if (state.prev)`, i.e. everything from that comment down to and including the `Pay/VelPlayerToBall Per Step Grounded` block.

Keep `Touch/Above 200`, `Touch/Above 300`, `Touch/Above 450`, `Touch/Had Jumped`, `Touch/Had Flipped`, `Touch/Rate Airborne`, `Touch/Rate Grounded` and `Player/Touch Height` — they measure behaviour, not deleted reward terms. Relocate them above the deleted block so they survive, wrapped in `if (state.prev)` where they need it.

- [ ] **Step 4: Run tests to verify they pass**

```bash
scripts/build.sh 2>&1 | tail -5 && cd bot/build && ./HiveTests
```

Expected: all tests pass, including the five new budget/spec cases.

- [ ] **Step 5: Confirm the reward stack loads at runtime**

```bash
cd bot/build && ./HivemindBot train --games 4 --max-steps 20000 --no-metrics --label smoke-rewards 2>&1 | tail -30
```

Expected: trains and exits on the step budget with no crash. If `RewardShare/*` names appear in console output they should be the eight new ones.

- [ ] **Step 6: Commit**

```bash
git add bot/src/Config.h bot/src/env/Rewards.h bot/src/env/Rewards.cpp \
        bot/src/main.cpp bot/src/train/Train.cpp bot/tests/test_rewards.cpp
git commit -m "Replace the reward stack with goal-referenced budgets

Weights are now declared as per-episode or per-event budgets in
goal-units and converted at exactly one site. A term can no longer
integrate to nine goals per episode without someone typing 9.0 into a
field labelled goal-units per episode, which is what p1air's
grounded=0.05 did unnoticed.

Eight terms: Goal (the unit, 1.0), Touch, CleanLanding, WrongSurface,
HarshSpeedLoss, Speed, FaceBall, FaceBallAxis. Nothing but Goal is
zero-summed.

Deletes AirRecovery, GroundedBonus, GroundedReward, TouchHeight,
AimedStrongTouch, AimMultiplier, StrongTouchValue, BallProgress,
BallGoalProgress, PickupBoost, VelPlayerToBall, the RewardPhase enum,
--reward-phase, and Train.cpp's Pay/* block (which recomputed the
deleted terms).

Accepted consequences: the p1air comparison baseline is gone, and boost
is unrewarded so the bot may run itself dry.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Metrics for the new terms

Without these the run is unreadable — every pre-committed trigger in the spec names a metric that does not exist yet.

**Files:**
- Modify: `bot/src/train/Train.cpp` (episode-age block ~line 175; player loop ~line 215)
- Modify: `scripts/summarize_runs.py` (`INTERESTING` list ~line 29)

**Interfaces:**
- Consumes: `Hive::THROTTLE_TOP_SPEED` and `Hive::HARSH_LOSS_THRESHOLD` from Task 2. `Train.cpp` already includes `../env/Rewards.h`.
- Produces: metric names `Surface/Wrong Contact Rate`, `Surface/Wrong Contact While Flipping`, `Landing/Rate`, `Landing/Clean Share`, `Landing/Impact Speed`, `Speed/Mean`, `Speed/Above Throttle Cap Share`, `Speed/Harsh Loss Rate`, `Speed/Max Step Decel`, `FaceBall/Mean Cos`, `FaceBall/Axis Share`, `Touch/Edge Rate`, `Episode/Mean Steps`.

- [ ] **Step 1: Add `Episode/Mean Steps`**

In `Train.cpp`'s episode-age block, replace:

```cpp
			g_EpisodeAge[a]++;
			if (a < es.terminals.size() && es.terminals[a])
				g_EpisodeAge[a] = 0;
```

with:

```cpp
			g_EpisodeAge[a]++;
			if (a < es.terminals.size() && es.terminals[a]) {
				// Re-derives REFERENCE_EPISODE_STEPS, which ships as a working
				// figure of 150. Recorded at the terminal step so it is a real
				// episode length rather than a running age, and outside the
				// sampling gate so no episode is missed.
				report.AddAvg("Episode/Mean Steps", static_cast<float>(g_EpisodeAge[a]));
				g_EpisodeAge[a] = 0;
			}
```

- [ ] **Step 2: Add the per-player metrics**

In the player loop, immediately after the `Player/Speed Towards Ball` block (which already computes `toBall` and `dist`), insert:

```cpp
			// --- Surface contact ---------------------------------------------
			// The WrongSurface term as a rate. If this does not fall over a
			// run, the bot is not learning to land, whatever the reward says.
			const bool wrongSurface = player.worldContact.hasContact && !player.isOnGround;
			report.AddAvg("Surface/Wrong Contact Rate", wrongSurface ? 1.f : 0.f);

			// A front flip drives the nose into the floor for a few ticks, and
			// Player samples the final tick of eight, so the scrape is caught
			// ~25% of the time. The design prices that at ~11x cheaper than the
			// flip's own speed gain; this is the check on that arithmetic.
			if (player.isFlipping)
				report.AddAvg("Surface/Wrong Contact While Flipping", wrongSurface ? 1.f : 0.f);

			// --- Landings -----------------------------------------------------
			if (player.prev && player.isOnGround && !player.prev->isOnGround) {
				report.AddAvg("Landing/Rate", 1.f);
				report.AddAvg("Landing/Clean Share", player.worldContact.hasContact ? 0.f : 1.f);
				report.AddAvg("Landing/Impact Speed", RS_MAX(0.f, -player.prev->vel.z));
			} else {
				report.AddAvg("Landing/Rate", 0.f);
			}

			// --- Speed --------------------------------------------------------
			const float speed = player.vel.Length();
			report.AddAvg("Speed/Mean", speed);

			// Is the squared term buying boost- and flip-derived speed, or is
			// the bot parked on the free coasting floor? This is half of the
			// pre-committed farm trigger: this rising while Touch/Edge Rate
			// stays flat means the speed term is being farmed.
			report.AddAvg("Speed/Above Throttle Cap Share",
			              speed > THROTTLE_TOP_SPEED ? 1.f : 0.f);

			if (player.prev) {
				const float lost = player.prev->vel.Length() - speed;
				// Validates HARSH_LOSS_THRESHOLD against the real
				// distribution. The 400 uu/s figure rests on an empirical
				// ~3500 uu/s^2 braking rate, not a RocketSim constant, so it
				// has to be checked rather than trusted. Hits are excluded for
				// the same reason the reward excludes them.
				if (!player.ballTouchedStep) {
					report.AddAvg("Speed/Max Step Decel", RS_MAX(0.f, lost));
					report.AddAvg("Speed/Harsh Loss Rate",
					              lost > HARSH_LOSS_THRESHOLD ? 1.f : 0.f);
				}
			}

			// --- Facing -------------------------------------------------------
			// If the |c| lobe is being farmed, Mean Cos goes negative while
			// Axis Share stays high: the bot is driving away from the ball and
			// being paid for it. RewardShare cannot show this, because it
			// reports mean |r*w| and cannot tell a signed term from a
			// rectified one.
			if (dist > 1.f) {
				const float cosToBall = player.rotMat.forward.Dot(toBall / dist);
				report.AddAvg("FaceBall/Mean Cos", cosToBall);
				report.AddAvg("FaceBall/Axis Share", std::fabs(cosToBall));
			}

			// --- Touch edge ---------------------------------------------------
			// The rate the Touch term actually pays at, as opposed to
			// Player/Ball Touch Ratio which counts every step of contact. The
			// gap between them is how much carrying is happening.
			report.AddAvg("Touch/Edge Rate",
			              (player.ballTouchedStep
			               && !(player.prev && player.prev->ballTouchedStep)) ? 1.f : 0.f);
```

- [ ] **Step 3: Add the new metrics to the run summary**

In `scripts/summarize_runs.py`, replace the `INTERESTING` list with:

```python
INTERESTING = [
    ("Average Step Reward", "reward", "{:.4f}"),
    ("Policy Entropy", "entropy", "{:.4f}"),
    ("Player/Ball Touch Ratio", "touch ratio", "{:.4f}"),
    ("Touch/Edge Rate", "touch edges", "{:.5f}"),
    ("Player/Touch Height", "touch height", "{:.1f}"),
    ("Surface/Wrong Contact Rate", "wrong surface", "{:.4f}"),
    ("Landing/Clean Share", "clean landings", "{:.3f}"),
    ("Landing/Impact Speed", "landing impact", "{:.1f}"),
    ("Speed/Mean", "speed", "{:.1f}"),
    ("Speed/Above Throttle Cap Share", "above coast", "{:.3f}"),
    ("Speed/Harsh Loss Rate", "harsh losses", "{:.4f}"),
    ("FaceBall/Mean Cos", "facing", "{:.3f}"),
    ("Player/In Air Ratio", "air ratio", "{:.4f}"),
    ("Player/Boost", "boost", "{:.1f}"),
    ("Episode/Mean Steps", "episode steps", "{:.1f}"),
    ("Overall Steps/Second", "steps/sec", "{:,.0f}"),
]
```

- [ ] **Step 4: Build and verify the metrics emit**

```bash
scripts/build.sh 2>&1 | tail -5 && cd bot/build && \
  ./HivemindBot train --games 4 --max-steps 40000 --no-metrics --label smoke-metrics 2>&1 | tail -40
```

Expected: runs to the budget and exits cleanly. Then confirm the names reached the CSV:

```bash
head -1 bot/build/metrics/main-smoke-metrics.csv | tr ',' '\n' | grep -E 'Surface/|Landing/|Speed/|FaceBall/|Touch/Edge|Episode/Mean'
```

Expected: all 13 new names present. **A missing name here is the failure mode this project cannot afford** — a metric that silently never emits looks exactly like a metric that is flat.

- [ ] **Step 5: Run the full test suite**

```bash
cd bot/build && ./HiveTests
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add bot/src/train/Train.cpp scripts/summarize_runs.py
git commit -m "Add metrics for the new reward terms

Every pre-committed trigger in the design doc names a metric, so without
these the run is unreadable. Speed/Above Throttle Cap Share rising while
Touch/Edge Rate stays flat is the speed-farm signature; FaceBall/Mean Cos
going negative while Axis Share stays high is the |c|-lobe farm, which
RewardShare cannot detect because it reports mean |r*w|.

Speed/Max Step Decel exists to validate HARSH_LOSS_THRESHOLD, which rests
on an empirical braking figure rather than a RocketSim constant.
Episode/Mean Steps re-derives REFERENCE_EPISODE_STEPS.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Verify, document, and launch run A

**Files:**
- Modify: `CLAUDE.md` (the "Verified working" section and the reward-related standing decisions)
- Modify: `runs/RUNLOG.md` (append the p6 row after the run)

- [ ] **Step 1: Confirm the external patches are all applied**

The epsilon-floor is what keeps jump from extinguishing, and it is the one mechanism this design depends on that lives outside our tree.

```bash
scripts/apply_external_patches.py --check
grep -n 'ACTION_EXPLORE_EPS' external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp
grep -n 'advantages.mean()' external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp
```

Expected: the check reports all patches applied; `ACTION_EXPLORE_EPS = 0.02` present; advantage standardization present. **If the advantage-standardization patch is missing, stop and reapply it** — it is a manual patch and a re-clone drops it silently.

- [ ] **Step 2: Full clean verification**

```bash
scripts/build.sh 2>&1 | tail -5 && cd bot/build && ./HiveTests
```

Expected: build succeeds, all tests pass.

- [ ] **Step 3: Update CLAUDE.md**

In the **Layout / standing decisions** area, replace the sentence about rewards with a pointer to the new spec, and add to the standing decisions:

```markdown
Three standing decisions from that spec worth knowing before touching
rewards: no dribble/possession reward terms, ever (D4 — the flick-bot local
optimum); no magic numbers without measurement behind them (D6); and, from
`docs/superpowers/specs/2026-08-18-reward-redesign-design.md`, **every reward
weight is a budget in goal-units, converted in exactly one place**. Do not add
a per-step float. Do not add ball-directed dense shaping — the product form's
cross term charges a turn on both speed and alignment, which is what
extinguished steering on p3strike (D3).
```

Update the **Verified working** date line to note the reward rewrite.

- [ ] **Step 4: Launch run A**

```bash
cd bot/build && nohup ./HivemindBot train \
  --games 128 \
  --max-steps 100000000 \
  --track-skill \
  --label p6budget \
  > "$PWD/../../runs/p6budget.log" 2>&1 &
echo "started pid $!"
```

**Expect ~31 minutes.** Measured from `main-p5goalpot.csv`, which ran this same
shape: 100.2M steps at a final `Overall Steps/Second` of 53,990 (mean 51,949)
= 0.52 h. Note this is well below CLAUDE.md's "~81k steps/sec at 128 games" —
that figure predates skill tracking, whose 8 evaluation arenas compete with
training for CPU. Use ~52k for planning, not 81k.

`--track-skill` without `--self-play` matches the comparable-baseline
convention in the usage text, and is what p5goalpot used, so ratings are
comparable.

- [ ] **Step 5: Monitor to completion**

Check progress periodically:

```bash
tail -5 runs/p6budget.log
python3 scripts/summarize_runs.py --trend bot/build/metrics/main-p6budget.csv
```

- [ ] **Step 6: Analyse against the pre-committed criteria**

From the spec's success criteria — these are decision rules, not judgement calls:

| Check | Threshold | Meaning if it fails |
|---|---|---|
| `Player/Ball Touch Ratio` | beat 0.0021 at equal steps | the redesign did not help |
| `Action/Jump When Grounded Upright` | stay above ~0.011 | air play died again despite D5/D6 |
| `RewardShare/Speed` | under 0.25 | trigger the D8 above-floor swap |
| `Speed/Above Throttle Cap Share` rising while `Touch/Edge Rate` flat | must not co-occur | speed farm confirmed |
| `Surface/Wrong Contact While Flipping` | flips not suppressed | the flip tax was mispriced |
| `Speed/Max Step Decel` distribution | validates `T = 400` | threshold needs re-deriving |
| `FaceBall/Mean Cos` negative while `Axis Share` high | must not co-occur | the \|c\| lobe is being farmed |
| `Episode/Mean Steps` | re-derive `REFERENCE_EPISODE_STEPS` | budgets need rescaling |

- [ ] **Step 7: Record the run and commit**

Append a row to the top of the `runs/RUNLOG.md` table using the established format (`date | label | config delta | why | outcome`), stating what moved, what did not, and what the next run should change. Then:

```bash
git add runs/RUNLOG.md CLAUDE.md
git commit -m "Record run A (p6budget): first run on the rebuilt reward stack

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** D1 → Task 3 (`RewardBudget`, `RateWeight`, `PerSecondWeight`) and its budget tests. D2 → no code; recorded as the `GOAL` constant's comment in Task 3. D3 → Task 2 (`SpeedSquaredReward` comment) and D12's deletions in Task 3. D4 → Task 2 (`FaceBallAxisReward`) plus the decomposition test. D5 → Task 1 (`WrongSurfaceReward`). D6 → Task 1 (`CleanLandingReward`) plus the hop-vs-aerial test. D7 → Task 3's budget table; the flip-tax check is Task 4's `Surface/Wrong Contact While Flipping`. D8 → Task 2 (squared) and Task 4's trigger metrics. D9 → Task 2 (`TouchEdgeReward`) plus the dribble guard test. D10 → Task 2 (`HarshSpeedLossReward`) plus `Speed/Max Step Decel`. D11 → asserted by Task 3's spec-list test (nothing wrapped). D12 → Task 3, step 3b. D13 → **deliberately not implemented**; it is run B and the plan stops before it.

**Placeholder scan.** No TBD/TODO. Every code step carries real code. Test bodies are complete, not described.

**Type consistency.** `LANDING_REF_IMPACT`, `THROTTLE_TOP_SPEED`, `HARSH_LOSS_THRESHOLD` are declared in Tasks 1-2 and consumed in Tasks 3-4 under the same names. Spec name strings are fixed in Task 3 and depended on by nothing later except the RUNLOG. `TrainConfig::rewards` keeps its name across the type change, so `cfg.rewards` reads the same before and after.

**One gap deliberately left open:** the Task 3 step 3d instruction to "relocate" the surviving `Touch/*` metrics above the deleted block requires judgement about which need `state.prev`. `Touch/Above *`, `Touch/Had Jumped` and `Touch/Had Flipped` do not; `Touch/HitForce *` does and should be deleted with the block, since it exists only to explain a reward term that no longer exists.
