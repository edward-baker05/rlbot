#pragma once

#include "../Config.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <cmath>

#include <functional>
#include <string>
#include <vector>

namespace Hive {

// ============================================================================
// Custom rewards
// ============================================================================

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
// MEASURED SCOPE (p6budget): this fires on 2.3% of airborne steps. Free flight
// produces no chassis manifold, so it is a wheels-up-against-a-wall penalty and
// NOT an air tax. It was read as an air tax when the p6budget stack was
// designed, and that misreading is half of why that bot spent 90% of its life
// floating. Do not rely on it to price air time.
//
// Returns a NEGATIVE value and carries a POSITIVE weight (see the sign
// convention in GeneralRewardSpecs).
class WrongSurfaceReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return (player.worldContact.hasContact && !player.isOnGround) ? -1.f : 0.f;
	}
};

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

// max(0, v . dirToBall) / CAR_MAX_SPEED -- how fast we are closing on the ball.
//
// This is Zealan's SpeedTowardBallReward (RLGym-PPO-Guide, rewards.md), and it
// is upstream's VelocityPlayerToBallReward RECTIFIED at zero. The rectification
// is the whole difference and it matters twice:
//
//  1. Retreating is free rather than punished. Plenty of correct play moves
//     away from the ball, and the guide is explicit about not charging for it.
//  2. The signed form is dominated by cancellation: p1air's RUNLOG row records
//     RewardShare 0.482 against a near-zero NET, because circling generates
//     large +/- values that sum away. Rectified, the mass is the signal.
//
// WHY THIS EXISTS AGAIN, against the p6budget design's D3. That decision banned
// ball-directed dense shaping and replaced it with the additive factoring
// SpeedSquared + FaceBall, on the argument that the product form's cross term
// charges a turn twice. p6budget measured the result: over 100M steps the
// bot's rectified NOSE-to-ball cosine rose 0.338 -> 0.741 while its
// VELOCITY-to-ball cosine went 0.299 -> 0.300, dead flat, with the two facing
// terms taking 62% of net earnings. The cross term is not a defect; it is the
// coupling that makes alignment worth buying. Factored apart, the bot bought
// the cheap factor (rotation, which is free in the air) and never the
// expensive one.
//
// KNOWN AND ACCEPTED: this term is farmable around a chase-hit-chase cycle.
// That is deliberate. The standing lesson from p4pbrs is that the no-farm
// guarantee and the teaching signal are the same property seen from opposite
// sides -- the potential-based version telescopes to zero over exactly that
// cycle and taught nothing. TouchEdgeReward's budget is what makes finishing
// an approach worth more than repeating one.
class SpeedToBallReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const Vec toBall = state.ball.pos - player.pos;
		const float len = toBall.Length();
		if (len < 1e-4f)
			return 0.f;

		const float closing = player.vel.Dot(toBall / len);
		return RS_MAX(0.f, closing) / RLGC::CommonValues::CAR_MAX_SPEED;
	}
};

// max(0, forward . dirToBall) -- pays for pointing at the ball, pays nothing
// for pointing away.
//
// This is the asymmetric FaceBall the p6budget design wanted, at the limit
// w- = 0, shipped as ONE term instead of two. The identity that design was
// built on still holds:
//
//   w+ * max(0,c) + w- * min(0,c)  ==  ((w+ + w-)/2) * c  +  ((w+ - w-)/2) * |c|
//
// so at w- = 0 the signed lobe and the |c| lobe carry EQUAL weight, w+/2 each.
// Splitting them was worth doing when the asymmetry was 2:1 and the |c| annuity
// needed its own RewardShare line to be visible; at w- = 0 the split buys two
// metric lines and a second budget for a reward that is one clamp. The
// decomposition survives as an executable assertion in the tests.
//
// Note what this does NOT do, because the two are easy to conflate: the p6
// stack did not PAY for facing away, it punished it at half rate (w- = 0.133).
// This removes the punishment as well, which is what the guide recommends --
// shadow defence and retreating for a bounce both need the nose off the ball.
//
// Budget is deliberately small. p6budget measured this exact quantity being
// optimized to convergence (nose cosine 0.741) while buying zero approach, so
// it is a tiebreaker against driving backwards at the ball and nothing more.
class FaceBallRectifiedReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		const Vec toBall = state.ball.pos - player.pos;
		const float len = toBall.Length();
		if (len < 1e-4f)
			return 0.f;

		return RS_MAX(0.f, player.rotMat.forward.Dot(toBall / len));
	}
};

// ============================================================================
// Diagnostic constants
// ============================================================================
// Neither of these is a reward weight any more; both are thresholds for
// metrics in Train.cpp, kept here so the derivations stay with the physics.

// Throttle-only top speed: DRIVE_SPEED_TORQUE_FACTOR_CURVE reaches zero here,
// so any car can hold this indefinitely with no boost and no skill.
inline constexpr float THROTTLE_TOP_SPEED = 1410.f;

// Speed that can be lost in one decision step without a collision. RL brakes at
// roughly 3500 uu/s^2, which over a 1/15 s step is 233 uu/s, so 400 is clear of
// any input-driven deceleration and only a collision reaches it. That 3500 is
// EMPIRICAL, not a RocketSim constant.
inline constexpr float HARSH_LOSS_THRESHOLD = 400.f;

// Returns heap-allocated rewards; GigaLearn's EnvSet takes ownership, so the
// caller does not free them.

// One reward term with a stable metric name, since Reward::GetName() is
// useless for that on GCC (near-mangled typeid strings).
struct RewardSpec {
	std::string name;   // metric label, e.g. "SpeedToBall"
	float weight;
	std::function<RLGC::Reward*()> make;
};

// The order of specs matches the order of envSet->rewards[arena] and
// envSet->state.lastRewards[arena]; the reward-share metrics index by it.
std::vector<RewardSpec> GeneralRewardSpecs(const TrainConfig& cfg);

// Materializes the specs, same order.
std::vector<RLGC::WeightedReward> BuildGeneralRewards(const TrainConfig& cfg);

} // namespace Hive
