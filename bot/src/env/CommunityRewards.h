#pragma once

// Community reward functions, imported from rishieissocool/GigaLearn-Rewards.
//
// Source:  https://github.com/rishieissocool/GigaLearn-Rewards
// Imported: 2026-08-25
//
// The upstream pack advertises 56 rewards. What actually arrived:
//
//   39 imported here.
//   13 are STUBS -- `// TODO: Implement reward logic; return 0.0f` -- in BOTH
//   the
//      C++ and the Python copies. They are name-only placeholders. They are
//      deliberately NOT imported: a reward that silently returns zero while
//      appearing in the metrics under a plausible name is worse than an absent
//      one. Named below so nothing looks lost.
//   4 are dropped as unportable, superseded, or dependent on one of the above.
//      Named below with reasons.
//
// Classes are emitted in dependency order, not alphabetically: several of these
// compose others as members (ModifiedTouchReward holds a PowerShotReward,
// CradleFlickReward holds a CradleReward), so declaration order is
// load-bearing.
//
// Note the stubs are disproportionately the Control and Positioning entries,
// i.e. exactly the categories our own reward set is thinnest in. Boost
// discipline and rotation/spacing shaping remain unsolved by this import --
// TeamSpacingReward and PositioningReward below are the only real positioning
// entries that landed.
//
// Mechanical adaptations applied during import (the pack targets RLGym Python
// naming in places, even in its C++ files):
//
//   BLUE_TEAM / ORANGE_TEAM  ->  Team::BLUE / Team::ORANGE  (ours is an enum
//                                class; CommonValues' float 0/1 will not
//                                compare)
//   player.has_flip          ->  player.HasFlipOrJump()
//   state.last_touch         ->  state.lastTouchCarID
//
// Constants (SIDE_WALL_X, BACK_WALL_Y, GOAL_HEIGHT, BALL_RADIUS, CAR_MAX_SPEED,
// SUPERSONIC_THRESHOLD, ...) come from RLGC::CommonValues via the
// using-directive below rather than being redefined, so there is one source of
// truth. The pack redefined them per-file with matching values; that was an
// obvious drift hazard.
//
// Weighting: per Zealan's rewards.md a reward should output within [-1, 1] so
// the weight alone determines its impact. Several of these do NOT respect that
// -- TeamSpacingReward accumulates one penalty per teammate, so its range
// scales with team size. Check the range before you pick a weight.

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace Dash {
namespace Community {

// Contained on purpose: this is a vendored blob written against unqualified
// RLGC names, and qualifying every one of them by hand would make future
// re-imports from upstream a manual diff instead of a re-run of the importer.
using namespace RLGC;
using namespace RLGC::CommonValues;

// [aerial]
class AirReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (!player.isOnGround) {
			return player.HasFlipOrJump() ? 0.5f : 1.0f;
		}
		return 0.0f;
	}
};

// [ball]
class BallYCoordinateReward : public Reward {

  public:
	float exponent;

	BallYCoordinateReward(float exponent = 1.0f) : exponent(exponent) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float y =
			(player.team == Team::BLUE) ? state.ball.pos.y : -state.ball.pos.y;
		return std::pow(y / (BACK_WALL_Y + BALL_RADIUS), exponent);
	}
};

// [event]
class BoostAcquisitions : public Reward {
  public:
	float boost_reserves;

	BoostAcquisitions() : boost_reserves(1.0f) {}

	virtual void Reset(const GameState &initialState) override {
		boost_reserves = 1.0f;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float boost_gain = player.boost - boost_reserves;
		boost_reserves = player.boost;
		return boost_gain > 0 ? boost_gain : 0.0f;
	}
};

// [positioning]
class CenterReward : public Reward {

  public:
	float centered_distance;
	bool punish_area_exit;
	float non_participation_reward;
	bool centered;
	Vec goal_spot;

	CenterReward(float centered_distance = 1200.0f,
				 bool punish_area_exit = false,
				 float non_participation_reward = 0.0f)
		: centered_distance(centered_distance),
		  punish_area_exit(punish_area_exit),
		  non_participation_reward(non_participation_reward), centered(false),
		  goal_spot(0.0f, 5120.0f, 0.0f) {}

	virtual void Reset(const GameState &initialState) override {
		centered = false;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec ball_loc =
			(player.team == Team::BLUE) ? state.ball.pos : -state.ball.pos;
		float ball_dist_2d = (goal_spot - ball_loc).Length2D();
		float reward = 0.0f;
		if (centered) {
			if (ball_dist_2d > centered_distance) {
				centered = false;
				if (punish_area_exit) {
					reward -= 1.0f;
				}
			}
		} else {
			if (ball_dist_2d < centered_distance) {
				centered = true;
				reward += (state.lastTouchCarID == player.carId)
							  ? 1.0f
							  : non_participation_reward;
			}
		}
		return reward;
	}
};

// [aerial]
class ChallengeReward : public Reward {

  public:
	float challenge_distance;

	ChallengeReward(float challenge_distance = 300.0f)
		: challenge_distance(challenge_distance) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (!player.isOnGround &&
			player.pos.Dist(state.ball.pos) < challenge_distance) {
			for (const auto &p : state.players) {
				if (p.team != player.team &&
					p.pos.Dist(state.ball.pos) < challenge_distance) {
					reward += 0.1f;
					if (!player.HasFlipOrJump()) {
						reward += 0.9f;
					}
					break;
				}
			}
		}
		return reward;
	}
};

// [touch]
class ClearReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (player.ballTouchedStep) {
			float vel_difference =
				(state.prev->ball.vel - state.ball.vel).Length();
			reward = vel_difference / 4600.0f;
		}
		return reward;
	}
};

// [ball]
class CradleReward : public Reward {

  public:
	float min_distance;

	CradleReward(float minimum_barrier = 200.0f)
		: min_distance(minimum_barrier) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.pos.z < state.ball.pos.z &&
			BALL_RADIUS + 20.0f < state.ball.pos.z &&
			state.ball.pos.z < BALL_RADIUS + 200.0f &&
			player.pos.Dist2D(state.ball.pos) <= 170.0f) {
			if (std::abs(state.ball.pos.x) < 3946.0f &&
				std::abs(state.ball.pos.y) < 4970.0f) {
				if (min_distance > 0.0f) {
					for (const auto &p : state.players) {
						if (p.team != player.team &&
							p.pos.Dist(state.ball.pos) < min_distance) {
							return 0.0f;
						}
					}
				}
				return 1.0f;
			}
		}
		return 0.0f;
	}
};

// [event]
class DemoPunish : public Reward {
  public:
	std::vector<bool> demo_statuses;

	DemoPunish() : demo_statuses(64, true) {}

	virtual void Reset(const GameState &initialState) override {
		demo_statuses.assign(64, true);
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (player.eventState.demoed && !demo_statuses[player.carId]) {
			reward = -1.0f;
		}
		demo_statuses[player.carId] = player.eventState.demoed;
		return reward;
	}
};

// [positioning]
class DistanceReward : public Reward {

  public:
	float dist_max;
	float max_reward;

	DistanceReward(float dist_max = 1000.0f, float max_reward = 2.0f)
		: dist_max(dist_max), max_reward(max_reward) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float distance = (state.ball.pos - player.pos).Length() - 110.0f;
		if (distance > dist_max) {
			return 0.0f;
		}
		return max_reward * (1.0f - (distance / dist_max));
	}
};

// [velocity]
class FlatSpeedReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec flat_vel = player.vel;
		flat_vel.z = 0.0f;
		return std::abs(flat_vel.Length()) / 2300.0f;
	}
};

// [velocity]
class ForwardBiasReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		return player.rotMat.forward.Dot(player.vel.Normalized());
	}
};

// [positioning]
class GroundedReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		return player.isOnGround ? 1.0f : 0.0f;
	}
};

// [touch]
class HeightTouchReward : public Reward {

  public:
	float min_height, exp, cooperation_dist;

	HeightTouchReward(float min_height = 92.0f, float exp = 0.2f,
					  float cooperation_dist = 0.0f)
		: min_height(min_height), exp(exp), cooperation_dist(cooperation_dist) {
	}

	bool cooperation_detector(const Player &player, const GameState &state) {
		for (const auto &p : state.players) {
			if (p.carId != player.carId &&
				player.pos.Dist(p.pos) < cooperation_dist) {
				return true;
			}
		}
		return false;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (player.ballTouchedStep) {
			if (state.ball.pos.z >= min_height) {
				if (!player.isOnGround || cooperation_dist < 90.0f ||
					!cooperation_detector(player, state)) {
					if (player.isOnGround) {
						reward += std::pow(std::clamp(state.ball.pos.z - 92.0f,
													  0.0001f, 5000.0f),
										   exp);
					} else {
						reward +=
							std::pow(std::clamp(state.ball.pos.z, 1.0f, 500.0f),
									 exp * 2.0f);
					}
				}
			} else if (!player.isOnGround) {
				reward += 1.0f;
			}
		}
		return reward;
	}
};

// [touch]
class JumpTouchReward : public Reward {

  public:
	float min_height, max_height, range;

	JumpTouchReward(float min_height = 92.75f)
		: min_height(min_height), max_height(2044.0f - 92.75f),
		  range(max_height - min_height) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.ballTouchedStep && !player.isOnGround &&
			state.ball.pos.z >= min_height) {
			return (state.ball.pos.z - min_height) / range;
		}
		return 0.0f;
	}
};

// [aerial]
class LandingRecoveryReward : public Reward {

  public:
	Vec up;

	LandingRecoveryReward() : up(0.0f, 0.0f, 1.0f) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (!player.isOnGround && player.vel.z < 0.0f &&
			player.pos.z > 250.0f) {
			Vec flattened_vel = player.vel;
			flattened_vel.z = 0.0f;
			flattened_vel = flattened_vel.Normalized();
			Vec forward = player.rotMat.forward;
			Vec flattened_forward =
				Vec(forward.x, forward.y, 0.0f).Normalized();
			reward += flattened_vel.Dot(flattened_forward);
			reward += up.Dot(player.rotMat.up);
			reward /= 2.0f;
		}
		return reward;
	}
};

// [control]
class LavaFloorReward : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.isOnGround && player.pos.z < 50.0f) {
			return -1.0f;
		}
		return 0.0f;
	}
};

// [control]
class MillennialKickoffReward : public Reward {

  public:
	bool closest_to_ball(const Player &player, const GameState &state) {
		float player_dist = player.pos.Dist(state.ball.pos);
		for (const auto &p : state.players) {
			if (p.team == player.team && p.carId != player.carId) {
				float dist = p.pos.Dist(state.ball.pos);
				if (dist < player_dist) {
					return false;
				}
			}
		}
		return true;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (state.ball.pos.x == 0.0f && state.ball.pos.y == 0.0f &&
			closest_to_ball(player, state)) {
			return -1.0f;
		}
		return 0.0f;
	}
};

// [velocity]
class NaiveSpeedReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		return std::abs(player.vel.Length()) / 2300.0f;
	}
};

// [positioning]
class PositioningReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec ball =
			(player.team == Team::BLUE) ? state.ball.pos : -state.ball.pos;
		Vec pos = (player.team == Team::BLUE) ? player.pos : -player.pos;
		float reward = 0.0f;
		if (ball.y < pos.y) {
			float diff = ball.y - pos.y;
			reward = -std::clamp(std::abs(diff) / 5000.0f, 0.0f, 1.0f);
		}
		return reward;
	}
};

// [utility]
class PositiveWrapperReward : public Reward {

  public:
	Reward *rew;

	PositiveWrapperReward(Reward *base_reward) : rew(base_reward) {}

	virtual void Reset(const GameState &initialState) override {
		rew->Reset(initialState);
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = rew->GetReward(player, state, isFinal);
		return reward < 0.0f ? 0.0f : reward;
	}
};

// [touch]
class PowerShotReward : public Reward {

  public:
	float min_change;
	Vec last_velocity;

	PowerShotReward(float min_change = 300.0f)
		: min_change(min_change), last_velocity(0.0f, 0.0f, 0.0f) {}

	virtual void Reset(const GameState &initialState) override {
		last_velocity = Vec(0.0f, 0.0f, 0.0f);
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		Vec cur_vel = Vec(state.ball.vel.x, state.ball.vel.y, 0.0f);
		if (player.ballTouchedStep) {
			float vel_change = (last_velocity - cur_vel).Length();
			if (vel_change > min_change) {
				reward = vel_change / (2300.0f * 2.0f);
			}
		}
		last_velocity = cur_vel;
		return reward;
	}
};

// [ball]
class PushReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec pos =
			(player.team == Team::BLUE) ? state.ball.pos : -state.ball.pos;
		if (pos.y > 0.0f) {
			float y_scale = pos.y / 5213.0f;
			if (std::abs(pos.x) > 800.0f) {
				float x_scale = (std::abs(pos.x) / 4096.0f) * y_scale;
				return y_scale - x_scale;
			}
			return y_scale;
		} else if (pos.y < 0.0f) {
			float y_scale = pos.y / 5213.0f;
			if (std::abs(pos.x) > 800.0f) {
				float x_scale = (std::abs(pos.x) / 4096.0f) * std::abs(y_scale);
				return y_scale + x_scale;
			}
			return y_scale;
		}
		return 0.0f;
	}
};

// [velocity]
class RetreatReward : public Reward {

  public:
	Vec defense_target;

	RetreatReward() : defense_target(BLUE_GOAL_BACK) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec ball =
			(player.team == Team::BLUE) ? state.ball.pos : -state.ball.pos;
		Vec pos = (player.team == Team::BLUE) ? player.pos : -player.pos;
		Vec vel = (player.team == Team::BLUE) ? player.vel : -player.vel;
		float reward = 0.0f;
		if (ball.y + 200.0f < pos.y) {
			Vec pos_diff = defense_target - pos;
			Vec norm_pos_diff = pos_diff.Normalized();
			Vec norm_vel = vel / CAR_MAX_SPEED;
			reward = norm_pos_diff.Dot(norm_vel);
		}
		return reward;
	}
};

// [control]
class SaveBoostReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		return (player.boost > 20 ? player.boost : 20 - player.boost) * 0.01;
	}
};

// [utility]
class SequentialRewards : public Reward {

  public:
	std::vector<Reward *> rewards_list;
	std::vector<int> step_counts;
	int step_count;
	int step_index;

	SequentialRewards(const std::vector<Reward *> &rewards,
					  const std::vector<int> &steps)
		: rewards_list(rewards), step_counts(steps), step_count(0),
		  step_index(0) {}

	virtual void Reset(const GameState &initialState) override {
		for (auto &rew : rewards_list) {
			rew->Reset(initialState);
		}
		step_count = 0;
		step_index = 0;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (step_index < step_counts.size() &&
			step_count > step_counts[step_index]) {
			step_index++;
		}
		step_count++;
		return rewards_list[step_index]->GetReward(player, state, isFinal);
	}
};

// [velocity]
class SpeedReward : public Reward {

  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float car_speed = player.vel.Length();
		float car_dir =
			player.rotMat.forward.Dot(player.vel) >= 0 ? 1.0f : -1.0f;
		car_speed *= car_dir / 2300.0f;
		return std::min(car_speed, 1.0f);
	}
};

// [positioning]
class TeamSpacingReward : public Reward {

  public:
	float min_spacing;

	TeamSpacingReward(float min_spacing = 1000.0f)
		: min_spacing(std::max(0.0000001f, min_spacing)) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (player.eventState.demoed) {
			return reward;
		}
		for (const auto &p : state.players) {
			if (p.carId != player.carId && p.team == player.team &&
				!p.eventState.demoed) {
				float separation = player.pos.Dist(p.pos);
				if (separation < min_spacing) {
					reward -= 1.0f - (separation / min_spacing);
				}
			}
		}
		return reward;
	}
};

// [touch]
class TouchBallReward : public Reward {

  public:
	float min_touch, min_height, min_distance, aerial_weight;
	bool air_reward, first_touch;

	TouchBallReward(float min_touch = 0.05f, float min_height = 170.0f,
					float min_distance = 300.0f, float aerial_weight = 0.15f,
					bool air_reward = true, bool first_touch = false)
		: min_touch(min_touch), min_height(min_height),
		  min_distance(min_distance), aerial_weight(aerial_weight),
		  air_reward(air_reward), first_touch(first_touch) {}

	float get_closest_enemy_distance(const Player &player,
									 const GameState &state) {
		float closest_dist = 50000.0f;
		for (const auto &p : state.players) {
			if (p.team != player.team) {
				float dist = player.pos.Dist2D(p.pos);
				if (dist < closest_dist) {
					closest_dist = dist;
				}
			}
		}
		return closest_dist;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (player.ballTouchedStep) {
			if (first_touch && state.ball.pos.x == 0.0f &&
				state.ball.pos.y == 0.0f) {
				reward += 5.0f;
			}
			if (state.ball.pos.z >= min_height &&
				(min_distance == 0.0f ||
				 get_closest_enemy_distance(player, state) > min_distance)) {
				reward +=
					std::max(min_touch,
							 std::pow(std::abs(state.ball.pos.z - BALL_RADIUS),
									  aerial_weight) -
								 1.0f);
			}
			if (air_reward && !player.isOnGround) {
				reward += player.HasFlipOrJump() ? 0.5f : 1.0f;
			}
		}
		return reward;
	}
};

// [touch]
class TouchBallTweakedReward : public Reward {

  public:
	float min_touch, min_height, min_distance, aerial_weight;
	bool air_reward, first_touch;
	Vec last_velocity;

	TouchBallTweakedReward(float min_touch = 0.05f, float min_height = 170.0f,
						   float min_distance = 300.0f,
						   float aerial_weight = 0.15f, bool air_reward = true,
						   bool first_touch = false)
		: min_touch(min_touch), min_height(min_height),
		  min_distance(min_distance), aerial_weight(aerial_weight),
		  air_reward(air_reward), first_touch(first_touch),
		  last_velocity(0.0f, 0.0f, 0.0f) {}

	virtual void Reset(const GameState &initialState) override {
		last_velocity = Vec(0.0f, 0.0f, 0.0f);
	}

	float get_closest_enemy_distance(const Player &player,
									 const GameState &state) {
		float closest_dist = 50000.0f;
		for (const auto &p : state.players) {
			if (p.team != player.team) {
				float dist = player.pos.Dist2D(p.pos);
				if (dist < closest_dist) {
					closest_dist = dist;
				}
			}
		}
		return closest_dist;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		Vec current_vel = state.ball.vel;
		if (player.ballTouchedStep) {
			if (state.ball.pos.z >= min_height ||
				(state.ball.pos.z >= BALL_RADIUS + 20.0f &&
				 (min_distance == 0.0f ||
				  get_closest_enemy_distance(player, state) > min_distance))) {
				reward +=
					std::max(min_touch,
							 std::pow(std::abs(state.ball.pos.z - BALL_RADIUS),
									  aerial_weight) -
								 1.0f);
				reward += (last_velocity - current_vel).Length() / 2300.0f;
			}
			if (air_reward && !player.isOnGround) {
				reward += 0.5f;
				if (!player.HasFlipOrJump()) {
					reward += 0.5f;
				}
			}
		}
		last_velocity = current_vel;
		return reward;
	}
};

// [velocity]
class TweakedVelocityPlayerToGoalReward : public Reward {

  public:
	float max_leeway, default_power;

	TweakedVelocityPlayerToGoalReward(float max_leeway = 100.0f,
									  float default_power = 0.0f)
		: max_leeway(max_leeway), default_power(default_power) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec ball_pos =
			(player.team == Team::BLUE) ? state.ball.pos : -state.ball.pos;
		Vec player_pos = (player.team == Team::BLUE) ? player.pos : -player.pos;
		Vec player_goal =
			(player.team == Team::BLUE) ? BLUE_GOAL_BACK : ORANGE_GOAL_BACK;
		Vec diff = player_pos - ball_pos;
		if (diff.y < max_leeway) {
			return 0.0f;
		}
		Vec direction = (player_goal - player_pos).Normalized();
		Vec vel = player.vel / CAR_MAX_SPEED;
		return direction.Dot(vel);
	}
};

// [ball]
class VelocityBallToGoalReward : public Reward {
  public:
	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec ball_pos =
			(player.team == Team::BLUE) ? state.ball.pos : -state.ball.pos;
		Vec goal_pos = (player.team == Team::BLUE)
						   ? Vec(0.0f, 6000.0f, 321.3875f)
						   : Vec(0.0f, -6000.0f, 321.3875f);
		Vec pos_diff = ball_pos - goal_pos;
		Vec norm_pos_diff = pos_diff.Normalized();
		Vec vel = state.ball.vel / 6000.0f;
		return norm_pos_diff.Dot(vel);
	}
};

// [velocity]
class VelocityPlayerToBallReward : public Reward {

  public:
	bool use_scalar_projection;

	VelocityPlayerToBallReward(bool use_scalar_projection = false)
		: use_scalar_projection(use_scalar_projection) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		Vec pos_diff = state.ball.pos - player.pos;
		Vec norm_pos_diff = pos_diff.Normalized();
		Vec norm_vel = player.vel / CAR_MAX_SPEED;
		return norm_pos_diff.Dot(norm_vel);
	}
};

// [ball]
class DribbleFlickReward : public Reward {

  public:
	float min_distance, max_vel_diff;
	bool training;
	CradleReward cradle_reward;

	DribbleFlickReward(float minimum_barrier = 300.0f,
					   float max_vel_diff = 400.0f, bool training = true)
		: min_distance(minimum_barrier), max_vel_diff(max_vel_diff),
		  training(training), cradle_reward(0.0f) {}

	virtual void Reset(const GameState &initialState) override {
		cradle_reward.Reset(initialState);
	}

	bool stable_carry(const Player &player, const GameState &state) {
		if (BALL_RADIUS + 20.0f < state.ball.pos.z &&
			state.ball.pos.z < BALL_RADIUS + 80.0f) {
			return (player.vel - state.ball.vel).Length() <= max_vel_diff;
		}
		return false;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = cradle_reward.GetReward(player, state, isFinal) * 0.1f;
		if (reward > 0.0f) {
			if (!training) {
				reward = 0.0f;
			}
			bool stable = stable_carry(player, state);
			bool challenged = false;
			for (const auto &p : state.players) {
				if (p.team != player.team &&
					p.pos.Dist(state.ball.pos) < min_distance) {
					challenged = true;
					break;
				}
			}
			if (challenged) {
				if (stable) {
					if (player.isOnGround) {
						return reward - 0.1f;
					} else {
						float speedRatio =
							RS_MIN(state.ball.vel.Length2D() /
									   RLGC::Math::KPHToVel(130.f),
								   1.f);
						return player.HasFlipOrJump()
								   ? reward + 0.3f
								   : reward + 0.4f +
										 (state.ball.vel.Length2D() /
										  speedRatio * 0.5);
					}
				}
			} else if (stable) {
				return reward + 0.2f;
			}
		}
		return reward;
	}
};

// [ball]
class KickoffReward : public Reward {
  public:
	VelocityPlayerToBallReward velDirReward;
	SpeedReward velReward;
	bool boostPunish;
	int ticks;

	KickoffReward(bool boostPunish = true)
		: boostPunish(boostPunish), ticks(0) {}

	virtual void Reset(const GameState &initialState) override { ticks = 0; }

	bool IsClosestToBall(const Player &player, const GameState &state) {
		float playerDist = player.pos.Dist(state.ball.pos);
		for (const auto &p : state.players) {
			if (p.carId != player.carId && p.team == player.team &&
				!p.eventState.demoed) {
				float dist = p.pos.Dist(state.ball.pos);
				if (dist < playerDist) {
					return false;
				}
			}
		}
		return true;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		if (fabs(state.ball.pos.x) < 0.01f && fabs(state.ball.pos.y) < 0.01f &&
			IsClosestToBall(player, state)) {
			reward += velReward.GetReward(player, state, isFinal);
			reward += velDirReward.GetReward(player, state, isFinal);
		}
		ticks++;
		return reward;
	}
};

// [touch]
class ModifiedTouchReward : public Reward {

  public:
	PowerShotReward psr;
	float min_height, height_cap, vel_scale, touch_scale, jump_scale;
	bool jump_reward;
	int tick_count, tick_min;

	ModifiedTouchReward(float min_change = 300.0f, float min_height = 200.0f,
						float vel_scale = 1.0f, float touch_scale = 1.0f,
						bool jump_reward = false, float jump_scale = 0.1f,
						int tick_min = 0)
		: psr(min_change), min_height(min_height), height_cap(2044.0f - 92.75f),
		  vel_scale(vel_scale), touch_scale(touch_scale),
		  jump_reward(jump_reward), jump_scale(jump_scale), tick_count(0),
		  tick_min(tick_min) {}

	virtual void Reset(const GameState &initialState) override {
		psr.Reset(initialState);
		tick_count = 0;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		float reward = 0.0f;
		float psr_val = psr.GetReward(player, state, isFinal);
		if (player.ballTouchedStep) {
			if (tick_count <= 0) {
				tick_count = tick_min;
				reward += std::abs(psr_val * vel_scale);
				if (!player.isOnGround) {
					if (jump_reward) {
						reward += jump_scale;
						if (!player.HasFlipOrJump()) {
							reward += jump_scale;
						}
					}
					if (state.ball.pos.z > min_height) {
						reward += std::abs((state.ball.pos.z / height_cap) *
										   touch_scale);
					}
				}
			} else {
				tick_count--;
			}
		} else {
			tick_count--;
		}
		return reward;
	}
};

// [ball]
class PositiveBallVelToGoalReward : public Reward {

  public:
	VelocityBallToGoalReward rew;

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		return std::clamp(rew.GetReward(player, state, isFinal), 0.0f, 1.0f);
	}
};

// [utility]
class PositivePlayerVelToBallReward : public Reward {

  public:
	VelocityPlayerToBallReward rew;

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		return std::clamp(rew.GetReward(player, state, isFinal), 0.0f, 1.0f);
	}
};

// [velocity]
class SelectiveChaseReward : public Reward {

  public:
	VelocityPlayerToBallReward vel_dir_reward;
	float distance_requirement;

	SelectiveChaseReward(float distance_req = 500.0f)
		: distance_requirement(distance_req) {}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		if (player.pos.Dist2D(state.ball.pos) > distance_requirement) {
			return vel_dir_reward.GetReward(player, state, isFinal);
		}
		return 0.0f;
	}
};

// [positioning]
class ThreeManRewards : public Reward {

  public:
	float min_spacing;
	VelocityBallToGoalReward vel_reward;
	KickoffReward kor;

	ThreeManRewards(float min_spacing = 1500.0f) : min_spacing(min_spacing) {}

	float spacing_reward(const Player &player, const GameState &state,
						 int role) {
		float reward = 0.0f;
		if (role != 0) {
			for (const auto &p : state.players) {
				if (p.team == player.team && p.carId != player.carId &&
					!p.eventState.demoed) {
					float separation = player.pos.Dist(p.pos);
					if (separation < min_spacing) {
						reward -= 1.0f - (separation / min_spacing);
					}
				}
			}
		}
		return reward;
	}

	virtual float GetReward(const Player &player, const GameState &state,
							bool isFinal) override {
		std::vector<std::pair<float, int>> player_distances;
		for (const auto &p : state.players) {
			if (p.team == player.team) {
				player_distances.emplace_back(p.pos.Dist(state.ball.pos),
											  p.carId);
			}
		}
		std::sort(player_distances.begin(), player_distances.end());
		int role = 0;
		for (size_t i = 0; i < player_distances.size(); ++i) {
			if (player_distances[i].second == player.carId) {
				role = i;
				break;
			}
		}
		float reward = spacing_reward(player, state, role);
		if (role == 1) {
			reward += vel_reward.GetReward(player, state, isFinal);
			reward += kor.GetReward(player, state, isFinal);
		}
		return reward;
	}
};

// ---------------------------------------------------------------------------
// NOT IMPORTED
// ---------------------------------------------------------------------------
//
// Stubs upstream (empty in BOTH the C++ and the Python copy -- `return 0.0f`):
//
//   AerialNavigation
//   AerialTraining
//   BoostDiscipline
//   BoostTrainer
//   ControllerModerator
//   DefenseTrainer
//   FlipCorrecter
//   GoalboxPenalty
//   OmniBoostDiscipline
//   OncePerStepRewardWrapper
//   PlayerAlignment
//   RuleOnePunishment
//   VelocityBallDefense
//
// Dropped because a dependency above is unavailable:
//
//   StarterReward                needs EventReward
//   VersatileBallVelocityReward  needs VelocityBallDefense
//
// Dropped as unportable or superseded:
//
//   EventReward     -- superseded by RLGC::PlayerDataEventReward in
//   CommonRewards.h,
//                      which does the same job natively off per-step
//                      `eventState` flags. The pack's version reconstructs
//                      cumulative match counters our Player does not carry
//                      (match_goals / match_shots / match_saves /
//                      match_demolishes) and reads team scores our GameState
//                      does not carry either. It also indexes
//                      `last_registered_values[player.carId]` into a vector
//                      sized by player COUNT -- carId is an id, not an index,
//                      so that is an out-of-bounds read waiting to happen. Use
//                      PlayerDataEventReward instead.
//
//   DefenderReward  -- needs a running scoreline to detect a concede. Our
//   GameState
//                      exposes only `goalScored` (a per-step bool) and no
//                      score, so a faithful port would have to infer the
//                      scoring team from ball position at the goal tick and
//                      keep its own counter. That is a rewrite, not an import,
//                      and it would inherit the original's flaw of sharing one
//                      `enemy_goals` counter across every player in the arena.
//                      Left out rather than half-ported.

} // namespace Community
} // namespace Dash
