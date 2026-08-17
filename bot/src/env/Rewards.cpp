#include "Rewards.h"

#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <stdexcept>

using namespace RLGC;

namespace Hive {

float TouchHeightReward::GetReward(const Player &player, const GameState &state,
                                   bool isFinal) {
  if (!player.ballTouchedStep)
    return 0.f;

  const float height = state.ball.pos.z - CommonValues::BALL_RADIUS;
  if (height <= 0.f)
    return 0.f;

  return RS_MIN(1.f, height / maxHeight);
}

static constexpr float TOUCH_MIN_KPH = 5.f;
static constexpr float TOUCH_MAX_KPH = 100.f;

std::vector<WeightedReward> BuildGeneralRewards(const TrainConfig &cfg) {
  const RewardWeights &w = cfg.rewards;

  if (cfg.rewardPhase != RewardPhase::Foundations) {
    throw std::runtime_error(
        "BuildGeneralRewards(): only RewardPhase::Foundations is designed. "
        "See docs/rewards.md -- later phases must be derived from the run "
        "that precedes them, not guessed at.");
  }

  return {{new VelocityPlayerToBallReward(), w.velPlayerToBall},
          {new ZeroSumReward(
               new StrongTouchReward(TOUCH_MIN_KPH, TOUCH_MAX_KPH), 0.2f),
           w.strongTouch},
          {new ZeroSumReward(new VelocityBallToGoalReward(), 0.3f),
           w.velBallToGoal},
          {new GoalReward(), w.goal},
          {new PickupBoostReward(), w.pickupBoost},
          {new FaceBallReward(), w.faceBall}};
}

} // namespace Hive
