#pragma once

#include "../Config.h"
#include "../env/Rewards.h"

#include <RLGymCPP/Gamestates/GameState.h>
#include <RLGymCPP/Rewards/Reward.h>

#include <deque>
#include <string>
#include <vector>

namespace Dash {

// Evaluates the training reward stack alongside a spectated game, so the
// rewards a checkpoint would be earning can be read off step by step. This is
// the "print each reward every step" debugging Zealan recommends in
// reference-guide/my_bot_stopped_improving.md.
//
// The numbers are only worth reading if they match what the learner computes,
// so this deliberately mirrors RLGymCPP's EnvSet: the reward stack is built
// ONCE (several rewards carry per-episode state, such as AirLaunchReward's
// launchZ, and rebuilding per step would silently zero it), Reset() runs on
// episode start, and every step runs PreStep() over all rewards before any
// GetAllRewards(). Callers must pass a GameState whose prev is populated --
// DirectionalTouchReward and the aerial rewards read it, and report 0 without.
//
// Owns the reward objects it builds.
class RewardProbe {
  public:
	// pauseThreshold is a weighted-contribution magnitude; the first Event
	// reward to exceed it in a step trips Tripped(). Zero or less never trips.
	// historySteps is how many past steps FormatHistory() can look back over.
	RewardProbe(const TrainConfig &cfg, float pauseThreshold,
				int historySteps);
	~RewardProbe();

	RewardProbe(const RewardProbe &) = delete;
	RewardProbe &operator=(const RewardProbe &) = delete;

	// Clears totals and history, and resets every reward's episode state.
	void BeginEpisode(const RLGC::GameState &initialState);

	// One step of the same work EnvSet does: PreStep over all rewards, then
	// GetAllRewards over all rewards. Call once per environment step, after
	// the gamestate has been updated and the terminal conditions evaluated.
	void Step(const RLGC::GameState &state, bool isFinal);

	// True when the last Step() crossed the auto-pause threshold.
	bool Tripped() const { return !history.empty() && history.back().trippedIdx >= 0; }

	// Steps recorded in the current episode.
	int StepCount() const { return stepCount; }

	// Full per-reward table for the most recent step. Empty string before the
	// first Step().
	std::string FormatLastStep(const char *header) const;

	// One line per retained step: totals plus which Event rewards fired. Shows
	// the run-up to a pause, which is usually where a bugged reward's cause is.
	std::string FormatHistory() const;

	// Full per-reward table of episode-to-date totals.
	std::string FormatEpisodeTotals() const;

  private:
	// Values are flattened as rewardIdx * numPlayers + playerIdx.
	struct StepRecord {
		int step = 0;
		float time = 0.f;
		std::vector<float> raw;
		std::vector<float> weighted;
		int trippedIdx = -1;
	};

	// Rows for zero-weight rewards are dropped entirely (they cannot affect
	// training), Continuous rows are always kept so they can be watched trend,
	// and Event rows appear only on the steps they fire.
	bool ShouldShowRow(int rewardIdx, const std::vector<float> &raw) const;

	std::string FormatTable(const char *header, const std::vector<float> &raw,
							const std::vector<float> &weighted) const;

	std::vector<RewardSpec> specs;
	std::vector<RLGC::Reward *> rewards;
	std::vector<bool> isZeroSum;

	float pauseThreshold;
	size_t historyLimit;
	float secondsPerStep;

	int numPlayers = 0;
	std::vector<std::string> playerLabels;

	int stepCount = 0;
	std::vector<float> episodeTotalsRaw;
	std::vector<float> episodeTotalsWeighted;

	std::deque<StepRecord> history;
};

} // namespace Dash
