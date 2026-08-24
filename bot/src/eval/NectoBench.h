#pragma once

#include "../Config.h"

#include <GigaLearnCPP/Util/Report.h>

#include <filesystem>
#include <memory>

namespace Dash {

struct NectoBenchResult {
	bool valid = false;

	int episodes = 0;  // Episodes played.
	int decisive = 0;  // Of those, the ones that ended in a goal.
	int goalsFor = 0;  // Scored by the checkpoint.
	int goalsAgainst = 0;

	float winRate = 0.f; // Over decisive episodes only.
	float elo = 0.f;     // Against Necto pinned at 0.
};

// Head-to-head against Necto on a fixed set of kickoff arenas.
//
// Separate from the training arenas on purpose. The training-arena numbers are
// free but confounded -- the policy is exploring with entropy, Necto is
// sampling stochastically, and spawns come from the randomised training
// distribution. This plays deterministic-vs-deterministic from kickoffs, so the
// curve moves when the policy changes rather than when the noise does.
//
// It scores the latest CHECKPOINT rather than the live policy: Dash::Policy
// already loads a checkpoint and applies its observation standardisation, so
// this needs no access to the learner's internals. At tsPerSave = 1M the
// checkpoint lags by at most 1M steps, which is nothing next to the benchmark
// interval.
class NectoBench {
  public:
	explicit NectoBench(const TrainConfig &cfg);
	~NectoBench();

	NectoBench(const NectoBench &) = delete;
	NectoBench &operator=(const NectoBench &) = delete;

	// Plays a full round. Returns an invalid result if no complete checkpoint
	// exists yet, which is normal before the first save.
	NectoBenchResult Run(const std::filesystem::path &checkpoint);

  private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

// Set, not averaged: these are computed once per round, matching how the
// existing Rating/* metric is emitted.
void ReportNectoBench(const NectoBenchResult &result, GGL::Report &report);

} // namespace Dash
