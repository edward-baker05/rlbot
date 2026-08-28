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
	float elo = 0.f;     // Against the opponent pinned at 0.

	// Over the trailing window of decisive games, across rounds. One round is
	// 16 games, which is far too few to read a win rate off.
	int windowGames = 0;
	float windowWinRate = 0.f;
	float windowElo = 0.f;
};

// Head-to-head against a Necto-family opponent, from the training scenario pool.
//
// Separate from the training arenas on purpose. The training-arena numbers are
// free but confounded -- the policy is exploring with entropy, the opponent is
// sampling stochastically, and only a slice of arenas has an opponent in it at
// all. This plays deterministic-vs-deterministic, so the curve moves when the
// policy changes rather than when the noise does.
//
// It does NOT use a fixed start. It used to, and that was the more obvious
// choice -- a kickoff is the lowest-variance start there is. But variance is not
// the only thing that matters: starting every episode at a kickoff measures
// kickoffs, and a bot can be much better than you over a whole game while being
// no better than you in the ten seconds after one. Measured against Nexto that
// error was large enough to invert the ranking. So episodes start from
// BuildSpawner(cfg) -- the same distribution the policy trains on -- and the
// variance is paid for with more episodes instead.
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

	// "Necto" or "Nexto", from the model file that actually loaded. Which one it
	// is changes what the Elo means, so callers should say it out loud.
	const char *OpponentName() const;

  private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

// Set, not averaged: these are computed once per round, matching how the
// existing Rating/* metric is emitted.
void ReportNectoBench(const NectoBenchResult &result, GGL::Report &report);

} // namespace Dash
