#include "Spectate.h"

#include "../Config.h"
#include "../env/Env.h"
#include "../env/Obs.h"
#include "../policy/Policy.h"
#include "Checkpoints.h"

#include <GigaLearnCPP/Util/RenderSender.h>

#include <RLGymCPP/ActionParsers/DefaultAction.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <pybind11/embed.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>

using namespace RLGC;
namespace fs = std::filesystem;

namespace Hive {

namespace {

// Resolve which checkpoint to play. Returns an empty path when following a run
// that has not saved one yet, which is a wait-and-retry, not an error.
fs::path ResolveCheckpoint(const SpectateConfig& cfg) {
	if (!cfg.followRun.empty())
		return FindLatestCheckpoint(cfg.followRun);
	return cfg.model;
}

} // namespace

void RunSpectate(const SpectateConfig& cfg) {
	if (cfg.model.empty() == cfg.followRun.empty())
		throw std::runtime_error("RunSpectate(): pass exactly one of --model or --follow");

	// Measured, not assumed: inferring for two cars at 15 Hz on libtorch's
	// default thread pool cost the concurrent 128-game training run 4.4% of its
	// throughput (65.6k -> 62.7k steps/sec) purely through CPU contention on a
	// 6-core box. Pinned to one thread it is 0.3%, inside the noise. A
	// spectator that slows the run it is watching is not safe to leave open.
	//
	// Set through the environment rather than at::set_num_threads() because
	// libtorch's headers are private to GigaLearnCPP and are not on this
	// target's include path. The pools read these when they first initialize,
	// which is during the first inference, well after this point. The 0 flag
	// leaves an explicit setting from the caller alone.
	if (!cfg.useGPU) {
		setenv("OMP_NUM_THREADS", "1", 0);
		setenv("MKL_NUM_THREADS", "1", 0);
	}

	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	RocketSim::Init(meshEnv ? meshEnv : "collision_meshes");

	if (cfg.seed >= 0)
		srand(static_cast<unsigned>(cfg.seed));

	fs::path checkpoint = ResolveCheckpoint(cfg);
	if (checkpoint.empty()) {
		throw std::runtime_error(
			"RunSpectate(): no complete checkpoint in " + cfg.followRun.string() +
			" yet. A run saves its first at tsPerSave steps; try again shortly.");
	}

	// Deployment-side values, so what is watched matches what is trained and
	// what is deployed. A divergence here would not crash -- it would just make
	// the bot look worse than it is, which is the whole class of bug the
	// `verify` subcommand exists to catch.
	TrainConfig tcfg = {};
	const int obsSize = ProbeObsSize(tcfg.maxPlayersPerTeam);
	auto obsBuilder = MakeObsBuilder(tcfg.maxPlayersPerTeam);
	DefaultAction parser;

	// Learner's constructor is what normally starts the interpreter; there is
	// no Learner here, and RenderSender needs it to import the receiver module.
	pybind11::initialize_interpreter();
	GGL::RenderSender sender(cfg.timeScale);

	std::unique_ptr<StateSetter> setter;
	if (cfg.spawns == SpectateSpawns::Curriculum)
		setter.reset(BuildGeneralCurriculum(tcfg.curriculum));

	NoTouchCondition noTouch(tcfg.noTouchTimeoutSeconds);
	GoalScoreCondition goalScored;

	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	auto policy = std::make_unique<Policy>(obsBuilder.get(), obsSize, &parser,
	                                       tcfg.modelShape, cfg.useGPU);
	policy->Load(checkpoint);
	std::printf("Spectating %s (%s, %s spawns) -> RocketSimVis on UDP 9273\n",
	            checkpoint.string().c_str(),
	            cfg.deterministic ? "deterministic" : "stochastic",
	            cfg.spawns == SpectateSpawns::Curriculum ? "curriculum" : "kickoff");

	for (int episode = 0; cfg.episodes == 0 || episode < cfg.episodes; episode++) {
		// Between episodes, not mid-episode: swapping the policy under a car
		// mid-play would show a discontinuity that is an artifact of watching,
		// not of the bot.
		if (!cfg.followRun.empty()) {
			fs::path latest = FindLatestCheckpoint(cfg.followRun);
			if (!latest.empty() && latest != checkpoint) {
				checkpoint = latest;
				policy = std::make_unique<Policy>(obsBuilder.get(), obsSize, &parser,
				                                  tcfg.modelShape, cfg.useGPU);
				policy->Load(checkpoint);
				std::printf("-> now playing %s\n", checkpoint.filename().string().c_str());
				std::fflush(stdout);
			}
		}

		if (setter)
			setter->ResetArena(arena);
		else
			arena->ResetToRandomKickoff();

		// One GameState reused for the whole episode. Constructing a fresh one
		// per step (as RunEval does, where it is harmless) makes deltaTime the
		// arena's whole lifetime rather than one step: it would break both the
		// renderer's wall-clock pacing and ballTouchedStep, and so the no-touch
		// timeout too.
		GameState gs;
		gs.UpdateFromArena(arena, std::vector<Action>(2), nullptr);

		noTouch.Reset(gs);
		goalScored.Reset(gs);

		while (true) {
			auto acts = policy->InferBatch({gs.players[0], gs.players[1]}, {gs, gs},
			                               cfg.deterministic);

			// Replay the training cadence exactly: hold the previous action for
			// actionDelay ticks, then apply the fresh one for the rest.
			gs.ResetBeforeStep();
			arena->Step(tcfg.actionDelay);

			std::vector<Action> applied(2);
			auto carItr = arena->_cars.begin();
			for (int i = 0; i < 2; i++, carItr++) {
				applied[i] = acts[i];
				(*carItr)->controls = (CarControls)applied[i];
			}
			arena->Step(tcfg.tickSkip - tcfg.actionDelay);
			gs.UpdateFromArena(arena, applied, nullptr);

			sender.Send(gs); // Also paces to wall-clock.

			if (goalScored.IsTerminal(gs) || noTouch.IsTerminal(gs))
				break;
		}
	}

	delete arena;
}

} // namespace Hive
