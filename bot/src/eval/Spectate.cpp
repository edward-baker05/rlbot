#include "Spectate.h"

#include "../Config.h"
#include "../env/Env.h"
#include "../env/Actions.h"
#include "../env/Obs.h"
#include "../policy/Policy.h"
#include "Checkpoints.h"

#include <GigaLearnCPP/Util/RenderSender.h>

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

// Empty path when following a run that has not saved a checkpoint yet.
fs::path ResolveCheckpoint(const SpectateConfig& cfg) {
	if (!cfg.followRun.empty())
		return FindLatestCheckpoint(cfg.followRun);
	return cfg.model;
}

} // namespace

void RunSpectate(const SpectateConfig& cfg) {
	if (cfg.model.empty() == cfg.followRun.empty())
		throw std::runtime_error("RunSpectate(): pass exactly one of --model or --follow");

	// Pin inference to one thread so a spectator cannot steal a run's CPU.
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

	// Deployment-side values, so what is watched matches what is deployed.
	TrainConfig tcfg = {};
	const int obsSize = ProbeObsSize(tcfg.maxPlayersPerTeam, tcfg.obs);
	auto obsBuilder = MakeObsBuilder(tcfg.maxPlayersPerTeam, tcfg.obs);
	auto parser = MakeActionParser(tcfg.maskActions);

	// Learner's constructor normally starts the interpreter; there is none here.
	pybind11::initialize_interpreter();
	GGL::RenderSender sender(cfg.timeScale);

	std::unique_ptr<StateSetter> setter;
	if (cfg.spawns == SpectateSpawns::Training)
		setter.reset(BuildSpawner(tcfg));

	NoTouchCondition noTouch(tcfg.noTouchTimeoutSeconds);
	GoalScoreCondition goalScored;

	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	auto policy = std::make_unique<Policy>(obsBuilder.get(), obsSize, parser.get(),
	                                       tcfg.modelShape, cfg.useGPU);
	policy->Load(checkpoint);
	std::printf("Spectating %s (%s, %s spawns) -> RocketSimVis on UDP 9273\n",
	            checkpoint.string().c_str(),
	            cfg.deterministic ? "deterministic" : "stochastic",
	            cfg.spawns == SpectateSpawns::Training ? "training" : "kickoff");

	for (int episode = 0; cfg.episodes == 0 || episode < cfg.episodes; episode++) {
		// Between episodes only: swapping the policy under a car mid-play misleads.
		if (!cfg.followRun.empty()) {
			fs::path latest = FindLatestCheckpoint(cfg.followRun);
			if (!latest.empty() && latest != checkpoint) {
				checkpoint = latest;
				policy = std::make_unique<Policy>(obsBuilder.get(), obsSize, parser.get(),
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

		// One GameState reused for the whole episode.
		GameState gs;
		gs.UpdateFromArena(arena, std::vector<Action>(2), nullptr);

		noTouch.Reset(gs);
		goalScored.Reset(gs);

		while (true) {
			auto acts = policy->InferBatch({gs.players[0], gs.players[1]}, {gs, gs},
			                               cfg.deterministic);

			// Replay the training cadence exactly: hold the action for actionDelay ticks.
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
