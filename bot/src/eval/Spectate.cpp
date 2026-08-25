#include "Spectate.h"

#include "../Config.h"
#include "../env/Actions.h"
#include "../env/Env.h"
#include "../env/Obs.h"
#include "../policy/Policy.h"
#include "Checkpoints.h"
#include "KeyPoller.h"
#include "RewardProbe.h"

#include <GigaLearnCPP/Util/RenderSender.h>

#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>

#include <pybind11/embed.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace RLGC;
namespace fs = std::filesystem;

namespace Dash {

namespace {

constexpr const char *KEY_HELP =
	"  [space] pause/resume  [.] step  [c] run  [b] history  [s] totals  "
	"[q] quit";

// Empty path when following a run that has not saved a checkpoint yet.
fs::path ResolveCheckpoint(const SpectateConfig &cfg) {
	if (!cfg.followRun.empty())
		return FindLatestCheckpoint(cfg.followRun);
	return cfg.model;
}

} // namespace

void RunSpectate(const SpectateConfig &cfg) {
	if (cfg.model.empty() == cfg.followRun.empty())
		throw std::runtime_error(
			"RunSpectate(): pass exactly one of --model or --follow");

	// Pin inference to one thread so a spectator cannot steal a run's CPU.
	if (!cfg.useGPU) {
		setenv("OMP_NUM_THREADS", "1", 0);
		setenv("MKL_NUM_THREADS", "1", 0);
	}

	const char *meshEnv = std::getenv("DASH_COLLISION_MESHES");
	if (!meshEnv)
		meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	RocketSim::Init(meshEnv ? meshEnv : "collision_meshes");

	if (cfg.seed >= 0)
		srand(static_cast<unsigned>(cfg.seed));

	fs::path checkpoint = ResolveCheckpoint(cfg);
	if (checkpoint.empty()) {
		throw std::runtime_error("RunSpectate(): no complete checkpoint in " +
								 cfg.followRun.string() +
								 " yet. A run saves its first at tsPerSave "
								 "steps; try again shortly.");
	}

	// Deployment-side values, so what is watched matches what is deployed.
	TrainConfig tcfg = {};
	const int obsSize = ProbeObsSize(tcfg.maxPlayersPerTeam, tcfg.obs);
	auto obsBuilder = MakeObsBuilder(tcfg.maxPlayersPerTeam, tcfg.obs);
	auto parser = MakeActionParser(tcfg.maskActions);

	// Learner's constructor normally starts the interpreter; there is none
	// here.
	pybind11::initialize_interpreter();
	GGL::RenderSender sender(cfg.timeScale);

	std::unique_ptr<StateSetter> setter;
	if (cfg.spawns == SpectateSpawns::Training)
		setter.reset(BuildSpawner(tcfg));

	NoTouchCondition noTouch(tcfg.noTouchTimeoutSeconds);
	GoalScoreCondition goalScored;

	Arena *arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	auto policy = std::make_unique<Policy>(
		obsBuilder.get(), obsSize, parser.get(), tcfg.modelShape, cfg.useGPU);
	policy->Load(checkpoint);
	std::printf("Spectating %s (%s, %s spawns) -> RocketSimVis on UDP 9273\n",
				checkpoint.string().c_str(),
				cfg.deterministic ? "deterministic" : "stochastic",
				cfg.spawns == SpectateSpawns::Training ? "training"
													   : "kickoff");

	// --rewards runs the training reward stack alongside the game and hands the
	// clock to the keyboard. Both stay null otherwise, and every guard below is
	// on `probe`, so plain spectating is untouched.
	std::unique_ptr<RewardProbe> probe;
	std::unique_ptr<KeyPoller> keys;
	bool paused = false;
	bool autoPause = true;
	bool quit = false;

	if (cfg.debugRewards) {
		probe = std::make_unique<RewardProbe>(tcfg, cfg.rewardPauseThreshold,
											  cfg.rewardHistorySteps);
		keys = std::make_unique<KeyPoller>();
		paused = cfg.rewardStartPaused;

		if (cfg.rewardPauseThreshold > 0.f)
			std::printf(
				"Auto-pausing when an event reward exceeds %.4f (weighted).\n",
				cfg.rewardPauseThreshold);
		else
			std::printf("Auto-pause disabled; stepping is manual.\n");

		if (keys->Active()) {
			std::printf("%s\n", KEY_HELP);
		} else {
			// Nothing could ever resume us, so pausing here would hang the
			// spectator outright. Fall back to a plain log of every step,
			// which is the more useful thing to pipe to a file anyway.
			paused = false;
			autoPause = false;
			std::printf("stdin is not a terminal, so keyboard control is off. "
						"Printing every step instead; run via "
						"scripts/spectate.sh for pause and step.\n");
		}
		std::fflush(stdout);
	}

	const bool interactive = keys && keys->Active();

	// Drains buffered keypresses and, while paused, blocks until the user
	// resumes or asks for a single step. Returns false when the user quits.
	// Pausing simply stops streaming: RocketSimVis clamps its interpolation
	// ratio at 1, so it holds the last frame it received.
	//
	// Leaving a pause resends the CURRENT state first. RocketSimVis paces its
	// interpolation by recv_interval, the wall-clock gap between the last two
	// packets, so after a long pause the next packet would be interpolated
	// over that whole pause -- the car would drift to its new position over
	// several seconds instead of stepping to it. Resending the unchanged state
	// costs nothing visually (read_from_json shifts next into prev, so prev
	// and next match and nothing moves) but resets that clock, leaving the
	// step's real packet a gap of milliseconds to interpolate across.
	auto handleInput = [&](const GameState &current) -> bool {
		const bool wasPaused = paused;

		auto resume = [&]() {
			if (wasPaused)
				sender.Send(current);
			return true;
		};

		for (;;) {
			switch (keys->Poll()) {
			case 'q':
				return false;
			case ' ':
				paused = !paused;
				if (paused) {
					std::printf("%s%s\n",
								probe->FormatLastStep("== PAUSED ==").c_str(),
								KEY_HELP);
				} else {
					autoPause = true;
					std::printf("-- running --\n");
				}
				std::fflush(stdout);
				break;
			case '.':
				// Advance exactly one step, then come back here.
				paused = true;
				return resume();
			case 'c':
				paused = false;
				autoPause = false;
				std::printf("-- running, auto-pause off until next keypress --\n");
				std::fflush(stdout);
				break;
			case 'b':
				std::printf("%s", probe->FormatHistory().c_str());
				std::fflush(stdout);
				break;
			case 's':
				std::printf("%s", probe->FormatEpisodeTotals().c_str());
				std::fflush(stdout);
				break;
			default:
				break;
			}

			if (!paused)
				return resume();

			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	};

	for (int episode = 0;
		 !quit && (cfg.episodes == 0 || episode < cfg.episodes);
		 episode++) {
		// Between episodes only: swapping the policy under a car mid-play
		// misleads.
		if (!cfg.followRun.empty()) {
			fs::path latest = FindLatestCheckpoint(cfg.followRun);
			if (!latest.empty() && latest != checkpoint) {
				checkpoint = latest;
				policy = std::make_unique<Policy>(obsBuilder.get(), obsSize,
												  parser.get(), tcfg.modelShape,
												  cfg.useGPU);
				policy->Load(checkpoint);
				const auto now = std::chrono::system_clock::now();
				const std::time_t t_c =
					std::chrono::system_clock::to_time_t(now);
				std::printf("-> now playing %s (%s)\n",
							checkpoint.filename().string().c_str(),
							std::ctime(&t_c));
				std::fflush(stdout);
			}
		}

		if (setter)
			setter->ResetArena(arena);
		else
			arena->ResetToRandomKickoff();

		// One GameState reused for the whole episode, plus the state it
		// replaced. Several rewards read state.prev and player.prev
		// (DirectionalTouchReward, the aerial rewards), and silently report 0
		// without it, so the probe would print fiction if this were not
		// threaded through the way EnvSet does it.
		GameState gs;
		GameState prevGs;
		prevGs.MakeEmpty();
		gs.UpdateFromArena(arena, std::vector<Action>(2), nullptr);

		noTouch.Reset(gs);
		goalScored.Reset(gs);
		if (probe)
			probe->BeginEpisode(gs);

		while (true) {
			if (probe && !handleInput(gs)) {
				quit = true;
				break;
			}

			auto acts = policy->InferBatch({gs.players[0], gs.players[1]},
										   {gs, gs}, cfg.deterministic);

			// Replay the training cadence exactly: hold the action for
			// actionDelay ticks.
			prevGs = gs;
			gs.ResetBeforeStep();
			arena->Step(tcfg.actionDelay);

			std::vector<Action> applied(2);
			auto carItr = arena->_cars.begin();
			for (int i = 0; i < 2; i++, carItr++) {
				applied[i] = acts[i];
				(*carItr)->controls = (CarControls)applied[i];
			}
			arena->Step(tcfg.tickSkip - tcfg.actionDelay);
			gs.UpdateFromArena(arena, applied,
							   prevGs.IsEmpty() ? nullptr : &prevGs);

			sender.Send(gs); // Also paces to wall-clock.

			// Both conditions are evaluated, never short-circuited: they are
			// stateful and EnvSet guarantees each is called once per step.
			const bool goalTerm = goalScored.IsTerminal(gs);
			const bool noTouchTerm = noTouch.IsTerminal(gs);
			const bool isFinal = goalTerm || noTouchTerm;

			if (probe) {
				probe->Step(gs, isFinal);

				char header[128];
				const float t = probe->StepCount() * tcfg.tickSkip / 120.f;

				if (paused || !interactive) {
					// Manual single-step, or the non-interactive log: always
					// print, even an all-zero step, so every keypress (or
					// every step) produces a visible frame marker.
					std::snprintf(header, sizeof(header),
								  "== ep %d  step %d  t=%.2fs ==", episode + 1,
								  probe->StepCount(), t);
					std::printf("%s", probe->FormatLastStep(header).c_str());
				} else if (autoPause && probe->Tripped()) {
					paused = true;
					std::snprintf(header, sizeof(header),
								  "== PAUSED  ep %d  step %d  t=%.2fs ==",
								  episode + 1, probe->StepCount(), t);
					std::printf("%s%s\n", probe->FormatLastStep(header).c_str(),
								KEY_HELP);
				}
				std::fflush(stdout);
			}

			if (isFinal)
				break;
		}

		if (probe && !quit) {
			std::printf("%s", probe->FormatEpisodeTotals().c_str());
			std::fflush(stdout);
		}
	}

	delete arena;
}

} // namespace Dash
