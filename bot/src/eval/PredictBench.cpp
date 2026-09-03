#include "PredictBench.h"

#include "../Config.h"
#include "../env/Obs.h"
#include "../env/PredictiveObs.h"

#include <RLGymCPP/Gamestates/GameState.h>

#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using namespace RLGC;

namespace Dash {

extern uint64_t g_predictNanos;
extern uint64_t g_assembleNanos;

namespace {

struct BenchArena {
	Arena* arena = nullptr;
	std::unique_ptr<ObsBuilder> obs;
	GameState state;
};

struct BenchResult {
	double envStepsPerSec = 0;
	uint64_t simulations = 0;   // full window re-simulations
	uint64_t simTicks = 0;      // ball-only arena ticks stepped
	uint64_t envSteps = 0;
};

// Random controller inputs. The cars are not trying to play, but they do move
// and hit the ball, which is the only thing the prediction cache cares about:
// an untouched ball is the cheap case, and a benchmark that never touches it
// measures the wrong regime.
Action RandomAction(std::mt19937& rng) {
	std::uniform_real_distribution<float> axis(-1.f, 1.f);
	std::uniform_int_distribution<int> flag(0, 1);

	Action a = {};
	a.throttle = axis(rng);
	a.steer = axis(rng);
	a.pitch = axis(rng);
	a.yaw = axis(rng);
	a.roll = axis(rng);
	a.jump = flag(rng);
	a.boost = flag(rng);
	a.handbrake = flag(rng);
	return a;
}

// One pass: step every arena by tickSkip, then build every player's obs.
// Mirrors what EnvSet does per env step, minus the policy.
BenchResult TimeMode(ObsMode mode, int numArenas, int steps, int tickSkip,
                     bool drive) {
	std::vector<BenchArena> arenas(numArenas);
	for (int i = 0; i < numArenas; i++) {
		arenas[i].arena = Arena::Create(GameMode::SOCCAR);
		arenas[i].arena->AddCar(Team::BLUE);
		arenas[i].arena->AddCar(Team::ORANGE);
		arenas[i].arena->ResetToRandomKickoff(i);
		arenas[i].obs = MakeObsBuilder(3, mode);
		arenas[i].state = GameState(arenas[i].arena);
		arenas[i].obs->Reset(arenas[i].state);
	}

	// Seeded, so the Advanced and Predictive arms see identical car inputs and
	// therefore identical ball trajectories. Without that the two arms would
	// diverge and the comparison would be between different games.
	std::mt19937 rng(12345);

	const auto start = std::chrono::steady_clock::now();

	for (int s = 0; s < steps; s++) {
		for (auto& a : arenas) {
			if (drive) {
				for (Car* car : a.arena->_cars)
					car->controls =
						static_cast<CarControls>(RandomAction(rng));
			}

			a.arena->Step(tickSkip);
			a.state.UpdateFromArena(a.arena,
			                        std::vector<Action>(a.arena->_cars.size()),
			                        nullptr);
			for (const auto& p : a.state.players)
				a.obs->BuildObs(p, a.state);
		}
	}

	const auto end = std::chrono::steady_clock::now();

	BenchResult out = {};
	out.envSteps = (uint64_t)numArenas * steps;
	for (auto& a : arenas) {
		if (auto* pred = dynamic_cast<PredictiveObs*>(a.obs.get())) {
			out.simulations += pred->SimulationCount();
			out.simTicks += pred->SimulatedTickCount();
		}
		delete a.arena;
	}

	const double seconds = std::chrono::duration<double>(end - start).count();
	out.envStepsPerSec = (double)out.envSteps / seconds;
	return out;
}

// Returns the throughput loss, as a percentage.
double ReportArm(const char* regime, int numArenas, int steps, int tickSkip,
                 bool drive) {
	const BenchResult advanced =
		TimeMode(ObsMode::Advanced, numArenas, steps, tickSkip, drive);
	g_predictNanos = 0;
	g_assembleNanos = 0;
	const BenchResult predictive =
		TimeMode(ObsMode::Predictive, numArenas, steps, tickSkip, drive);

	const double lossPct =
		100.0 * (1.0 - predictive.envStepsPerSec / advanced.envStepsPerSec);

	const double stepsPerSim =
		predictive.simulations
			? (double)predictive.envSteps / (double)predictive.simulations
			: 0.0;
	const double ballTicksPerEnvStep =
		predictive.envSteps
			? (double)predictive.simTicks / (double)predictive.envSteps
			: 0.0;

	std::printf("\n%s\n", regime);
	std::printf("  %-12s %10.0f env-steps/sec\n", "Advanced",
	            advanced.envStepsPerSec);
	std::printf("  %-12s %10.0f env-steps/sec\n", "Predictive",
	            predictive.envStepsPerSec);
	std::printf("  throughput loss: %.1f%%\n", lossPct);
	std::printf("  predictor.Get(): %.1f us/env-step   rest of BuildObs: %.1f us/env-step\n",
	            (double)g_predictNanos / 1000.0 / (double)predictive.envSteps,
	            (double)g_assembleNanos / 1000.0 / (double)predictive.envSteps);
	std::printf("  cache: 1 re-simulation per %.1f env-steps "
	            "(%.1f ball-only ticks simulated per env-step, "
	            "against %d full-arena ticks)\n",
	            stepsPerSim, ballTicksPerEnvStep, tickSkip);
	return lossPct;
}

}  // namespace

int RunPredictBench(int numArenas, int steps) {
	const int tickSkip = TrainConfig{}.tickSkip;

	std::printf("Warming up...\n");
	TimeMode(ObsMode::Advanced, numArenas, steps / 4, tickSkip, true);

	std::printf("\n%d arenas, %d steps, tickSkip %d, window %d ticks, "
	            "deepest sample %d ticks\n",
	            numArenas, steps, tickSkip,
	            BallPredictor::WINDOW_TICKS,
	            BallPredictor::SAMPLE_TICKS.back());

	// The regime that matters is the driven one: training arenas have the ball
	// touched every second or two, and every touch costs a full window
	// re-simulation. The idle arm is the floor, where sliding is all there is.
	ReportArm("IDLE (cars take no input; ball is never touched -- best case)",
	          numArenas, steps, tickSkip, false);
	const double lossPct =
		ReportArm("DRIVEN (random inputs; ball is touched -- representative)",
		          numArenas, steps, tickSkip, true);

	// A sliding window costs tickSkip ball-only ticks per env-step for an
	// untouched ball, plus one window re-simulation per touch.
	static constexpr double GATE_PCT = 25.0;

	std::printf("\n  GATE (driven): %s -- %.1f%% loss, budget %.0f%%\n",
	            lossPct <= GATE_PCT ? "PASS" : "FAIL", lossPct, GATE_PCT);

	return lossPct <= GATE_PCT ? 0 : 1;
}

}  // namespace Dash
