#include "HivemindBot.h"

#include "../env/Actions.h"
#include "../env/Obs.h"


#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace RLGC;

namespace Hive {

// ---------------------------------------------------------------------------
// Settings
// ---------------------------------------------------------------------------

static std::string EnvOr(const char* key, const std::string& fallback) {
	const char* v = std::getenv(key);
	return (v && *v) ? std::string(v) : fallback;
}

static int EnvIntOr(const char* key, int fallback) {
	const char* v = std::getenv(key);
	if (!v || !*v)
		return fallback;
	try {
		return std::stoi(v);
	} catch (...) {
		std::fprintf(stderr, "[HivemindBot] WARNING: %s is not an integer, using %d\n", key, fallback);
		return fallback;
	}
}

BotSettings BotSettings::FromEnvironment() {
	BotSettings s = {};

	const std::string model = EnvOr("HIVE_MODEL", "");
	if (model.empty()) {
		throw std::runtime_error(
			"HIVE_MODEL is not set. Point it at a GigaLearn checkpoint folder, "
			"e.g. checkpoints/main/50000000");
	}
	s.model = model;

	s.collisionMeshes = EnvOr("HIVE_COLLISION_MESHES", "collision_meshes");
	s.maxPlayersPerTeam = EnvIntOr("HIVE_MAX_PLAYERS_PER_TEAM", 1);
	s.tickSkip = EnvIntOr("HIVE_TICK_SKIP", 8);
	s.maskActions = EnvIntOr("HIVE_MASK_ACTIONS", 0) != 0;
	s.actionDelay = EnvIntOr("HIVE_ACTION_DELAY", 7);
	s.deterministic = EnvIntOr("HIVE_DETERMINISTIC", 1) != 0;
	s.useGPU = EnvIntOr("HIVE_USE_GPU", 1) != 0;

	return s;
}

// ---------------------------------------------------------------------------
// Shared context
// ---------------------------------------------------------------------------

SharedContext& Context() {
	static SharedContext ctx;
	return ctx;
}

void SharedContext::Initialize(const BotSettings& s) {
	settings = s;

	RocketSim::Init(settings.collisionMeshes);

	obsSize = ProbeObsSize(settings.maxPlayersPerTeam);
	obsBuilder = MakeObsBuilder(settings.maxPlayersPerTeam);
	actionParser = MakeActionParser(settings.maskActions);

	policy = std::make_unique<Policy>(
		obsBuilder.get(), obsSize, actionParser.get(), settings.modelShape, settings.useGPU);

	policy->Load(settings.model);
	std::printf("[HivemindBot] Loaded model from %s\n", settings.model.c_str());

	std::printf("[HivemindBot] Observation size %d, tickSkip %d, actionDelay %d, %s\n",
	            obsSize, settings.tickSkip, settings.actionDelay,
	            settings.useGPU ? "GPU" : "CPU");
}

// ---------------------------------------------------------------------------
// Bot
// ---------------------------------------------------------------------------

HivemindBot::HivemindBot(std::unordered_set<unsigned> indices,
                         unsigned team,
                         std::string name) noexcept
	: rlbot::Bot(std::move(indices), team, std::move(name)) {

	std::printf("[HivemindBot] Team %u controlling %zu car(s):", team, this->indices.size());
	for (unsigned idx : this->indices) {
		std::printf(" %u", idx);
		cars[idx] = {};
	}
	std::printf("\n");
}

HivemindBot::~HivemindBot() noexcept = default;

void HivemindBot::initialize(const rlbot::flat::ControllableTeamInfo* controllableTeamInfo,
                             const rlbot::flat::FieldInfo* fieldInfo,
                             const rlbot::flat::MatchConfiguration* matchConfiguration) noexcept {
	converter.Initialize(fieldInfo);
	converter.Reset();
	prevSeconds = -1.f;
}

void HivemindBot::update(const rlbot::flat::GamePacket* packet,
                         const rlbot::flat::BallPrediction* ballPrediction) noexcept {
	// The interface marks update() noexcept, so nothing may escape. A throwing
	// inference call must degrade to "keep holding the last controls" rather
	// than terminate the process mid-match.
	try {
		if (!packet || !packet->players()) {
			for (unsigned idx : indices)
				setOutput(idx, {});
			return;
		}

		// --- Tick accounting -------------------------------------------------
		const float now = packet->match_info() ? packet->match_info()->seconds_elapsed() : 0.f;
		int ticksElapsed = 1;
		if (prevSeconds >= 0.f) {
			ticksElapsed = static_cast<int>(std::lround((now - prevSeconds) * 120.f));
			// Clamp: a paused game or a goal replay can produce a huge or
			// negative delta, which would otherwise desync the action cadence.
			ticksElapsed = RS_MAX(0, RS_MIN(ticksElapsed, 32));
		}
		prevSeconds = now;

		const GameState gs = converter.Convert(packet);
		const auto& settings = Context().settings;

		// --- Decide which cars need a fresh action ---------------------------
		std::vector<unsigned> needInference;
		std::vector<Player> players;
		std::vector<GameState> states;

		for (unsigned idx : indices) {
			// A car we were assigned may not be in the packet yet (or at all,
			// briefly, during a match restart).
			if (idx >= packet->players()->size())
				continue;

			CarState& car = cars[idx];
			car.ticks += ticksElapsed;

			if (car.needsInference) {
				needInference.push_back(idx);
				players.push_back(gs.players[idx]);
				states.push_back(gs);
			}
		}

		// --- One batched forward pass for every car that needs it ------------
		if (!needInference.empty()) {
			auto actions = Context().policy->InferBatch(
				players, states, settings.deterministic, settings.temperature);

			for (size_t i = 0; i < needInference.size(); i++) {
				CarState& car = cars[needInference[i]];
				car.queued = actions[i];
				car.needsInference = false;
			}
		}

		// --- Apply the trained action cadence --------------------------------
		for (unsigned idx : indices) {
			CarState& car = cars[idx];

			// Latency: hold the previous action until actionDelay ticks have
			// passed, matching how the policy experienced the world in
			// training.
			if (car.ticks >= (settings.actionDelay - 1) || car.ticks == -1)
				car.applied = car.queued;

			// Repeat: infer again only every tickSkip ticks.
			if (car.ticks >= settings.tickSkip || car.ticks == -1) {
				car.ticks = 0;
				car.needsInference = true;
			}

			// ControllerState is a flatbuffers struct: build it in one shot.
			// Field order is throttle, steer, pitch, yaw, roll, then the
			// buttons -- matching RLGC::Action's own ordering.
			setOutput(idx, rlbot::flat::ControllerState(
				car.applied.throttle,
				car.applied.steer,
				car.applied.pitch,
				car.applied.yaw,
				car.applied.roll,
				car.applied.jump == 1.f,
				car.applied.boost == 1.f,
				car.applied.handbrake == 1.f,
				/*use_item=*/false));
		}

	} catch (const std::exception& e) {
		if (!loggedError) {
			loggedError = true; // Log once; a per-tick error would flood output
			std::fprintf(stderr, "[HivemindBot] ERROR in update(): %s\n", e.what());
		}
		for (unsigned idx : indices)
			setOutput(idx, {});
	} catch (...) {
		if (!loggedError) {
			loggedError = true;
			std::fprintf(stderr, "[HivemindBot] ERROR in update(): unknown exception\n");
		}
		for (unsigned idx : indices)
			setOutput(idx, {});
	}
}

} // namespace Hive
