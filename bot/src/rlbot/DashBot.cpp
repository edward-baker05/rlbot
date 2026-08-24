#include "DashBot.h"

#include "../env/Actions.h"
#include "../env/Obs.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

using namespace RLGC;

namespace Dash {

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
		std::fprintf(stderr, "[DashBot] WARNING: %s is not an integer, using %d\n", key, fallback);
		return fallback;
	}
}

BotSettings BotSettings::FromEnvironment() {
	BotSettings s = {};
	const std::string model = EnvOr("DASH_MODEL", EnvOr("HIVE_MODEL", ""));
	if (model.empty()) {
		throw std::runtime_error(
			"DASH_MODEL is not set. Point it at a GigaLearn checkpoint folder, "
			"e.g. checkpoints/main/50000000");
	}
	s.model = model;

	s.collisionMeshes = EnvOr("DASH_COLLISION_MESHES", EnvOr("HIVE_COLLISION_MESHES", "collision_meshes"));
	s.maxPlayersPerTeam = EnvIntOr("DASH_MAX_PLAYERS_PER_TEAM", EnvIntOr("HIVE_MAX_PLAYERS_PER_TEAM", 1));
	s.tickSkip = EnvIntOr("DASH_TICK_SKIP", EnvIntOr("HIVE_TICK_SKIP", 8));
	s.maskActions = EnvIntOr("DASH_MASK_ACTIONS", EnvIntOr("HIVE_MASK_ACTIONS", 0)) != 0;
	s.obs = ObsMode::Default;
	s.actionDelay = EnvIntOr("DASH_ACTION_DELAY", EnvIntOr("HIVE_ACTION_DELAY", 7));
	s.deterministic = EnvIntOr("DASH_DETERMINISTIC", EnvIntOr("HIVE_DETERMINISTIC", 1)) != 0;
	s.useGPU = EnvIntOr("DASH_USE_GPU", EnvIntOr("HIVE_USE_GPU", 1)) != 0;

	return s;
}

SharedContext& Context() {
	static SharedContext ctx;
	return ctx;
}

void SharedContext::Initialize(const BotSettings& s) {
	settings = s;

	RocketSim::Init(settings.collisionMeshes);

	obsSize = ProbeObsSize(settings.maxPlayersPerTeam, settings.obs);
	obsBuilder = MakeObsBuilder(settings.maxPlayersPerTeam, settings.obs);
	actionParser = MakeActionParser(settings.maskActions);

	policy = std::make_unique<Policy>(
		obsBuilder.get(), obsSize, actionParser.get(), settings.modelShape, settings.useGPU);

	policy->Load(settings.model);
	std::printf("[DashBot] Loaded model from %s\n", settings.model.c_str());

	std::printf("[DashBot] Observation size %d, tickSkip %d, actionDelay %d, %s\n",
	            obsSize, settings.tickSkip, settings.actionDelay,
	            settings.useGPU ? "GPU" : "CPU");
}

DashBot::DashBot(std::unordered_set<unsigned> indices,
                 unsigned team,
                 std::string name) noexcept
	: rlbot::Bot(std::move(indices), team, std::move(name)) {
	std::printf("[DashBot] Team %u controlling %zu car(s):", team, this->indices.size());
	for (unsigned idx : this->indices) {
		std::printf(" %u", idx);
		cars[idx] = {};
	}
	std::printf("\n");
}

DashBot::~DashBot() noexcept = default;

void DashBot::initialize(const rlbot::flat::ControllableTeamInfo* controllableTeamInfo,
                         const rlbot::flat::FieldInfo* fieldInfo,
                         const rlbot::flat::MatchConfiguration* matchConfiguration) noexcept {
	converter.Initialize(fieldInfo);
	converter.Reset();
	prevSeconds = -1.f;
}

void DashBot::update(const rlbot::flat::GamePacket* packet,
                     const rlbot::flat::BallPrediction* ballPrediction) noexcept {
	try {
		if (!packet || !packet->players()) {
			for (unsigned idx : indices)
				setOutput(idx, {});
			return;
		}

		const float now = packet->match_info() ? packet->match_info()->seconds_elapsed() : 0.f;
		int ticksElapsed = 1;
		if (prevSeconds >= 0.f) {
			ticksElapsed = static_cast<int>(std::lround((now - prevSeconds) * 120.f));
			ticksElapsed = RS_MAX(0, RS_MIN(ticksElapsed, 32));
		}
		prevSeconds = now;

		const GameState gs = converter.Convert(packet);
		const auto& settings = Context().settings;

		std::vector<unsigned> needInference;
		std::vector<Player> players;
		std::vector<GameState> states;

		for (unsigned idx : indices) {
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

		if (!needInference.empty()) {
			auto actions = Context().policy->InferBatch(
				players, states, settings.deterministic, settings.temperature);

			for (size_t i = 0; i < needInference.size(); i++) {
				CarState& car = cars[needInference[i]];
				car.queued = actions[i];
				car.needsInference = false;
			}
		}

		for (unsigned idx : indices) {
			CarState& car = cars[idx];

			if (car.ticks >= (settings.actionDelay - 1) || car.ticks == -1)
				car.applied = car.queued;

			if (car.ticks >= settings.tickSkip || car.ticks == -1) {
				car.ticks = 0;
				car.needsInference = true;
			}

			setOutput(idx, rlbot::flat::ControllerState(
				car.applied.throttle,
				car.applied.steer,
				car.applied.pitch,
				car.applied.yaw,
				car.applied.roll,
				car.applied.jump == 1.f,
				car.applied.boost == 1.f,
				car.applied.handbrake == 1.f,
				false));
		}
	} catch (const std::exception& e) {
		if (!loggedError) {
			loggedError = true;
			std::fprintf(stderr, "[DashBot] ERROR in update(): %s\n", e.what());
		}
		for (unsigned idx : indices)
			setOutput(idx, {});
	} catch (...) {
		if (!loggedError) {
			loggedError = true;
			std::fprintf(stderr, "[DashBot] ERROR in update(): unknown exception\n");
		}
		for (unsigned idx : indices)
			setOutput(idx, {});
	}
}

}  // namespace Dash
