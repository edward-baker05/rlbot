#pragma once

#include "../policy/Policy.h"
#include "PacketConvert.h"

#include <rlbot/Bot.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Hive {

// ============================================================================
// HivemindBot
// ============================================================================
// One process, one team, every car on it.
//
// RLBot v5 gives us this natively: with `hivemind = true` in bot.toml, the
// framework assigns all same-team cars sharing our agent_id to a single Bot
// instance, and `indices` holds all of them. There is no coordination protocol
// to write, and every car is inferred in ONE batched forward pass.
//
// Humans need no handling. RLBot only ever hands us indices for cars we
// control; every other car -- human or bot, our team or theirs -- arrives as an
// ordinary entry in the packet's player list and flows into the padded
// observation like any other.
// ============================================================================

// Settings loaded from the environment before the bot connects, since RLBot
// gives a bot no config channel of its own.
struct BotSettings {
	std::filesystem::path model;  // HIVE_MODEL
	std::filesystem::path collisionMeshes = "collision_meshes";

	int maxPlayersPerTeam = 1;
	int tickSkip = 8;
	int actionDelay = 7;

	ModelShape modelShape = {};

	// Deterministic play picks the highest-probability action every step. It is
	// noticeably stronger than sampling and is what you want in a real match;
	// sampling is only useful if you want variety.
	bool deterministic = true;
	float temperature = 1.f;

	bool useGPU = true;

	// Read settings from environment variables, applying the defaults above.
	// Throws std::runtime_error if HIVE_MODEL is unset.
	static BotSettings FromEnvironment();
};

// Process-wide state shared by every HivemindBot instance. RLBot constructs
// bots through a factory that takes no user parameters, so this is how settings
// and the loaded model reach them. The model is loaded once and shared, which
// also means a blue and an orange hivemind in the same process share weights.
struct SharedContext {
	BotSettings settings;
	std::unique_ptr<RLGC::ObsBuilder> obsBuilder;
	std::unique_ptr<RLGC::ActionParser> actionParser;
	std::unique_ptr<Policy> policy;
	int obsSize = 0;

	// Initialise RocketSim, build the obs/action pipeline, and load the model.
	// Call once before connecting. Throws on failure.
	void Initialize(const BotSettings& settings);
};

SharedContext& Context();

class HivemindBot final : public rlbot::Bot {
public:
	HivemindBot(std::unordered_set<unsigned> indices, unsigned team, std::string name) noexcept;
	~HivemindBot() noexcept override;

	void initialize(const rlbot::flat::ControllableTeamInfo* controllableTeamInfo,
	                const rlbot::flat::FieldInfo* fieldInfo,
	                const rlbot::flat::MatchConfiguration* matchConfiguration) noexcept override;

	void update(const rlbot::flat::GamePacket* packet,
	            const rlbot::flat::BallPrediction* ballPrediction) noexcept override;

private:
	// Per-car action repeat state. The policy was trained acting once every
	// `tickSkip` ticks with `actionDelay` ticks of latency; replaying that
	// cadence at deployment is not optional. Acting every tick instead makes
	// the bot behave measurably differently from the one that was trained.
	struct CarState {
		RLGC::Action queued = {};   // Freshly inferred, not yet applied
		RLGC::Action applied = {};  // Currently being held
		int ticks = -1;
		bool needsInference = true;
	};

	PacketConverter converter;
	std::unordered_map<unsigned, CarState> cars;

	float prevSeconds = -1.f;
	bool loggedError = false;
};

} // namespace Hive
