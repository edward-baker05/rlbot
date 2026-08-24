#pragma once

#include "../env/Obs.h"
#include "../policy/Policy.h"
#include "PacketConvert.h"

#include <rlbot/Bot.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace Hive {

struct BotSettings {
	std::filesystem::path model;
	std::filesystem::path collisionMeshes = "collision_meshes";
	int maxPlayersPerTeam = 1;
	int tickSkip = 8;
	int actionDelay = 7;
	bool maskActions = false;
	ObsMode obs = ObsMode::Default;
	ModelShape modelShape = {};
	bool deterministic = true;
	float temperature = 1.f;
	bool useGPU = true;

	static BotSettings FromEnvironment();
};

struct SharedContext {
	BotSettings settings;
	std::unique_ptr<RLGC::ObsBuilder> obsBuilder;
	std::unique_ptr<RLGC::ActionParser> actionParser;
	std::unique_ptr<Policy> policy;
	int obsSize = 0;

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
	struct CarState {
		RLGC::Action queued = {};
		RLGC::Action applied = {};
		int ticks = -1;
		bool needsInference = true;
	};

	PacketConverter converter;
	std::unordered_map<unsigned, CarState> cars;

	float prevSeconds = -1.f;
	bool loggedError = false;
};

}  // namespace Hive
