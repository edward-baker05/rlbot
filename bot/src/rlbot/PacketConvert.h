#pragma once

#include <RLGymCPP/Gamestates/GameState.h>

#include <rlbot/Bot.h>

#include <unordered_map>
#include <vector>

namespace Hive {

class PacketConverter {
public:
	void Initialize(const rlbot::flat::FieldInfo* fieldInfo);

	RLGC::GameState Convert(const rlbot::flat::GamePacket* packet);

	void Reset() { lastTouchTimes.clear(); }

private:
	std::vector<int> rlgymToRLBotPad;

	std::unordered_map<int, float> lastTouchTimes;
};

}  // namespace Hive
