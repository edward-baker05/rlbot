#include "PacketConvert.h"

#include <RLGymCPP/CommonValues.h>

#include <cmath>
#include <cstdio>
#include <limits>

using namespace RLGC;

namespace Dash {

static inline Vec ToVec(const rlbot::flat::Vector3& v) {
	return Vec(v.x(), v.y(), v.z());
}

void PacketConverter::Initialize(const rlbot::flat::FieldInfo* fieldInfo) {
	rlgymToRLBotPad.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, -1);

	if (!fieldInfo || !fieldInfo->boost_pads()) {
		std::fprintf(stderr,
			"[PacketConverter] WARNING: no FieldInfo; falling back to identity boost pad mapping. "
			"Boost pad observations may be wrong.\n");
		for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
			rlgymToRLBotPad[i] = i;
		return;
	}

	const auto* pads = fieldInfo->boost_pads();
	int matched = 0;
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		const Vec& want = CommonValues::BOOST_LOCATIONS[i];
		int best = -1;
		float bestDistSq = std::numeric_limits<float>::max();
		for (unsigned j = 0; j < pads->size(); j++) {
			const Vec have = ToVec(*pads->Get(j)->location());
			const Vec d = have - want;
			const float distSq = d.x * d.x + d.y * d.y;
			if (distSq < bestDistSq) {
				bestDistSq = distSq;
				best = static_cast<int>(j);
			}
		}

		if (best >= 0 && bestDistSq < 100.f * 100.f) {
			rlgymToRLBotPad[i] = best;
			matched++;
		}
	}

	if (matched != CommonValues::BOOST_LOCATIONS_AMOUNT) {
		std::fprintf(stderr,
			"[PacketConverter] WARNING: matched only %d of %d boost pads to RLBot's field info. "
			"Unmatched pads will read as always-available.\n",
			matched, CommonValues::BOOST_LOCATIONS_AMOUNT);
	}
}

GameState PacketConverter::Convert(const rlbot::flat::GamePacket* packet) {
	GameState gs = {};

	// RocketSim's fixed physics rate. seconds_elapsed is the only clock the
	// packet carries, and it is monotonic -- it ticks up through kickoffs,
	// replays and pauses alike -- so it is a sound basis for a tick count.
	//
	// This matters because PredictiveObs caches its ball trajectory against
	// this counter. Left at zero, every frame looks like tick 0 to the cache,
	// the live ball never matches the slice it is compared against, and the
	// predictor re-simulates from scratch on every inference. That is correct
	// but it is a different code path from the one training exercises, which is
	// exactly the asymmetry that hides bugs.
	//
	// A new match restarts seconds_elapsed near zero; BallPredictor treats a
	// rewound clock as a new episode and re-simulates, so no reset is needed.
	if (const auto* info = packet->match_info()) {
		const float seconds = info->seconds_elapsed();
		if (seconds > 0.f)
			gs.lastTickCount = static_cast<uint64_t>(
				std::llround(static_cast<double>(seconds) * 120.0));
	}

	if (packet->balls() && packet->balls()->size() > 0) {
		const auto* phys = packet->balls()->Get(0)->physics();
		gs.ball.pos = ToVec(phys->location());
		gs.ball.vel = ToVec(phys->velocity());
		gs.ball.angVel = ToVec(phys->angular_velocity());
	}

	const auto* padStates = packet->boost_pads();
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		bool active = true;
		float timer = 0.f;
		const int j = (i < static_cast<int>(rlgymToRLBotPad.size())) ? rlgymToRLBotPad[i] : -1;
		if (padStates && j >= 0 && j < static_cast<int>(padStates->size())) {
			active = padStates->Get(j)->is_active();
			timer = padStates->Get(j)->timer();
		}

		gs.boostPads[i] = active;
		gs.boostPadTimers[i] = timer;

		const int inv = CommonValues::BOOST_LOCATIONS_AMOUNT - i - 1;
		gs.boostPadsInv[inv] = active;
		gs.boostPadTimersInv[inv] = timer;
	}

	const auto* players = packet->players();
	const float now = packet->match_info() ? packet->match_info()->seconds_elapsed() : 0.f;

	if (players) {
		gs.players.reserve(players->size());

		for (unsigned i = 0; i < players->size(); i++) {
			const auto* info = players->Get(i);
			Player p = {};
			const auto* phys = info->physics();
			p.pos = ToVec(phys->location());
			p.vel = ToVec(phys->velocity());
			p.angVel = ToVec(phys->angular_velocity());
			p.rotMat = Angle(phys->rotation().yaw(),
			                 phys->rotation().pitch(),
			                 phys->rotation().roll()).ToRotMat();

			p.index = static_cast<int>(i);
			p.carId = static_cast<uint32_t>(info->player_id());
			p.team = static_cast<Team>(info->team());
			p.boost = info->boost();
			p.isSupersonic = info->is_supersonic();

			const auto airState = info->air_state();
			p.isOnGround = (airState == rlbot::flat::AirState::OnGround);
			p.isJumping = (airState == rlbot::flat::AirState::Jumping);
			p.isFlipping = (airState == rlbot::flat::AirState::Dodging);

			p.isDemoed = info->demolished_timeout() >= 0.f;
			p.demoRespawnTimer = RS_MAX(0.f, info->demolished_timeout());

			p.hasJumped = info->has_jumped();
			if (p.isOnGround) {
				p.hasDoubleJumped = false;
				p.hasFlipped = false;
				p.airTimeSinceJump = 0.f;
			} else if (info->dodge_timeout() >= 0.f) {
				p.hasDoubleJumped = false;
				p.hasFlipped = false;
				p.airTimeSinceJump = 0.f;
			} else {
				p.hasDoubleJumped = info->has_double_jumped();
				p.hasFlipped = true;
				p.airTimeSinceJump = 999.f;
			}

			if (const auto* last = info->last_input()) {
				p.prevAction = Action(
					last->throttle(), last->steer(),
					last->pitch(), last->yaw(), last->roll(),
					last->jump() ? 1.f : 0.f,
					last->boost() ? 1.f : 0.f,
					last->handbrake() ? 1.f : 0.f);
			} else {
				p.prevAction = Action(0, 0, 0, 0, 0, 0, 0, 0);
			}

			p.ballTouchedStep = false;
			if (const auto* touch = info->latest_touch()) {
				const float t = touch->game_seconds();
				auto it = lastTouchTimes.find(info->player_id());
				if (it == lastTouchTimes.end()) {
					lastTouchTimes[info->player_id()] = t;
				} else if (t > it->second) {
					it->second = t;
					p.ballTouchedStep = true;
					gs.lastTouchCarID = static_cast<int>(p.carId);
				}
			}
			p.ballTouchedTick = p.ballTouchedStep;

			gs.players.push_back(p);
		}
	}

	if (packet->match_info()) {
		gs.goalScored = (packet->match_info()->match_phase() == rlbot::flat::MatchPhase::GoalScored);
	}

	return gs;
}

}  // namespace Dash
