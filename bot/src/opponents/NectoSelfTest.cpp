#include "NectoSelfTest.h"

#include "NectoPolicy.h"

#include <RLGymCPP/CommonValues.h>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace RLGC;

namespace Dash {

namespace {

// A hand-built state, not a simulated one: the point is that both sides see
// byte-identical inputs, so the numbers only have to be distinct and awkward,
// not physically reachable. Deliberately asymmetric in x, y and z so a
// transposed column or a missed sign flip cannot pass.
GameState MakeProbeState() {
	GameState gs = {};

	gs.ball.pos = Vec(-431.5f, 1287.25f, 512.75f);
	gs.ball.vel = Vec(920.5f, -1455.f, 233.25f);
	gs.ball.angVel = Vec(0.75f, -2.125f, 3.5f);

	gs.players.resize(2);

	{ // Blue
		Player &p = gs.players[0];
		p.index = 0;
		p.carId = 1;
		p.team = Team::BLUE;
		p.pos = Vec(-1024.f, -2048.5f, 17.25f);
		p.vel = Vec(455.f, 1200.25f, -30.5f);
		p.angVel = Vec(-1.25f, 0.5f, 2.75f);
		p.rotMat = RotMat(Vec(0.6f, -0.8f, 0.f), Vec(0.8f, 0.6f, 0.f),
						  Vec(0.f, 0.f, 1.f));
		p.boost = 47.f; // 0-100 in the sim, 0-1 in the obs
		p.isOnGround = true;
		p.hasJumped = false;
		p.hasDoubleJumped = false;
		p.hasFlipped = false;
		p.demoRespawnTimer = 0.f;
	}

	{ // Orange -- the mirrored-field path, and airborne so the flags differ
		Player &p = gs.players[1];
		p.index = 1;
		p.carId = 2;
		p.team = Team::ORANGE;
		p.pos = Vec(2200.75f, 3100.f, 845.5f);
		p.vel = Vec(-1310.f, -220.5f, 640.f);
		p.angVel = Vec(3.25f, 1.75f, -0.5f);
		p.rotMat = RotMat(Vec(0.f, 0.28f, 0.96f), Vec(1.f, 0.f, 0.f),
						  Vec(0.f, 0.96f, -0.28f));
		p.boost = 88.f;
		p.isOnGround = false;
		p.hasJumped = true;
		p.hasDoubleJumped = false;
		p.hasFlipped = false;
		p.airTimeSinceJump = 0.05f; // still inside the double-jump window
		// Column 21 for a car: the timer is what Necto reads, the flag is what
		// Nexto reads, so set both rather than leaving either family's column
		// uniformly zero and untested.
		p.demoRespawnTimer = 1.75f;
		p.isDemoed = true;
	}

	gs.boostPads.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, true);
	gs.boostPadTimers.assign(CommonValues::BOOST_LOCATIONS_AMOUNT, 0.f);

	// A few pads on cooldown, big and small, so column 21 is not all zeros.
	gs.boostPads[3] = false;
	gs.boostPadTimers[3] = 7.5f; // big pad, 10s cooldown
	gs.boostPads[7] = false;
	gs.boostPadTimers[7] = 1.25f; // small pad, 4s cooldown
	gs.boostPads[30] = false;
	gs.boostPadTimers[30] = 9.f;

	return gs;
}

nlohmann::json VecToJson(const Vec &v) {
	return nlohmann::json::array({v.x, v.y, v.z});
}

nlohmann::json PlayerToJson(const Player &p) {
	nlohmann::json j;
	j["car_id"] = p.carId;
	j["team"] = p.team == Team::BLUE ? 0 : 1;
	j["pos"] = VecToJson(p.pos);
	j["vel"] = VecToJson(p.vel);
	j["ang_vel"] = VecToJson(p.angVel);
	j["forward"] = VecToJson(p.rotMat.forward);
	j["up"] = VecToJson(p.rotMat.up);
	j["boost_100"] = p.boost;
	j["on_ground"] = p.isOnGround;
	j["has_flip"] = p.HasFlipOrJump();
	j["demo_respawn_timer"] = p.demoRespawnTimer;
	j["is_demoed"] = p.isDemoed;
	return j;
}

} // namespace

int RunNectoSelfTest(NectoFamily family, const std::string &outPath) {
	const GameState gs = MakeProbeState();
	const int n = NectoPolicy::TokenCount(gs);

	nlohmann::json out;
	out["family"] = NectoFamilyName(family);
	out["features"] = NectoPolicy::FEATURES;
	out["q_size"] = NectoPolicy::Q_SIZE;
	out["tokens"] = n;

	out["ball"] = {{"pos", VecToJson(gs.ball.pos)},
				   {"vel", VecToJson(gs.ball.vel)},
				   {"ang_vel", VecToJson(gs.ball.angVel)}};

	out["players"] = nlohmann::json::array();
	for (const Player &p : gs.players)
		out["players"].push_back(PlayerToJson(p));

	out["boost_pads"] = nlohmann::json::array();
	for (int i = 0; i < CommonValues::BOOST_LOCATIONS_AMOUNT; i++) {
		const Vec &loc = CommonValues::BOOST_LOCATIONS[i];
		out["boost_pads"].push_back({{"loc", VecToJson(loc)},
									 {"active", gs.boostPads[i]},
									 {"cooldown", gs.boostPadTimers[i]},
									 {"is_big", loc.z > 72.f}});
	}

	// The previous action matters: it is appended raw to q, unnormalized, so a
	// wrong offset there is invisible in kv and fatal in play.
	Action prevAction = {};
	prevAction.throttle = 1.f;
	prevAction.steer = -1.f;
	prevAction.pitch = 0.5f;
	prevAction.yaw = -0.25f;
	prevAction.roll = 1.f;
	prevAction.jump = 1.f;
	prevAction.boost = 0.f;
	prevAction.handbrake = 1.f;

	out["prev_action"] = nlohmann::json::array();
	for (int k = 0; k < static_cast<int>(Action::ELEM_AMOUNT); k++)
		out["prev_action"].push_back(prevAction[k]);

	// Nexto's head emits an index into this, so the table is as much a part of
	// the port as the observation is, and just as silent when wrong.
	if (family == NectoFamily::Nexto) {
		out["lookup_table"] = nlohmann::json::array();
		for (const Action &a : NectoPolicy::NextoLookupTable()) {
			nlohmann::json row = nlohmann::json::array();
			for (int k = 0; k < static_cast<int>(Action::ELEM_AMOUNT); k++)
				row.push_back(a[k]);
			out["lookup_table"].push_back(row);
		}
	}

	out["obs"] = nlohmann::json::array();
	for (int playerIdx = 0; playerIdx < static_cast<int>(gs.players.size());
		 playerIdx++) {
		std::vector<float> q(NectoPolicy::Q_SIZE, 0.f);
		std::vector<float> kv(static_cast<size_t>(n) * NectoPolicy::FEATURES,
							  0.f);
		NectoPolicy::BuildObs(family, gs, playerIdx, prevAction, q.data(),
							  kv.data());

		nlohmann::json kvJson = nlohmann::json::array();
		for (int i = 0; i < n; i++) {
			nlohmann::json rowJson = nlohmann::json::array();
			for (int c = 0; c < NectoPolicy::FEATURES; c++)
				rowJson.push_back(kv[static_cast<size_t>(i) *
										 NectoPolicy::FEATURES +
									 c]);
			kvJson.push_back(rowJson);
		}

		out["obs"].push_back({{"player_idx", playerIdx}, {"q", q}, {"kv", kvJson}});
	}

	const std::string text = out.dump(1);
	if (outPath.empty()) {
		std::printf("%s\n", text.c_str());
	} else {
		std::ofstream f(outPath);
		if (!f) {
			std::fprintf(stderr, "Could not open %s for writing\n",
						 outPath.c_str());
			return 1;
		}
		f << text << "\n";
		std::printf("Wrote %s (%s, %d tokens, %zu players)\n", outPath.c_str(),
					NectoFamilyName(family), n, gs.players.size());
	}
	return 0;
}

} // namespace Dash
