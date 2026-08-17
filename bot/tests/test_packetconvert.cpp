#include "doctest/doctest.h"
#include "TestCommon.h"

#include <rlbot/PacketConvert.h>

#include <RLGymCPP/CommonValues.h>

#include <memory>

using namespace Hive;

namespace {

// Build a serialized GamePacket via the flatbuffers object API, then view it
// through the same accessor types the live RLBot connection delivers.
struct PacketFixture {
	flatbuffers::FlatBufferBuilder fbb;
	const rlbot::flat::GamePacket* packet = nullptr;

	void Build(rlbot::flat::GamePacketT& pkt) {
		fbb.Clear();
		fbb.Finish(rlbot::flat::GamePacket::Pack(fbb, &pkt));
		packet = flatbuffers::GetRoot<rlbot::flat::GamePacket>(fbb.GetBufferPointer());
	}
};

std::unique_ptr<rlbot::flat::PlayerInfoT> MakePlayer(float x, float y, float z, uint32_t team, int32_t id) {
	auto p = std::make_unique<rlbot::flat::PlayerInfoT>();
	p->physics = std::make_unique<rlbot::flat::Physics>(
		rlbot::flat::Vector3(x, y, z),
		rlbot::flat::Rotator(0.f, 0.f, 0.f),
		rlbot::flat::Vector3(100.f, 0.f, 0.f),
		rlbot::flat::Vector3(0.f, 0.f, 0.f));
	p->team = team;
	p->player_id = id;
	p->boost = 33.f;
	p->air_state = rlbot::flat::AirState::OnGround;
	p->demolished_timeout = -1.f;
	p->dodge_timeout = -1.f;
	return p;
}

rlbot::flat::GamePacketT MakeBasePacket() {
	rlbot::flat::GamePacketT pkt;
	pkt.match_info = std::make_unique<rlbot::flat::MatchInfoT>();
	pkt.match_info->seconds_elapsed = 10.f;
	pkt.match_info->match_phase = rlbot::flat::MatchPhase::Active;

	auto ball = std::make_unique<rlbot::flat::BallInfoT>();
	ball->physics = std::make_unique<rlbot::flat::Physics>(
		rlbot::flat::Vector3(500.f, -1000.f, 93.f),
		rlbot::flat::Rotator(0.f, 0.f, 0.f),
		rlbot::flat::Vector3(-250.f, 400.f, 10.f),
		rlbot::flat::Vector3(0.f, 0.f, 0.f));
	pkt.balls.push_back(std::move(ball));

	pkt.players.push_back(MakePlayer(-2000.f, -3000.f, 17.f, 0, 7));
	pkt.players.push_back(MakePlayer(2000.f, 3000.f, 17.f, 1, 8));

	for (int i = 0; i < RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT; i++)
		pkt.boost_pads.push_back(rlbot::flat::BoostPadState(true, 0.f));
	return pkt;
}

} // namespace

TEST_CASE("Convert carries ball and player physics through") {
	PacketConverter conv;
	conv.Initialize(nullptr); // identity pad mapping

	PacketFixture fx;
	auto pkt = MakeBasePacket();
	fx.Build(pkt);

	RLGC::GameState gs = conv.Convert(fx.packet);

	CHECK(gs.ball.pos.x == doctest::Approx(500.f));
	CHECK(gs.ball.vel.y == doctest::Approx(400.f));
	REQUIRE(gs.players.size() == 2);
	CHECK(gs.players[0].pos.y == doctest::Approx(-3000.f));
	CHECK(gs.players[0].team == Team::BLUE);
	CHECK(gs.players[1].team == Team::ORANGE);
	CHECK(gs.players[0].boost == doctest::Approx(33.f));
	CHECK(gs.players[0].isOnGround);
	CHECK(!gs.players[0].isDemoed);
}

TEST_CASE("touch timestamps become per-step touch flags exactly once") {
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;

	auto pkt = MakeBasePacket();
	auto touch = std::make_unique<rlbot::flat::TouchT>();
	touch->game_seconds = 9.5f;
	pkt.players[0]->latest_touch = std::move(touch);
	fx.Build(pkt);
	RLGC::GameState first = conv.Convert(fx.packet);
	// First sighting of a touch time only seeds the baseline.
	CHECK(!first.players[0].ballTouchedStep);

	auto pkt2 = MakeBasePacket();
	auto touch2 = std::make_unique<rlbot::flat::TouchT>();
	touch2->game_seconds = 10.5f;
	pkt2.players[0]->latest_touch = std::move(touch2);
	fx.Build(pkt2);
	RLGC::GameState second = conv.Convert(fx.packet);
	CHECK(second.players[0].ballTouchedStep);

	fx.Build(pkt2);
	RLGC::GameState third = conv.Convert(fx.packet);
	// Same timestamp again: not a new touch.
	CHECK(!third.players[0].ballTouchedStep);
}

TEST_CASE("an inactive pad reads back inactive through the identity mapping") {
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;

	auto pkt = MakeBasePacket();
	pkt.boost_pads[5] = rlbot::flat::BoostPadState(false, 3.5f);
	fx.Build(pkt);
	RLGC::GameState gs = conv.Convert(fx.packet);

	CHECK(!gs.boostPads[5]);
	CHECK(gs.boostPadTimers[5] == doctest::Approx(3.5f));
	const int inv = RLGC::CommonValues::BOOST_LOCATIONS_AMOUNT - 5 - 1;
	CHECK(!gs.boostPadsInv[inv]);
}
