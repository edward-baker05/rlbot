#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/BallPredictor.h>
#include <rlbot/PacketConvert.h>

#include <RLGymCPP/CommonValues.h>

#include <memory>

using namespace Dash;

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

std::unique_ptr<rlbot::flat::PlayerInfoT> MakePlayer(float x, float y, float z,
                                                     uint32_t team, int32_t id) {
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

rlbot::flat::GamePacketT MakeBasePacket(float secondsElapsed = 10.f) {
	rlbot::flat::GamePacketT pkt;
	pkt.match_info = std::make_unique<rlbot::flat::MatchInfoT>();
	pkt.match_info->seconds_elapsed = secondsElapsed;
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

TEST_CASE("Convert derives a physics tick count from seconds_elapsed") {
	PacketConverter conv;
	conv.Initialize(nullptr); // identity pad mapping

	PacketFixture fx;
	auto pkt = MakeBasePacket(10.f);
	fx.Build(pkt);

	CHECK(conv.Convert(fx.packet).lastTickCount == 1200);

	// One env step at tickSkip 8 is 8 ticks later.
	auto pkt2 = MakeBasePacket(10.f + 8.f / 120.f);
	fx.Build(pkt2);
	CHECK(conv.Convert(fx.packet).lastTickCount == 1208);
}

TEST_CASE("Convert clamps a zero or negative match clock to tick zero") {
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;

	// lastTickCount is unsigned; a negative clock must not wrap to ~1.8e19.
	auto pkt = MakeBasePacket(-1.f);
	fx.Build(pkt);
	CHECK(conv.Convert(fx.packet).lastTickCount == 0);

	auto zero = MakeBasePacket(0.f);
	fx.Build(zero);
	CHECK(conv.Convert(fx.packet).lastTickCount == 0);
}

// The reason the tick count exists at all: without it every frame looks like
// tick 0 and the predictor re-simulates on every single inference.
TEST_CASE("the deploy clock lets BallPredictor reuse its trajectory") {
	Dash::Test::EnsureRocketSim();
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;
	BallPredictor pred;

	auto pkt = MakeBasePacket(10.f);
	fx.Build(pkt);
	RLGC::GameState first = conv.Convert(fx.packet);
	const BallTrajectory& traj = pred.Get(first);
	REQUIRE(pred.SimulationCount() == 1);

	// The next inference frame, with the ball exactly where the prediction put
	// it -- which is what an untouched ball does.
	auto pkt2 = MakeBasePacket(10.f + 8.f / 120.f);
	pkt2.balls[0]->physics = std::make_unique<rlbot::flat::Physics>(
		rlbot::flat::Vector3(traj.pos[8].x, traj.pos[8].y, traj.pos[8].z),
		rlbot::flat::Rotator(0.f, 0.f, 0.f),
		rlbot::flat::Vector3(traj.vel[8].x, traj.vel[8].y, traj.vel[8].z),
		rlbot::flat::Vector3(0.f, 0.f, 0.f));
	fx.Build(pkt2);

	pred.Get(conv.Convert(fx.packet));
	CHECK(pred.SimulationCount() == 1); // reused, not re-simulated
}

TEST_CASE("a match restart rewinds the clock and invalidates the cache") {
	Dash::Test::EnsureRocketSim();
	PacketConverter conv;
	conv.Initialize(nullptr);
	PacketFixture fx;
	BallPredictor pred;

	auto late = MakeBasePacket(300.f);
	fx.Build(late);
	pred.Get(conv.Convert(fx.packet));
	REQUIRE(pred.SimulationCount() == 1);

	// New match: seconds_elapsed restarts near zero.
	auto fresh = MakeBasePacket(0.5f);
	fx.Build(fresh);
	pred.Get(conv.Convert(fx.packet));
	CHECK(pred.SimulationCount() == 2);
}
