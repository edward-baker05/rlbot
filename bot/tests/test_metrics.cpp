#include "doctest/doctest.h"
#include "TestCommon.h"

#include <env/Curriculum.h>
#include <env/StateSetters.h>
#include <train/Metrics.h>

#include <string>

using namespace Hive;

TEST_CASE("NormalizeShares divides by the total") {
	auto shares = NormalizeShares({3.f, 1.f});
	REQUIRE(shares.size() == 2);
	CHECK(shares[0] == doctest::Approx(0.75f));
	CHECK(shares[1] == doctest::Approx(0.25f));
}

TEST_CASE("NormalizeShares of all zeros is empty") {
	CHECK(NormalizeShares({0.f, 0.f, 0.f}).empty());
	CHECK(NormalizeShares({}).empty());
}

TEST_CASE("CurriculumState picks children and records their names") {
	Hive::Test::EnsureRocketSim();
	Arena* arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);

	CurriculumState cs({
		{new NeutralPlayState(), 1.f, "NeutralPlay"},
		{new AerialState(), 1.f, "Aerial"},
		{new BallContactState(), 0.f, "NeverPicked"}, // dropped: zero weight
	});

	CHECK(cs.LastPickedName().empty());

	bool sawNeutral = false, sawAerial = false;
	for (int i = 0; i < 200; i++) {
		cs.ResetArena(arena);
		const std::string& name = cs.LastPickedName();
		CHECK(name != "NeverPicked");
		if (name == "NeutralPlay") sawNeutral = true;
		if (name == "Aerial") sawAerial = true;
	}
	CHECK(sawNeutral);
	CHECK(sawAerial);
	delete arena;
}

TEST_CASE("CurriculumState with no positive weights throws") {
	CHECK_THROWS(CurriculumState({{new NeutralPlayState(), 0.f, "x"}}));
}
