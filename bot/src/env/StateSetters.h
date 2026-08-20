#pragma once

#include <RLGymCPP/StateSetters/StateSetter.h>

#include <memory>

namespace Hive {

class InfiniteBoostState : public RLGC::StateSetter {
public:
	InfiniteBoostState(RLGC::StateSetter* inner, float chance)
		: inner(inner), chance(chance) {}

	void ResetArena(Arena* arena) override;

	bool LastWasInfinite() const { return lastWasInfinite; }

private:
	std::unique_ptr<RLGC::StateSetter> inner;
	float chance;
	bool lastWasInfinite = false;
};

class StrikeState : public RLGC::StateSetter {
public:
	float minBallZ = 250.f, maxBallZ = 550.f;
	float minDist = 700.f, maxDist = 1400.f;
	float minSpeed = 900.f, maxSpeed = 1600.f;
	float fullBoostChance = 0.4f;
	float minBoost = 20.f, maxBoost = 100.f;

	void ResetArena(Arena* arena) override;
};

class AerialState : public RLGC::StateSetter {
public:
	float minBallZ = 700.f, maxBallZ = 1700.f;
	float minCarDist = 1200.f, maxCarDist = 2600.f;
	float minBoost = 40.f, maxBoost = 100.f;

	void ResetArena(Arena* arena) override;
};

class AirDribbleState : public RLGC::StateSetter {
public:
	float minZ = 400.f, maxZ = 1400.f;

	void ResetArena(Arena* arena) override;
};

class FlipResetState : public RLGC::StateSetter {
public:
	float minBallZ = 1100.f, maxBallZ = 1750.f;
	float carBelowBall = 300.f;

	void ResetArena(Arena* arena) override;
};

class GroundDribbleState : public RLGC::StateSetter {
public:
	float minSpeed = 700.f, maxSpeed = 1600.f;

	void ResetArena(Arena* arena) override;
};

class DemoState : public RLGC::StateSetter {
public:
	float minSpeed = 1200.f, maxSpeed = 2200.f;
	float separation = 2500.f;

	void ResetArena(Arena* arena) override;
};

class DefendState : public RLGC::StateSetter {
public:
	float minBallSpeed = 800.f, maxBallSpeed = 2600.f;

	void ResetArena(Arena* arena) override;
};

class RecoverState : public RLGC::StateSetter {
public:
	float minZ = 300.f, maxZ = 1500.f;

	void ResetArena(Arena* arena) override;
};

class NeutralPlayState : public RLGC::StateSetter {
public:
	void ResetArena(Arena* arena) override;
};

class BallContactState : public RLGC::StateSetter {
public:
	float minDist = 250.f, maxDist = 700.f;
	float movingBallChance = 0.6f;

	void ResetArena(Arena* arena) override;
};

}  // namespace Hive
