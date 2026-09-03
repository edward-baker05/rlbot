#pragma once

#include <RLGymCPP/StateSetters/StateSetter.h>

#include <memory>

namespace Dash {

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

class AerialHoverState : public RLGC::StateSetter {
public:
	explicit AerialHoverState(float initialBoost = 100.f)
		: initialBoost(initialBoost) {}

	void ResetArena(Arena* arena) override;

private:
	float initialBoost;
};

class HighBallPopUpState : public RLGC::StateSetter {
public:
	explicit HighBallPopUpState(float initialBoost = 100.f)
		: initialBoost(initialBoost) {}

	void ResetArena(Arena* arena) override;

private:
	float initialBoost;
};

class BackboardFollowState : public RLGC::StateSetter {
public:
	explicit BackboardFollowState(float initialBoost = 100.f)
		: initialBoost(initialBoost) {}

	void ResetArena(Arena* arena) override;

private:
	float initialBoost;
};

}  // namespace Dash
