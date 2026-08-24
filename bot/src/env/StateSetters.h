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

}  // namespace Dash
