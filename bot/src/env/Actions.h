#pragma once

#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include <memory>

namespace Hive {

// RLGymCPP's DefaultAction with the situational mask removed.
//
// WHY THIS EXISTS. `DefaultAction::GetActionMask` restricts a grounded car to
// 42 of the 90 actions, of which 18 press jump -- a 42.9% jump prior. Python
// RLGym's `LookupAction`, which every reference bot and Zealan's guide use,
// applies no mask at all, so its grounded jump prior is 18/90 = 20%.
//
// That factor of 2.1 is not cosmetic. Ground dwell is 1/p_jump decision steps
// and an air stint is ~15, so a UNIFORM policy is airborne 87% masked against
// 75% unmasked -- and p1advnorm measured `Player/In Air Ratio` 0.886 at a jump
// rate of 0.43, matching the masked prediction. This project has spent eight
// runs fighting air time that its own action mask doubles the prior of.
//
// The mask is not a bug. It stops the policy spending capacity on actions that
// do nothing (air-roll while grounded), which is real. It is simply an
// undocumented divergence from every implementation this project compares
// itself against, so the reproduction removes it and phase B measures whether
// putting it back helps.
class UnmaskedAction : public RLGC::DefaultAction {
public:
	std::vector<uint8_t> GetActionMask(const RLGC::Player& player,
	                                   const RLGC::GameState& state) override {
		return std::vector<uint8_t>(GetActionAmount(), true);
	}
};

// The ONLY place an action parser is constructed. Training, deployment,
// `verify`, `eval` and `spectate` all route through here, because a parser
// mismatch between training and deployment does not crash -- the bot loads,
// plays, and is quietly worse. Same reasoning as ModelShape.
std::unique_ptr<RLGC::DefaultAction> MakeActionParser(bool masked);

} // namespace Hive
