#include "Actions.h"

namespace Dash {

std::unique_ptr<RLGC::DefaultAction>
MakeActionParser(bool masked, const NectoArenaState *nectoArena) {
	return std::make_unique<DashAction>(masked, nectoArena);
}

} // namespace Dash
