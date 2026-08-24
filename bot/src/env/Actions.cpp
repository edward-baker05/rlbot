#include "Actions.h"

namespace Dash {

std::unique_ptr<RLGC::DefaultAction> MakeActionParser(bool masked) {
	if (masked)
		return std::make_unique<RLGC::DefaultAction>();

	return std::make_unique<UnmaskedAction>();
}

} // namespace Dash
