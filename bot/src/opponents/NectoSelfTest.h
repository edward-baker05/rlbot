#pragma once

#include "NectoPolicy.h"

#include <string>

namespace Dash {

// Dumps a fixed synthetic game state and the observation one family builds from
// it, so scripts/necto_obs_check.py (or nexto_obs_check.py) can feed the
// identical state to the real Python NectoObsBuilder / NextoObsBuilder and diff
// the two.
//
// This exists because a silent observation mismatch is the failure mode that
// costs the most: everything still runs, the opponent just plays badly, and the
// whole exercise looks fine while being worthless. It matters more for Nexto
// than it did for Necto -- Nexto's relative pass rotates the whole frame, so a
// mistake there is a plausible-looking observation of the wrong world.
int RunNectoSelfTest(NectoFamily family, const std::string &outPath);

} // namespace Dash
