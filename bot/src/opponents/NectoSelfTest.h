#pragma once

#include <string>

namespace Dash {

// Dumps a fixed synthetic game state and the Necto observation built from it,
// so scripts/necto_obs_check.py can feed the identical state to the real Python
// NectoObsBuilder and diff the two.
//
// This exists because a silent observation mismatch is the failure mode that
// costs the most: everything still runs, Necto just plays badly, and the whole
// exercise looks fine while being worthless.
int RunNectoSelfTest(const std::string &outPath);

} // namespace Dash
