#pragma once

#include <vector>

namespace Hive {

// Convert per-term sum(|weighted reward|) into fractions of the total.
// Returns an empty vector when the total is zero.
std::vector<float> NormalizeShares(const std::vector<float>& weightedAbsTotals);

} // namespace Hive
