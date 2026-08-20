#include "Metrics.h"

namespace Hive {

std::vector<float> NormalizeShares(const std::vector<float>& totals) {
	float sum = 0.f;
	for (float t : totals)
		sum += t;
	if (sum <= 0.f)
		return {};
	std::vector<float> out;
	out.reserve(totals.size());
	for (float t : totals)
		out.push_back(t / sum);
	return out;
}

}  // namespace Hive
