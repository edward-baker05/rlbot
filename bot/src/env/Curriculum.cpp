#include "Curriculum.h"

#include <stdexcept>

namespace Hive {

CurriculumState::CurriculumState(std::vector<CurriculumEntry> in) {
	for (auto& e : in) {
		if (e.weight > 0.f) {
			total += e.weight;
			cumulative.push_back(total);
			entries.push_back(std::move(e));
		} else {
			delete e.setter;
		}
	}
	if (entries.empty())
		throw std::runtime_error("CurriculumState: every entry has zero weight");
}

void CurriculumState::ResetArena(Arena* arena) {
	// Child setters are process-lifetime, matching upstream ownership.
	const float f = RocketSim::Math::RandFloat(0, total);
	for (size_t i = 0; i < entries.size(); i++) {
		if (f <= cumulative[i]) {
			lastPicked = entries[i].name;
			entries[i].setter->ResetArena(arena);
			return;
		}
	}
	lastPicked = entries.back().name;
	entries.back().setter->ResetArena(arena);
}

} // namespace Hive
