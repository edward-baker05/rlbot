#pragma once

#include <RLGymCPP/StateSetters/StateSetter.h>

#include <string>
#include <vector>

namespace Hive {

struct CurriculumEntry {
	RLGC::StateSetter* setter;
	float weight;
	std::string name;
};

// Drop-in replacement for RLGC::CombinedState that remembers which child it
// last picked, so episode outcomes can be attributed to the scenario that
// spawned them. One instance per arena (CreateEnv makes one per env), so the
// last-picked name needs no locking.
class CurriculumState : public RLGC::StateSetter {
public:
	// Drops weight<=0 entries, deleting their setters; throws if none remain.
	explicit CurriculumState(std::vector<CurriculumEntry> entries);

	void ResetArena(Arena* arena) override;

	// Empty until the first reset.
	const std::string& LastPickedName() const { return lastPicked; }

	// Names of the entries that survived the zero-weight filter. Metrics
	// report a zero sample for scenarios not picked this step; without that,
	// rare scenarios' Share averages are biased upward.
	std::vector<std::string> EntryNames() const {
		std::vector<std::string> names;
		names.reserve(entries.size());
		for (const auto& e : entries)
			names.push_back(e.name);
		return names;
	}

private:
	std::vector<CurriculumEntry> entries;
	std::vector<float> cumulative;
	float total = 0.f;
	std::string lastPicked;
};

} // namespace Hive
