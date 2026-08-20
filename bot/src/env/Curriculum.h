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

class CurriculumState : public RLGC::StateSetter {
public:
	explicit CurriculumState(std::vector<CurriculumEntry> entries);

	void ResetArena(Arena* arena) override;

	const std::string& LastPickedName() const { return lastPicked; }

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

}  // namespace Hive
