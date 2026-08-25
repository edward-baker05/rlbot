#include "RewardProbe.h"

#include <RLGymCPP/Rewards/ZeroSumReward.h>

#include <cmath>
#include <cstdio>

using namespace RLGC;

namespace Dash {

namespace {

constexpr int NAME_WIDTH = 16;
constexpr int CELL_WIDTH = 20;

// "+0.41200 (+0.620)": the weighted contribution that actually reaches the
// learner, and in parentheses the reward's own output before its weight. A
// reward can look harmless in the weighted column purely because its weight is
// small, so both numbers are needed to judge whether it is behaving. Five
// decimals because the weights span four orders of magnitude, from goal at 1.0
// down to possession at 1e-4.
std::string FormatCell(float weighted, float raw) {
	if (weighted == 0.f && raw == 0.f)
		return ".";

	char buf[64];
	std::snprintf(buf, sizeof(buf), "%+.5f (%+.3f)", weighted, raw);
	return buf;
}

void AppendPadded(std::string &out, const std::string &text, int width) {
	out += text;
	for (int i = (int)text.size(); i < width; i++)
		out += ' ';
}

// Cells are space-padded, so the last one on every row would otherwise trail
// whitespace into the terminal.
void EndLine(std::string &out) {
	while (!out.empty() && out.back() == ' ')
		out.pop_back();
	out += '\n';
}

} // namespace

RewardProbe::RewardProbe(const TrainConfig &cfg, float pauseThreshold,
						 int historySteps)
	: specs(GeneralRewardSpecs(cfg)), pauseThreshold(pauseThreshold),
	  historyLimit(historySteps > 0 ? (size_t)historySteps : 0),
	  secondsPerStep((float)cfg.tickSkip / 120.f) {

	rewards.reserve(specs.size());
	for (const RewardSpec &spec : specs)
		rewards.push_back(spec.make());

	isZeroSum.reserve(rewards.size());
	for (Reward *reward : rewards)
		isZeroSum.push_back(dynamic_cast<ZeroSumReward *>(reward) != nullptr);
}

RewardProbe::~RewardProbe() {
	for (Reward *reward : rewards)
		delete reward;
}

void RewardProbe::BeginEpisode(const GameState &initialState) {
	for (Reward *reward : rewards)
		reward->Reset(initialState);

	numPlayers = (int)initialState.players.size();

	// Name each column by team, disambiguated by index only when a team has
	// more than one car. Spectate is 1v1 today, but nothing here assumes it.
	int teamCounts[2] = {0, 0};
	for (const Player &player : initialState.players)
		teamCounts[(int)player.team]++;

	int teamSeen[2] = {0, 0};
	playerLabels.clear();
	for (const Player &player : initialState.players) {
		const int team = (int)player.team;
		const char *base = player.team == Team::BLUE ? "BLUE" : "ORANGE";
		if (teamCounts[team] > 1) {
			playerLabels.push_back(std::string(base) + " " +
								   std::to_string(++teamSeen[team]));
		} else {
			playerLabels.push_back(base);
		}
	}

	stepCount = 0;
	episodeTotalsRaw.assign(rewards.size() * numPlayers, 0.f);
	episodeTotalsWeighted.assign(rewards.size() * numPlayers, 0.f);
	history.clear();
}

void RewardProbe::Step(const GameState &state, bool isFinal) {
	// EnvSet runs every reward's PreStep before any GetAllRewards, and some
	// rewards depend on that ordering: the aerial rewards latch launchZ in
	// PreStep and read it in GetReward. Two separate loops, not one fused one.
	for (Reward *reward : rewards)
		reward->PreStep(state);

	StepRecord record;
	record.step = ++stepCount;
	record.time = stepCount * secondsPerStep;
	record.raw.assign(rewards.size() * numPlayers, 0.f);
	record.weighted.assign(rewards.size() * numPlayers, 0.f);

	for (size_t rewardIdx = 0; rewardIdx < rewards.size(); rewardIdx++) {
		const std::vector<float> output =
			rewards[rewardIdx]->GetAllRewards(state, isFinal);

		for (int i = 0; i < numPlayers && i < (int)output.size(); i++) {
			const size_t flat = rewardIdx * numPlayers + i;
			const float raw = output[i];
			const float weighted = raw * specs[rewardIdx].weight;

			record.raw[flat] = raw;
			record.weighted[flat] = weighted;
			episodeTotalsRaw[flat] += raw;
			episodeTotalsWeighted[flat] += weighted;

			// Continuous rewards swing every step by design, so arming the
			// auto-pause on them would stop on nothing in particular.
			if (record.trippedIdx < 0 && pauseThreshold > 0.f &&
				specs[rewardIdx].kind == RewardKind::Event &&
				specs[rewardIdx].weight != 0.f &&
				std::fabs(weighted) > pauseThreshold) {
				record.trippedIdx = (int)rewardIdx;
			}
		}
	}

	if (historyLimit > 0) {
		history.push_back(std::move(record));
		while (history.size() > historyLimit)
			history.pop_front();
	} else {
		history.clear();
		history.push_back(std::move(record));
	}
}

bool RewardProbe::ShouldShowRow(int rewardIdx,
								const std::vector<float> &raw) const {
	if (specs[rewardIdx].weight == 0.f)
		return false;

	if (specs[rewardIdx].kind == RewardKind::Continuous)
		return true;

	for (int i = 0; i < numPlayers; i++)
		if (raw[rewardIdx * numPlayers + i] != 0.f)
			return true;

	return false;
}

std::string RewardProbe::FormatTable(const char *header,
									 const std::vector<float> &raw,
									 const std::vector<float> &weighted) const {
	std::string out = header;
	out += '\n';

	AppendPadded(out, "  ", NAME_WIDTH + 4);
	for (const std::string &label : playerLabels)
		AppendPadded(out, label, CELL_WIDTH);
	EndLine(out);

	int rowsShown = 0;
	for (int rewardIdx = 0; rewardIdx < (int)rewards.size(); rewardIdx++) {
		if (!ShouldShowRow(rewardIdx, raw))
			continue;

		rowsShown++;
		out += "  ";
		AppendPadded(out, specs[rewardIdx].name, NAME_WIDTH);
		AppendPadded(out, isZeroSum[rewardIdx] ? "Z " : "  ", 2);
		for (int i = 0; i < numPlayers; i++) {
			const size_t flat = (size_t)rewardIdx * numPlayers + i;
			AppendPadded(out, FormatCell(weighted[flat], raw[flat]),
						 CELL_WIDTH);
		}
		EndLine(out);
	}

	if (rowsShown == 0)
		out += "  (all zero)\n";

	// Totals are of the weighted column only: that sum is the scalar the
	// learner actually optimizes.
	AppendPadded(out, "  ", NAME_WIDTH + 4);
	for (int i = 0; i < numPlayers * CELL_WIDTH; i++)
		out += '-';
	out += '\n';

	out += "  ";
	AppendPadded(out, "total", NAME_WIDTH + 2);
	for (int i = 0; i < numPlayers; i++) {
		float sum = 0.f;
		for (int rewardIdx = 0; rewardIdx < (int)rewards.size(); rewardIdx++)
			sum += weighted[(size_t)rewardIdx * numPlayers + i];

		char buf[64];
		std::snprintf(buf, sizeof(buf), "%+.5f", sum);
		AppendPadded(out, buf, CELL_WIDTH);
	}
	EndLine(out);

	return out;
}

std::string RewardProbe::FormatLastStep(const char *header) const {
	if (history.empty())
		return {};

	const StepRecord &record = history.back();
	std::string out = FormatTable(header, record.raw, record.weighted);

	if (record.trippedIdx >= 0) {
		char buf[160];
		float peak = 0.f;
		for (int i = 0; i < numPlayers; i++) {
			const float v =
				record.weighted[(size_t)record.trippedIdx * numPlayers + i];
			if (std::fabs(v) > std::fabs(peak))
				peak = v;
		}
		std::snprintf(buf, sizeof(buf), "\n  tripped by: %s  (|%+.5f| > %.4f)\n",
					  specs[record.trippedIdx].name.c_str(), peak,
					  pauseThreshold);
		out += buf;
	}

	return out;
}

std::string RewardProbe::FormatHistory() const {
	if (history.empty())
		return {};

	char buf[128];
	std::snprintf(buf, sizeof(buf), "== last %d steps ==\n",
				  (int)history.size());
	std::string out = buf;

	out += "  ";
	AppendPadded(out, "step", 8);
	AppendPadded(out, "t", 9);
	for (const std::string &label : playerLabels)
		AppendPadded(out, label, 12);
	out += "events fired\n";

	for (const StepRecord &record : history) {
		out += "  ";
		std::snprintf(buf, sizeof(buf), "%d", record.step);
		AppendPadded(out, buf, 8);
		std::snprintf(buf, sizeof(buf), "%.2fs", record.time);
		AppendPadded(out, buf, 9);

		for (int i = 0; i < numPlayers; i++) {
			float sum = 0.f;
			for (int r = 0; r < (int)rewards.size(); r++)
				sum += record.weighted[(size_t)r * numPlayers + i];
			std::snprintf(buf, sizeof(buf), "%+.5f", sum);
			AppendPadded(out, buf, 12);
		}

		// Only Event rewards are listed: naming the Continuous ones on every
		// line would bury the timeline this view exists to show.
		std::string fired;
		for (int r = 0; r < (int)rewards.size(); r++) {
			if (specs[r].kind != RewardKind::Event || specs[r].weight == 0.f)
				continue;

			bool any = false;
			for (int i = 0; i < numPlayers; i++)
				if (record.raw[(size_t)r * numPlayers + i] != 0.f)
					any = true;

			if (any) {
				if (!fired.empty())
					fired += ", ";
				fired += specs[r].name;
			}
		}

		out += fired.empty() ? "-" : fired;
		if (record.trippedIdx >= 0)
			out += "   <-- paused here";
		out += '\n';
	}

	return out;
}

std::string RewardProbe::FormatEpisodeTotals() const {
	if (stepCount == 0)
		return {};

	char header[128];
	std::snprintf(header, sizeof(header), "== episode totals (%d steps) ==",
				  stepCount);
	return FormatTable(header, episodeTotalsRaw, episodeTotalsWeighted);
}

} // namespace Dash
