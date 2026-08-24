#include "Train.h"

#include "../env/Env.h"
#include "../env/Obs.h"
#include "../env/Rewards.h"
#include "../eval/Checkpoints.h"
#include "../eval/NectoBench.h"
#include "../opponents/NectoArena.h"
#include "../opponents/NectoDriver.h"

#include <GigaLearnCPP/Learner.h>
#include <RLGymCPP/EnvSet/EnvSet.h>

#include <nlohmann/json.hpp>

#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>

using namespace GGL;
using namespace RLGC;

namespace Dash {

static int64_t g_MaxSteps = 0;

static volatile std::sig_atomic_t g_StopRequested = 0;
static void HandleSigint(int) { g_StopRequested = 1; }

static std::vector<std::pair<std::string, float>> g_RewardLabels;
static std::vector<int> g_EpisodeAge;
static bool g_NectoEnabled = false;

// The benchmark blocks collection while it runs, the same way the built-in
// skill tracker does. A few seconds every benchInterval iterations.
static std::unique_ptr<NectoBench> g_NectoBench;
static std::filesystem::path g_NectoBenchRunFolder;
static int64_t g_NectoBenchIntervalTs = 0;
static int64_t g_NectoBenchNextTs = 0;

static void SaveAndExit(Learner *learner, const char *reason) {
	std::cout << "\n" << reason << ". Saving and exiting.\n";
	std::cout.flush();
	learner->Save();
	if (learner->metricSender)
		learner->metricSender->Finish();
	std::_Exit(0);
}

// Per-iteration metrics for one arena that has Necto in it.
//
// Everything is AddAvg, so each key comes out as a mean over whatever it was
// recorded against -- per step for the rate signals, per episode for the rest.
static void RecordNectoArena(const GameState &gs, const NectoArenaState &arena,
							 bool terminal, int episodeAge, Report &report) {
	const Team learnerTeam =
		(arena.nectoTeam == Team::BLUE) ? Team::ORANGE : Team::BLUE;

	int touches = 0, learnerCars = 0;
	for (const Player &p : gs.players) {
		if (p.team != learnerTeam)
			continue;
		learnerCars++;
		touches += p.ballTouchedStep ? 1 : 0;
	}
	if (learnerCars > 0)
		report.AddAvg("Necto/Train/TouchRate",
					  static_cast<float>(touches) / learnerCars);

	// Blue defends -y. The direct read on "is it being pinned back".
	const bool ballInOwnHalf =
		(learnerTeam == Team::BLUE) ? (gs.ball.pos.y < 0) : (gs.ball.pos.y > 0);
	report.AddAvg("Necto/Train/BallInOwnHalfFrac", ballInOwnHalf ? 1.f : 0.f);

	if (!terminal)
		return;

	report.AddAvg("Necto/Train/EpisodeSteps", static_cast<float>(episodeAge));

	// Early on, most episodes end on the no-touch timeout rather than a goal.
	// Without this the win rate is a mean over a handful of samples and reads
	// as noise; with it, the two together are interpretable.
	report.AddAvg("Necto/Train/DecisiveRate", gs.goalScored ? 1.f : 0.f);

	if (!gs.goalScored) {
		report.AddAvg("Necto/Train/GoalsFor", 0.f);
		report.AddAvg("Necto/Train/GoalsAgainst", 0.f);
		report.AddAvg("Necto/Train/GoalDiffPerEp", 0.f);
		return;
	}

	// The ball is sitting in the conceding team's net. This is how the
	// framework itself attributes goals (CommonRewards.h,
	// PolicyVersionManager). PlayerEventState::goal is not used on purpose: it
	// credits the last toucher on the scoring team, so an own goal sets no
	// player's flag at all.
	const Team conceding = RS_TEAM_FROM_Y(gs.ball.pos.y);
	const bool learnerScored = (conceding != learnerTeam);

	report.AddAvg("Necto/Train/WinRate", learnerScored ? 1.f : 0.f);
	report.AddAvg("Necto/Train/GoalsFor", learnerScored ? 1.f : 0.f);
	report.AddAvg("Necto/Train/GoalsAgainst", learnerScored ? 0.f : 1.f);
	report.AddAvg("Necto/Train/GoalDiffPerEp", learnerScored ? 1.f : -1.f);
}

static void StepCallback(Learner *learner, const std::vector<GameState> &states,
						 Report &report) {
	if (g_StopRequested) {
		SaveAndExit(learner, "Interrupted (Ctrl-C or wandb Stop)");
	}
	if (g_MaxSteps > 0 &&
		static_cast<int64_t>(learner->totalTimesteps) >= g_MaxSteps) {
		std::ostringstream reason;
		reason << "Reached step budget (" << learner->totalTimesteps
			   << " >= " << g_MaxSteps << ")";
		SaveAndExit(learner, reason.str().c_str());
	}

	auto &es = learner->envSet->state;
	if (g_EpisodeAge.size() != states.size())
		g_EpisodeAge.assign(states.size(), 0);
	for (size_t a = 0; a < states.size(); a++) {
		g_EpisodeAge[a]++;
		const bool terminal = a < es.terminals.size() && es.terminals[a];

		if (terminal)
			report.AddAvg("Episode/Mean Steps",
						  static_cast<float>(g_EpisodeAge[a]));

		// Necto arenas get their own accounting, folded in here so the episode
		// age is still readable -- it is zeroed at the bottom of this loop.
		if (g_NectoEnabled) {
			const auto *nectoArena = static_cast<const NectoArenaState *>(
				learner->envSet->userInfos[a]);
			if (nectoArena && nectoArena->active)
				RecordNectoArena(states[a], *nectoArena, terminal,
								 g_EpisodeAge[a], report);
		}

		if (terminal)
			g_EpisodeAge[a] = 0;
	}

	if (g_NectoBench && g_NectoBenchIntervalTs > 0 &&
		static_cast<int64_t>(learner->totalTimesteps) >= g_NectoBenchNextTs) {
		// Advance first: a benchmark that throws or finds no checkpoint must
		// not retry on every single step for the rest of the run.
		g_NectoBenchNextTs = static_cast<int64_t>(learner->totalTimesteps) +
							 g_NectoBenchIntervalTs;

		const std::filesystem::path checkpoint =
			FindLatestCheckpoint(g_NectoBenchRunFolder);
		if (!checkpoint.empty()) {
			const NectoBenchResult result = g_NectoBench->Run(checkpoint);
			ReportNectoBench(result, report);
			if (result.valid) {
				std::cout << "Necto benchmark @ " << learner->totalTimesteps
						  << ": " << result.goalsFor << "-"
						  << result.goalsAgainst << " over " << result.episodes
						  << " episodes, Elo " << static_cast<int>(result.elo)
						  << "\n";
			}
		}
	}

	auto &envSet = *learner->envSet;
	if (!g_RewardLabels.empty()) {
		std::vector<float> totals(g_RewardLabels.size(), 0.f);
		bool any = false;
		for (size_t a = 0; a < envSet.state.lastRewards.size(); a++) {
			const auto &last = envSet.state.lastRewards[a];
			if (last.size() != totals.size())
				continue;
			for (size_t j = 0; j < totals.size(); j++)
				totals[j] += std::fabs(last[j] * g_RewardLabels[j].second);
			any = true;
		}
		if (any) {
			for (size_t j = 0; j < totals.size(); j++)
				report.AddAvg("RewardMass/" + g_RewardLabels[j].first,
							  totals[j] / static_cast<float>(
											  envSet.state.lastRewards.size()));
		}
	}
}

static nlohmann::json ConfigToJson(const TrainConfig &cfg) {
	const RewardBudget &b = cfg.rewards;
	nlohmann::json j;

	j["rewards"] = {
		{"goal", b.goal},
		{"strongTouch", b.strongTouch},
		{"air", b.airTouch},
		{"pickupBoost", b.pickupBoost},
		{"saveBoost", b.saveBoost},
		{"bump", b.bump},
		{"demo", b.demo},
		{"save", b.save},
		{"awkwardContact", b.awkwardContact},
		{"possession", b.possession},
	};

	j["ppo"] = {
		{"tsPerItr", cfg.tsPerItr},
		{"miniBatchSize", cfg.miniBatchSize},
		{"epochs", cfg.epochs},
		{"entropyScale", cfg.entropyScale},
		{"entropyTarget", cfg.entropyTarget},
		{"entropyAdjustRate", cfg.entropyAdjustRate},
		{"gaeGamma", cfg.gaeGamma},
		{"policyLR", cfg.policyLR},
		{"criticLR", cfg.criticLR},
	};

	j["env"] = {
		{"maxPlayersPerTeam", cfg.maxPlayersPerTeam},
		{"maskActions", cfg.maskActions},
		{"obs", cfg.obs == ObsMode::Advanced ? "Advanced" : "Default"},
		{"infiniteBoostChance", cfg.infiniteBoostChance},
		{"teamSpirit", cfg.teamSpirit},
		{"noTouchTimeoutSeconds", cfg.noTouchTimeoutSeconds},
		{"numGames", cfg.numGames},
		{"tickSkip", cfg.tickSkip},
		{"actionDelay", cfg.actionDelay},
		{"teamDistribution",
		 {
			 {"p1v1", cfg.teamDistribution.p1v1},
			 {"p2v2", cfg.teamDistribution.p2v2},
			 {"p3v3", cfg.teamDistribution.p3v3},
		 }},
	};

	j["model"] = {
		{"sharedHeadLayers", cfg.modelShape.sharedHeadLayers},
		{"policyLayers", cfg.modelShape.policyLayers},
		{"addLayerNorm", cfg.modelShape.addLayerNorm},
	};

	j["necto"] = {
		{"enabled", cfg.necto.enabled},
		{"arenaFraction", cfg.necto.arenaFraction},
		{"trainBeta", cfg.necto.trainBeta},
		{"benchmark", cfg.necto.benchmark},
		{"benchInterval", cfg.necto.benchInterval},
		{"benchArenas", cfg.necto.benchArenas},
	};

	j["selfPlay"] = {
		{"trainAgainstOldVersions", cfg.selfPlay.trainAgainstOldVersions},
		{"trainAgainstOldChance", cfg.selfPlay.trainAgainstOldChance},
		{"tsPerVersion", cfg.selfPlay.tsPerVersion},
		{"maxOldVersions", cfg.selfPlay.maxOldVersions},
		{"trackSkill", cfg.selfPlay.trackSkill},
	};

	return j;
}

// Flattens to "section.field" so a diff names the exact knob that moved.
static void FlattenConfig(const nlohmann::json &j,
						  std::map<std::string, std::string> &out) {
	for (auto &section : j.items())
		for (auto &field : section.value().items())
			out[section.key() + "." + field.key()] = field.value().dump();
}

static int64_t NewestCheckpointSteps(const std::filesystem::path &folder) {
	int64_t newest = 0;
	std::error_code ec;
	for (const auto &e : std::filesystem::directory_iterator(folder, ec)) {
		if (!e.is_directory())
			continue;
		const std::string name = e.path().filename().string();
		if (name.empty() ||
			name.find_first_not_of("0123456789") != std::string::npos)
			continue;
		newest = RS_MAX(newest, std::stoll(name));
	}
	return newest;
}

static void RecordConfig(const TrainConfig &cfg) {
	const std::filesystem::path folder = cfg.CheckpointFolder();
	std::error_code ec;
	std::filesystem::create_directories(folder, ec);

	const nlohmann::json current = ConfigToJson(cfg);
	const std::filesystem::path historyPath = folder / "CONFIG_HISTORY.json";

	nlohmann::json history = nlohmann::json::array();
	if (std::filesystem::exists(historyPath)) {
		std::ifstream in(historyPath);
		try {
			in >> history;
		} catch (const std::exception &e) {
			std::cerr << "CONFIG_HISTORY.json unreadable (" << e.what()
					  << "); starting a new one.\n";
			history = nlohmann::json::array();
		}
	}

	std::map<std::string, std::string> now, before;
	FlattenConfig(current, now);
	if (!history.empty() && history.back().contains("config"))
		FlattenConfig(history.back()["config"], before);

	nlohmann::json changed = nlohmann::json::object();
	for (const auto &[k, v] : now) {
		auto it = before.find(k);
		if (it == before.end()) {
			if (!before.empty())
				changed[k] = {nullptr, v};
		} else if (it->second != v) {
			changed[k] = {it->second, v};
		}
	}

	{
		std::ofstream out(folder / "CONFIG.json");
		out << current.dump(2) << "\n";
	}

	if (!history.empty() && changed.empty())
		return;

	nlohmann::json entry;
	entry["total_timesteps_at_start"] = NewestCheckpointSteps(folder);
	entry["changed"] = changed;
	entry["config"] = current;
	history.push_back(entry);

	std::ofstream out(historyPath);
	out << history.dump(2) << "\n";

	if (!changed.empty()) {
		std::cout << "Config change recorded at "
				  << entry["total_timesteps_at_start"].get<int64_t>()
				  << " steps:\n";
		for (auto &c : changed.items())
			std::cout << "  " << c.key() << ": " << c.value()[0] << " -> "
					  << c.value()[1] << "\n";
	}
}

void RunTraining(const TrainConfig &cfg) {
	const char *meshEnv = std::getenv("DASH_COLLISION_MESHES");
	if (!meshEnv)
		meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	const std::string meshPath = meshEnv ? meshEnv : "collision_meshes";
	RocketSim::Init(meshPath);

	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam, cfg.obs);
	int n1 = 0, n2 = 0, n3 = 0;
	for (int i = 0; i < cfg.numGames; i++) {
		int sz = cfg.teamDistribution.SampleTeamSize(i, cfg.numGames);
		if (sz == 1)
			n1++;
		else if (sz == 2)
			n2++;
		else if (sz == 3)
			n3++;
	}
	std::cout << "Observation size: " << obsSize << " (mode="
			  << (cfg.obs == ObsMode::Advanced ? "Advanced" : "Default")
			  << ", maxPlayersPerTeam=" << cfg.maxPlayersPerTeam << ")\n";
	std::cout << "Arenas:           " << cfg.numGames << " total (" << n1
			  << "x 1v1, " << n2 << "x 2v2, " << n3 << "x 3v3)\n";
	std::cout << "Run:              " << cfg.RunName() << "\n";
	std::cout << "Checkpoints:      " << cfg.CheckpointFolder() << "\n";
	std::cout << "Self-play:        "
			  << (cfg.selfPlay.trainAgainstOldVersions
					  ? "on (" +
							std::to_string(static_cast<int>(
								cfg.selfPlay.trainAgainstOldChance * 100)) +
							"% of iterations, snapshot every " +
							std::to_string(cfg.selfPlay.tsPerVersion /
										   1'000'000) +
							"M steps)"
					  : "off")
			  << "\n";
	std::cout << "Skill tracking:   "
			  << (cfg.selfPlay.trackSkill ? "on" : "off") << "\n";
	if (cfg.necto.enabled) {
		int nectoArenas = 0;
		for (int i = 0; i < cfg.numGames; i++)
			if (NectoArenaAssignment(i, cfg.necto.arenaFraction, nullptr))
				nectoArenas++;
		// The share of ARENAS is not the share of data: only the learner's half
		// of a Necto arena produces trajectories.
		const float f =
			static_cast<float>(nectoArenas) / RS_MAX(cfg.numGames, 1);
		std::cout << "Necto opponent:   on (" << nectoArenas << " arenas, "
				  << static_cast<int>(100.f * f + 0.5f) << "% of arenas = "
				  << static_cast<int>(100.f * f / (2.f - f) + 0.5f)
				  << "% of training data, beta " << cfg.necto.trainBeta
				  << ")\n";
	}
	if (cfg.maxSteps > 0)
		std::cout << "Step budget:      " << cfg.maxSteps << "\n";

	RecordConfig(cfg);

	g_MaxSteps = cfg.maxSteps;

	g_RewardLabels.clear();
	for (auto &s : GeneralRewardSpecs(cfg))
		g_RewardLabels.push_back({s.name, s.weight});

	LearnerConfig lc = {};

	lc.deviceType =
		cfg.useGPU ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;
	lc.numGames = cfg.numGames;
	lc.tickSkip = cfg.tickSkip;
	lc.actionDelay = cfg.actionDelay;
	// lc.randomSeed = cfg.randomSeed;

	lc.checkpointFolder = cfg.CheckpointFolder();
	lc.tsPerSave = cfg.tsPerSave;
	lc.checkpointsToKeep = cfg.checkpointsToKeep;

	lc.ppo.tsPerItr = cfg.tsPerItr;
	lc.ppo.batchSize = cfg.tsPerItr;
	lc.ppo.miniBatchSize = cfg.miniBatchSize;
	lc.ppo.epochs = cfg.epochs;
	lc.ppo.entropyScale = cfg.entropyScale;

	lc.ppo.entropyTarget = cfg.entropyTarget;
	lc.ppo.entropyAdjustRate = cfg.entropyAdjustRate;
	lc.ppo.gaeGamma = cfg.gaeGamma;
	lc.ppo.policyLR = cfg.policyLR;
	lc.ppo.criticLR = cfg.criticLR;

	lc.ppo.sharedHead.layerSizes = cfg.modelShape.sharedHeadLayers;
	lc.ppo.sharedHead.activationType = cfg.modelShape.activation;
	lc.ppo.sharedHead.addLayerNorm = cfg.modelShape.addLayerNorm;
	lc.ppo.sharedHead.addOutputLayer = false;

	lc.ppo.policy.layerSizes = cfg.modelShape.policyLayers;
	lc.ppo.policy.activationType = cfg.modelShape.activation;
	lc.ppo.policy.addLayerNorm = cfg.modelShape.addLayerNorm;

	lc.ppo.critic.layerSizes = cfg.modelShape.policyLayers;
	lc.ppo.critic.activationType = cfg.modelShape.activation;
	lc.ppo.critic.addLayerNorm = cfg.modelShape.addLayerNorm;

	lc.trainAgainstOldVersions = cfg.selfPlay.trainAgainstOldVersions;
	lc.trainAgainstOldChance = cfg.selfPlay.trainAgainstOldChance;
	lc.savePolicyVersions =
		cfg.selfPlay.trainAgainstOldVersions || cfg.selfPlay.trackSkill;
	lc.tsPerVersion = cfg.selfPlay.tsPerVersion;
	lc.maxOldVersions = cfg.selfPlay.maxOldVersions;

	lc.skillTracker.enabled = cfg.selfPlay.trackSkill;
	lc.skillTracker.numArenas = cfg.selfPlay.skillArenas;
	lc.skillTracker.updateInterval = cfg.selfPlay.skillUpdateInterval;
	lc.skillTracker.simTime = cfg.selfPlay.skillSimTime;
	lc.skillTracker.maxSimTime = cfg.selfPlay.skillMaxSimTime;

	lc.sendMetrics = cfg.sendMetrics;
	lc.metricsProjectName = cfg.wandbProject;
	lc.metricsGroupName = cfg.wandbGroup;
	lc.metricsRunName = cfg.RunName();

	lc.renderMode = cfg.renderMode;
	lc.renderTimeScale = cfg.renderTimeScale;

	// Necto opponent. Built before the Learner because the Learner calls the
	// mask hook during construction of its EnvSet.
	std::unique_ptr<NectoDriver> nectoDriver;
	if (cfg.necto.enabled) {
		std::filesystem::path modelPath = cfg.necto.modelPath;
		if (const char *envModel = std::getenv("DASH_NECTO_MODEL"))
			modelPath = envModel;

		nectoDriver = std::make_unique<NectoDriver>(
			modelPath, cfg.necto.trainBeta, cfg.randomSeed);

		NectoDriver *driver = nectoDriver.get();
		lc.externalPlayerMaskFn = [driver](RLGC::EnvSet *envSet,
										   std::vector<uint8_t> &mask) {
			driver->BuildMask(envSet, mask);
		};
		lc.preStepFn = [driver](RLGC::EnvSet *envSet) { driver->Step(envSet); };
	}
	g_NectoEnabled = cfg.necto.enabled;

	if (cfg.necto.enabled && cfg.necto.benchmark && !cfg.renderMode) {
		g_NectoBench = std::make_unique<NectoBench>(cfg);
		g_NectoBenchRunFolder = cfg.CheckpointFolder();
		// benchInterval is in iterations; the callback sees timesteps.
		g_NectoBenchIntervalTs =
			static_cast<int64_t>(cfg.necto.benchInterval) * cfg.tsPerItr;
		// Skip the very first crossing: nothing has been saved yet on a fresh
		// run, and on a resumed one the first iteration is not a useful sample.
		g_NectoBenchNextTs = g_NectoBenchIntervalTs;
	}

	auto envCreateFn = [cfg](int index) -> EnvCreateResult {
		return CreateEnv(index, cfg);
	};

	Learner learner(envCreateFn, lc, StepCallback);

	std::signal(SIGINT, HandleSigint);

	learner.Start();
}

} // namespace Dash
