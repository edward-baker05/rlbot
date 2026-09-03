#include "Config.h"
#include "eval/Checkpoints.h"
#include "eval/MatchBench.h"
#include "eval/MigrateObs.h"
#include "eval/NectoBench.h"
#include "eval/PredictBench.h"
#include "eval/Spectate.h"
#include "eval/WinMatrix.h"
#include "opponents/NectoSelfTest.h"
#include "rlbot/DashBot.h"
#include "train/Train.h"

#include <rlbot/BotManager.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

void PrintUsage(const char *argv0) {
	std::printf(
		"Usage: %s <command> [options]\n"
		"\n"
		"Commands:\n"
		"  train            Train the policy\n"
		"  play             Connect to RLBot v5 and play a match\n"
		"  spectate         Watch a checkpoint or live run in RocketSimVis\n"
		"  match            Play standard 5-minute head-to-head games between "
		"two checkpoints\n"
		"  benchmark        Score a checkpoint head-to-head against Necto\n"
		"  win-matrix       Round-robin every saved policy version against every\n"
		"                   other, to test whether the lineage is transitive\n"
		"  predict-bench    Measure the throughput cost of the prediction obs "
		"block\n"
		"  migrate-obs      Widen a checkpoint's input layer for a new obs "
		"mode\n"
		"  necto-selftest   Dump the Necto observation for a fixed state, to "
		"diff\n"
		"                   against the Python reference\n"
		"  nexto-selftest   The same for Nexto, whose observation differs in "
		"three\n"
		"                   places -- see scripts/nexto_obs_check.py\n"
		"\n"
		"Training options:\n"
		"  --games N            Number of simultaneous games (default 128)\n"
		"  --obs MODE           Observation builder: default, advanced, "
		"predictive or\n"
		"                       padgeometry (default: padgeometry)\n"
		"  --p1 X               Weight/probability of 1v1 arenas (default: "
		"1.0)\n"
		"  --p2 X               Weight/probability of 2v2 arenas (default: "
		"0.0)\n"
		"  --p3 X               Weight/probability of 3v3 arenas (default: "
		"0.0)\n"
		"  --max-players N      Max players per team for padding capacity "
		"(default: 3)\n"
		"  --render             Stream to RocketSimVis instead of training at "
		"speed\n"
		"  --cpu                Train on CPU instead of CUDA\n"
		"  --no-metrics         Do not send metrics to wandb\n"
		"  --seed N             Random seed (default: clock)\n"
		"  --max-steps N        Stop after N timesteps (default: run until Q)\n"
		"  --label NAME         Suffix run and checkpoint names, to keep runs "
		"apart\n"
		"  --entropy X          Entropy bonus scale (default 0.020)\n"
		"  --entropy-target X   Starting target normalized entropy (0 disables "
		"controller,\n"
		"                       default 0.49)\n"
		"  --entropy-target-min X   Floor the target decays to (>= target "
		"disables\n"
		"                       the decay, default 0.25)\n"
		"  --entropy-decay X    Target drop per 1B steps (default 0.02)\n"
		"  --entropy-decay-from N   Step count the decay is measured from "
		"(default 0)\n"
		"  --lr X               Policy and critic learning rate; 0 freezes "
		"the\n"
		"                       policy, for calibration probes\n"
		"  --aerial-spawns X    Fraction of episodes in aerial state (default: "
		"0.65)\n"
		"  --aerial-hover-frac X Fraction of aerial spawns in hover state "
		"(default: "
		"0.60)\n"
		"  --fresh              Start over instead of resuming --label's "
		"checkpoints\n"
		"\n"
		"Spectate options:\n"
		"  --rewards            Print the training reward breakdown each step "
		"and\n"
		"                       take keyboard control: [space] pause/resume, "
		"[.]\n"
		"                       step, [c] run, [b] history, [s] totals, [q] "
		"quit\n"
		"  --reward-start-paused  Imply --rewards and begin paused at step 0\n"
		"  --reward-pause X     Auto-pause when an event reward's weighted\n"
		"                       contribution exceeds X (default 0.25, 0 "
		"disables)\n"
		"  --reward-history N   Steps of breakdown kept for [b] (default 30)\n"
		"\n"
		"Benchmark options:\n"
		"  --run LABEL          Score the newest checkpoint of this run\n"
		"  --model PATH         Score this checkpoint folder instead\n"
		"  --arenas N           Parallel arenas, also the goals per round\n"
		"                       (default 16)\n"
		"  --rounds N           Play N rounds (default 1)\n"
		"\n"
		"Match options (head-to-head standard games):\n"
		"  --m1 / --p1 PATH     First checkpoint path or run label (default: "
		"t1)\n"
		"  --m2 / --p2 PATH     Second checkpoint path or run label (default: "
		"t2)\n"
		"  --label1 NAME        Label for bot 1 (default: folder name)\n"
		"  --label2 NAME        Label for bot 2 (default: folder name)\n"
		"  --games N            Total 5-minute games to play (default: 500)\n"
		"  --arenas N           Parallel arenas (default: 64)\n"
		"  --game-time X        Game clock length in seconds (default: 300 = 5 "
		"min)\n"
		"  --overtime           Enable sudden death overtime on ties (default: "
		"true)\n"
		"  --no-overtime        Disable overtime (allow ties)\n"
		"  --deterministic      Use argmax actions (default: true)\n"
		"  --stochastic         Sample actions with temperature\n"
		"  --temp X             Temperature for stochastic sampling (default: "
		"1.0)\n"
		"  --cpu                Run on CPU instead of CUDA\n"
		"  --json PATH          Export full per-game results to JSON\n"
		"\n"
		"Necto opponent:\n"
		"  --necto              Put Necto in a slice of the training arenas\n"
		"  --necto-fraction X   Fraction of ARENAS with Necto in them "
		"(default\n"
		"                       0.20). The share of training DATA is X/(2-X),\n"
		"                       since only the learner's half of those arenas\n"
		"                       produces trajectories\n"
		"  --no-necto-bench     Skip the periodic head-to-head benchmark\n"
		"  --necto-bench-interval N  Iterations between benchmarks (default "
		"100)\n"
		"  --necto-bench-arenas N    Arenas per benchmark round (default 16)\n"
		"\n"
		"Self-play options:\n"
		"  --self-play          Train against saved old versions, and track "
		"skill\n"
		"  --track-skill        Track ELO against old versions without "
		"self-play\n"
		"                       (use this for a comparable baseline)\n"
		"  --ts-per-version N   Snapshot the policy every N steps (default "
		"5M)\n"
		"  --old-version-chance X  Fraction of iterations played against a "
		"saved\n"
		"                       old version (default 0.20)\n"
		"  --ts-per-itr N       Timesteps collected per PPO iteration "
		"(default\n"
		"                       250k). Lower it for short shakedown runs on a "
		"busy GPU\n"
		"\n"
		"Environment (play):\n"
		"  DASH_MODEL           Checkpoint folder for the policy (required)\n"
		"  DASH_OBS             Observation builder: default, advanced or "
		"predictive (default: advanced)\n"
		"  RLBOT_AGENT_ID       Set by RLBot when it launches the bot\n"
		"\n"
		"Environment (both):\n"
		"  DASH_COLLISION_MESHES  Path to RocketSim collision meshes\n"
		"                         (default: collision_meshes)\n"
		"  DASH_NECTO_MODEL       Path to necto-model.pt\n",
		argv0);
}

int RunPlay(int argc, char *argv[]) {
	// RLBot launches us with these set; the defaults let you run by hand.
	const char *host = std::getenv("RLBOT_SERVER_IP");
	const char *port = std::getenv("RLBOT_SERVER_PORT");
	const char *agentId = std::getenv("RLBOT_AGENT_ID");

	if (!host || !*host)
		host = "127.0.0.1";
	if (!port || !*port)
		port = "23234";

	if (!agentId || !*agentId) {
		std::fprintf(stderr, "RLBOT_AGENT_ID is not set.\n"
							 "RLBot normally sets this when it launches the "
							 "bot. To run by hand, set it to the\n"
							 "agent_id from bot.toml, e.g.:  "
							 "RLBOT_AGENT_ID=dash/bot ./DashBot play\n");
		return EXIT_FAILURE;
	}

	try {
		// RLBot has a connection timeout that lazy GPU loading would trip.
		Dash::Context().Initialize(Dash::BotSettings::FromEnvironment());
	} catch (const std::exception &e) {
		std::fprintf(stderr, "Failed to initialize: %s\n", e.what());
		return EXIT_FAILURE;
	}

	rlbot::BotManager<Dash::DashBot> manager{true};

	if (!manager.connect(host, port, agentId, /*ballPrediction=*/false)) {
		std::fprintf(stderr, "Failed to connect to RLBotServer at %s:%s\n",
					 host, port);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int RunTrain(int argc, char *argv[]) {
	Dash::TrainConfig cfg = {};
	bool fresh = false;

	for (int i = 2; i < argc; i++) {
		const std::string arg = argv[i];

		if (arg == "--games" && i + 1 < argc) {
			cfg.numGames = std::atoi(argv[++i]);
		} else if ((arg == "--obs" || arg == "--obs-mode") && i + 1 < argc) {
			std::string modeStr = argv[++i];
			if (modeStr == "default" || modeStr == "Default") {
				cfg.obs = Dash::ObsMode::Default;
			} else if (modeStr == "advanced" || modeStr == "Advanced") {
				cfg.obs = Dash::ObsMode::Advanced;
			} else if (modeStr == "predictive" || modeStr == "Predictive") {
				cfg.obs = Dash::ObsMode::Predictive;
			} else if (modeStr == "padgeometry" || modeStr == "PadGeometry") {
				cfg.obs = Dash::ObsMode::PadGeometry;
			} else {
				std::fprintf(stderr,
							 "Unknown obs mode: %s (expected 'default', "
							 "'advanced', 'predictive' or 'padgeometry')\n",
							 modeStr.c_str());
				return EXIT_FAILURE;
			}
		} else if ((arg == "--p1" || arg == "--prob-1v1") && i + 1 < argc) {
			cfg.teamDistribution.p1v1 =
				static_cast<float>(std::atof(argv[++i]));
		} else if ((arg == "--p2" || arg == "--prob-2v2") && i + 1 < argc) {
			cfg.teamDistribution.p2v2 =
				static_cast<float>(std::atof(argv[++i]));
		} else if ((arg == "--p3" || arg == "--prob-3v3") && i + 1 < argc) {
			cfg.teamDistribution.p3v3 =
				static_cast<float>(std::atof(argv[++i]));
		} else if ((arg == "--max-players" ||
					arg == "--max-players-per-team") &&
				   i + 1 < argc) {
			cfg.maxPlayersPerTeam = std::atoi(argv[++i]);
		} else if (arg == "--seed" && i + 1 < argc) {
			cfg.randomSeed = std::atoll(argv[++i]);
		} else if (arg == "--max-steps" && i + 1 < argc) {
			cfg.maxSteps = std::atoll(argv[++i]);
		} else if (arg == "--label" && i + 1 < argc) {
			cfg.runLabel = argv[++i];
		} else if (arg == "--entropy" && i + 1 < argc) {
			cfg.entropyScale = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--entropy-target" && i + 1 < argc) {
			// 0 disables the controller and pins entropyScale to --entropy.
			cfg.entropyTarget = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--entropy-target-min" && i + 1 < argc) {
			cfg.entropyTargetMin = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--entropy-decay" && i + 1 < argc) {
			cfg.entropyTargetDecayPerB =
				static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--entropy-decay-from" && i + 1 < argc) {
			cfg.entropyDecayFromSteps = std::atoll(argv[++i]);
		} else if (arg == "--lr" && i + 1 < argc) {
			// 0 freezes the policy, which is what a calibration probe needs.
			const float lr = static_cast<float>(std::atof(argv[++i]));
			cfg.policyLR = lr;
			cfg.criticLR = lr;
		} else if (arg == "--infinite-boost" && i + 1 < argc) {
			cfg.infiniteBoostChance = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--aerial-spawns" && i + 1 < argc) {
			cfg.aerial.aerialSpawnChance =
				static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--aerial-hover-frac" && i + 1 < argc) {
			cfg.aerial.hoverFraction = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--backboard-follow-frac" && i + 1 < argc) {
			cfg.aerial.backboardFollowFraction =
				static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--fresh") {
			fresh = true;
		} else if (arg == "--self-play") {
			cfg.selfPlay.trainAgainstOldVersions = true;
			// Skill tracking is what makes the result readable.
			cfg.selfPlay.trackSkill = true;
		} else if (arg == "--track-skill") {
			cfg.selfPlay.trackSkill = true;
		} else if (arg == "--ts-per-version" && i + 1 < argc) {
			cfg.selfPlay.tsPerVersion = std::atoll(argv[++i]);
		} else if (arg == "--ts-per-itr" && i + 1 < argc) {
			// Smaller iterations mean a smaller experience buffer and smaller
			// minibatches, which is what makes a short shakedown run fit on a
			// GPU that is already busy.
			cfg.tsPerItr = std::atoi(argv[++i]);
			cfg.miniBatchSize = RS_MIN(cfg.miniBatchSize, cfg.tsPerItr);
		} else if (arg == "--old-version-chance" && i + 1 < argc) {
			cfg.selfPlay.trainAgainstOldChance =
				static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--necto") {
			cfg.necto.enabled = true;
		} else if (arg == "--necto-fraction" && i + 1 < argc) {
			cfg.necto.enabled = true;
			cfg.necto.arenaFraction = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--no-necto-bench") {
			cfg.necto.benchmark = false;
		} else if (arg == "--necto-bench-interval" && i + 1 < argc) {
			cfg.necto.benchInterval = std::atoi(argv[++i]);
		} else if (arg == "--necto-bench-arenas" && i + 1 < argc) {
			cfg.necto.benchArenas = std::atoi(argv[++i]);
		} else if (arg == "--render") {
			// Wall-clock speed for RocketSimVis; useless for actually training.
			cfg.renderMode = true;
			cfg.numGames = 1;
			cfg.sendMetrics = false;
		} else if (arg == "--cpu") {
			cfg.useGPU = false;
		} else if (arg == "--no-metrics") {
			cfg.sendMetrics = false;
		} else {
			std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
			return EXIT_FAILURE;
		}
	}

	if (cfg.numGames < 1) {
		std::fprintf(stderr, "--games must be at least 1\n");
		return EXIT_FAILURE;
	}

	if (cfg.maxPlayersPerTeam <
		cfg.teamDistribution.MaxActivePlayersPerTeam()) {
		std::fprintf(stderr,
					 "WARNING: maxPlayersPerTeam (%d) is smaller than active "
					 "team distribution max (%d). Raising to %d.\n",
					 cfg.maxPlayersPerTeam,
					 cfg.teamDistribution.MaxActivePlayersPerTeam(),
					 cfg.teamDistribution.MaxActivePlayersPerTeam());
		cfg.maxPlayersPerTeam = cfg.teamDistribution.MaxActivePlayersPerTeam();
	}

	// Reusing a label CONTINUES that run rather than starting fresh.
	const std::filesystem::path checkpoints = cfg.CheckpointFolder();
	const bool hasExisting = std::filesystem::exists(checkpoints) &&
							 !std::filesystem::is_empty(checkpoints);

	if (hasExisting) {
		if (fresh) {
			// Renamed, never deleted: a crashed run's last save is often worth
			// keeping.
			std::filesystem::path archived = checkpoints;
			archived += "-archived";
			for (int n = 2; std::filesystem::exists(archived); n++)
				archived =
					checkpoints.string() + "-archived" + std::to_string(n);

			std::error_code ec;
			std::filesystem::rename(checkpoints, archived, ec);
			if (ec) {
				std::fprintf(stderr, "--fresh: could not move %s aside: %s\n",
							 checkpoints.string().c_str(),
							 ec.message().c_str());
				return EXIT_FAILURE;
			}
			std::printf("--fresh: moved existing checkpoints to %s\n",
						archived.string().c_str());

			// The receiver appends to an existing CSV, which would merge two
			// runs silently.
			const char *metricsEnv = std::getenv("DASH_METRICS_DIR");
			if (!metricsEnv)
				metricsEnv = std::getenv("HIVE_METRICS_DIR");
			const std::filesystem::path metricsCsv =
				std::filesystem::path(metricsEnv ? metricsEnv : "metrics") /
				(cfg.RunName() + ".csv");
			if (std::filesystem::exists(metricsCsv)) {
				std::filesystem::path archivedCsv =
					metricsCsv.parent_path() /
					(cfg.RunName() + "-archived.csv");
				for (int n = 2; std::filesystem::exists(archivedCsv); n++)
					archivedCsv = metricsCsv.parent_path() /
								  (cfg.RunName() + "-archived" +
								   std::to_string(n) + ".csv");

				std::error_code csvEc;
				std::filesystem::rename(metricsCsv, archivedCsv, csvEc);
				if (csvEc)
					std::fprintf(
						stderr,
						"--fresh: WARNING could not move %s aside: %s\n",
						metricsCsv.string().c_str(), csvEc.message().c_str());
				else
					std::printf("--fresh: moved existing metrics to %s\n",
								archivedCsv.string().c_str());
			}
		} else {
			std::printf("NOTE: resuming from existing checkpoints in %s.\n"
						"      Pass --fresh to start over (the old ones are "
						"archived, not deleted).\n",
						checkpoints.string().c_str());
		}
	}

	try {
		Dash::RunTraining(cfg);
	} catch (const std::exception &e) {
		std::fprintf(stderr, "Training failed: %s\n", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int RunSpectate(int argc, char *argv[]) {
	Dash::SpectateConfig scfg = {};
	Dash::TrainConfig defaults = {};
	for (int i = 2; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--follow" && i + 1 < argc) {
			defaults.runLabel = argv[++i];
			scfg.followRun = defaults.CheckpointFolder();
		} else if (arg == "--model" && i + 1 < argc) {
			scfg.model = argv[++i];
		} else if (arg == "--spawns" && i + 1 < argc) {
			const std::string mode = argv[++i];
			if (mode == "training" || mode == "curriculum") {
				scfg.spawns = Dash::SpectateSpawns::Training;
			} else if (mode == "kickoff") {
				scfg.spawns = Dash::SpectateSpawns::Kickoff;
			} else {
				std::fprintf(stderr, "--spawns must be training or kickoff\n");
				return EXIT_FAILURE;
			}
		} else if (arg == "--time-scale" && i + 1 < argc) {
			scfg.timeScale = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--episodes" && i + 1 < argc) {
			scfg.episodes = std::atoi(argv[++i]);
		} else if (arg == "--seed" && i + 1 < argc) {
			scfg.seed = std::atoll(argv[++i]);
		} else if (arg == "--reward-pause" && i + 1 < argc) {
			scfg.rewardPauseThreshold =
				static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--reward-history" && i + 1 < argc) {
			scfg.rewardHistorySteps = std::atoi(argv[++i]);
		} else if (arg == "--rewards") {
			scfg.debugRewards = true;
		} else if (arg == "--reward-start-paused") {
			scfg.debugRewards = true;
			scfg.rewardStartPaused = true;
		} else if (arg == "--deterministic") {
			scfg.deterministic = true;
		} else if (arg == "--gpu") {
			scfg.useGPU = true;
		} else {
			std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
			return EXIT_FAILURE;
		}
	}
	if (scfg.model.empty() == scfg.followRun.empty()) {
		std::fprintf(stderr,
					 "Usage: %s spectate --follow <label> | --model <ckpt>\n"
					 "       [--spawns training|kickoff] [--deterministic]\n"
					 "       [--time-scale X] [--episodes N] [--gpu]\n"
					 "       [--rewards] [--reward-start-paused]\n"
					 "       [--reward-pause X] [--reward-history N]\n",
					 argv[0]);
		return EXIT_FAILURE;
	}
	try {
		Dash::RunSpectate(scfg);
	} catch (const std::exception &e) {
		std::fprintf(stderr, "Spectate failed: %s\n", e.what());
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

} // namespace

std::string FindCollisionMeshes() {
	const char *meshEnv = std::getenv("DASH_COLLISION_MESHES");
	if (!meshEnv)
		meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	if (meshEnv &&
		std::filesystem::exists(std::filesystem::path(meshEnv) / "soccar"))
		return meshEnv;

	for (const auto &p : {"collision_meshes", "tools/collision_meshes",
						  "bot/build/collision_meshes",
						  "../tools/collision_meshes", "../collision_meshes"}) {
		if (std::filesystem::exists(std::filesystem::path(p) / "soccar"))
			return p;
	}
	return "collision_meshes";
}

int RunBenchmark(int argc, char *argv[]) {
	Dash::TrainConfig cfg = {};
	cfg.necto.enabled = true;

	std::filesystem::path model;
	std::string run;
	int rounds = 1;

	for (int i = 2; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--model" && i + 1 < argc) {
			model = argv[++i];
		} else if (arg == "--run" && i + 1 < argc) {
			run = argv[++i];
		} else if (arg == "--arenas" && i + 1 < argc) {
			cfg.necto.benchArenas = std::atoi(argv[++i]);
		} else if (arg == "--rounds" && i + 1 < argc) {
			rounds = std::atoi(argv[++i]);
		} else {
			std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
			return EXIT_FAILURE;
		}
	}

	if (!run.empty() && model.empty()) {
		cfg.runLabel = run;
		model = Dash::FindLatestCheckpoint(cfg.CheckpointFolder());
		if (model.empty()) {
			std::fprintf(stderr, "No complete checkpoint under %s\n",
						 cfg.CheckpointFolder().string().c_str());
			return EXIT_FAILURE;
		}
	}

	if (model.empty()) {
		std::fprintf(stderr, "Pass --model <checkpoint> or --run <label>\n");
		return EXIT_FAILURE;
	}

	// Keep the Elo file next to the run the checkpoint came from, so repeated
	// benchmarks of the same run accumulate rather than starting over.
	if (run.empty()) {
		cfg.checkpointRoot = model.parent_path().parent_path();
		cfg.runLabel = model.parent_path().filename().string();
	}

	RocketSim::Init(FindCollisionMeshes());

	Dash::NectoBench bench(cfg);
	std::printf("Benchmarking %s vs %s (%d arenas, training scenario pool)\n",
				model.string().c_str(), bench.OpponentName(),
				cfg.necto.benchArenas);

	for (int r = 0; r < rounds; r++) {
		const Dash::NectoBenchResult result = bench.Run(model);
		if (!result.valid) {
			std::fprintf(stderr, "Benchmark produced no episodes\n");
			return EXIT_FAILURE;
		}
		std::printf("round %d: %d-%d over %d episodes (%d decisive), "
					"win rate %.3f | last %d: %.3f, Elo %.1f\n",
					r + 1, result.goalsFor, result.goalsAgainst,
					result.episodes, result.decisive, result.winRate,
					result.windowGames, result.windowWinRate,
					result.windowElo);
	}
	return EXIT_SUCCESS;
}

int RunWinMatrix(int argc, char *argv[]) {
	Dash::WinMatrixConfig cfg = {};
	std::string run = "t1";

	for (int i = 2; i < argc; i++) {
		const std::string arg = argv[i];
		if (arg == "--run" && i + 1 < argc)
			run = argv[++i];
		else if (arg == "--versions" && i + 1 < argc)
			cfg.maxVersions = std::atoi(argv[++i]);
		else if (arg == "--games-per-pair" && i + 1 < argc)
			cfg.gamesPerPair = std::atoi(argv[++i]);
		else if (arg == "--arenas" && i + 1 < argc)
			cfg.arenas = std::atoi(argv[++i]);
		else if (arg == "--duration" && i + 1 < argc)
			cfg.gameDuration = static_cast<float>(std::atof(argv[++i]));
		else if (arg == "--stochastic")
			cfg.deterministic = false;
		else if (arg == "--cpu")
			cfg.useGPU = false;
		else if (arg == "--json" && i + 1 < argc)
			cfg.jsonOutput = argv[++i];
		else {
			std::fprintf(stderr, "Unknown win-matrix option: %s\n", arg.c_str());
			return 1;
		}
	}

	cfg.runFolder = std::filesystem::path("checkpoints") / run;
	if (!std::filesystem::is_directory(cfg.runFolder)) {
		std::fprintf(stderr, "No such run folder: %s\n", cfg.runFolder.c_str());
		return 1;
	}

	RocketSim::Init(FindCollisionMeshes());

	Dash::WinMatrixResult res = Dash::RunWinMatrix(cfg);
	if (!res.valid)
		return 1;

	Dash::PrintWinMatrix(res);
	return 0;
}

int RunMatch(int argc, char *argv[]) {
	Dash::MatchRunnerConfig cfg = {};
	std::string m1Input = "t1";
	std::string m2Input = "t2";

	for (int i = 2; i < argc; i++) {
		const std::string arg = argv[i];
		if ((arg == "--m1" || arg == "--p1" || arg == "--run1" ||
			 arg == "--bot1") &&
			i + 1 < argc) {
			m1Input = argv[++i];
		} else if ((arg == "--m2" || arg == "--p2" || arg == "--run2" ||
					arg == "--bot2") &&
				   i + 1 < argc) {
			m2Input = argv[++i];
		} else if (arg == "--label1" && i + 1 < argc) {
			cfg.label1 = argv[++i];
		} else if (arg == "--label2" && i + 1 < argc) {
			cfg.label2 = argv[++i];
		} else if (arg == "--games" && i + 1 < argc) {
			cfg.totalGames = std::atoi(argv[++i]);
		} else if (arg == "--arenas" && i + 1 < argc) {
			cfg.arenas = std::atoi(argv[++i]);
		} else if ((arg == "--game-time" || arg == "--time") && i + 1 < argc) {
			cfg.gameDuration = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--overtime") {
			cfg.suddenDeathOvertime = true;
		} else if (arg == "--no-overtime") {
			cfg.suddenDeathOvertime = false;
		} else if (arg == "--max-ot" && i + 1 < argc) {
			cfg.maxOvertime = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--deterministic") {
			cfg.deterministic = true;
		} else if (arg == "--stochastic") {
			cfg.deterministic = false;
		} else if ((arg == "--temp" || arg == "--temperature") &&
				   i + 1 < argc) {
			cfg.temperature = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--cpu") {
			cfg.useGPU = false;
		} else if (arg == "--gpu") {
			cfg.useGPU = true;
		} else if (arg == "--json" && i + 1 < argc) {
			cfg.jsonOutput = argv[++i];
		} else {
			std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
			return EXIT_FAILURE;
		}
	}

	if (cfg.label1.empty() || cfg.label1 == "t1") {
		cfg.label1 = std::filesystem::path(m1Input).filename().string();
	}
	if (cfg.label2.empty() || cfg.label2 == "t2") {
		cfg.label2 = std::filesystem::path(m2Input).filename().string();
	}

	cfg.model1 = Dash::ResolveCheckpoint(m1Input);
	if (cfg.model1.empty()) {
		std::fprintf(stderr,
					 "Error: Could not resolve checkpoint for model 1: '%s'\n",
					 m1Input.c_str());
		return EXIT_FAILURE;
	}

	cfg.model2 = Dash::ResolveCheckpoint(m2Input);
	if (cfg.model2.empty()) {
		std::fprintf(stderr,
					 "Error: Could not resolve checkpoint for model 2: '%s'\n",
					 m2Input.c_str());
		return EXIT_FAILURE;
	}

	RocketSim::Init(FindCollisionMeshes());

	std::printf("=============================================================="
				"==================\n");
	std::printf("HEAD-TO-HEAD MATCH EVALUATION\n");
	std::printf("  Bot 1 (%s): %s\n", cfg.label1.c_str(),
				cfg.model1.string().c_str());
	std::printf("  Bot 2 (%s): %s\n", cfg.label2.c_str(),
				cfg.model2.string().c_str());
	std::printf(
		"  Volume: %d games across %d arenas (%.1f min regulation games)\n",
		cfg.totalGames, cfg.arenas, cfg.gameDuration / 60.0f);
	std::printf("  Overtime: %s (max %.1f min), Device: %s, Inference: %s\n",
				cfg.suddenDeathOvertime ? "Sudden Death" : "Off",
				cfg.maxOvertime / 60.0f, cfg.useGPU ? "GPU (CUDA)" : "CPU",
				cfg.deterministic
					? "Deterministic (Argmax)"
					: ("Stochastic (T=" + std::to_string(cfg.temperature) + ")")
						  .c_str());
	std::printf("=============================================================="
				"==================\n");

	Dash::MatchBench matchBench(cfg);
	Dash::MatchSummaryResult result = matchBench.Run();

	if (!result.valid) {
		std::fprintf(stderr, "Match evaluation failed or produced no games.\n");
		return EXIT_FAILURE;
	}

	std::printf("\n============================================================"
				"====================\n");
	std::printf("MATCH RESULTS SUMMARY (%d Games Played)\n", result.totalGames);
	std::printf("=============================================================="
				"==================\n");
	std::printf("  %-10s : %4d wins (%.2f%%) [95%% CI: %.2f%% - %.2f%%]\n",
				result.label1.c_str(), result.bot1Wins,
				result.bot1WinRate * 100.0f,
				(1.0f - result.winRateCiUpper95) * 100.0f,
				(1.0f - result.winRateCiLower95) * 100.0f);
	std::printf("  %-10s : %4d wins (%.2f%%) [95%% CI: %.2f%% - %.2f%%]\n",
				result.label2.c_str(), result.bot2Wins,
				result.bot2WinRate * 100.0f, result.winRateCiLower95 * 100.0f,
				result.winRateCiUpper95 * 100.0f);
	if (result.ties > 0) {
		std::printf("  Ties       : %4d (%.2f%%)\n", result.ties,
					(result.ties * 100.0f) / result.totalGames);
	}

	std::printf("\n--- GOALS & SCORING ---\n");
	std::printf("  %-10s : %4d total goals (Mean: %.2f +/- %.2f per game)\n",
				result.label1.c_str(), result.bot1Goals, result.bot1MeanGoals,
				result.bot1StdGoals);
	std::printf("  %-10s : %4d total goals (Mean: %.2f +/- %.2f per game)\n",
				result.label2.c_str(), result.bot2Goals, result.bot2MeanGoals,
				result.bot2StdGoals);
	std::printf(
		"  Goal Diff  : %+.2f (%s minus %s) [95%% CI: %+.2f to %+.2f]\n",
		result.meanGoalDiff, result.label2.c_str(), result.label1.c_str(),
		result.goalDiffCiLower95, result.goalDiffCiUpper95);

	std::printf("\n--- STATISTICAL SIGNIFICANCE TESTS ---\n");
	std::printf(
		"  Binomial Test (H0: p = 0.5)      : p = %.4e %s\n",
		result.binomialPValue,
		result.binomialPValue < 0.001
			? "(p < 0.001, highly significant)"
			: (result.binomialPValue < 0.01
				   ? "(p < 0.01, highly significant)"
				   : (result.binomialPValue < 0.05
						  ? "(p < 0.05, statistically significant)"
						  : "(not statistically significant at alpha=0.05)")));
	std::printf("  Paired t-test (Goal Differential): p = %.4e\n",
				result.pairedTPValue);
	std::printf("  Wilcoxon Signed-Rank Test        : p = %.4e\n",
				result.wilcoxonPValue);

	std::printf("\n--- OVERTIME & SIDE ANALYSIS ---\n");
	std::printf("  Overtime Games: %d / %d (%.1f%%), Mean OT Duration: %.1fs\n",
				result.overtimeGames, result.totalGames,
				(result.overtimeGames * 100.0f) / result.totalGames,
				result.meanOtDuration);
	std::printf("    OT Wins: %s: %d, %s: %d\n", result.label1.c_str(),
				result.bot1OtWins, result.label2.c_str(), result.bot2OtWins);
	std::printf("  Side Balance:\n");
	std::printf(
		"    %s as BLUE:   %d/%d (%.1f%%) | %s as ORANGE: %d/%d (%.1f%%)\n",
		result.label1.c_str(), result.bot1BlueWins, result.bot1BlueGames,
		result.bot1BlueGames > 0
			? (result.bot1BlueWins * 100.0f) / result.bot1BlueGames
			: 0.f,
		result.label1.c_str(), result.bot1OrangeWins, result.bot1OrangeGames,
		result.bot1OrangeGames > 0
			? (result.bot1OrangeWins * 100.0f) / result.bot1OrangeGames
			: 0.f);
	std::printf(
		"    %s as BLUE:   %d/%d (%.1f%%) | %s as ORANGE: %d/%d (%.1f%%)\n",
		result.label2.c_str(), result.bot2BlueWins, result.bot2BlueGames,
		result.bot2BlueGames > 0
			? (result.bot2BlueWins * 100.0f) / result.bot2BlueGames
			: 0.f,
		result.label2.c_str(), result.bot2OrangeWins, result.bot2OrangeGames,
		result.bot2OrangeGames > 0
			? (result.bot2OrangeWins * 100.0f) / result.bot2OrangeGames
			: 0.f);
	std::printf("    Overall Blue Win Rate: %.1f%% | Orange Win Rate: %.1f%%\n",
				result.blueWinRate * 100.0f, result.orangeWinRate * 100.0f);

	std::printf("\n============================================================"
				"====================\n");
	if (result.binomialPValue < 0.05) {
		const std::string &winner =
			(result.bot2Wins > result.bot1Wins) ? result.label2 : result.label1;
		float winRate =
			std::max(result.bot1WinRate, result.bot2WinRate) * 100.0f;
		std::printf("VERDICT: %s is STATISTICALLY SIGNIFICANTLY BETTER (Win "
					"Rate: %.2f%%, p = %.2e)\n",
					winner.c_str(), winRate, result.binomialPValue);
	} else {
		std::printf("VERDICT: NO STATISTICALLY SIGNIFICANT DIFFERENCE detected "
					"between %s and %s (p = %.4f)\n",
					result.label1.c_str(), result.label2.c_str(),
					result.binomialPValue);
	}
	std::printf("=============================================================="
				"==================\n\n");

	return EXIT_SUCCESS;
}

int main(int argc, char *argv[]) {
	// Unbuffered output so logs interleave correctly when RLBot captures them.
	std::setvbuf(stdout, nullptr, _IONBF, 0);
	std::setvbuf(stderr, nullptr, _IONBF, 0);

	if (argc < 2) {
		PrintUsage(argv[0]);
		return EXIT_FAILURE;
	}

	const std::string command = argv[1];

	if (command == "train")
		return RunTrain(argc, argv);

	if (command == "play")
		return RunPlay(argc, argv);

	if (command == "spectate")
		return RunSpectate(argc, argv);

	if (command == "match")
		return RunMatch(argc, argv);

	if (command == "benchmark")
		return RunBenchmark(argc, argv);

	if (command == "win-matrix")
		return RunWinMatrix(argc, argv);

	if (command == "migrate-obs") {
		std::filesystem::path src, dst;
		int oldObs = 225, newObs = 249;
		for (int i = 2; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--src" && i + 1 < argc)
				src = argv[++i];
			else if (arg == "--dst" && i + 1 < argc)
				dst = argv[++i];
			else if (arg == "--old-obs" && i + 1 < argc)
				oldObs = std::atoi(argv[++i]);
			else if (arg == "--new-obs" && i + 1 < argc)
				newObs = std::atoi(argv[++i]);
			else {
				std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
				return EXIT_FAILURE;
			}
		}
		if (src.empty() || dst.empty()) {
			std::fprintf(stderr,
						 "Usage: %s migrate-obs --src <run folder> "
						 "--dst <run folder>\n"
						 "       [--old-obs N] [--new-obs N]\n", argv[0]);
			return EXIT_FAILURE;
		}
		if (std::filesystem::exists(dst)) {
			std::fprintf(stderr, "Refusing to overwrite existing %s\n",
						 dst.string().c_str());
			return EXIT_FAILURE;
		}
		return Dash::RunMigrateObs(src, dst, oldObs, newObs);
	}

	if (command == "predict-bench") {
		int arenas = 64;
		int steps = 400;
		for (int i = 2; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--arenas" && i + 1 < argc)
				arenas = std::atoi(argv[++i]);
			else if (arg == "--steps" && i + 1 < argc)
				steps = std::atoi(argv[++i]);
			else {
				std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
				return EXIT_FAILURE;
			}
		}
		RocketSim::Init(FindCollisionMeshes());
		return Dash::RunPredictBench(arenas, steps);
	}

	if (command == "necto-selftest" || command == "nexto-selftest") {
		// Optional output path; defaults to stdout.
		const Dash::NectoFamily family = command == "nexto-selftest"
											 ? Dash::NectoFamily::Nexto
											 : Dash::NectoFamily::Necto;
		return Dash::RunNectoSelfTest(family, argc > 2 ? argv[2] : "");
	}

	if (command == "-h" || command == "--help" || command == "help") {
		PrintUsage(argv[0]);
		return EXIT_SUCCESS;
	}

	std::fprintf(stderr, "Unknown command: %s\n\n", command.c_str());
	PrintUsage(argv[0]);
	return EXIT_FAILURE;
}
