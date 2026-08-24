#include "Config.h"
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
		"\n"
		"Training options:\n"
		"  --games N            Number of simultaneous games (default 128)\n"
		"  --obs MODE           Observation builder: default or advanced "
		"(default: advanced)\n"
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
		"  --entropy-target X   Target normalized entropy (0 disables "
		"controller, default 0.0)\n"
		"  --lr X               Policy and critic learning rate; 0 freezes "
		"the\n"
		"                       policy, for calibration probes\n"
		"  --fresh              Start over instead of resuming --label's "
		"checkpoints\n"
		"\n"
		"Self-play options:\n"
		"  --self-play          Train against saved old versions, and track "
		"skill\n"
		"  --track-skill        Track ELO against old versions without "
		"self-play\n"
		"                       (use this for a comparable baseline)\n"
		"  --ts-per-version N   Snapshot the policy every N steps (default "
		"5M)\n"
		"\n"
		"Environment (play):\n"
		"  DASH_MODEL           Checkpoint folder for the policy (required)\n"
		"  DASH_OBS             Observation builder: default or advanced "
		"(default: advanced)\n"
		"  RLBOT_AGENT_ID       Set by RLBot when it launches the bot\n"
		"\n"
		"Environment (both):\n"
		"  DASH_COLLISION_MESHES  Path to RocketSim collision meshes\n"
		"                         (default: collision_meshes)\n",
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

	// batchHivemind = true delivers all our cars in one update(), for one
	// batch.
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
			} else {
				std::fprintf(
					stderr,
					"Unknown obs mode: %s (expected 'default' or 'advanced')\n",
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
		} else if (arg == "--lr" && i + 1 < argc) {
			// 0 freezes the policy, which is what a calibration probe needs.
			const float lr = static_cast<float>(std::atof(argv[++i]));
			cfg.policyLR = lr;
			cfg.criticLR = lr;
		} else if (arg == "--infinite-boost" && i + 1 < argc) {
			cfg.infiniteBoostChance = static_cast<float>(std::atof(argv[++i]));
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

} // namespace

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

	if (command == "-h" || command == "--help" || command == "help") {
		PrintUsage(argv[0]);
		return EXIT_SUCCESS;
	}

	std::fprintf(stderr, "Unknown command: %s\n\n", command.c_str());
	PrintUsage(argv[0]);
	return EXIT_FAILURE;
}
