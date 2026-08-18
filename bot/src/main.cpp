#include "Config.h"
#include "Verify.h"
#include "eval/Eval.h"
#include "eval/Spectate.h"
#include "rlbot/HivemindBot.h"
#include "train/Train.h"

#include <rlbot/BotManager.h>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void PrintUsage(const char* argv0) {
	std::printf(
		"Usage: %s <command> [options]\n"
		"\n"
		"Commands:\n"
		"  train            Train the policy\n"
		"  play             Connect to RLBot v5 and play a match\n"
		"  verify <folder>  Check a checkpoint loads and infers sanely before deploying it\n"
		"  eval --blue A --orange B   Play two checkpoints against each other in RocketSim\n"
		"  spectate         Watch a checkpoint play live in RocketSimVis (no training)\n"
		"\n"
		"Spectate options (pass exactly one of --follow / --model):\n"
		"  --follow LABEL       Follow a run's newest checkpoint, reloading between\n"
		"                       episodes -- safe to point at a run in progress\n"
		"  --model FOLDER       Play one specific checkpoint folder\n"
		"  --spawns MODE        training (default) or kickoff\n"
		"  --deterministic      Take the argmax action instead of sampling\n"
		"  --time-scale X       Speed multiplier (default 1.0 = real time)\n"
		"  --episodes N         Stop after N episodes (default: run until Ctrl-C)\n"
		"  --gpu                Use CUDA (default CPU, to leave the GPU to training)\n"
		"\n"
		"Training options:\n"
		"  --games N            Number of simultaneous games (default 128)\n"
		"  --render             Stream to RocketSimVis instead of training at speed\n"
		"  --cpu                Train on CPU instead of CUDA\n"
		"  --no-metrics         Do not send metrics to wandb\n"
		"  --seed N             Random seed (default: clock)\n"
		"  --max-steps N        Stop after N timesteps (default: run until Q)\n"
		"  --label NAME         Suffix run and checkpoint names, to keep runs apart\n"
		"  --entropy X          Entropy bonus scale (default 0.002)\n"
		"\n"
		"Self-play options:\n"
		"  --self-play          Train against saved old versions, and track skill\n"
		"  --track-skill        Track ELO against old versions without self-play\n"
		"                       (use this for a comparable baseline)\n"
		"  --ts-per-version N   Snapshot the policy every N steps (default 5M)\n"
		"\n"
		"Environment (play):\n"
		"  HIVE_MODEL           Checkpoint folder for the policy (required)\n"
		"  RLBOT_AGENT_ID       Set by RLBot when it launches the bot\n"
		"\n"
		"Environment (both):\n"
		"  HIVE_COLLISION_MESHES  Path to RocketSim collision meshes\n"
		"                         (default: collision_meshes)\n",
		argv0);
}

int RunPlay(int argc, char* argv[]) {
	// RLBot launches us with these set; the defaults let you run by hand.
	const char* host = std::getenv("RLBOT_SERVER_IP");
	const char* port = std::getenv("RLBOT_SERVER_PORT");
	const char* agentId = std::getenv("RLBOT_AGENT_ID");

	if (!host || !*host) host = "127.0.0.1";
	if (!port || !*port) port = "23234";

	if (!agentId || !*agentId) {
		std::fprintf(stderr,
			"RLBOT_AGENT_ID is not set.\n"
			"RLBot normally sets this when it launches the bot. To run by hand, set it to the\n"
			"agent_id from bot.toml, e.g.:  RLBOT_AGENT_ID=hivemind/bot ./HivemindBot play\n");
		return EXIT_FAILURE;
	}

	try {
		// Load models before connecting. RLBot has a connection timeout, and
		// loading a policy onto the GPU is slow enough to trip it if done
		// lazily on the first packet.
		Hive::Context().Initialize(Hive::BotSettings::FromEnvironment());
	} catch (const std::exception& e) {
		std::fprintf(stderr, "Failed to initialize: %s\n", e.what());
		return EXIT_FAILURE;
	}

	// batchHivemind = true asks RLBot to deliver all our cars in one update()
	// call, which is what lets us run a single batched inference pass.
	rlbot::BotManager<Hive::HivemindBot> manager{true};

	if (!manager.connect(host, port, agentId, /*ballPrediction=*/false)) {
		std::fprintf(stderr, "Failed to connect to RLBotServer at %s:%s\n", host, port);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int RunTrain(int argc, char* argv[]) {
	Hive::TrainConfig cfg = {};

	for (int i = 2; i < argc; i++) {
		const std::string arg = argv[i];

		if (arg == "--games" && i + 1 < argc) {
			cfg.numGames = std::atoi(argv[++i]);
		} else if (arg == "--seed" && i + 1 < argc) {
			cfg.randomSeed = std::atoll(argv[++i]);
		} else if (arg == "--max-steps" && i + 1 < argc) {
			cfg.maxSteps = std::atoll(argv[++i]);
		} else if (arg == "--label" && i + 1 < argc) {
			cfg.runLabel = argv[++i];
		} else if (arg == "--entropy" && i + 1 < argc) {
			cfg.entropyScale = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--infinite-boost" && i + 1 < argc) {
			cfg.infiniteBoostChance = static_cast<float>(std::atof(argv[++i]));
		} else if (arg == "--self-play") {
			cfg.selfPlay.trainAgainstOldVersions = true;
			// Skill tracking is what makes the result readable, so turn it on
			// with self-play unless it was already requested.
			cfg.selfPlay.trackSkill = true;
		} else if (arg == "--track-skill") {
			cfg.selfPlay.trackSkill = true;
		} else if (arg == "--ts-per-version" && i + 1 < argc) {
			cfg.selfPlay.tsPerVersion = std::atoll(argv[++i]);
		} else if (arg == "--render") {
			// Rendering runs the sim at wall-clock speed so you can watch it in
			// RocketSimVis. Useful for sanity-checking state setters and
			// rewards; useless for actually training.
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

	try {
		Hive::RunTraining(cfg);
	} catch (const std::exception& e) {
		std::fprintf(stderr, "Training failed: %s\n", e.what());
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char* argv[]) {
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

	if (command == "verify") {
		if (argc < 3) {
			std::fprintf(stderr, "Usage: %s verify <checkpoint-folder>\n", argv[0]);
			return EXIT_FAILURE;
		}
		return Hive::RunVerify(argv[2]);
	}

	if (command == "eval") {
		Hive::EvalConfig ecfg = {};
		for (int i = 2; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--blue" && i + 1 < argc) {
				ecfg.blueModel = argv[++i];
			} else if (arg == "--orange" && i + 1 < argc) {
				ecfg.orangeModel = argv[++i];
			} else if (arg == "--games" && i + 1 < argc) {
				ecfg.games = std::atoi(argv[++i]);
			} else if (arg == "--seconds" && i + 1 < argc) {
				ecfg.maxSeconds = static_cast<float>(std::atof(argv[++i]));
			} else if (arg == "--seed" && i + 1 < argc) {
				ecfg.seed = std::atoll(argv[++i]);
			} else if (arg == "--cpu") {
				ecfg.useGPU = false;
			} else {
				std::fprintf(stderr, "Unknown option: %s\n", arg.c_str());
				return EXIT_FAILURE;
			}
		}
		if (ecfg.blueModel.empty() || ecfg.orangeModel.empty()) {
			std::fprintf(stderr, "Usage: %s eval --blue <ckpt> --orange <ckpt> [--games N] [--seconds S] [--cpu]\n", argv[0]);
			return EXIT_FAILURE;
		}
		try {
			Hive::RunEval(ecfg);
		} catch (const std::exception& e) {
			std::fprintf(stderr, "Eval failed: %s\n", e.what());
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	if (command == "spectate") {
		Hive::SpectateConfig scfg = {};
		Hive::TrainConfig defaults = {};
		for (int i = 2; i < argc; i++) {
			const std::string arg = argv[i];
			if (arg == "--follow" && i + 1 < argc) {
				// Take a label, not a path: the point of this command is to
				// watch a run, and the caller should not have to know how
				// checkpoint folders are named.
				defaults.runLabel = argv[++i];
				scfg.followRun = defaults.CheckpointFolder();
			} else if (arg == "--model" && i + 1 < argc) {
				scfg.model = argv[++i];
			} else if (arg == "--spawns" && i + 1 < argc) {
				const std::string mode = argv[++i];
				if (mode == "training" || mode == "curriculum") {
					scfg.spawns = Hive::SpectateSpawns::Training;
				} else if (mode == "kickoff") {
					scfg.spawns = Hive::SpectateSpawns::Kickoff;
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
			             "       [--time-scale X] [--episodes N] [--gpu]\n",
			             argv[0]);
			return EXIT_FAILURE;
		}
		try {
			Hive::RunSpectate(scfg);
		} catch (const std::exception& e) {
			std::fprintf(stderr, "Spectate failed: %s\n", e.what());
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	if (command == "-h" || command == "--help" || command == "help") {
		PrintUsage(argv[0]);
		return EXIT_SUCCESS;
	}

	std::fprintf(stderr, "Unknown command: %s\n\n", command.c_str());
	PrintUsage(argv[0]);
	return EXIT_FAILURE;
}
