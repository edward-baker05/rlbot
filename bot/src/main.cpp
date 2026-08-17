#include "Config.h"
#include "Verify.h"
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
		"\n"
		"Training options:\n"
		"  --games N            Number of simultaneous games (default 128)\n"
		"  --render             Stream to RocketSimVis instead of training at speed\n"
		"  --cpu                Train on CPU instead of CUDA\n"
		"  --no-metrics         Do not send metrics to wandb\n"
		"  --seed N             Random seed (default: clock)\n"
		"  --max-steps N        Stop after N timesteps (default: run until Q)\n"
		"  --label NAME         Suffix run and checkpoint names, to keep runs apart\n"
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

	if (command == "-h" || command == "--help" || command == "help") {
		PrintUsage(argv[0]);
		return EXIT_SUCCESS;
	}

	std::fprintf(stderr, "Unknown command: %s\n\n", command.c_str());
	PrintUsage(argv[0]);
	return EXIT_FAILURE;
}
