#include "Train.h"

#include "../env/Curriculum.h"
#include "../env/Env.h"
#include "../env/Obs.h"
#include "../env/PlayPhase.h"
#include "../env/Rewards.h"
#include "Metrics.h"

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/PPO/PPOLearner.h> // private GGL header; see bot/CMakeLists.txt
#include <RLGymCPP/CommonValues.h>
#include <RLGymCPP/EnvSet/EnvSet.h>
#include <RLGymCPP/Math.h>

#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>

using namespace GGL;
using namespace RLGC;

namespace Hive {

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
// These track HOW the bot earns reward, not just how much -- reward goes up
// in every run, but a healthy curve can still hide a collapsing touch height.
// Sampled on a fraction of steps, since iterating every player of every game
// every step is a real cost at 128 games.

// Step budget for bounded runs. Set from RunTraining(); 0 disables.
static int64_t g_MaxSteps = 0;

// Set by HandleSigint, checked in StepCallback. wandb's "Stop run" button
// sends SIGINT to this process; a plain Ctrl-C does too. Only async-signal-
// safe work happens in the handler itself -- the actual save/exit runs from
// StepCallback on the main thread.
static volatile std::sig_atomic_t g_StopRequested = 0;
static void HandleSigint(int) { g_StopRequested = 1; }

// (name, weight) per reward term, in spec order -- the same order EnvSet
// stores per-term rewards in. Set once in RunTraining() before the learner
// starts; no rewards are allocated for this.
static std::vector<std::pair<std::string, float>> g_RewardLabels;

// Needed to rebuild observations inside the metrics callback; EnvSetConfig does
// not carry it. Set once in RunTraining() before the learner starts.
static int g_MaxPlayersPerTeam = 1;

// Decision steps since each arena last reset, for the Episode/* buckets.
static std::vector<int> g_EpisodeAge;


// Runs on a fraction of sampled iterations: an extra critic forward pass is not
// free, and this exists to answer a question, not to run forever. Same thread as
// collection (Learner.cpp calls the step callback after envSet->Sync()), so
// touching the models here does not race the workers.
static void CriticValueMetrics(Learner* learner, const std::vector<GameState>& states, Report& report) {
	static int callCount = 0;
	if ((callCount++ % 8) != 0)
		return;

	auto obsBuilder = MakeObsBuilder(g_MaxPlayersPerTeam);

	// Flat [N, obsSize] buffer plus, per row, what it is so the values can be
	// bucketed after one batched forward pass.
	struct Tag {
		bool isPrev;   // row is the state the policy chose FROM
		bool grounded; // plain wheels-down, for the V(ground) vs V(air) split
		bool decision; // grounded AND upright AND jump legal: the jump choice
		bool jumped;   // and the policy pressed jump
	};

	std::vector<float> flat;
	std::vector<Tag> tags;
	int obsSize = 0;

	auto push = [&](const Player& p, const GameState& gs, Tag tag) {
		FList obs = obsBuilder->BuildObs(p, gs);
		if (obsSize == 0)
			obsSize = static_cast<int>(obs.size());
		if (static_cast<int>(obs.size()) != obsSize)
			return; // ragged obs would corrupt the batch; skip rather than guess
		flat.insert(flat.end(), obs.begin(), obs.end());
		tags.push_back(tag);
	};

	for (const GameState& state : states) {
		if (!state.prev)
			continue;

		for (const Player& player : state.players) {
			if (!player.prev)
				continue;
			const Player& before = *player.prev;

			const bool turtled =
				before.worldContact.hasContact && before.worldContact.contactNormal.z > 0.9f;
			const bool decision = before.isOnGround && before.rotMat.up.z > 0.7f &&
			                      (before.HasFlipOrJump() || turtled);

			const bool jumped = player.prevAction.jump != 0.f;
			push(before, *state.prev, {true, before.isOnGround, decision, jumped});
			push(player, state, {false, player.isOnGround, decision, jumped});
		}
	}

	if (tags.empty() || obsSize == 0)
		return;

	torch::NoGradGuard noGrad;
	const int rows = static_cast<int>(tags.size());
	torch::Tensor obs =
		torch::from_blob(flat.data(), {rows, obsSize}, torch::kFloat32).clone();
	torch::Tensor vals = learner->ppo->InferCritic(obs.to(learner->ppo->device)).cpu().flatten();
	const float* v = vals.const_data_ptr<float>();

	const float gamma = learner->config.ppo.gaeGamma;

	for (int i = 0; i + 1 < rows; i += 2) {
		// Rows were pushed in (before, after) pairs, so i is the chosen-from
		// state and i+1 is where it led.
		if (!tags[i].isPrev || tags[i + 1].isPrev)
			continue;

		report.AddAvg("Critic/V All", v[i]);
		// The plain split answers (a) directly: is being airborne worth more?
		report.AddAvg(tags[i].grounded ? "Critic/V Grounded" : "Critic/V Airborne", v[i]);

		if (tags[i].decision) {
			const float tdDelta = gamma * v[i + 1] - v[i];
			report.AddAvg(tags[i].jumped ? "Critic/TD Delta Jump" : "Critic/TD Delta NoJump",
			              tdDelta);
			report.AddAvg(tags[i].jumped ? "Critic/V After Jump" : "Critic/V After NoJump",
			              v[i + 1]);
		}
	}
}

// Save first, then _exit rather than return: unwinding out of a callback
// mid-collection would race the worker threads, and there is nothing left to
// clean up once the checkpoint is on disk.
static void SaveAndExit(Learner* learner, const char* reason) {
	std::cout << "\n" << reason << ". Saving and exiting.\n";
	std::cout.flush();
	learner->Save();
	if (learner->metricSender)
		learner->metricSender->Finish();
	std::_Exit(0);
}

static void StepCallback(Learner* learner, const std::vector<GameState>& states, Report& report) {
	// GigaLearn's training loop runs until the user presses Q; there is no
	// timestep limit and no documented way to break out of it. The step
	// callback is the only hook that runs inside the loop with access to the
	// learner, so the budget and the SIGINT check are both enforced here.
	if (g_StopRequested) {
		SaveAndExit(learner, "Interrupted (Ctrl-C or wandb Stop)");
	}
	if (g_MaxSteps > 0 && static_cast<int64_t>(learner->totalTimesteps) >= g_MaxSteps) {
		std::ostringstream reason;
		reason << "Reached step budget (" << learner->totalTimesteps << " >= " << g_MaxSteps << ")";
		SaveAndExit(learner, reason.str().c_str());
	}

	// --- Episode age --------------------------------------------------------
	// Tracked OUTSIDE the sampling gate below, because it has to count every
	// step to stay accurate. Buckets behaviour by time since spawn, so an
	// approach failure right after spawn is distinguishable from a recovery
	// failure later in the episode.
	{
		auto& es = learner->envSet->state;
		if (g_EpisodeAge.size() != states.size())
			g_EpisodeAge.assign(states.size(), 0);
		for (size_t a = 0; a < states.size(); a++) {
			// Terminals are flagged for the step that ended the episode; the
			// reset happens after. Count first, then zero, so the final step of
			// an episode is still attributed to that episode.
			g_EpisodeAge[a]++;
			if (a < es.terminals.size() && es.terminals[a])
				g_EpisodeAge[a] = 0;
		}
	}

	// Sample roughly a quarter of steps. Averages over an iteration are just
	// as accurate and cost a quarter as much.
	const bool sample = (rand() % 4) == 0;
	if (!sample)
		return;

	PhaseCounts phases;

	for (size_t arenaIdx = 0; arenaIdx < states.size(); arenaIdx++) {
		const GameState& state = states[arenaIdx];
		// Buckets are decision steps at 15 Hz: the first second off the spawn,
		// the next three, then everything after.
		const int age = arenaIdx < g_EpisodeAge.size() ? g_EpisodeAge[arenaIdx] : 0;
		const char* ageBucket = (age < 15) ? "Early" : (age < 60 ? "Mid" : "Late");

		for (const Player& player : state.players) {
			// --- Play phase distribution -------------------------------------
			// Shows what the policy actually spends its time doing. If you
			// bump the aerial weight in the curriculum and the Aerial share
			// does not move, the setter is not doing what you think it is.
			const PlayPhase phase = ClassifyPhase(player, state);
			phases.Add(phase);

			// --- Core behaviour ----------------------------------------------
			report.AddAvg("Player/In Air Ratio", !player.isOnGround);
			report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
			report.AddAvg("Player/Demoed Ratio", player.isDemoed);
			report.AddAvg("Player/Speed", player.vel.Length());
			report.AddAvg("Player/Boost", player.boost);

			const Vec toBall = state.ball.pos - player.pos;
			const float dist = toBall.Length();
			if (dist > 1.f)
				report.AddAvg("Player/Speed Towards Ball", RS_MAX(0.f, player.vel.Dot(toBall / dist)));

			// Same three quantities, split by how old the episode is. If the
			// Early numbers are strong and Mid/Late collapse, the bot can
			// approach a ball exactly once per spawn.
			report.AddAvg(std::string("Episode/") + ageBucket + "/Touch Rate",
			              player.ballTouchedStep ? 1.f : 0.f);
			report.AddAvg(std::string("Episode/") + ageBucket + "/In Air Ratio", !player.isOnGround);
			report.AddAvg(std::string("Episode/") + ageBucket + "/Ball Dist", dist);
			if (dist > 1.f)
				report.AddAvg(std::string("Episode/") + ageBucket + "/Approach Speed",
				              RS_MAX(0.f, player.vel.Dot(toBall / dist)));

			// Touch height is the clearest single indicator of whether the bot
			// is developing an air game. Watch it more than the reward.
			if (player.ballTouchedStep)
				report.AddAvg("Player/Touch Height", state.ball.pos.z);

			// --- Touch distribution -------------------------------------------
			// Distribution rather than the mean touch height (~147 measured):
			// the mean hides the thing that actually matters, which is whether
			// ANY touches are happening in the jump-only band at all. None of
			// these depend on state.prev -- they read the current touch and
			// grounded state, not a velocity delta.
			{
				const bool air = !player.isOnGround;
				if (player.ballTouchedStep) {
					const float h = state.ball.pos.z;
					report.AddAvg("Touch/Above 200", h > 200.f ? 1.f : 0.f);
					report.AddAvg("Touch/Above 300", h > 300.f ? 1.f : 0.f);
					report.AddAvg("Touch/Above 450", h > 450.f ? 1.f : 0.f);

					// Did it get there with a jump, and did it flip into the
					// ball? This is the target skill, stated as a metric.
					report.AddAvg("Touch/Had Jumped", player.hasJumped ? 1.f : 0.f);
					report.AddAvg("Touch/Had Flipped", player.hasFlipped ? 1.f : 0.f);
				}
				report.AddAvg(air ? "Touch/Rate Airborne" : "Touch/Rate Grounded",
				              player.ballTouchedStep ? 1.f : 0.f);
			}

			// --- What the policy actually DID --------------------------------
			// Everything above is a state statistic: it says where the car
			// ended up, not what the policy chose. "In Air Ratio 0.91" is
			// consistent with a policy that jumps constantly AND with one that
			// never jumps but keeps getting launched, which need opposite fixes.
			//
			// prevAction is the action applied during this step, so it must be
			// conditioned on the PREVIOUS state -- that is the state the policy
			// saw when it chose. Without prev there is no decision to attribute.
			if (!player.prev)
				continue;

			const Player& before = *player.prev;

			// Mirrors DefaultAction::GetActionMask: jump actions are offered
			// while a flip/jump remains, and also while turtled (upside down),
			// which is how a stuck car rights itself. If jump was not on the
			// menu, the step says nothing about whether the policy wants it.
			const bool turtled =
				before.worldContact.hasContact && before.worldContact.contactNormal.z > 0.9f;
			const bool couldJump = before.HasFlipOrJump() || turtled;

			// A car resting upside down on the floor is "grounded" and jump is
			// the correct way out of it, so an upright split is needed before
			// a high grounded jump rate can be read as a farm rather than as
			// recovery. Wheels-down is rotMat.up.z near +1.
			const bool upright = before.rotMat.up.z > 0.7f;

			if (couldJump) {
				if (before.isOnGround)
					report.AddAvg(upright ? "Action/Jump When Grounded Upright"
					                      : "Action/Jump When Grounded Inverted",
					              player.prevAction.jump);
				else
					report.AddAvg("Action/Jump When Airborne", player.prevAction.jump);
			}

			// --- Is it just standing there? ----------------------------------
			// The old stack had a flat `Grounded` bonus that made standing
			// still a risk-free annuity; that term is gone (deleted with
			// GroundedBonusReward), but the ratio is still worth watching --
			// SpeedSquaredReward pays zero for a motionless car, so a nonzero
			// stationary share now means the *rest* of the stack isn't moving
			// the policy off it either.
			if (player.isOnGround) {
				const float sp = player.vel.Length();
				report.AddAvg("Player/Grounded Speed", sp);
				report.AddAvg("Player/Grounded Stationary Ratio", sp < 200.f ? 1.f : 0.f);
			}

			// --- Is it actually driving? -------------------------------------
			// Added while diagnosing a policy that boosts in straight lines and
			// never turns. Everything else here measures resulting STATE; these
			// measure the grounded control inputs directly, because "it does not
			// steer" and "it steers but cannot hold a line" look identical from
			// speed and position alone.
			//
			// Split by whether boost is even available: DefaultAction masks out
			// every boost action at zero boost, so a raw boost rate conflates
			// "chose not to boost" with "could not".
			if (before.isOnGround && upright) {
				report.AddAvg("Action/Steer Nonzero", player.prevAction.steer != 0.f ? 1.f : 0.f);
				report.AddAvg("Action/Throttle Forward", player.prevAction.throttle > 0.f ? 1.f : 0.f);
				report.AddAvg("Action/Throttle Zero", player.prevAction.throttle == 0.f ? 1.f : 0.f);
				report.AddAvg("Action/Handbrake", player.prevAction.handbrake);
				if (before.boost > 0.f)
					report.AddAvg("Action/Boost When Available", player.prevAction.boost);
			}

			// --- What KIND of flip? ------------------------------------------
			// DefaultAction's jump entries always have yaw == 0 (jump+yaw
			// combinations are skipped when the table is built), so a diagonal
			// flip is pitch and roll together; a straight flip is one axis.
			if (player.prevAction.jump != 0.f) {
				const bool pitching = player.prevAction.pitch != 0.f;
				const bool rolling = player.prevAction.roll != 0.f;
				report.AddAvg("Flip/Diagonal Share", (pitching && rolling) ? 1.f : 0.f);
				report.AddAvg("Flip/Neutral Share", (!pitching && !rolling) ? 1.f : 0.f);

				// airTimeSinceJump is the gap between leaving the ground and
				// now, so on the step a second jump is pressed in the air it IS
				// the flip delay. A deliberate stall shows up as a delay well
				// past the ~0.1s a reflexive double-jump would give.
				if (!before.isOnGround && before.hasJumped)
					report.AddAvg("Flip/Delay Seconds", before.airTimeSinceJump);
			}
			if (before.isOnGround)
				report.AddAvg("Player/Grounded Upright Ratio", upright);

			report.AddAvg(before.isOnGround ? "Action/Boost When Grounded"
			                                : "Action/Boost When Airborne",
			              player.prevAction.boost);

			// Where does the air time come from? Of every ground->air
			// transition, how many did the policy cause by pressing jump, as
			// opposed to driving off a ramp, being bumped, or a curriculum
			// spawn. If this is low the flip-spam reading is simply wrong.
			if (before.isOnGround) {
				report.AddAvg("Player/Leave Ground Rate", !player.isOnGround);
				if (!player.isOnGround)
					report.AddAvg("Player/Takeoff Was Jump", player.prevAction.jump);
			}

			// How long a single airborne stint lasts, in seconds. Pairs with
			// Leave Ground Rate: together they say whether 91% air time is many
			// short hops or a few very long tumbles.
			if (!player.isOnGround)
				report.AddAvg("Player/Air Time", player.airTime);

			// --- What does a flip actually buy? ------------------------------
			// Landing vs sustained split: tests whether a flip is a speed pump
			// or just a heading randomizer. Not a clean counterfactual --
			// "sustained" cars have had time to accelerate -- but it bounds the
			// effect.
			if (player.isOnGround && dist > 1.f) {
				const float towards = RS_MAX(0.f, player.vel.Dot(toBall / dist));
				const bool landed = !before.isOnGround;
				report.AddAvg(landed ? "Player/Approach Speed On Landing"
				                     : "Player/Approach Speed Sustained", towards);
				report.AddAvg(landed ? "Player/Speed On Landing"
				                     : "Player/Speed Sustained", player.vel.Length());
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());

		report.AddAvg("Game/Ball Height", state.ball.pos.z);
		report.AddAvg("Game/Players", static_cast<float>(state.players.size()));
	}


	// Report each phase as a fraction of sampled player-steps.
	const int64_t total = phases.Total();
	if (total > 0) {
		for (int i = 0; i < PLAY_PHASE_COUNT; i++) {
			const auto phase = static_cast<PlayPhase>(i);
			report.AddAvg(std::string("Phase/") + PlayPhaseName(phase),
			              static_cast<float>(phases.counts[i]) / static_cast<float>(total));
		}
	}

	// --- Reward shares ------------------------------------------------------
	// lastRewards holds each term's raw (unweighted, pre-zero-sum) reward for
	// one sampled player per arena; |r * w| across terms approximates where
	// the realized reward mass is going. This is the farming detector.
	auto& envSet = *learner->envSet;
	if (!g_RewardLabels.empty()) {
		std::vector<float> totals(g_RewardLabels.size(), 0.f);
		bool any = false;
		for (size_t a = 0; a < envSet.state.lastRewards.size(); a++) {
			const auto& last = envSet.state.lastRewards[a];
			if (last.size() != totals.size())
				continue;
			for (size_t j = 0; j < totals.size(); j++)
				totals[j] += std::fabs(last[j] * g_RewardLabels[j].second);
			any = true;
		}
		if (any) {
			auto shares = NormalizeShares(totals);
			for (size_t j = 0; j < shares.size(); j++)
				report.AddAvg("RewardShare/" + g_RewardLabels[j].first, shares[j]);
		}
	}

	// --- Scenario outcomes --------------------------------------------------
	// Terminal arenas have not been reset yet at callback time, so the
	// curriculum's last-picked name still labels the episode that just ended.
	std::map<std::string, int> scenarioCounts;
	for (size_t a = 0; a < envSet.stateSetters.size(); a++) {
		auto* cs = dynamic_cast<CurriculumState*>(envSet.stateSetters[a]);
		if (!cs || cs->LastPickedName().empty())
			continue;
		if (scenarioCounts.empty()) {
			// Seed every configured scenario with zero so names not picked
			// this step still contribute a sample; otherwise rare scenarios'
			// Share averages are biased upward.
			for (const auto& name : cs->EntryNames())
				scenarioCounts[name] = 0;
		}
		scenarioCounts[cs->LastPickedName()]++;
		if (envSet.state.terminals[a]) {
			const bool goal = states[a].goalScored;
			report.AddAvg("Scenario/" + cs->LastPickedName() + "/EndedInGoal", goal ? 1.f : 0.f);
		}
	}
	if (!envSet.stateSetters.empty()) {
		// A true share: count per name across all arenas, so a scenario that
		// never runs is distinguishable from one that always does.
		const float arenaCount = static_cast<float>(envSet.stateSetters.size());
		for (const auto& [name, count] : scenarioCounts)
			report.AddAvg("Scenario/" + name + "/Share", static_cast<float>(count) / arenaCount);
	}

	// --- What does the CRITIC think? ----------------------------------------
	// V(grounded) vs V(airborne) says whether the critic favours air time on
	// its own, independent of what the reward does. The TD split goes further:
	// out of a grounded state, does bootstrapping through the jump action look
	// better than not jumping? Note the TD figure omits the immediate reward
	// (it's gamma*V(s') - V(s), not the full residual), so read it as "where
	// does jumping take me", not as an advantage.
	CriticValueMetrics(learner, states, report);
}

// ---------------------------------------------------------------------------

void RunTraining(const TrainConfig& cfg) {
	// RocketSim needs the collision meshes to simulate the arena geometry.
	// Without them cars fall through the world, which presents as a bot that
	// learns nothing rather than as an obvious error.
	const char* meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	const std::string meshPath = meshEnv ? meshEnv : "collision_meshes";
	RocketSim::Init(meshPath);

	// Probe the observation width rather than deriving it. See env/Obs.h.
	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam);
	std::cout << "Observation size: " << obsSize
	          << " (maxPlayersPerTeam=" << cfg.maxPlayersPerTeam << ")\n";
	std::cout << "Run:              " << cfg.RunName() << "\n";
	std::cout << "Checkpoints:      " << cfg.CheckpointFolder() << "\n";
	std::cout << "Self-play:        "
	          << (cfg.selfPlay.trainAgainstOldVersions
	                  ? "on (" + std::to_string(static_cast<int>(cfg.selfPlay.trainAgainstOldChance * 100)) +
	                        "% of iterations, snapshot every " +
	                        std::to_string(cfg.selfPlay.tsPerVersion / 1'000'000) + "M steps)"
	                  : "off")
	          << "\n";
	std::cout << "Skill tracking:   " << (cfg.selfPlay.trackSkill ? "on" : "off") << "\n";
	if (cfg.maxSteps > 0)
		std::cout << "Step budget:      " << cfg.maxSteps << "\n";

	g_MaxSteps = cfg.maxSteps;
	g_MaxPlayersPerTeam = cfg.maxPlayersPerTeam;

	g_RewardLabels.clear();
	for (auto& s : GeneralRewardSpecs(cfg))
		g_RewardLabels.push_back({s.name, s.weight});

	LearnerConfig lc = {};

	lc.deviceType = cfg.useGPU ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;
	lc.numGames = cfg.numGames;
	lc.tickSkip = cfg.tickSkip;
	lc.actionDelay = cfg.actionDelay;
	lc.randomSeed = cfg.randomSeed;

	lc.checkpointFolder = cfg.CheckpointFolder();
	lc.tsPerSave = cfg.tsPerSave;
	lc.checkpointsToKeep = cfg.checkpointsToKeep;

	lc.ppo.tsPerItr = cfg.tsPerItr;
	lc.ppo.batchSize = cfg.tsPerItr;
	lc.ppo.miniBatchSize = cfg.miniBatchSize;
	lc.ppo.epochs = cfg.epochs;
	lc.ppo.entropyScale = cfg.entropyScale;
	lc.ppo.gaeGamma = cfg.gaeGamma;
	lc.ppo.policyLR = cfg.policyLR;
	lc.ppo.criticLR = cfg.criticLR;

	// The policy and shared-head shapes here MUST match ModelShape in
	// Config.h, because that is what the RLBot client rebuilds at load time.
	// Mismatch means the deployed bot silently loads garbage weights.
	lc.ppo.sharedHead.layerSizes = cfg.modelShape.sharedHeadLayers;
	lc.ppo.sharedHead.activationType = cfg.modelShape.activation;
	lc.ppo.sharedHead.addLayerNorm = cfg.modelShape.addLayerNorm;
	lc.ppo.sharedHead.addOutputLayer = false;

	lc.ppo.policy.layerSizes = cfg.modelShape.policyLayers;
	lc.ppo.policy.activationType = cfg.modelShape.activation;
	lc.ppo.policy.addLayerNorm = cfg.modelShape.addLayerNorm;

	// The critic is training-only, so it never has to match the client. Give it
	// the same shape as the policy; there is rarely a reason to differ.
	lc.ppo.critic.layerSizes = cfg.modelShape.policyLayers;
	lc.ppo.critic.activationType = cfg.modelShape.activation;
	lc.ppo.critic.addLayerNorm = cfg.modelShape.addLayerNorm;

	// --- Self-play ----------------------------------------------------------
	// The learner forces savePolicyVersions on if either of these is enabled,
	// since both need the version pool. Setting it explicitly documents the
	// dependency rather than relying on that.
	lc.trainAgainstOldVersions = cfg.selfPlay.trainAgainstOldVersions;
	lc.trainAgainstOldChance = cfg.selfPlay.trainAgainstOldChance;
	lc.savePolicyVersions = cfg.selfPlay.trainAgainstOldVersions || cfg.selfPlay.trackSkill;
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

	// Capture cfg by value: the learner calls this for every game at startup,
	// and outliving the caller's stack frame is not worth risking.
	auto envCreateFn = [cfg](int index) -> EnvCreateResult {
		return CreateEnv(index, cfg);
	};

	Learner learner(envCreateFn, lc, StepCallback);

	// Installed after the Learner (and its embedded Python interpreter) is
	// constructed, since Python init can otherwise install its own handler.
	std::signal(SIGINT, HandleSigint);

	learner.Start();
}

} // namespace Hive
