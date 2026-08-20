#include "Train.h"

#include "../env/Curriculum.h"
#include "../env/Env.h"
#include "../env/Obs.h"
#include "../env/PlayPhase.h"
#include "../env/Rewards.h"
#include "../env/StateSetters.h"
#include "Metrics.h"

#include <GigaLearnCPP/Learner.h>
#include <GigaLearnCPP/PPO/PPOLearner.h>
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

static int64_t g_MaxSteps = 0;

static volatile std::sig_atomic_t g_StopRequested = 0;
static void HandleSigint(int) { g_StopRequested = 1; }

static std::vector<std::pair<std::string, float>> g_RewardLabels;

static int g_MaxPlayersPerTeam = 1;
static ObsMode g_ObsMode = ObsMode::Relative;
static float g_TouchAccelExponent = 2.f;
static float g_AirTouchDirectionExponent = 1.f;
static std::vector<int> g_EpisodeAge;

static void CriticValueMetrics(Learner *learner,
							   const std::vector<GameState> &states,
							   Report &report) {
	static int callCount = 0;
	if ((callCount++ % 8) != 0)
		return;

	auto obsBuilder = MakeObsBuilder(g_MaxPlayersPerTeam, g_ObsMode);

	struct Tag {
		bool isPrev;
		bool grounded;
		bool decision;
		bool jumped;
	};

	std::vector<float> flat;
	std::vector<Tag> tags;
	int obsSize = 0;

	auto push = [&](const Player &p, const GameState &gs, Tag tag) {
		FList obs = obsBuilder->BuildObs(p, gs);
		if (obsSize == 0)
			obsSize = static_cast<int>(obs.size());
		if (static_cast<int>(obs.size()) != obsSize)
			return;

		flat.insert(flat.end(), obs.begin(), obs.end());
		tags.push_back(tag);
	};

	for (const GameState &state : states) {
		if (!state.prev)
			continue;

		for (const Player &player : state.players) {
			if (!player.prev)
				continue;
			const Player &before = *player.prev;

			const bool turtled = before.worldContact.hasContact &&
								 before.worldContact.contactNormal.z > 0.9f;
			const bool decision = before.isOnGround &&
								  before.rotMat.up.z > 0.7f &&
								  (before.HasFlipOrJump() || turtled);

			const bool jumped = player.prevAction.jump != 0.f;
			push(before, *state.prev,
				 {true, before.isOnGround, decision, jumped});
			push(player, state, {false, player.isOnGround, decision, jumped});
		}
	}

	if (tags.empty() || obsSize == 0)
		return;

	torch::NoGradGuard noGrad;
	const int rows = static_cast<int>(tags.size());
	torch::Tensor obs =
		torch::from_blob(flat.data(), {rows, obsSize}, torch::kFloat32).clone();
	torch::Tensor vals =
		learner->ppo->InferCritic(obs.to(learner->ppo->device)).cpu().flatten();
	const float *v = vals.const_data_ptr<float>();

	const float gamma = learner->config.ppo.gaeGamma;

	for (int i = 0; i + 1 < rows; i += 2) {
		if (!tags[i].isPrev || tags[i + 1].isPrev)
			continue;

		report.AddAvg("Critic/V All", v[i]);

		report.AddAvg(
			tags[i].grounded ? "Critic/V Grounded" : "Critic/V Airborne", v[i]);

		if (tags[i].decision) {
			const float tdDelta = gamma * v[i + 1] - v[i];
			report.AddAvg(tags[i].jumped ? "Critic/TD Delta Jump"
										 : "Critic/TD Delta NoJump",
						  tdDelta);
			report.AddAvg(tags[i].jumped ? "Critic/V After Jump"
										 : "Critic/V After NoJump",
						  v[i + 1]);
		}
	}
}

static void SaveAndExit(Learner *learner, const char *reason) {
	std::cout << "\n" << reason << ". Saving and exiting.\n";
	std::cout.flush();
	learner->Save();
	if (learner->metricSender)
		learner->metricSender->Finish();
	std::_Exit(0);
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

	{
		auto &es = learner->envSet->state;
		if (g_EpisodeAge.size() != states.size())
			g_EpisodeAge.assign(states.size(), 0);
		for (size_t a = 0; a < states.size(); a++) {
			g_EpisodeAge[a]++;
			if (a < es.terminals.size() && es.terminals[a]) {
				report.AddAvg("Episode/Mean Steps",
							  static_cast<float>(g_EpisodeAge[a]));
				g_EpisodeAge[a] = 0;
			}
		}
	}

	const bool sample = (rand() % 4) == 0;
	if (!sample)
		return;

	PhaseCounts phases;

	for (size_t arenaIdx = 0; arenaIdx < states.size(); arenaIdx++) {
		const GameState &state = states[arenaIdx];

		const int age =
			arenaIdx < g_EpisodeAge.size() ? g_EpisodeAge[arenaIdx] : 0;
		const char *ageBucket =
			(age < 15) ? "Early" : (age < 60 ? "Mid" : "Late");

		for (const Player &player : state.players) {
			const PlayPhase phase = ClassifyPhase(player, state);
			phases.Add(phase);

			report.AddAvg("Player/In Air Ratio", !player.isOnGround);
			report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);
			report.AddAvg("Player/Demoed Ratio", player.isDemoed);
			report.AddAvg("Player/Speed", player.vel.Length());
			report.AddAvg("Player/Boost", player.boost);

			const Vec toBall = state.ball.pos - player.pos;
			const float dist = toBall.Length();
			if (dist > 1.f)
				report.AddAvg("Player/Speed Towards Ball",
							  RS_MAX(0.f, player.vel.Dot(toBall / dist)));

			const bool wrongSurface =
				player.worldContact.hasContact && !player.isOnGround;
			report.AddAvg("Surface/Wrong Contact Rate",
						  wrongSurface ? 1.f : 0.f);

			if (player.isFlipping)
				report.AddAvg("Surface/Wrong Contact While Flipping",
							  wrongSurface ? 1.f : 0.f);

			if (player.prev && player.isOnGround && !player.prev->isOnGround) {
				report.AddAvg("Landing/Rate", 1.f);
				report.AddAvg("Landing/Clean Share",
							  player.worldContact.hasContact ? 0.f : 1.f);
				report.AddAvg("Landing/Impact Speed",
							  RS_MAX(0.f, -player.prev->vel.z));
			} else {
				report.AddAvg("Landing/Rate", 0.f);
			}

			const float speed = player.vel.Length();
			report.AddAvg("Speed/Mean", speed);

			report.AddAvg("Speed/Above Throttle Cap Share",
						  speed > THROTTLE_TOP_SPEED ? 1.f : 0.f);

			if (player.prev) {
				const float lost = player.prev->vel.Length() - speed;

				if (!player.ballTouchedStep) {
					report.AddAvg("Speed/Mean Step Decel", RS_MAX(0.f, lost));
					report.AddAvg("Speed/Decel Above 200",
								  lost > 200.f ? 1.f : 0.f);
					report.AddAvg("Speed/Decel Above 400",
								  lost > HARSH_LOSS_THRESHOLD ? 1.f : 0.f);
					report.AddAvg("Speed/Decel Above 800",
								  lost > 800.f ? 1.f : 0.f);
				}
			}

			if (dist > 1.f) {
				const Vec dirToBall = toBall / dist;
				const float cosToBall = player.rotMat.forward.Dot(dirToBall);
				report.AddAvg("FaceBall/Mean Cos", cosToBall);
				report.AddAvg("FaceBall/Axis Share", std::fabs(cosToBall));

				report.AddAvg("FaceBall/Rectified", RS_MAX(0.f, cosToBall));
				report.AddAvg(player.isOnGround ? "FaceBall/Rectified Grounded"
												: "FaceBall/Rectified Airborne",
							  RS_MAX(0.f, cosToBall));

				const float sp = player.vel.Length();
				if (sp > 1.f)
					report.AddAvg("Player/Velocity Alignment",
								  RS_MAX(0.f, player.vel.Dot(dirToBall)) / sp);
			}

			report.AddAvg("Player/Ball Far Share",
						  dist >= FlipSpeedReward::MIN_BALL_DIST ? 1.f : 0.f);

			report.AddAvg("Touch/Edge Rate",
						  (player.ballTouchedStep &&
						   !(player.prev && player.prev->ballTouchedStep))
							  ? 1.f
							  : 0.f);

			report.AddAvg(std::string("Episode/") + ageBucket + "/Touch Rate",
						  player.ballTouchedStep ? 1.f : 0.f);
			report.AddAvg(std::string("Episode/") + ageBucket + "/In Air Ratio",
						  !player.isOnGround);
			report.AddAvg(std::string("Episode/") + ageBucket + "/Ball Dist",
						  dist);
			if (dist > 1.f)
				report.AddAvg(std::string("Episode/") + ageBucket +
								  "/Approach Speed",
							  RS_MAX(0.f, player.vel.Dot(toBall / dist)));

			if (player.ballTouchedStep)
				report.AddAvg("Player/Touch Height", state.ball.pos.z);

			{
				const bool air = !player.isOnGround;
				if (player.ballTouchedStep) {
					const float h = state.ball.pos.z;
					report.AddAvg("Touch/Above 200", h > 200.f ? 1.f : 0.f);
					report.AddAvg("Touch/Above 300", h > 300.f ? 1.f : 0.f);
					report.AddAvg("Touch/Above 450", h > 450.f ? 1.f : 0.f);

					report.AddAvg("Touch/Had Jumped",
								  player.hasJumped ? 1.f : 0.f);
					report.AddAvg("Touch/Had Flipped",
								  player.hasFlipped ? 1.f : 0.f);

					// The air-direction readout, gated on the same rising edge
					// and airborne condition AirTouchReward pays on, and
					// computed through the same helper so the metric cannot
					// drift from the term it audits.
					//
					// Backward Share has a null: a policy indifferent to
					// direction reads 0.5, and Direction Factor reads 0.5 with
					// it. Anything at those values means the direction factor
					// is buying nothing.
					if (!(player.prev && player.prev->ballTouchedStep) &&
						player.airTime > 0.f) {
						const float goalward =
							BallGoalwardCos(state, player.team);
						report.AddAvg("AirTouch/Goalward Cos", goalward);
						report.AddAvg(
							"AirTouch/Direction Factor",
							GoalwardFactor(goalward,
										   g_AirTouchDirectionExponent));
						report.AddAvg("AirTouch/Backward Share",
									  goalward < 0.f ? 1.f : 0.f);
					}

					// The accuracy readout. Gated on the same rising edge as
					// ShotOnTargetReward via the shared projection, so the
					// metric cannot disagree with the reward it audits.
					if (!(player.prev && player.prev->ballTouchedStep)) {
						const ShotProjection shot =
							ProjectShot(state, player.team);
						report.AddAvg("Shot/Toward Net Rate",
									  shot.valid ? 1.f : 0.f);
						if (shot.valid) {
							// What ShotOnTargetReward's strength factor is
							// actually worth. The budget was sized against an
							// ASSUMED mean strength of 0.25; this is the
							// number that replaces the assumption.
							report.AddAvg(
								"Shot/Strength",
								RS_MAX(0.f, GoalwardDeltaV(state, player.team)));
							report.AddAvg("Shot/Ball Speed",
										  state.ball.vel.Length());
							report.AddAvg("Shot/On Target Share",
										  shot.missDist <= 0.f ? 1.f : 0.f);
							report.AddAvg("Shot/Miss Distance", shot.missDist);
							// A post shot in the sense the bot is accused of:
							// aimed at the frame, not the net.
							report.AddAvg("Shot/Post Share",
										  (shot.missDist > 0.f &&
										   shot.missDist <= 200.f)
											  ? 1.f
											  : 0.f);
						}
					}

					if (state.prev) {
						const float hitForce =
							(state.ball.vel - state.prev->ball.vel).Length();
						report.AddAvg("Touch/Hit Force", hitForce);

						{
							const float x = GoalwardDeltaV(state, player.team);
							report.AddAvg("Touch/Goal Accel Raw", x);
							report.AddAvg(
								"Touch/Goal Accel Value",
								std::copysign(std::pow(std::fabs(x),
													   g_TouchAccelExponent),
											  x));
						}
						report.AddAvg(
							"Touch/Strong Value",
							hitForce < RLGC::Math::KPHToVel(20)
								? 0.f
								: RS_MIN(1.f,
										 hitForce / RLGC::Math::KPHToVel(130)));
					}
				}
				report.AddAvg(air ? "Touch/Rate Airborne"
								  : "Touch/Rate Grounded",
							  player.ballTouchedStep ? 1.f : 0.f);
			}

			if (!player.prev)
				continue;

			const Player &before = *player.prev;

			const bool turtled = before.worldContact.hasContact &&
								 before.worldContact.contactNormal.z > 0.9f;
			const bool couldJump = before.HasFlipOrJump() || turtled;
			const bool upright = before.rotMat.up.z > 0.7f;

			if (before.isOnGround)
				report.AddAvg("Player/Grounded Tilted Ratio",
							  upright ? 0.f : 1.f);

			if (couldJump) {
				if (before.isOnGround)
					report.AddAvg(upright ? "Action/Jump When Grounded Upright"
										  : "Action/Jump When Grounded Tilted",
								  player.prevAction.jump);
				else
					report.AddAvg("Action/Jump When Airborne",
								  player.prevAction.jump);
			}

			if (player.isOnGround) {
				const float sp = player.vel.Length();
				report.AddAvg("Player/Grounded Speed", sp);
				report.AddAvg("Player/Grounded Stationary Ratio",
							  sp < 200.f ? 1.f : 0.f);
			}

			if (before.isOnGround && upright) {
				report.AddAvg("Action/Steer Nonzero",
							  player.prevAction.steer != 0.f ? 1.f : 0.f);
				report.AddAvg("Action/Throttle Forward",
							  player.prevAction.throttle > 0.f ? 1.f : 0.f);
				report.AddAvg("Action/Throttle Zero",
							  player.prevAction.throttle == 0.f ? 1.f : 0.f);
				report.AddAvg("Action/Handbrake", player.prevAction.handbrake);
				if (before.boost > 0.f)
					report.AddAvg("Action/Boost When Available",
								  player.prevAction.boost);
			}

			if (player.isFlipping && !before.isFlipping) {
				const float ballDist = (state.ball.pos - player.pos).Length();
				const bool travel = ballDist >= FlipSpeedReward::MIN_BALL_DIST;
				report.AddAvg("Flip/Travel Share", travel ? 1.f : 0.f);
				report.AddAvg("Flip/Ball Dist", ballDist);
			}

			if (player.prevAction.jump != 0.f) {
				const bool pitching = player.prevAction.pitch != 0.f;
				const bool rolling = player.prevAction.roll != 0.f;
				report.AddAvg("Flip/Diagonal Share",
							  (pitching && rolling) ? 1.f : 0.f);
				report.AddAvg("Flip/Neutral Share",
							  (!pitching && !rolling) ? 1.f : 0.f);

				report.AddAvg("Flip/Pitch Only Share",
							  (pitching && !rolling) ? 1.f : 0.f);
				report.AddAvg("Flip/Roll Only Share",
							  (!pitching && rolling) ? 1.f : 0.f);

				if (!before.isOnGround && before.hasJumped)
					report.AddAvg("Flip/Delay Seconds",
								  before.airTimeSinceJump);
			}
			if (before.isOnGround)
				report.AddAvg("Player/Grounded Upright Ratio", upright);

			report.AddAvg(before.isOnGround ? "Action/Boost When Grounded"
											: "Action/Boost When Airborne",
						  player.prevAction.boost);

			if (before.isOnGround) {
				report.AddAvg("Player/Leave Ground Rate", !player.isOnGround);
				if (!player.isOnGround)
					report.AddAvg("Player/Takeoff Was Jump",
								  player.prevAction.jump);
			}

			if (!player.isOnGround)
				report.AddAvg("Player/Air Time", player.airTime);

			if (player.isOnGround && dist > 1.f) {
				const float towards =
					RS_MAX(0.f, player.vel.Dot(toBall / dist));
				const bool landed = !before.isOnGround;
				report.AddAvg(landed ? "Player/Approach Speed On Landing"
									 : "Player/Approach Speed Sustained",
							  towards);
				report.AddAvg(landed ? "Player/Speed On Landing"
									 : "Player/Speed Sustained",
							  player.vel.Length());
			}
		}

		if (state.goalScored)
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());

		report.AddAvg("Game/Ball Height", state.ball.pos.z);
		report.AddAvg("Game/Players", static_cast<float>(state.players.size()));
	}

	const int64_t total = phases.Total();
	if (total > 0) {
		for (int i = 0; i < PLAY_PHASE_COUNT; i++) {
			const auto phase = static_cast<PlayPhase>(i);
			report.AddAvg(std::string("Phase/") + PlayPhaseName(phase),
						  static_cast<float>(phases.counts[i]) /
							  static_cast<float>(total));
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
			// RewardShare normalizes BEFORE averaging, so it reports
			// E[x/sum] rather than E[x]/E[sum]. For a spiky event term the
			// same sample that raises the numerator raises the denominator,
			// which truncates the spike and biases the share DOWN -- measured
			// at up to 2.9x against the raw `Rewards/*` means on p15's event
			// terms. The share is kept for continuity with the run log;
			// RewardMass is the one to solve budgets against.
			for (size_t j = 0; j < totals.size(); j++)
				report.AddAvg("RewardMass/" + g_RewardLabels[j].first,
							  totals[j] / static_cast<float>(
											  envSet.state.lastRewards.size()));

			auto shares = NormalizeShares(totals);
			for (size_t j = 0; j < shares.size(); j++)
				report.AddAvg("RewardShare/" + g_RewardLabels[j].first,
							  shares[j]);
		}
	}

	{
		const ObsHealth health = ConsumeObsHealth();
		if (health.checked > 0)
			report.AddAvg("Obs/Non-Finite Rate",
						  static_cast<float>(health.nonFinite) /
							  static_cast<float>(health.checked));
	}

	for (size_t a = 0; a < envSet.stateSetters.size(); a++) {
		auto *ib = dynamic_cast<InfiniteBoostState *>(envSet.stateSetters[a]);
		if (ib)
			report.AddAvg("Boost/Infinite Episode Share",
						  ib->LastWasInfinite() ? 1.f : 0.f);
	}

	std::map<std::string, int> scenarioCounts;
	for (size_t a = 0; a < envSet.stateSetters.size(); a++) {
		auto *cs = dynamic_cast<CurriculumState *>(envSet.stateSetters[a]);
		if (!cs || cs->LastPickedName().empty())
			continue;
		if (scenarioCounts.empty()) {
			for (const auto &name : cs->EntryNames())
				scenarioCounts[name] = 0;
		}
		scenarioCounts[cs->LastPickedName()]++;
		if (envSet.state.terminals[a]) {
			const bool goal = states[a].goalScored;
			report.AddAvg("Scenario/" + cs->LastPickedName() + "/EndedInGoal",
						  goal ? 1.f : 0.f);
		}
	}
	if (!envSet.stateSetters.empty()) {
		const float arenaCount = static_cast<float>(envSet.stateSetters.size());
		for (const auto &[name, count] : scenarioCounts)
			report.AddAvg("Scenario/" + name + "/Share",
						  static_cast<float>(count) / arenaCount);
	}

	CriticValueMetrics(learner, states, report);
}

void RunTraining(const TrainConfig &cfg) {
	const char *meshEnv = std::getenv("HIVE_COLLISION_MESHES");
	const std::string meshPath = meshEnv ? meshEnv : "collision_meshes";
	RocketSim::Init(meshPath);

	const int obsSize = ProbeObsSize(cfg.maxPlayersPerTeam, cfg.obs);
	std::cout << "Observation size: " << obsSize
			  << " (maxPlayersPerTeam=" << cfg.maxPlayersPerTeam << ")\n";
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
	if (cfg.maxSteps > 0)
		std::cout << "Step budget:      " << cfg.maxSteps << "\n";

	g_MaxSteps = cfg.maxSteps;
	g_MaxPlayersPerTeam = cfg.maxPlayersPerTeam;
	g_ObsMode = cfg.obs;
	g_TouchAccelExponent = cfg.rewards.touchAccelExponent;
	g_AirTouchDirectionExponent = cfg.rewards.airTouchDirectionExponent;

	g_RewardLabels.clear();
	for (auto &s : GeneralRewardSpecs(cfg))
		g_RewardLabels.push_back({s.name, s.weight});

	LearnerConfig lc = {};

	lc.deviceType =
		cfg.useGPU ? LearnerDeviceType::GPU_CUDA : LearnerDeviceType::CPU;
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

	auto envCreateFn = [cfg](int index) -> EnvCreateResult {
		return CreateEnv(index, cfg);
	};

	Learner learner(envCreateFn, lc, StepCallback);

	std::signal(SIGINT, HandleSigint);

	learner.Start();
}

}  // namespace Hive
