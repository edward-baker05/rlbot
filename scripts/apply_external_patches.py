#!/usr/bin/env python3
"""Reapply this repo's local patches to external/.

`external/*/` is gitignored -- each dependency is a nested repo with its own
history -- so edits made there are invisible to this repo's git and are lost
the moment anyone re-clones a dependency. That is a real hazard: the patches
are not cosmetic, and a silently-unpatched build trains a measurably worse bot
without failing.

This script carries the patch bodies in-tree so they survive, and is invoked by
scripts/build.sh before configuring. Idempotent: it detects an already-applied
patch by its marker and does nothing.

Run directly to check status:  scripts/apply_external_patches.py --check
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --- Patch bodies -----------------------------------------------------------
# Each patch is (path, marker, anchor, replacement). `anchor` is matched
# literally and replaced by `replacement`; `marker` is the string whose presence
# means the patch is already in. Anchors are deliberately short and distinctive
# so upstream reformatting elsewhere in the file cannot break them the way a
# context diff would.

EXPLORATION_FLOOR_ANCHOR = '''	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);
	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);'''

EXPLORATION_FLOOR_BODY = '''	auto result = torch::softmax(logits + ACTION_DISABLED_LOGIT * actionMasks.logical_not(), -1);

	// --- HIVE LOCAL PATCH: exploration floor --------------------------------
	// Mix the policy with a uniform distribution over the VALID actions, so no
	// action's probability can decay to zero.
	//
	// Why: this project has lost three separate control dimensions to
	// extinction -- jump (p1air, 0.49 -> 0.0000 by 20M), then steer and
	// throttle (p3strike, steer rate 0.0006). PPO's gradient is proportional
	// to the probability of the action taken, so once an action's probability
	// reaches ~0 it receives no gradient and cannot come back, whatever the
	// reward says. p2entropy confirmed the entropy bonus does not fix this:
	// 2.5x the coefficient moved measured entropy 0.0665 -> 0.0659. Entropy is
	// a property of the whole distribution and says nothing about whether one
	// particular action still has support.
	//
	// ACTION_MIN_PROB above cannot serve: it is applied after masking, so
	// raising it would give DISABLED actions sampling probability too.
	//
	// Sizing: with ~24 valid ground actions, eps 0.02 floors each valid action
	// at 0.02/24 ~= 8.3e-4, i.e. once per ~1200 steps. At 250k steps per
	// iteration that is ~200 samples per action per iteration -- enough to keep
	// a gradient alive without meaningfully perturbing a converged policy.
	//
	// Applied inside this function on purpose: every call site (collection,
	// the update's log-probs, the KL computations) then uses the identical
	// distribution, which PPO's importance ratio requires.
	//
	// Verified live: resuming the EXTINCT p3strike policy with this patch moved
	// Action/Steer Nonzero 0.0005 -> 0.0082 and Policy Entropy 0.155 -> 0.224
	// with no retraining at all.
	constexpr float ACTION_EXPLORE_EPS = 0.02f;
	{
		auto validF = actionMasks.to(result.dtype());
		auto validCount = validF.sum(-1, true).clamp_min(1);
		result = result * (1 - ACTION_EXPLORE_EPS) + (validF / validCount) * ACTION_EXPLORE_EPS;
	}
	// --- END HIVE LOCAL PATCH -----------------------------------------------

	return result.view({ -1, models["policy"]->config.numOutputs }).clamp(ACTION_MIN_PROB, 1);'''

# ---------------------------------------------------------------------------

ADV_STD_ANCHOR = """				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);"""

ADV_STD_BODY = """				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);

				// --- HIVE LOCAL PATCH: standardize advantages ---------------
				// Upstream feeds RAW advantages into the clipped objective, so
				// the size of the policy step scales with their absolute
				// magnitude -- and that magnitude is not a free parameter, it
				// is whatever the current reward scale happens to be divided by
				// a running return normalizer.
				//
				// Two ways this bites, both measured in this project:
				//
				// 1. The step decays on its own as the critic improves, because
				//    a better critic means smaller TD residuals. p1-validate
				//    over 117M steps: GAE/Avg Advantage 0.151 -> 0.088, Mean KL
				//    1.25e-3 -> 6.9e-4, SB3 Clip Fraction 6.1e-3 -> 4.1e-3
				//    against a healthy 0.05-0.2. What looked like "learning
				//    stops at ~40M" was an update size decaying to nothing.
				//
				// 2. Every reward-weight edit becomes a silent learning-rate
				//    edit. Learner.cpp's returnStat is a CUMULATIVE Welford over
				//    the whole run, so it barely moves after a few hundred
				//    million steps; cutting reward weights shrinks the numerator
				//    and leaves the divisor fossilized. p15manual at ~530M:
				//    GAE/Avg Advantage 0.205 -> 0.028 and SB3 Clip Fraction
				//    0.00098, i.e. ~30M steps of near-zero learning after what
				//    was meant to be a reward rebalance. Across that whole run
				//    log(KL) regressed on log(advantage) with slope 0.93,
				//    r = 0.73 -- which is the signature this patch removes.
				//
				// Standardizing per minibatch makes the step size depend on the
				// SHAPE of the advantages rather than their scale, which is what
				// PPO's trust region assumes and what every reference
				// implementation does.
				advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8);
				// --- END HIVE LOCAL PATCH -----------------------------------"""


# ---------------------------------------------------------------------------

SKIP_NONFINITE_ANCHOR = """			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			models.StepOptims();"""

SKIP_NONFINITE_BODY = """			if (trainSharedHead)
				nn::utils::clip_grad_norm_(models["shared_head"]->parameters(), 0.5f);

			// --- HIVE LOCAL PATCH: skip non-finite updates ----------------------
			// Discard the whole update if any gradient is NaN or inf, instead of
			// stepping the optimizer with it.
			//
			// Why: clip_grad_norm_ above does NOT stop this. It computes a total
			// norm and rescales by max_norm/norm -- and if any gradient is NaN the
			// norm is NaN, so the scale is NaN, and it multiplies NaN into EVERY
			// parameter. Clipping converts one bad gradient into a fully destroyed
			// network.
			//
			// Observed on p11boost, which died at 29.8M steps with
			// `Policy Update Magnitude: nan` followed by a CUDA device-side assert
			// (`probability tensor contains either inf, nan or element < 0`) when
			// the poisoned policy was next sampled. Everything upstream was
			// healthy: entropy 0.538, KL 0.0053, reward 0.0624, `GAE/Returns STD`
			// 2.113, all finite.
			//
			// A skipped update costs one minibatch of experience. The alternative
			// costs the run, and every run after it that resumes the checkpoint.
			{
				bool gradsFinite = true;
				for (Model* model : models) {
					for (auto& param : model->seq->parameters()) {
						if (!param.grad().defined())
							continue;
						if (!torch::isfinite(param.grad()).all().item<bool>()) {
							gradsFinite = false;
							break;
						}
					}
					if (!gradsFinite)
						break;
				}

				if (!gradsFinite) {
					RG_LOG("WARNING: non-finite gradient detected, skipping this optimizer step");
					for (Model* model : models)
						model->optim->zero_grad();
					continue;
				}
			}
			// --- END HIVE LOCAL PATCH -------------------------------------------

			models.StepOptims();"""

# ---------------------------------------------------------------------------

RETURN_STD_ANCHOR = """		float curReward;
		if (returnStd != 0) {"""

RETURN_STD_BODY = """		float curReward;
		// HIVE LOCAL PATCH: finite guard. `returnStd != 0` is TRUE for NaN, so
		// the original test let a NaN standardizer through and turned every
		// reward in the batch into NaN. isfinite() excludes NaN and inf both.
		if (std::isfinite(returnStd) && returnStd != 0) {"""


ENTROPY_CONFIG_ANCHOR = '\t\tfloat entropyScale = 0.018f; // The scale of the normalized entropy loss'

ENTROPY_CONFIG_BODY = '\t\tfloat entropyScale = 0.018f; // The scale of the normalized entropy loss\n\n\t\t// --- HIVE LOCAL PATCH: target-entropy controller -----------------------\n\t\t// entropyScale above is a FIXED coefficient on a quantity that shrinks.\n\t\t// The bonus is proportional to H, so as H falls the bonus weakens, which\n\t\t// is self-accelerating: p12goal at scale 0.002 ran `Policy Relative\n\t\t// Entropy Loss` 1.01 -> 0.12 while `Policy Entropy` fell 0.71 -> 0.146\n\t\t// (= 1.9 effective actions out of 90, i.e. near-deterministic). A fixed\n\t\t// coefficient cannot hold a floor; it only slows the descent.\n\t\t//\n\t\t// entropyTarget > 0 turns entropyScale into a controlled variable: it is\n\t\t// adjusted each iteration toward whatever value holds measured entropy at\n\t\t// the target. This is SAC\'s automatic temperature tuning (Haarnoja et al.\n\t\t// 2018, "Soft Actor-Critic Algorithms and Applications", sec. 5) applied\n\t\t// to PPO\'s entropy bonus: gradient descent on J(a) = a * (H - H_target),\n\t\t// done in log-space so the step is multiplicative and `a` stays positive.\n\t\t//\n\t\t// Entropy here is GigaLearn\'s NORMALIZED entropy in [0,1] (divided by\n\t\t// log(numActions)), so the target is a fraction of maximum, not nats.\n\t\t// 0 disables the controller and entropyScale stays fixed.\n\t\tfloat entropyTarget = 0.f;\n\n\t\t// Multiplicative gain, per iteration, in log-space.\n\t\t//\n\t\t// 0.05 shipped first and was too slow to be useful. p12 showed that\n\t\t// entropyScale 0.002 does NOT hold entropy at 0.40 -- it sails past and\n\t\t// keeps falling to 0.146 -- so the controller has to be able to RAISE\n\t\t// the scale by an order of magnitude or more. At 0.05 and a residual\n\t\t// error of 0.05, a 10x move takes ln(10)/0.0025 = 920 iterations = 46M\n\t\t// steps: half a run spent merely reaching the operating point.\n\t\t//\n\t\t// 0.15 makes that 15M steps while staying gentle per step (0.75% per\n\t\t// iteration at error 0.05), which matters: entropy responds to a scale\n\t\t// change with a lag of tens of iterations, and a controller that\n\t\t// outruns its own plant oscillates.\n\t\tfloat entropyAdjustRate = 0.15f;\n\n\t\t// --- ANTI-WINDUP ------------------------------------------------------\n\t\t// Without this the controller is BROKEN on a fresh run, and p13strike\'s\n\t\t// first attempt proved it in three minutes. A new policy starts at\n\t\t// entropy ~0.98, far ABOVE target, so the controller correctly wants no\n\t\t// bonus and winds the scale down -- but it keeps integrating that large\n\t\t// one-sided error for the whole transient. Measured: 0.002 -> 5.5e-5 by\n\t\t// 8.4M steps and still falling, heading for a floor from which recovery\n\t\t// would take longer than the run itself. The run silently becomes an\n\t\t// entropyScale ~ 0 run, i.e. LESS exploration than the fixed-coefficient\n\t\t// baseline it was built to improve on.\n\t\t//\n\t\t// Fix: hold the scale at its initial value until measured entropy first\n\t\t// REACHES the target, and only then start integrating. While the policy\n\t\t// is more random than asked for there is nothing to correct, and the\n\t\t// transient carries no information about the scale needed to hold the\n\t\t// target afterwards. On a resumed policy already below target this\n\t\t// engages on the first iteration, which is also correct.\n\t\t//\n\t\t// Runtime state, not settings. They live here because PPOLearner owns\n\t\t// its config by value and already mutates entropyScale through it.\n\t\tbool entropyControllerEngaged = false;\n\t\tfloat entropyScaleMin = 0.f; // set on engagement\n\n\t\t// The floor is a FRACTION of the scale at engagement, not an absolute.\n\t\t// An absolute 1e-5 sits 200x below nominal -- a hole deep enough that\n\t\t// falling in ends the experiment, which is exactly what happened. 0.2x\n\t\t// of nominal is a real correction without being a trap. The ceiling\n\t\t// stays generous, because raising the scale is the whole job.\n\t\tfloat entropyScaleMinFrac = 0.2f;\n\t\tfloat entropyScaleMax = 0.2f;\n\t\t// --- END HIVE LOCAL PATCH ---------------------------------------------'

ENTROPY_CONTROLLER_ANCHOR = '\t// Assemble and return report\n\treport["Policy Entropy"] = avgEntropy.Get();'

ENTROPY_CONTROLLER_BODY = '\t// --- HIVE LOCAL PATCH: target-entropy controller ------------------------\n\t// See PPOLearnerConfig.h for the derivation. Applied AFTER the epochs, on\n\t// the iteration\'s mean entropy, so one adjustment per iteration.\n\t//\n\t// Skipped on the first iteration, whose averages are not trustworthy, and\n\t// guarded on isfinite for the same reason the NaN patches exist: a NaN\n\t// entropy would otherwise turn entropyScale into NaN and poison every\n\t// subsequent loss.\n\tif (config.entropyTarget > 0 && !isFirstIteration) {\n\t\tconst float measured = avgEntropy.Get();\n\t\tif (std::isfinite(measured)) {\n\t\t\t// Anti-windup: there is nothing to correct while the policy is MORE\n\t\t\t// random than asked for. See PPOLearnerConfig.h -- integrating the\n\t\t\t// fresh-init transient buries the scale somewhere it cannot climb\n\t\t\t// back out of inside one run.\n\t\t\tif (!config.entropyControllerEngaged && measured <= config.entropyTarget) {\n\t\t\t\tconfig.entropyControllerEngaged = true;\n\t\t\t\tconfig.entropyScaleMin = config.entropyScale * config.entropyScaleMinFrac;\n\t\t\t}\n\n\t\t\tif (config.entropyControllerEngaged) {\n\t\t\t\tconst float err = config.entropyTarget - measured;\n\t\t\t\tconfig.entropyScale = std::clamp(\n\t\t\t\t\tconfig.entropyScale * std::exp(config.entropyAdjustRate * err),\n\t\t\t\t\tconfig.entropyScaleMin, config.entropyScaleMax);\n\t\t\t}\n\t\t}\n\t}\n\t// 0 until the controller takes over, so "is it driving yet" is readable off\n\t// the graph instead of inferred from the scale sitting still.\n\treport["Entropy Controller Engaged"] = config.entropyControllerEngaged ? 1.f : 0.f;\n\t// The scale is now a moving quantity, so it MUST be observable -- a\n\t// controlled variable that nobody can read is not controlled.\n\treport["Entropy Scale"] = config.entropyScale;\n\treport["Entropy Target"] = config.entropyTarget;\n\t// --- END HIVE LOCAL PATCH -----------------------------------------------\n\n\t// Assemble and return report\n\treport["Policy Entropy"] = avgEntropy.Get();'

AIR_MASK_OFF_BY_ONE_ANCHOR = """		if (i > numGroundActions && !action.jump)
			airMask[i] = true;"""

AIR_MASK_OFF_BY_ONE_BODY = """		// HIVE LOCAL PATCH: off-by-one in the air mask. Ground actions occupy
		// indices [0, numGroundActions - 1], so the FIRST aerial action sits at
		// index numGroundActions exactly, and `>` excluded it.
		//
		// Effect: action 24 (pitch -1, yaw -1, roll -1, no jump, no boost) was
		// unreachable under every mask -- it fails groundMask (index >= 24),
		// failed airMask here, and fails jumpMask (jump == 0). 89 of the 90
		// actions were selectable.
		//
		// It is not a junk entry. It is one half of a symmetric pair: the
		// +1,+1,+1 diagonal air-roll is reachable and its mirror was not, so the
		// aerial action set was asymmetric. Orange is handled by inverting the
		// OBSERVATION, not the action table, so that asymmetry did not cancel
		// out -- it was baked into the control set for both teams.
		//
		// Verified by regenerating the table and both masks: with `>` exactly
		// one index (24) is unreachable; with `>=` none are.
		//
		// Note this changes the action distribution. A policy trained under the
		// old mask has never seen action 24 and will need a few iterations to
		// place it.
		if (i >= numGroundActions && !action.jump)
			airMask[i] = true;"""

# ---------------------------------------------------------------------------

PPO_TUNING_CONFIG_ANCHOR = """		float clipRange = 0.2f;"""

PPO_TUNING_CONFIG_BODY = """
		// --- HIVE LOCAL PATCH: critic loss scale --------------------------------
		// PPO's vf_coef. Upstream forms `combinedLoss = ppoLoss + criticLoss` with
		// no coefficient, i.e. an implicit 1.0.
		//
		// With separate policy and critic towers that barely matters -- they have
		// separate optimizers and separate LRs. It matters here because this
		// project runs a shared trunk of {1024, 1024, 512} against {512} heads, so
		// ~95% of the parameters take gradient from BOTH losses, summed.
		//
		// Measured on the current run: `Policy Loss` 0.0015, `Critic Loss` 0.0037.
		// Those two were NOT comparable as upstream reported them -- see the
		// reporting patch in PPOLearner.cpp -- because avgCriticLoss accumulated
		// AFTER the batchSizeRatio multiply while avgPolicyLoss accumulates BEFORE
		// it. At miniBatchSize 50k / batchSize 250k that ratio is 0.2, so the true
		// critic MSE was ~0.0185, not 0.0037. The reporting fix stands on its own.
		//
		// What does NOT follow is a ratio of gradient influence. The policy loss is
		// near zero BY CONSTRUCTION once advantages are standardized: they are
		// mean-zero, so -mean(min(ratio*A, clipped*A)) ~= -mean(A) ~= 0 whenever
		// ratio ~= 1, while its GRADIENT scales with std(A) = 1. Comparing the two
		// loss magnitudes measures nothing about which one moves the trunk, in
		// either direction. Tune this from `GAE/Explained Variance` and the two
		// `* Update Magnitude` metrics, never from the losses.
		//
		// NOTE it does NOT act as a critic-LR change, and criticLR needs no
		// compensation. Adam normalises per parameter by its own running gradient
		// magnitude, so scaling a loss by k scales m and sqrt(v) together and the
		// step for a parameter fed only by that loss is unchanged. Every weight in
		// the critic head is fed only by this loss.
		//
		// The scale acts in exactly one place: the shared trunk, where the two
		// gradients are SUMMED before Adam sees them. Watch
		// `GAE/Explained Variance` anyway -- it is what says whether the critic
		// still predicts.
		float criticLossScale = 1.f;
		// --- END HIVE LOCAL PATCH -----------------------------------------------

		// --- HIVE LOCAL PATCH: KL-targeting LR controller -----------------------
		// Same control pattern as the target-entropy controller above, applied to
		// a quantity that is a better proxy for "is this run still learning".
		//
		// Why KL and not entropy: this project's own advantage-standardization
		// work established that the observed failure -- "learning stops at ~40M"
		// -- was an UPDATE SIZE decaying to nothing, measured as Mean KL
		// 1.25e-3 -> 6.9e-4 with SB3 Clip Fraction 6.1e-3 against a healthy
		// 0.05-0.2. Entropy is a property of the action distribution's shape and
		// can sit still while the policy stops moving; KL measures the policy
		// actually moving. Action SUPPORT is already protected separately by the
		// exploration floor -- the job people usually hand the entropy bonus, and
		// which p2entropy showed it does badly (2.5x the coefficient moved
		// measured entropy 0.0665 -> 0.0659).
		//
		// Control law, in log space because KL spans orders of magnitude:
		//   lr *= (klTarget / measuredKL) ^ klAdjustRate
		// Measured above target shrinks the LR, below target grows it. The
		// exponent form is scale-free, so klAdjustRate means the same thing at
		// KL 1e-4 as at KL 1e-2.
		//
		// Applies to policyLR ONLY. KL is a property of the policy; the critic's
		// step size has nothing to do with it. Note the shared head takes
		// min(policyLR, criticLR), so the trunk follows whichever is lower.
		//
		// 0 disables and the LR stays fixed. THERE IS NO SAFE DEFAULT TARGET --
		// the textbook 0.01 assumes ~10 epochs and this project runs 2, which
		// produces KL an order of magnitude smaller. Calibrate before enabling:
		// run with it off, find the KL that coincided with a healthy SB3 Clip
		// Fraction, and set that.
		float klTarget = 0.f;
		float klAdjustRate = 0.3f;

		// Bounds, as multiples of the LR configured at startup. The controller may
		// not run away in either direction: too low stalls the run silently, too
		// high destroys the policy in a single iteration.
		float klLRScaleMin = 0.25f;
		float klLRScaleMax = 4.f;

		// Runtime state, not settings -- the base LR the bounds are measured
		// against, captured on the first controlled iteration because
		// SetLearningRates() overwrites policyLR in place.
		bool  klBaseLRCaptured = false;
		float klBasePolicyLR = 0.f;
		// --- END HIVE LOCAL PATCH -----------------------------------------------

		float clipRange = 0.2f;"""

# ---------------------------------------------------------------------------

CRITIC_LOSS_SCALE_ANCHOR = """					criticLoss = mseLoss(vals, targetValues) * batchSizeRatio;
					avgCriticLoss += criticLoss.detach().cpu().item<float>();"""

CRITIC_LOSS_SCALE_BODY = """					// --- HIVE LOCAL PATCH: critic loss scale + comparable reporting -----
					// Two changes to one line.
					//
					// 1. config.criticLossScale is PPO's vf_coef, absent upstream.
					//    See PPOLearnerConfig.h for why it matters under a shared trunk.
					//
					// 2. avgCriticLoss now records the UNSCALED MSE. Upstream recorded
					//    it after the batchSizeRatio multiply while avgPolicyLoss is
					//    recorded before it, so `Critic Loss` and `Policy Loss` were off
					//    by a factor of batchSizeRatio (0.2 here) relative to each other
					//    and could not be compared -- which is exactly the comparison
					//    anyone tuning vf_coef has to make. Recording the raw MSE also
					//    keeps the metric independent of the new scale knob, so it still
					//    means \\"how wrong is the critic\\" rather than \\"how wrong is the
					//    critic times how much we care\\".
					//
					// This is a metric DISCONTINUITY: `Critic Loss` steps up by 1/0.2 = 5x
					// at the iteration this lands. The critic did not get worse.
					auto criticMSE = mseLoss(vals, targetValues);
					avgCriticLoss += criticMSE.detach().cpu().item<float>();
					criticLoss = criticMSE * batchSizeRatio * config.criticLossScale;
					// --- END HIVE LOCAL PATCH -------------------------------------------"""

# ---------------------------------------------------------------------------

KL_CONTROLLER_ANCHOR = """	report["Policy Entropy"] = avgEntropy.Get();
	report["Mean KL Divergence"] = avgDivergence.Get();"""

KL_CONTROLLER_BODY = """	report["Policy Entropy"] = avgEntropy.Get();
	report["Mean KL Divergence"] = avgDivergence.Get();

	// --- HIVE LOCAL PATCH: KL-targeting LR controller ---------------------------
	// See PPOLearnerConfig.h for the derivation. One adjustment per iteration, on
	// the iteration's mean KL, after the epochs have run.
	//
	// Skipped on the first iteration for the same reason the entropy controller
	// skips it -- its averages are not trustworthy -- and guarded on isfinite and
	// on measured > 0, since the control law takes a log of the measurement.
	if (config.klTarget > 0 && !isFirstIteration) {
		const float measuredKL = avgDivergence.Get();
		if (std::isfinite(measuredKL) && measuredKL > 0) {
			if (!config.klBaseLRCaptured) {
				config.klBaseLRCaptured = true;
				config.klBasePolicyLR = config.policyLR;
			}

			// (klTarget / measured) ^ klAdjustRate, written as an exp of a log so
			// the step is multiplicative and the LR cannot go negative.
			const float logErr = std::log(config.klTarget) - std::log(measuredKL);
			const float newPolicyLR = std::clamp(
				config.policyLR * std::exp(config.klAdjustRate * logErr),
				config.klBasePolicyLR * config.klLRScaleMin,
				config.klBasePolicyLR * config.klLRScaleMax);

			// SetLearningRates also re-derives the shared head's LR from
			// min(policyLR, criticLR), which is what we want -- the trunk should
			// follow whichever tower is stepping more cautiously.
			if (newPolicyLR != config.policyLR)
				SetLearningRates(newPolicyLR, config.criticLR);
		}
	}
	// A controlled variable that nobody can read is not controlled.
	report["KL Target"] = config.klTarget;
	report["Policy LR"] = config.policyLR;
	// --- END HIVE LOCAL PATCH ---------------------------------------------------"""

# ---------------------------------------------------------------------------

EXPLAINED_VAR_ANCHOR = """					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();"""

EXPLAINED_VAR_BODY = """					report["GAE/Avg Val Target"] = tTargetVals.abs().mean().item<float>();

					// --- HIVE LOCAL PATCH: explained variance -----------------------
					// EV = 1 - Var(target - prediction) / Var(target), the standard
					// scalar for value-function quality:
					//
					//   EV = 1   critic predicts the target exactly
					//   EV = 0   critic is no better than predicting the mean
					//   EV < 0   critic is actively worse than a constant -- and PPO
					//            can sit here for a long time while `Critic Loss` looks
					//            unremarkable, because standardizeReturns holds the
					//            scale roughly fixed no matter how bad the fit is.
					//
					// GAE.cpp:103 forms outTargetValues = valPreds + outAdvantages, so
					// the residual (target - prediction) IS the advantage exactly. No
					// extra tensors and no re-inference needed.
					//
					// Why this matters more than Critic Loss: a decaying advantage
					// signal is this project's known failure mode, and there are two
					// different causes with opposite fixes. A GOOD critic (EV -> 1)
					// shrinks advantages legitimately, because TD residuals get small.
					// A BAD critic (EV -> 0) produces small advantages that are noise.
					// `GAE/Avg Advantage` alone cannot tell those apart; EV can.
					{
						auto varTarget = tTargetVals.var();
						auto varResidual = tAdvantages.var();
						report["GAE/Explained Variance"] =
							1.f - (varResidual / varTarget.clamp_min(1e-8f)).item<float>();
					}
					// --- END HIVE LOCAL PATCH ---------------------------------------"""

# ---------------------------------------------------------------------------

BOOSTPAD_NULL_ANCHOR = """	Car* curLockedCar = NULL;
	uint32_t prevLockedCarID = NULL;"""

BOOSTPAD_NULL_BODY = """	Car* curLockedCar = NULL;
	uint32_t prevLockedCarID = 0;"""

GIGALEARN_MINIMUM_ANCHOR = 'cmake_minimum_required (VERSION 3.8)'

GIGALEARN_MINIMUM_BODY = 'cmake_minimum_required (VERSION 3.12)'

GIGALEARN_PYTHON_ANCHOR = """find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
target_link_libraries(GigaLearnCPP PUBLIC ${PYTHON_LIBRARIES})"""

GIGALEARN_PYTHON_BODY = """find_package(Python COMPONENTS Interpreter Development REQUIRED)
include_directories(${Python_INCLUDE_DIRS})
target_link_libraries(GigaLearnCPP PUBLIC ${Python_LIBRARIES})"""

GIGALEARN_COPY_ANCHOR = """configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPY)
configure_file("./python_scripts/render_receiver.py" "../python_scripts/render_receiver.py" COPY)"""

GIGALEARN_COPY_BODY = """configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPYONLY)
configure_file("./python_scripts/render_receiver.py" "../python_scripts/render_receiver.py" COPYONLY)"""

CPP_INTERFACE_CMP0169_ANCHOR = """cmake_minimum_required(VERSION 3.22)

project(RLBotCPP VERSION 2.0.0)"""

CPP_INTERFACE_CMP0169_BODY = """cmake_minimum_required(VERSION 3.22)

if(POLICY CMP0169)
	cmake_policy(SET CMP0169 OLD)
endif()

project(RLBotCPP VERSION 2.0.0)"""

PPOLEARNER_INCLUDE_ANCHOR = '#include "ExperienceBuffer.h";'

PPOLEARNER_INCLUDE_BODY = '#include "ExperienceBuffer.h"'

REPORT_PRAGMA_ONCE_ANCHOR = '#pragma once\n\n#include "Report.h"'

REPORT_PRAGMA_ONCE_BODY = '#include "Report.h"'

METRIC_SENDER_H_ANCHOR = """namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);
		void Finish();

		~MetricSender();
	};
}"""

METRIC_SENDER_H_BODY = """namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		void* pyMod = nullptr;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);
		void Finish();

		~MetricSender();
	};
}"""

METRIC_SENDER_CPP_ANCHOR = """#include "MetricSender.h"

#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		pyMod = py::module::import("python_scripts.metric_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = pyMod.attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		pyMod.attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

void GGL::MetricSender::Finish() {
	try {
		pyMod.attr("finish")();
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to finish, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {

}"""

METRIC_SENDER_CPP_BODY = """#include "MetricSender.h"

#include <pybind11/pybind11.h>
#include "Timer.h"

namespace py = pybind11;
using namespace GGL;

GGL::MetricSender::MetricSender(std::string _projectName, std::string _groupName, std::string _runName, std::string runID) :
	projectName(_projectName), groupName(_groupName), runName(_runName) {

	RG_LOG("Initializing MetricSender...");

	try {
		pyMod = new py::module(py::module::import("python_scripts.metric_receiver"));
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to import metrics receiver, exception: " << e.what());
	}

	try {
		auto returedRunID = static_cast<py::module*>(pyMod)->attr("init")(PY_EXEC_PATH, projectName, groupName, runName, runID);
		curRunID = returedRunID.cast<std::string>();
		RG_LOG(" > " << (runID.empty() ? "Starting" : "Continuing") << " run with ID : \"" << curRunID << "\"...");

	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to initialize in Python, exception: " << e.what());
	}

	RG_LOG(" > MetricSender initalized.");
}

void GGL::MetricSender::Send(const Report& report) {
	py::dict reportDict = {};

	for (auto& pair : report.data)
		reportDict[pair.first.c_str()] = pair.second;

	try {
		static_cast<py::module*>(pyMod)->attr("add_metrics")(reportDict);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to add metrics, exception: " << e.what());
	}
}

void GGL::MetricSender::Finish() {
	try {
		static_cast<py::module*>(pyMod)->attr("finish")();
	} catch (std::exception& e) {
		RG_ERR_CLOSE("MetricSender: Failed to finish, exception: " << e.what());
	}
}

GGL::MetricSender::~MetricSender() {
	if (pyMod) {
		delete static_cast<py::module*>(pyMod);
		pyMod = nullptr;
	}
}"""

RENDER_SENDER_H_ANCHOR = """namespace GGL {
	struct RG_IMEXPORT RenderSender {
		pybind11::module pyMod;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		void Send(const RLGC::GameState& state);

		~RenderSender();
	};
}"""

RENDER_SENDER_H_BODY = """namespace GGL {
	struct RG_IMEXPORT RenderSender {
		void* pyMod = nullptr;

		float timeScale;
		double adaptiveRenderDelay = -1;
		Timer renderTimer = {};

		RenderSender(float timeScale);

		RG_NO_COPY(RenderSender);

		void Send(const RLGC::GameState& state);

		~RenderSender();
	};
}"""

RENDER_SENDER_CPP_ANCHOR = """#include "RenderSender.h"

#include <nlohmann/json.hpp>

using namespace nlohmann;
using namespace RLGC;

GGL::RenderSender::RenderSender(float timeScale) : timeScale(timeScale) {
	RG_LOG("Initializing RenderSender...");

	try {
		RG_LOG("Current dir: " << std::filesystem::current_path());
		pyMod = pybind11::module::import("python_scripts.render_receiver");
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to import render receiver, exception: " << e.what());
	}

	RG_LOG(" > RenderSender initalized.");
}"""

RENDER_SENDER_CPP_BODY = """#include "RenderSender.h"

#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>

using namespace nlohmann;
using namespace RLGC;

GGL::RenderSender::RenderSender(float timeScale) : timeScale(timeScale) {
	RG_LOG("Initializing RenderSender...");

	try {
		RG_LOG("Current dir: " << std::filesystem::current_path());
		pyMod = new pybind11::module(pybind11::module::import("python_scripts.render_receiver"));
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to import render receiver, exception: " << e.what());
	}

	RG_LOG(" > RenderSender initalized.");
}"""

RENDER_SENDER_CPP_SEND_ANCHOR = """	try {
		pyMod.attr("render_state")(jStr);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to send gamestate, exception: " << e.what());
	}"""

RENDER_SENDER_CPP_SEND_BODY = """	try {
		static_cast<pybind11::module*>(pyMod)->attr("render_state")(jStr);
	} catch (std::exception& e) {
		RG_ERR_CLOSE("RenderSender: Failed to send gamestate, exception: " << e.what());
	}"""

RENDER_SENDER_CPP_DTOR_ANCHOR = """GGL::RenderSender::~RenderSender() {}"""

RENDER_SENDER_CPP_DTOR_BODY = """GGL::RenderSender::~RenderSender() {
	if (pyMod) {
		delete static_cast<pybind11::module*>(pyMod);
		pyMod = nullptr;
	}
}"""

MODEL_ITERATOR_ANCHOR = """		class ModelIterator : public std::iterator<std::forward_iterator_tag, Model*> {
		public:
			using MapItr = std::map<std::string, Model*>::iterator;
			MapItr _mapItr;"""

MODEL_ITERATOR_BODY = """		class ModelIterator {
		public:
			using iterator_category = std::forward_iterator_tag;
			using value_type = Model*;
			using difference_type = std::ptrdiff_t;
			using pointer = Model**;
			using reference = Model*&;

			using MapItr = std::map<std::string, Model*>::iterator;
			MapItr _mapItr;"""

BALL_ENUM_BITWISE_ANCHOR = """bulletWorld->addRigidBody(&_rigidBody, btBroadphaseProxy::DefaultFilter | CollisionMasks::HOOPS_NET, btBroadphaseProxy::AllFilter);"""

BALL_ENUM_BITWISE_BODY = """bulletWorld->addRigidBody(&_rigidBody, (int)btBroadphaseProxy::DefaultFilter | (int)CollisionMasks::HOOPS_NET, btBroadphaseProxy::AllFilter);"""

BOTCONTEXT_REDUNDANT_MOVE_ANCHOR = """	// collect desired game state
	auto const gameState = m_bot->getDesiredGameState ();
	if (gameState.has_value () && m_matchConfiguration->enable_state_setting ())
		m_connection.sendDesiredGameState (std::move (gameState.value ()));"""

BOTCONTEXT_REDUNDANT_MOVE_BODY = """	// collect desired game state
	auto const gameState = m_bot->getDesiredGameState ();
	if (gameState.has_value () && m_matchConfiguration->enable_state_setting ())
		m_connection.sendDesiredGameState (gameState.value ());"""

BULLET_MARGIN_RETURN_ANCHOR = """	case CONVEX_HULL_SHAPE_PROXYTYPE:
		return ((btConvexHullShape*)this)->getMargin();
	default:
		btAssert(false);
	}
}"""

BULLET_MARGIN_RETURN_BODY = """	case CONVEX_HULL_SHAPE_PROXYTYPE:
		return ((btConvexHullShape*)this)->getMargin();
	default:
		btAssert(false);
		return 0;
	}
}"""

TORCH_KINETO_OPTIONAL_ANCHOR = """if(ON)
  append_torchlib_if_found(kineto)
endif()"""

TORCH_KINETO_OPTIONAL_BODY = """if(ON)
  find_library(kineto_LIBRARY kineto PATHS "${TORCH_INSTALL_PREFIX}/lib")
  if(kineto_LIBRARY)
    list(APPEND TORCH_LIBRARIES ${kineto_LIBRARY})
  endif()
endif()"""

ROCKETSIMVIS_MAIN_ANCHOR = """    print("Starting socket thread...")
    socket_thread = threading.Thread(target=run_socket_thread, args=(int(port),))
    socket_thread.start()

    print("Starting visualizer window...")

    app = QtWidgets.QApplication([])
    ui.update_scaling_factor(app)

    window = QRSVWindow(QRSVGLWidget(app.primaryScreen()))
    window.showNormal()
    app.exec_()

    print("Shutting down...")
    g_socket_listener.stop_async()
    exit()"""

ROCKETSIMVIS_MAIN_BODY = """    print("Starting socket thread...")
    socket_thread = threading.Thread(target=run_socket_thread, args=(int(port),), daemon=True)
    socket_thread.start()

    print("Starting visualizer window...")

    app = QtWidgets.QApplication([])
    ui.update_scaling_factor(app)

    window = QRSVWindow(QRSVGLWidget(app.primaryScreen()))
    window.showNormal()
    app.exec_()

    print("Shutting down...")
    if g_socket_listener:
        g_socket_listener.stop_async()
    sys.exit(0)"""

ROCKETSIMVIS_SOCKET_ANCHOR = """    def stop_async(self):
        self.should_run = False"""

ROCKETSIMVIS_SOCKET_BODY = """    def stop_async(self):
        self.should_run = False
        if hasattr(self, 'sock') and self.sock:
            try:
                self.sock.close()
            except Exception:
                pass"""



# --- External opponents (Necto) ---------------------------------------------
# Lets a non-GigaLearn opponent play in a slice of the training arenas without
# its experience entering the PPO update. See bot/src/opponents/ for the
# Dash-side driver that fills these hooks in.

EXT_CFG_INCLUDE_ANCHOR = '#include "SkillTrackerConfig.h"'

EXT_CFG_INCLUDE_BODY = '#include "SkillTrackerConfig.h"\n\n// --- HIVE LOCAL PATCH: external opponent includes -------------------------\n#include <functional>\n#include <vector>\nnamespace RLGC { struct EnvSet; }\n// --- END HIVE LOCAL PATCH -------------------------------------------------'

EXT_CFG_HOOKS_ANCHOR = '\t\tSkillTrackerConfig skillTracker = {};'

EXT_CFG_HOOKS_BODY = "\t\tSkillTrackerConfig skillTracker = {};\n\n\t\t// --- HIVE LOCAL PATCH: external opponent hooks --------------------------\n\t\t// Hooks for opponents that are NOT GigaLearn models, and so cannot ride the\n\t\t// old-version path: they have their own observation, their own action head,\n\t\t// or both. This project uses them to put Necto in a slice of the arenas.\n\t\t//\n\t\t// externalPlayerMaskFn is called once, after the EnvSet exists, and flags\n\t\t// the players the learner does not own. Flagged players are excluded from\n\t\t// the update -- their actions did not come from the policy, so PPO's\n\t\t// importance ratio for them is meaningless.\n\t\t//\n\t\t// preStepFn runs immediately before each StepSecondHalf, which is where an\n\t\t// external opponent computes its controls for every arena in one batch.\n\t\tstd::function<void(RLGC::EnvSet*, std::vector<uint8_t>&)> externalPlayerMaskFn = nullptr;\n\t\tstd::function<void(RLGC::EnvSet*)> preStepFn = nullptr;\n\t\t// --- END HIVE LOCAL PATCH -----------------------------------------------"

EXT_MASK_ANCHOR = '\t\tint numPlayers = envSet->state.numPlayers;'

EXT_MASK_BODY = '\t\tint numPlayers = envSet->state.numPlayers;\n\n\t\t// --- HIVE LOCAL PATCH: external opponent mask -------------------------\n\t\t// Players driven by something that is neither this policy nor an old\n\t\t// version of it. They must be kept out of the update: their actions did\n\t\t// not come from the policy, so PPO\'s importance ratio on them is\n\t\t// meaningless, and the gradient it produces is noise wearing a\n\t\t// confident-looking magnitude.\n\t\t//\n\t\t// The assignment is fixed for the whole run, so this is built once here\n\t\t// rather than re-derived per iteration.\n\t\tstd::vector<uint8_t> externalPlayerMask(numPlayers, 0);\n\t\tint numExternalPlayers = 0;\n\t\tif (config.externalPlayerMaskFn) {\n\t\t\tconfig.externalPlayerMaskFn(envSet, externalPlayerMask);\n\t\t\texternalPlayerMask.resize(numPlayers, 0);\n\t\t\tfor (uint8_t flagged : externalPlayerMask)\n\t\t\t\tnumExternalPlayers += flagged ? 1 : 0;\n\n\t\t\tRG_LOG(" > External opponents: " << numExternalPlayers << " of " << numPlayers << " players");\n\t\t\tif (numExternalPlayers >= numPlayers)\n\t\t\t\tRG_ERR_CLOSE("Every player is an external opponent; nothing would be learned");\n\t\t}\n\t\t// --- END HIVE LOCAL PATCH ---------------------------------------------'

EXT_NEWIDX_ANCHOR = '\t\t\tfor (int i = 0; i < numPlayers; i++)\n\t\t\t\tnewPlayerIndices.push_back(i);'

EXT_NEWIDX_BODY = '\t\t\t// --- HIVE LOCAL PATCH: external opponents excluded ---------------\n\t\t\t// newPlayerIndices is the only list the trajectory appends iterate, so\n\t\t\t// leaving external players out of it here is the entire mechanism that\n\t\t\t// keeps their experience out of the update.\n\t\t\tfor (int i = 0; i < numPlayers; i++)\n\t\t\t\tif (!externalPlayerMask[i])\n\t\t\t\t\tnewPlayerIndices.push_back(i);\n\t\t\t// --- END HIVE LOCAL PATCH ----------------------------------------'

EXT_OLDMASK_ANCHOR = '\t\t\t\t\tnewPlayerIndices.clear();\n\t\t\t\t\toldVersionPlayerMask.resize(numPlayers);\n\t\t\t\t\tint i = 0;\n\t\t\t\t\tfor (auto& state : envSet->state.gameStates) {\n\t\t\t\t\t\tfor (auto& player : state.players) {\n\t\t\t\t\t\t\tif (player.team == oldVersionTeam) {\n\t\t\t\t\t\t\t\toldVersionPlayerMask[i] = true;\n\t\t\t\t\t\t\t\toldPlayerIndices.push_back(i);\n\t\t\t\t\t\t\t} else {\n\t\t\t\t\t\t\t\toldVersionPlayerMask[i] = false;\n\t\t\t\t\t\t\t\tnewPlayerIndices.push_back(i);\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\ti++;\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\n\t\t\t\t\ttNewPlayerIndices = torch::tensor(newPlayerIndices);\n\t\t\t\t\ttOldPlayerIndices = torch::tensor(oldPlayerIndices);'

EXT_OLDMASK_BODY = '\t\t\t\t\tnewPlayerIndices.clear();\n\t\t\t\t\toldVersionPlayerMask.resize(numPlayers);\n\t\t\t\t\tint i = 0;\n\t\t\t\t\tfor (auto& state : envSet->state.gameStates) {\n\t\t\t\t\t\t// --- HIVE LOCAL PATCH: external opponent arenas keep their learner\n\t\t\t\t\t\t// An arena that already holds an external opponent is left as it\n\t\t\t\t\t\t// is. Handing its remaining player to an old version would leave\n\t\t\t\t\t\t// that arena with nobody learning in it -- sim time spent for no\n\t\t\t\t\t\t// gradient, and silently, since nothing downstream would notice.\n\t\t\t\t\t\tbool arenaHasExternal = false;\n\t\t\t\t\t\tfor (int j = 0; j < (int)state.players.size(); j++)\n\t\t\t\t\t\t\tarenaHasExternal |= externalPlayerMask[i + j] != 0;\n\t\t\t\t\t\t// --- END HIVE LOCAL PATCH ---\n\n\t\t\t\t\t\tfor (auto& player : state.players) {\n\t\t\t\t\t\t\tif (!arenaHasExternal && player.team == oldVersionTeam) {\n\t\t\t\t\t\t\t\toldVersionPlayerMask[i] = true;\n\t\t\t\t\t\t\t\toldPlayerIndices.push_back(i);\n\t\t\t\t\t\t\t} else {\n\t\t\t\t\t\t\t\toldVersionPlayerMask[i] = false;\n\t\t\t\t\t\t\t\tif (!externalPlayerMask[i])\n\t\t\t\t\t\t\t\t\tnewPlayerIndices.push_back(i);\n\t\t\t\t\t\t\t}\n\t\t\t\t\t\t\ti++;\n\t\t\t\t\t\t}\n\t\t\t\t\t}\n\n\t\t\t\t\tif (oldPlayerIndices.empty()) {\n\t\t\t\t\t\t// Every arena holds an external opponent, so there is no\n\t\t\t\t\t\t// old-version side to field this iteration.\n\t\t\t\t\t\toldVersion = NULL;\n\t\t\t\t\t} else {\n\t\t\t\t\t\ttNewPlayerIndices = torch::tensor(newPlayerIndices);\n\t\t\t\t\t\ttOldPlayerIndices = torch::tensor(oldPlayerIndices);\n\t\t\t\t\t}'

EXT_NUMREAL_ANCHOR = '\t\t\tint numRealPlayers = oldVersion ? newPlayerIndices.size() : envSet->state.numPlayers;'

EXT_NUMREAL_BODY = '\t\t\t// --- HIVE LOCAL PATCH: split inference flag ----------------------\n\t\t\t// newPlayerIndices is authoritative in every case now: it already\n\t\t\t// excludes both old-version and external players.\n\t\t\tint numRealPlayers = newPlayerIndices.size();\n\n\t\t\t// Take the split path whenever ANY player is not ours. That keeps\n\t\t\t// tLogProbs aligned 1:1 with newPlayerIndices, which is exactly what\n\t\t\t// the trajectory appends further down assume -- so they need no change.\n\t\t\tconst bool splitInference = oldVersion || numExternalPlayers > 0;\n\t\t\ttorch::Tensor tdNewPlayerIndices, tdOldPlayerIndices;\n\t\t\tif (splitInference) {\n\t\t\t\tif (!tNewPlayerIndices.defined())\n\t\t\t\t\ttNewPlayerIndices = torch::tensor(newPlayerIndices);\n\n\t\t\t\t// The split happens on the DEVICE (see the inference branch\n\t\t\t\t// below), and these index tensors are constant for the whole\n\t\t\t\t// iteration -- so upload them once here rather than per step.\n\t\t\t\ttdNewPlayerIndices = tNewPlayerIndices.to(ppo->device);\n\t\t\t\tif (oldVersion)\n\t\t\t\t\ttdOldPlayerIndices = tOldPlayerIndices.to(ppo->device);\n\t\t\t}\n\t\t\t// --- END HIVE LOCAL PATCH ----------------------------------------'

EXT_INFER_ANCHOR = '\t\t\t\t\t\tif (oldVersion) {\n\t\t\t\t\t\t\ttorch::Tensor tdNewStates = tStates.index_select(0, tNewPlayerIndices).to(ppo->device, true);\n\t\t\t\t\t\t\ttorch::Tensor tdOldStates = tStates.index_select(0, tOldPlayerIndices).to(ppo->device, true);\n\t\t\t\t\t\t\ttorch::Tensor tdNewActionMasks = tActionMasks.index_select(0, tNewPlayerIndices).to(ppo->device, true);\n\t\t\t\t\t\t\ttorch::Tensor tdOldActionMasks = tActionMasks.index_select(0, tOldPlayerIndices).to(ppo->device, true);\n\n\t\t\t\t\t\t\ttorch::Tensor tNewActions;\n\t\t\t\t\t\t\ttorch::Tensor tOldActions;\n\n\t\t\t\t\t\t\tppo->InferActions(tdNewStates, tdNewActionMasks, &tNewActions, &tLogProbs);\n\t\t\t\t\t\t\tppo->InferActions(tdOldStates, tdOldActionMasks, &tOldActions, NULL, &oldVersion->models);\n\n\t\t\t\t\t\t\ttActions = torch::zeros(numPlayers, tNewActions.dtype());\n\t\t\t\t\t\t\ttActions.index_copy_(0, tNewPlayerIndices, tNewActions.cpu());\n\t\t\t\t\t\t\ttActions.index_copy_(0, tOldPlayerIndices, tOldActions.cpu());\n\t\t\t\t\t\t} else {'

EXT_INFER_BODY = '\t\t\t\t\t\tif (splitInference) {\n\t\t\t\t\t\t\t// --- HIVE LOCAL PATCH: split inference branch ----------\n\t\t\t\t\t\t\t// One transfer per step, and every narrowing happens on\n\t\t\t\t\t\t\t// the device.\n\t\t\t\t\t\t\t//\n\t\t\t\t\t\t\t// The previous version index_select\'d the CPU tensors\n\t\t\t\t\t\t\t// before transferring, which looks cheaper -- fewer rows\n\t\t\t\t\t\t\t// cross the bus -- but was the dominant cost of this\n\t\t\t\t\t\t\t// block. Each CPU index_select over [numPlayers, obsSize]\n\t\t\t\t\t\t\t// and [numPlayers, numActions] is a libtorch parallel_for\n\t\t\t\t\t\t\t// launch on the collection thread, and reassembling the\n\t\t\t\t\t\t\t// action vector needed a CPU index_copy_ on top. Measured\n\t\t\t\t\t\t\t// with a Necto opponent in 20% of arenas, "Inference Time"\n\t\t\t\t\t\t\t// roughly doubled purely from taking this branch, while\n\t\t\t\t\t\t\t// the GPU sat at ~11% utilisation throughout collection.\n\t\t\t\t\t\t\t//\n\t\t\t\t\t\t\t// Inferring the whole batch instead wastes a forward pass\n\t\t\t\t\t\t\t// on rows we discard, which on an idle GPU is close to\n\t\t\t\t\t\t\t// free, and reduces the split to device-side index_selects.\n\t\t\t\t\t\t\ttorch::Tensor tdStates = tStates.to(ppo->device, true);\n\t\t\t\t\t\t\ttorch::Tensor tdActionMasks = tActionMasks.to(ppo->device, true);\n\n\t\t\t\t\t\t\t// External players\' entries are computed and thrown away:\n\t\t\t\t\t\t\t// their arena\'s action parser ignores the index entirely\n\t\t\t\t\t\t\t// and applies the opponent\'s own controls instead.\n\t\t\t\t\t\t\t// Old-version players\' entries are overwritten below.\n\t\t\t\t\t\t\ttorch::Tensor tAllLogProbs;\n\t\t\t\t\t\t\tppo->InferActions(tdStates, tdActionMasks, &tActions, &tAllLogProbs);\n\n\t\t\t\t\t\t\t// tLogProbs must come back in newPlayerIndices order --\n\t\t\t\t\t\t\t// the trajectory appends below walk that list and index\n\t\t\t\t\t\t\t// this 1:1. (Undefined when config.deterministic is set,\n\t\t\t\t\t\t\t// which InferActions signals by not writing it.)\n\t\t\t\t\t\t\tif (tAllLogProbs.defined())\n\t\t\t\t\t\t\t\ttLogProbs = tAllLogProbs.index_select(0, tdNewPlayerIndices);\n\n\t\t\t\t\t\t\tif (oldVersion) {\n\t\t\t\t\t\t\t\ttorch::Tensor tOldActions;\n\t\t\t\t\t\t\t\tppo->InferActions(\n\t\t\t\t\t\t\t\t\ttdStates.index_select(0, tdOldPlayerIndices),\n\t\t\t\t\t\t\t\t\ttdActionMasks.index_select(0, tdOldPlayerIndices),\n\t\t\t\t\t\t\t\t\t&tOldActions, NULL, &oldVersion->models);\n\t\t\t\t\t\t\t\ttActions = tActions.index_copy(0, tdOldPlayerIndices, tOldActions);\n\t\t\t\t\t\t\t}\n\n\t\t\t\t\t\t\ttActions = tActions.cpu();\n\t\t\t\t\t\t\t// --- END HIVE LOCAL PATCH -----------------------------\n\t\t\t\t\t\t} else {'

EXT_PRESTEP_ANCHOR = '\t\t\t\t\t\tenvSet->Sync(); // Make sure the first half is done\n\t\t\t\t\t\tenvSet->StepSecondHalf(curActions, false);'

EXT_PRESTEP_BODY = "\t\t\t\t\t\tenvSet->Sync(); // Make sure the first half is done\n\n\t\t\t\t\t\t// --- HIVE LOCAL PATCH: pre-step hook -----------------------\n\t\t\t\t\t\t// Where a non-GigaLearn opponent computes its controls, in ONE\n\t\t\t\t\t\t// batched forward covering every arena it plays in. The arenas'\n\t\t\t\t\t\t// action parsers pick those up during the step below.\n\t\t\t\t\t\t//\n\t\t\t\t\t\t// This has to be a hook rather than lazy work inside the parser:\n\t\t\t\t\t\t// the parser is called once per car, so inference would degrade\n\t\t\t\t\t\t// to one tiny forward per arena (~40ms per step measured at 51\n\t\t\t\t\t\t// arenas) instead of a single batched one (~3ms).\n\t\t\t\t\t\tif (config.preStepFn)\n\t\t\t\t\t\t\tconfig.preStepFn(envSet);\n\t\t\t\t\t\t// --- END HIVE LOCAL PATCH ---------------------------------\n\n\t\t\t\t\t\tenvSet->StepSecondHalf(curActions, false);"

PATCHES = [
	{
		"name": "exploration-floor",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: exploration floor",
		"anchor": EXPLORATION_FLOOR_ANCHOR,
		"body": EXPLORATION_FLOOR_BODY,
	},
	{
		"name": "standardize-advantages",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: standardize advantages",
		"anchor": ADV_STD_ANCHOR,
		"body": ADV_STD_BODY,
	},
	{
		"name": "skip-non-finite-updates",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: skip non-finite updates",
		"anchor": SKIP_NONFINITE_ANCHOR,
		"body": SKIP_NONFINITE_BODY,
	},
	{
		"name": "return-std-finite-guard",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/GAE.cpp",
		"marker": "HIVE LOCAL PATCH: finite guard",
		"anchor": RETURN_STD_ANCHOR,
		"body": RETURN_STD_BODY,
	},
	{
		"name": "entropy-target-config",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/PPO/PPOLearnerConfig.h",
		"marker": "HIVE LOCAL PATCH: target-entropy controller",
		"anchor": ENTROPY_CONFIG_ANCHOR,
		"body": ENTROPY_CONFIG_BODY,
	},
	{
		"name": "entropy-target-controller",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: target-entropy controller",
		"anchor": ENTROPY_CONTROLLER_ANCHOR,
		"body": ENTROPY_CONTROLLER_BODY,
	},
	{
		"name": "air-mask-off-by-one",
		"path": "external/GigaLearnCPP/GigaLearnCPP/RLGymCPP/src/RLGymCPP/ActionParsers/DefaultAction.cpp",
		"marker": "HIVE LOCAL PATCH: off-by-one in the air mask",
		"anchor": AIR_MASK_OFF_BY_ONE_ANCHOR,
		"body": AIR_MASK_OFF_BY_ONE_BODY,
	},
	{
		"name": "ppo-tuning-config",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/PPO/PPOLearnerConfig.h",
		"marker": "HIVE LOCAL PATCH: critic loss scale",
		"anchor": PPO_TUNING_CONFIG_ANCHOR,
		"body": PPO_TUNING_CONFIG_BODY,
	},
	{
		"name": "critic-loss-scale",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: critic loss scale + comparable reporting",
		"anchor": CRITIC_LOSS_SCALE_ANCHOR,
		"body": CRITIC_LOSS_SCALE_BODY,
	},
	{
		"name": "kl-lr-controller",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: KL-targeting LR controller",
		"anchor": KL_CONTROLLER_ANCHOR,
		"body": KL_CONTROLLER_BODY,
	},
	{
		"name": "explained-variance",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp",
		"marker": "HIVE LOCAL PATCH: explained variance",
		"anchor": EXPLAINED_VAR_ANCHOR,
		"body": EXPLAINED_VAR_BODY,
	},
	{
		"name": "boostpad-null-to-zero",
		"path": "external/GigaLearnCPP/GigaLearnCPP/RLGymCPP/RocketSim/src/Sim/BoostPad/BoostPad.h",
		"marker": "uint32_t prevLockedCarID = 0;",
		"anchor": BOOSTPAD_NULL_ANCHOR,
		"body": BOOSTPAD_NULL_BODY,
	},
	{
		"name": "gigalearn-cmake-minimum",
		"path": "external/GigaLearnCPP/GigaLearnCPP/CMakeLists.txt",
		"marker": "cmake_minimum_required (VERSION 3.12)",
		"anchor": GIGALEARN_MINIMUM_ANCHOR,
		"body": GIGALEARN_MINIMUM_BODY,
	},
	{
		"name": "gigalearn-python",
		"path": "external/GigaLearnCPP/GigaLearnCPP/CMakeLists.txt",
		"marker": "find_package(Python COMPONENTS Interpreter Development REQUIRED)",
		"anchor": GIGALEARN_PYTHON_ANCHOR,
		"body": GIGALEARN_PYTHON_BODY,
	},
	{
		"name": "gigalearn-configure-copy",
		"path": "external/GigaLearnCPP/GigaLearnCPP/CMakeLists.txt",
		"marker": 'configure_file("./python_scripts/metric_receiver.py" "../python_scripts/metric_receiver.py" COPYONLY)',
		"anchor": GIGALEARN_COPY_ANCHOR,
		"body": GIGALEARN_COPY_BODY,
	},
	{
		"name": "cpp-interface-cmp0169",
		"path": "external/cpp-interface/CMakeLists.txt",
		"marker": "cmake_policy(SET CMP0169 OLD)",
		"anchor": CPP_INTERFACE_CMP0169_ANCHOR,
		"body": CPP_INTERFACE_CMP0169_BODY,
	},
	{
		"name": "ppolearner-include-semicolon",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.h",
		"marker": '#include "ExperienceBuffer.h"\n',
		"anchor": PPOLEARNER_INCLUDE_ANCHOR,
		"body": PPOLEARNER_INCLUDE_BODY,
	},
	{
		"name": "report-pragma-once",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/Report.cpp",
		"marker": '#include "Report.h"\n\nvoid GGL::Report::Display',
		"anchor": REPORT_PRAGMA_ONCE_ANCHOR,
		"body": REPORT_PRAGMA_ONCE_BODY,
	},
	{
		"name": "metric-sender-h",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/MetricSender.h",
		"marker": "void* pyMod = nullptr;",
		"anchor": METRIC_SENDER_H_ANCHOR,
		"body": METRIC_SENDER_H_BODY,
	},
	{
		"name": "metric-sender-cpp",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/MetricSender.cpp",
		"marker": "new py::module(py::module::import",
		"anchor": METRIC_SENDER_CPP_ANCHOR,
		"body": METRIC_SENDER_CPP_BODY,
	},
	{
		"name": "render-sender-h",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/RenderSender.h",
		"marker": "void* pyMod = nullptr;",
		"anchor": RENDER_SENDER_H_ANCHOR,
		"body": RENDER_SENDER_H_BODY,
	},
	{
		"name": "render-sender-cpp",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/RenderSender.cpp",
		"marker": "new pybind11::module(pybind11::module::import",
		"anchor": RENDER_SENDER_CPP_ANCHOR,
		"body": RENDER_SENDER_CPP_BODY,
	},
	{
		"name": "render-sender-cpp-send",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/RenderSender.cpp",
		"marker": "static_cast<pybind11::module*>(pyMod)->attr(\"render_state\")",
		"anchor": RENDER_SENDER_CPP_SEND_ANCHOR,
		"body": RENDER_SENDER_CPP_SEND_BODY,
	},
	{
		"name": "render-sender-cpp-dtor",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Util/RenderSender.cpp",
		"marker": "delete static_cast<pybind11::module*>(pyMod);",
		"anchor": RENDER_SENDER_CPP_DTOR_ANCHOR,
		"body": RENDER_SENDER_CPP_DTOR_BODY,
	},
	{
		"name": "model-iterator-std-iterator",
		"path": "external/GigaLearnCPP/GigaLearnCPP/src/private/GigaLearnCPP/Util/Models.h",
		"marker": "using iterator_category = std::forward_iterator_tag;",
		"anchor": MODEL_ITERATOR_ANCHOR,
		"body": MODEL_ITERATOR_BODY,
	},
	{
		"name": "ball-enum-bitwise",
		"path": "external/GigaLearnCPP/GigaLearnCPP/RLGymCPP/RocketSim/src/Sim/Ball/Ball.cpp",
		"marker": "(int)btBroadphaseProxy::DefaultFilter | (int)CollisionMasks::HOOPS_NET",
		"anchor": BALL_ENUM_BITWISE_ANCHOR,
		"body": BALL_ENUM_BITWISE_BODY,
	},
	{
		"name": "botcontext-redundant-move",
		"path": "external/cpp-interface/library/BotContext.cpp",
		"marker": "m_connection.sendDesiredGameState (gameState.value ());",
		"anchor": BOTCONTEXT_REDUNDANT_MOVE_ANCHOR,
		"body": BOTCONTEXT_REDUNDANT_MOVE_BODY,
	},
	{
		"name": "bullet-margin-return",
		"path": "external/GigaLearnCPP/GigaLearnCPP/RLGymCPP/RocketSim/libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btCollisionShape.cpp",
		"marker": "return 0;\n\t}\n}",
		"anchor": BULLET_MARGIN_RETURN_ANCHOR,
		"body": BULLET_MARGIN_RETURN_BODY,
	},
	{
		"name": "torch-kineto-optional",
		"path": "libs/libtorch/share/cmake/Torch/TorchConfig.cmake",
		"marker": 'find_library(kineto_LIBRARY kineto PATHS "${TORCH_INSTALL_PREFIX}/lib")',
		"anchor": TORCH_KINETO_OPTIONAL_ANCHOR,
		"body": TORCH_KINETO_OPTIONAL_BODY,
		"optional": True,
	},
	{
		"name": "rocketsimvis-main-clean-exit",
		"path": "external/RocketSimVis/src/main.py",
		"marker": "daemon=True",
		"anchor": ROCKETSIMVIS_MAIN_ANCHOR,
		"body": ROCKETSIMVIS_MAIN_BODY,
	},
	{
		"name": "rocketsimvis-socket-clean-exit",
		"path": "external/RocketSimVis/src/socket_listener.py",
		"marker": "self.sock.close()",
		"anchor": ROCKETSIMVIS_SOCKET_ANCHOR,
		"body": ROCKETSIMVIS_SOCKET_BODY,
	},
	{
		"name": 'external-opponent-includes',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h',
		"marker": 'HIVE LOCAL PATCH: external opponent includes',
		"anchor": EXT_CFG_INCLUDE_ANCHOR,
		"body": EXT_CFG_INCLUDE_BODY,
	},
	{
		"name": 'external-opponent-hooks',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/LearnerConfig.h',
		"marker": 'HIVE LOCAL PATCH: external opponent hooks',
		"anchor": EXT_CFG_HOOKS_ANCHOR,
		"body": EXT_CFG_HOOKS_BODY,
	},
	{
		"name": 'external-opponent-mask',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp',
		"marker": 'HIVE LOCAL PATCH: external opponent mask',
		"anchor": EXT_MASK_ANCHOR,
		"body": EXT_MASK_BODY,
	},
	{
		"name": 'external-opponents-excluded',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp',
		"marker": 'HIVE LOCAL PATCH: external opponents excluded',
		"anchor": EXT_NEWIDX_ANCHOR,
		"body": EXT_NEWIDX_BODY,
	},
	{
		"name": 'external-opponent-arenas-keep-learner',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp',
		"marker": 'HIVE LOCAL PATCH: external opponent arenas keep their learner',
		"anchor": EXT_OLDMASK_ANCHOR,
		"body": EXT_OLDMASK_BODY,
	},
	{
		"name": 'split-inference-flag',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp',
		"marker": 'HIVE LOCAL PATCH: split inference flag',
		"anchor": EXT_NUMREAL_ANCHOR,
		"body": EXT_NUMREAL_BODY,
	},
	{
		"name": 'split-inference-branch',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp',
		"marker": 'HIVE LOCAL PATCH: split inference branch',
		"anchor": EXT_INFER_ANCHOR,
		"body": EXT_INFER_BODY,
	},
	{
		"name": 'external-opponent-pre-step-hook',
		"path": 'external/GigaLearnCPP/GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp',
		"marker": 'HIVE LOCAL PATCH: pre-step hook',
		"anchor": EXT_PRESTEP_ANCHOR,
		"body": EXT_PRESTEP_BODY,
	},
]


def process(patch, check_only):
	path = REPO / patch["path"]
	name = patch["name"]

	if not path.exists():
		if patch.get("optional"):
			return "skipped", f"{name}: target file not present (skipped)"
		return "missing", f"{name}: target file not found ({patch['path']})"

	text = path.read_text()

	if patch["marker"] in text:
		return "ok", ""

	if check_only:
		return "absent", f"{name}: NOT APPLIED"

	if patch["anchor"] not in text:
		# Neither applied nor applicable: upstream changed under us. Say so
		# loudly rather than writing a half-patched file.
		return "failed", (
			f"{name}: anchor not found -- upstream has changed. "
			f"Reapply by hand and update {Path(__file__).name}."
		)

	path.write_text(text.replace(patch["anchor"], patch["body"], 1))
	return "applied", f"{name}: applied"


def main():
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--check", action="store_true",
	                help="report status without modifying anything")
	args = ap.parse_args()

	bad = False
	for patch in PATCHES:
		status, message = process(patch, args.check)
		print(f"  {message}")
		if status in ("failed", "missing") or (args.check and status == "absent"):
			bad = True

	if bad:
		print("\nexternal/ is not fully patched.", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
