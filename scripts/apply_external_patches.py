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


PATCHES = [
	{
		"name": "exploration-floor",
		"path": "external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: exploration floor",
		"anchor": EXPLORATION_FLOOR_ANCHOR,
		"body": EXPLORATION_FLOOR_BODY,
	},
	{
		"name": "skip-non-finite-updates",
		"path": "external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: skip non-finite updates",
		"anchor": SKIP_NONFINITE_ANCHOR,
		"body": SKIP_NONFINITE_BODY,
	},
	{
		"name": "return-std-finite-guard",
		"path": "external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/GAE.cpp",
		"marker": "HIVE LOCAL PATCH: finite guard",
		"anchor": RETURN_STD_ANCHOR,
		"body": RETURN_STD_BODY,
	},
	{
		"name": "entropy-target-config",
		"path": "external/GigaLearnCPP-Leak/GigaLearnCPP/src/public/GigaLearnCPP/PPO/PPOLearnerConfig.h",
		"marker": "HIVE LOCAL PATCH: target-entropy controller",
		"anchor": ENTROPY_CONFIG_ANCHOR,
		"body": ENTROPY_CONFIG_BODY,
	},
	{
		"name": "entropy-target-controller",
		"path": "external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: target-entropy controller",
		"anchor": ENTROPY_CONTROLLER_ANCHOR,
		"body": ENTROPY_CONTROLLER_BODY,
	},
]


def process(patch, check_only):
	path = REPO / patch["path"]
	name = patch["name"]

	if not path.exists():
		return "missing", f"{name}: target file not found ({patch['path']})"

	text = path.read_text()

	if patch["marker"] in text:
		return "ok", f"{name}: already applied"

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
		print("\nexternal/ is not fully patched. See CLAUDE.md.", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
