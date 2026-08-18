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

PATCHES = [
	{
		"name": "exploration-floor",
		"path": "external/GigaLearnCPP-Leak/GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp",
		"marker": "HIVE LOCAL PATCH: exploration floor",
		"anchor": EXPLORATION_FLOOR_ANCHOR,
		"body": EXPLORATION_FLOOR_BODY,
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
