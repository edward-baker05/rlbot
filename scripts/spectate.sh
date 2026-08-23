#!/usr/bin/env bash
# Watch a checkpoint play live in RocketSimVis, without training anything.
#
#   scripts/spectate.sh <run-label> [extra args...]
#   scripts/spectate.sh --model /path/to/checkpoints/main-foo/50000000
#
# Safe to run against a training run that is in flight. This is a pure
# spectator: it loads a checkpoint, plays it against itself in ONE arena at
# wall-clock speed, and streams the gamestate to RocketSimVis. No learner, no
# optimizer, no checkpoint writes. Measured cost to a concurrent 128-game run:
# 0.45% of throughput (it pins libtorch to one thread; see docs/architecture.md).
#
# With a run label it follows that run: between episodes it picks up the newest
# complete checkpoint, so you watch the bot improve as training proceeds.
#
# Contrast with scripts/watch.sh, which runs `train --render` -- that IS a
# learner, and is for inspecting a state setter or reward change, not for
# watching an existing run.
#
# Start RocketSimVis first (scripts/vis.sh), then run this.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/HivemindBot"

if [[ ! -x "$BIN" ]]; then
	echo "HivemindBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

if [[ $# -eq 0 ]]; then
	echo "Usage: $0 <run-label> [extra args...]" >&2
	echo "       $0 --model <checkpoint-folder> [extra args...]" >&2
	echo >&2
	echo "Available runs:" >&2
	ls -1d "$REPO"/checkpoints/main-*/ "$BUILD_DIR"/checkpoints/main-*/ 2>/dev/null |
		sed 's|.*/checkpoints/main-|  |; s|/$||' | sort -u >&2 || echo "  (none yet)" >&2
	exit 1
fi

# A bare first argument is a run label; anything starting with - is passed through.
ARGS=()
if [[ "$1" != -* ]]; then
	ARGS+=(--follow "$1")
	shift
fi
ARGS+=("$@")

echo "Streaming to RocketSimVis on UDP 9273. Start it with scripts/vis.sh if it is not running."
# cd matters: RenderSender imports python_scripts.render_receiver relative to
# the working directory, and collision meshes resolve from here too.
cd "$BUILD_DIR"
exec "$BIN" spectate "${ARGS[@]}"
