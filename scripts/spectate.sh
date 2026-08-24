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
# Automatically launches RocketSimVis (scripts/vis.sh) in the background if
# not already running, and terminates it when spectating ends.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/DashBot"

if [[ ! -x "$BIN" ]]; then
	echo "DashBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

if [[ $# -eq 0 ]]; then
	echo "Usage: $0 <run-label> [extra args...]" >&2
	echo "       $0 --model <checkpoint-folder> [extra args...]" >&2
	echo >&2
	echo "Available runs:" >&2
	RUNS=$(find "$REPO"/checkpoints "$BUILD_DIR"/checkpoints -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sed 's|.*/||' | sort -u || true)
	if [[ -n "$RUNS" ]]; then
		echo "$RUNS" | sed 's/^/  /' >&2
	else
		echo "  (none yet)" >&2
	fi
	exit 1
fi

# A bare first argument is a run label; anything starting with - is passed through.
ARGS=()
if [[ "$1" != -* ]]; then
	ARGS+=(--follow "$1")
	shift
fi
ARGS+=("$@")

# Launch RocketSimVis in the background if not already active, and ensure clean exit.
VIS_PID=""
BOT_PID=""
cleanup() {
	trap - EXIT INT TERM
	if [[ -n "$BOT_PID" ]] && kill -0 "$BOT_PID" 2>/dev/null; then
		kill "$BOT_PID" 2>/dev/null || true
		wait "$BOT_PID" 2>/dev/null || true
	fi
	if [[ -n "$VIS_PID" ]] && kill -0 "$VIS_PID" 2>/dev/null; then
		kill "$VIS_PID" 2>/dev/null || true
		wait "$VIS_PID" 2>/dev/null || true
	fi
}
trap cleanup EXIT INT TERM

if ! pgrep -f "RocketSimVis.*src/main\.py" >/dev/null 2>&1; then
	"$REPO/scripts/vis.sh" &
	VIS_PID=$!
fi

echo "Streaming to RocketSimVis on UDP 9273."
# cd matters: RenderSender imports python_scripts.render_receiver relative to
# the working directory, and collision meshes resolve from here too.
(cd "$BUILD_DIR" && exec "$BIN" spectate "${ARGS[@]}") &
BOT_PID=$!

if [[ -n "$VIS_PID" ]]; then
	# Wait for either RocketSimVis or DashBot to exit.
	# If the user closes the visualizer window, kill DashBot and exit cleanly.
	# If DashBot exits, kill RocketSimVis and propagate DashBot's exit status.
	wait -n "$BOT_PID" "$VIS_PID" || true
	if kill -0 "$VIS_PID" 2>/dev/null; then
		set +e
		wait "$BOT_PID"
		BOT_EXIT=$?
		set -e
		exit "$BOT_EXIT"
	fi
else
	wait "$BOT_PID"
fi

