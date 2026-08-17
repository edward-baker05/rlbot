#!/usr/bin/env bash
# Watch training live in RocketSimVis.
#
#   scripts/watch.sh [extra args...]
#
# Runs a single game at wall-clock speed and streams state to RocketSimVis over
# UDP port 9273. This is the fastest way to find out whether a state setter or
# reward is doing what you intended -- reward curves will not tell you that a
# setter is spawning the car inside the ball.
#
# It is NOT training in any useful sense: one game at real time collects
# experience thousands of times slower than a real run. Use it to look, then
# stop it.
#
# To watch an EXISTING run instead, use scripts/spectate.sh -- this script
# builds a learner, which is wasted work if all you want is to see a checkpoint
# play, and it does not follow a run's checkpoints.
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

echo "Streaming to RocketSimVis on UDP 9273. Start it with scripts/vis.sh if it is not running."
cd "$BUILD_DIR"
exec "$BIN" train --render "$@"
