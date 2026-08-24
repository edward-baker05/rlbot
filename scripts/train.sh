#!/usr/bin/env bash
# Start a training run.
#
#   scripts/train.sh [extra args...]
#
# Training runs from bot/build, so checkpoints land in bot/build/checkpoints/
# and the collision meshes resolve on their default relative path.
#
# Press Q in the terminal to save and quit cleanly. Killing the process loses
# progress since the last checkpoint (default: every 1M steps).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/DashBot"

if [[ ! -x "$BIN" ]]; then
	echo "DashBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

# Metrics go to wandb. Without a login the run still works -- wandb falls back
# to offline mode and writes to bot/build/wandb/.
if ! python3 -c "import wandb" 2>/dev/null; then
	echo "NOTE: wandb is not installed; pass --no-metrics to silence its errors." >&2
fi

cd "$BUILD_DIR"
"$BIN" train "$@"
