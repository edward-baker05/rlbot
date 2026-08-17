#!/usr/bin/env bash
# Start a training run.
#
#   scripts/train.sh general [extra args...]
#   scripts/train.sh kickoff [extra args...]
#
# Training runs from bot/build, so checkpoints land in bot/build/checkpoints/
# and the collision meshes resolve on their default relative path.
#
# Press Q in the terminal to save and quit cleanly. Killing the process loses
# progress since the last checkpoint (default: every 1M steps).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/HivemindBot"

TARGET="${1:-general}"
shift || true

case "$TARGET" in
	general) COMMAND="train-general" ;;
	kickoff) COMMAND="train-kickoff" ;;
	*)
		echo "Usage: $0 {general|kickoff} [extra args...]" >&2
		exit 1
		;;
esac

if [[ ! -x "$BIN" ]]; then
	echo "HivemindBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

# Metrics go to wandb. Without a login the run still works -- wandb falls back
# to offline mode and writes to bot/build/wandb/.
if ! python3 -c "import wandb" 2>/dev/null; then
	echo "NOTE: wandb is not installed; pass --no-metrics to silence its errors." >&2
fi

cd "$BUILD_DIR"
exec "$BIN" "$COMMAND" "$@"
