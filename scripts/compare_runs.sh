#!/usr/bin/env bash
# Run two training configurations over an identical step budget and compare.
#
#   scripts/compare_runs.sh [STEPS] [GAMES]
#
# Run A: baseline    -- skill tracking on, self-play off
# Run B: self-play   -- skill tracking on, self-play on
#
# Both use the same seed and the same step budget. Budget, not wall clock:
# self-play costs time per step, and comparing by clock would penalise it for
# having seen less data rather than for being worse.
#
# Runs are sequential on purpose. In parallel they would contend for CPU and GPU
# and neither throughput number would mean anything.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/HivemindBot"
LOG_DIR="$REPO/runs"

STEPS="${1:-50000000}"
GAMES="${2:-128}"
SEED=42

if [[ ! -x "$BIN" ]]; then
	echo "HivemindBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

mkdir -p "$LOG_DIR"
cd "$BUILD_DIR"

# GigaLearn's keypress detector calls tcsetattr every poll, and when stdout is
# not a TTY every call fails and prints. Over a long run that is gigabytes of
# noise, so it is filtered at the point of writing rather than after.
run_one() {
	local label="$1"; shift
	echo "==> Run '$label': $STEPS steps, $GAMES games, seed $SEED"
	"$BIN" train-general \
		--games "$GAMES" \
		--seed "$SEED" \
		--max-steps "$STEPS" \
		--label "$label" \
		"$@" 2>&1 \
		| grep --line-buffered -v "tcsetattr" \
		> "$LOG_DIR/$label.log"
	echo "    log:     $LOG_DIR/$label.log"
	echo "    metrics: $BUILD_DIR/metrics/general-$label.csv"
}

run_one baseline  --track-skill
run_one selfplay  --self-play

echo
echo "==> Done. Compare with:"
echo "    scripts/summarize_runs.py $BUILD_DIR/metrics/general-baseline.csv $BUILD_DIR/metrics/general-selfplay.csv"
