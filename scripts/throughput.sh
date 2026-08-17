#!/usr/bin/env bash
# Measure training steps/sec across --games values. Wall-clock over a fixed
# step budget, so startup cost is included -- keep the budget large enough
# (5M) that it amortizes. Results are for RUNLOG.md; pick the numGames that
# wins and note it there.
#
#   scripts/throughput.sh [game-counts...]   # default: 64 128 192 256 320
#   STEPS=10000000 scripts/throughput.sh 128 256
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
BIN="$BUILD_DIR/HivemindBot"
STEPS="${STEPS:-5000000}"

[[ -x "$BIN" ]] || { echo "Build first: scripts/build.sh" >&2; exit 1; }

GAMES=(64 128 192 256 320)
[[ $# -gt 0 ]] && GAMES=("$@")

cd "$BUILD_DIR"
echo "games  steps  seconds  steps/sec"
for n in "${GAMES[@]}"; do
	label="throughput-$n-$(date +%s)"
	start=$(date +%s)
	"$BIN" train --max-steps "$STEPS" --games "$n" --no-metrics --label "$label" > /dev/null
	end=$(date +%s)
	secs=$((end - start))
	rm -rf "checkpoints/main-$label"
	echo "$n  $STEPS  $secs  $((STEPS / secs))"
done
