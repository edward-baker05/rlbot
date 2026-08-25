#!/usr/bin/env bash
# Configure and build the bot.
#
# The first run takes a while: the RLBot v5 C++ interface fetches flatbuffers,
# the RLBot schema and liburing at configure time, and RocketSim plus libtorch
# are not small. Later builds are incremental and fast.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO/bot/build"
JOBS="${JOBS:-$(nproc)}"

# external/*/ is gitignored, so a re-clone of any dependency silently drops our
# local patches -- and an unpatched build does not fail, it just trains a worse
# bot. Reapplying here makes that impossible to forget.
echo "==> Checking external/ patches"
"$REPO/scripts/apply_external_patches.py"

echo "==> Configuring ($BUILD_DIR)"
cmake -S "$REPO/bot" -B "$BUILD_DIR" \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DRLBOT_CPP_ENABLE_LTO=OFF

echo "==> Building with $JOBS jobs"
# DashTests as well as DashBot: both link the same DashCore, so the test binary
# costs a couple of extra translation units and nothing more. Building it here
# means a plain `scripts/build.sh` never leaves a stale test binary behind.
cmake --build "$BUILD_DIR" --parallel "$JOBS" --target DashBot DashTests

echo
echo "Built: $BUILD_DIR/DashBot"
echo "       $BUILD_DIR/DashTests"
echo "Try:   $BUILD_DIR/DashBot --help"
