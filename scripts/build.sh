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

echo "==> Configuring ($BUILD_DIR)"
cmake -S "$REPO/bot" -B "$BUILD_DIR" \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DCMAKE_BUILD_TYPE=Release \
	-DRLBOT_CPP_ENABLE_LTO=OFF

echo "==> Building with $JOBS jobs"
cmake --build "$BUILD_DIR" --parallel "$JOBS" --target HivemindBot

echo
echo "Built: $BUILD_DIR/HivemindBot"
echo "Try:   $BUILD_DIR/HivemindBot --help"
