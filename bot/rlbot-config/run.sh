#!/usr/bin/env bash
# Launcher used by RLBot v5 (referenced as run_command_linux in bot.toml).
#
# RLBot starts the bot process itself and passes RLBOT_SERVER_IP,
# RLBOT_SERVER_PORT and RLBOT_AGENT_ID in the environment. Everything else the
# bot needs has to be set here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

BUILD_DIR="${DASH_BUILD_DIR:-${HIVE_BUILD_DIR:-$REPO/bot/build}}"
BIN="$BUILD_DIR/DashBot"

if [[ ! -x "$BIN" ]]; then
	echo "DashBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

# --- Model selection -------------------------------------------------------
# Point DASH_MODEL at a GigaLearn checkpoint folder -- the numbered
# subdirectory holding the .lt files, not the parent. Override by exporting it
# before launching RLBot; the default picks the most recently written
# checkpoint. Only numbered step folders count -- the same level also holds
# policy_versions (the self-play snapshot pool), which is not a loadable
# checkpoint.
#
# Sort by MTIME, not by name. `sort -V` compares the whole path, so the label
# dominates the step number and any label sorting late in the alphabet wins
# outright: a 250k-step main-smoke-rewards folder beat every main-p1x run,
# and being trained under the old 89-float obs it failed to load at all.
: "${DASH_MODEL:=${HIVE_MODEL:-$(ls -dt "$BUILD_DIR"/checkpoints/*/[0-9]*/ "$REPO"/checkpoints/*/[0-9]*/ 2>/dev/null | head -1)}}"

if [[ -z "${DASH_MODEL:-}" || ! -d "${DASH_MODEL}" ]]; then
	echo "No model found. Train one first, or set DASH_MODEL to a checkpoint folder." >&2
	echo "(DASH_MODEL must be an ABSOLUTE path: RLBot launches this script from its own cwd, so a relative path set in your shell will not resolve here.)" >&2
	exit 1
fi
export DASH_MODEL

# --- Runtime settings ------------------------------------------------------
# These MUST match what the model was trained with, or the bot will act on a
# different cadence and read a differently-shaped observation than it learned.
export DASH_COLLISION_MESHES="${DASH_COLLISION_MESHES:-${HIVE_COLLISION_MESHES:-$BUILD_DIR/collision_meshes}}"
export DASH_MAX_PLAYERS_PER_TEAM="${DASH_MAX_PLAYERS_PER_TEAM:-${HIVE_MAX_PLAYERS_PER_TEAM:-1}}"
export DASH_TICK_SKIP="${DASH_TICK_SKIP:-${HIVE_TICK_SKIP:-8}}"
export DASH_ACTION_DELAY="${DASH_ACTION_DELAY:-${HIVE_ACTION_DELAY:-7}}"

# Action masking. MUST match the value training used (TrainConfig::maskActions).
export DASH_MASK_ACTIONS="${DASH_MASK_ACTIONS:-${HIVE_MASK_ACTIONS:-0}}"

# Deterministic play is meaningfully stronger than sampling in a real match.
export DASH_DETERMINISTIC="${DASH_DETERMINISTIC:-${HIVE_DETERMINISTIC:-1}}"

# Inference for at most 3 cars is tiny. The CPU avoids competing with the game
# for the GPU, which matters when Rocket League is rendering on the same card.
export DASH_USE_GPU="${DASH_USE_GPU:-${HIVE_USE_GPU:-0}}"

cd "$BUILD_DIR"
exec "$BIN" play
