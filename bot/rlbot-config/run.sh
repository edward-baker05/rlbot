#!/usr/bin/env bash
# Launcher used by RLBot v5 (referenced as run_command_linux in bot.toml).
#
# RLBot starts the bot process itself and passes RLBOT_SERVER_IP,
# RLBOT_SERVER_PORT and RLBOT_AGENT_ID in the environment. Everything else the
# bot needs has to be set here.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

BUILD_DIR="${HIVE_BUILD_DIR:-$REPO/bot/build}"
BIN="$BUILD_DIR/HivemindBot"

if [[ ! -x "$BIN" ]]; then
	echo "HivemindBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

# --- Model selection -------------------------------------------------------
# Point these at GigaLearn checkpoint folders -- the numbered subdirectories
# holding the .lt files, not the parent. Override either by exporting it before
# launching RLBot.
#
# If HIVE_KICKOFF_MODEL is unset or missing, the general model drives kickoffs
# too, which is the correct setup until a kickoff model has been trained.
: "${HIVE_GENERAL_MODEL:=$(ls -d "$BUILD_DIR"/checkpoints/general/*/ 2>/dev/null | sort -V | tail -1)}"
: "${HIVE_KICKOFF_MODEL:=$(ls -d "$BUILD_DIR"/checkpoints/kickoff/*/ 2>/dev/null | sort -V | tail -1)}"

if [[ -z "${HIVE_GENERAL_MODEL:-}" || ! -d "${HIVE_GENERAL_MODEL}" ]]; then
	echo "No general model found. Train one first, or set HIVE_GENERAL_MODEL to a checkpoint folder." >&2
	exit 1
fi

export HIVE_GENERAL_MODEL
[[ -n "${HIVE_KICKOFF_MODEL:-}" && -d "${HIVE_KICKOFF_MODEL}" ]] && export HIVE_KICKOFF_MODEL || unset HIVE_KICKOFF_MODEL

# --- Runtime settings ------------------------------------------------------
# These MUST match what the models were trained with, or the bot will act on a
# different cadence and read a differently-shaped observation than it learned.
export HIVE_COLLISION_MESHES="${HIVE_COLLISION_MESHES:-$BUILD_DIR/collision_meshes}"
export HIVE_MAX_PLAYERS_PER_TEAM="${HIVE_MAX_PLAYERS_PER_TEAM:-3}"
export HIVE_TICK_SKIP="${HIVE_TICK_SKIP:-8}"
export HIVE_ACTION_DELAY="${HIVE_ACTION_DELAY:-7}"

# Deterministic play is meaningfully stronger than sampling in a real match.
export HIVE_DETERMINISTIC="${HIVE_DETERMINISTIC:-1}"

# Inference for at most 3 cars is tiny. The CPU avoids competing with the game
# for the GPU, which matters when Rocket League is rendering on the same card.
export HIVE_USE_GPU="${HIVE_USE_GPU:-0}"

cd "$BUILD_DIR"
exec "$BIN" play
