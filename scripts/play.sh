#!/usr/bin/env bash
# Start a Rocket League match against the bot via RLBot v5.
#
#   scripts/play.sh                       # 1v1, you vs the bot
#   scripts/play.sh match-3v3-human.toml  # 3v3 with you on blue
#
# Requires the RLBot v5 CLI. If it is missing, install it with:
#   pipx install rlbot   (or: pip install --user rlbot)
#
# The bot process is launched by RLBot itself using bot.toml's
# run_command_linux, which points at bot/rlbot-config/run.sh.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="$REPO/bot/rlbot-config"
MATCH="${1:-match-1v1.toml}"

if [[ ! -f "$CONFIG_DIR/$MATCH" ]]; then
	echo "No such match config: $CONFIG_DIR/$MATCH" >&2
	echo "Available:" >&2
	ls "$CONFIG_DIR"/match-*.toml 2>/dev/null | xargs -n1 basename >&2
	exit 1
fi

if ! command -v rlbot >/dev/null 2>&1; then
	echo "The 'rlbot' CLI was not found." >&2
	echo "Install it with:  pipx install rlbot" >&2
	exit 1
fi

if [[ ! -x "$REPO/bot/build/HivemindBot" ]]; then
	echo "HivemindBot not built. Run scripts/build.sh first." >&2
	exit 1
fi

cd "$CONFIG_DIR"
exec rlbot run "$MATCH"
