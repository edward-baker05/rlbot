#!/usr/bin/env bash
# Launch RocketSimVis, the live viewer.
#
# It listens on UDP 9273 and renders whatever gamestate arrives, with no
# handshake or syncing. Start it whenever you like -- before or after a training
# run -- and it will pick up the stream.
#
# Pair with scripts/watch.sh, which runs the trainer in render mode.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIS_DIR="$REPO/external/RocketSimVis"
VENV="$VIS_DIR/.venv"

if [[ ! -d "$VENV" ]]; then
	echo "==> Creating RocketSimVis virtualenv"
	python3 -m venv "$VENV"
	"$VENV/bin/pip" install --quiet --upgrade pip
	"$VENV/bin/pip" install --quiet -r "$VIS_DIR/requirements.txt"
fi

cd "$VIS_DIR"
exec "$VENV/bin/python" src/main.py
