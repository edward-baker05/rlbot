#!/usr/bin/env python3
"""Start an RLBot v5 match from a match config toml.

RLBot v5's Python package (rlbot >= 2.0.0b*, installed with
`pip install --user --pre rlbot`) is a library, not a CLI -- there is no
`rlbot run` command. This script is the missing launcher: it starts
RLBotServer (downloaded to libs/rlbot/RLBotServer by scripts/setup_libs-style
one-liner below), submits the match config, and keeps running until Ctrl-C.

    scripts/run_match.py bot/rlbot-config/match-1v1.toml

Server binary, if missing:
    curl -fsSL -o libs/rlbot/RLBotServer \
      https://github.com/RLBot/core/releases/download/v5.0.0-rc17/RLBotServer
    chmod +x libs/rlbot/RLBotServer
"""

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SERVER = REPO / "libs" / "rlbot" / "RLBotServer"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 1

    match_toml = Path(sys.argv[1]).resolve()
    if not match_toml.is_file():
        print(f"No such match config: {match_toml}", file=sys.stderr)
        return 1

    try:
        from rlbot.managers import MatchManager
    except ImportError:
        print(
            "The rlbot v5 package is missing. Install it with:\n"
            "  pip install --user --pre rlbot",
            file=sys.stderr,
        )
        return 1

    if not SERVER.is_file():
        print(
            f"RLBotServer not found at {SERVER}.\n"
            "Download it (see the header of this script).",
            file=sys.stderr,
        )
        return 1

    manager = MatchManager(SERVER)
    try:
        # Relative config_file paths inside the toml resolve against the
        # toml's own directory, so no cwd games are needed.
        manager.start_match(match_toml)
        print("Match started. Ctrl-C to stop the match and shut the server down.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping match...")
        manager.shut_down()
    return 0


if __name__ == "__main__":
    sys.exit(main())
