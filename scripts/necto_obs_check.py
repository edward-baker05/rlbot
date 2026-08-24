#!/usr/bin/env python3
"""Diff the C++ Necto observation against the real Python NectoObsBuilder.

    bot/build/DashBot necto-selftest /tmp/necto_obs_cpp.json
    scripts/necto_obs_check.py /tmp/necto_obs_cpp.json

Why this exists: a silent observation mismatch is the worst failure mode in the
Necto opponent. Nothing crashes, nothing logs an error -- Necto just plays
badly, and 20% of training arenas quietly become noise while the metrics look
plausible. So the port is checked against the upstream implementation itself,
not against a re-derivation of it.

The C++ dump carries the exact state it used, so both sides observe identical
inputs by construction rather than by two transcriptions agreeing.

TWO COLUMNS ARE SUBSTITUTED on the reference side before diffing, both
deliberate divergences from necto_obs.py rather than bugs:

  * car column 21 (demo timer). necto_obs.py re-seeds each car's timer to 3
    whenever it reaches 0 and never resets it on an actual demolition -- an
    artifact of its RLBot port, and effectively a constant-zero column. The C++
    uses the sim's real CarState::demoRespawnTimer, which is what the model was
    trained on.
  * boost pad column 21 (pad cooldown). necto_obs.py re-derives cooldowns by
    watching the binary pad states decay at its own fixed rate. RocketSim
    reports the true remaining cooldown, so the C++ reads that directly.

Column 21 is untouched by normalization (_norm[21] == 1), by the orange
inversion (_invert[21] == 1) and by the relative-coordinate pass (which only
covers 5:11), so substituting it in the final array is exact.

Everything else -- normalization, the blue/orange flag swap, the field
inversion, relative coordinates, the q layout and the appended previous action
-- is compared with no allowance at all.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NECTO_DIR = REPO / "libs" / "opponents" / "NectoFamily"

TOL = 1e-5


class Vec:
    """Stands in for rlgym_compat's PhysicsObject vectors (numpy-compatible)."""

    def __init__(self, xyz):
        self.x, self.y, self.z = (float(v) for v in xyz)


class CarData:
    def __init__(self, p):
        import numpy as np

        self.position = np.array(p["pos"], dtype=np.float64)
        self.linear_velocity = np.array(p["vel"], dtype=np.float64)
        self.angular_velocity = np.array(p["ang_vel"], dtype=np.float64)
        self._forward = np.array(p["forward"], dtype=np.float64)
        self._up = np.array(p["up"], dtype=np.float64)

    def forward(self):
        return self._forward

    def up(self):
        return self._up


class PlayerData:
    def __init__(self, p):
        self.car_id = int(p["car_id"])
        self.team_num = int(p["team"])
        self.car_data = CarData(p)
        # rlgym_compat reports boost 0-1; the sim stores it 0-100.
        self.boost_amount = float(p["boost_100"]) / 100.0
        self.on_ground = bool(p["on_ground"])
        self.has_flip = bool(p["has_flip"])
        self.is_demoed = False


class BallData:
    def __init__(self, b):
        import numpy as np

        self.position = np.array(b["pos"], dtype=np.float64)
        self.linear_velocity = np.array(b["vel"], dtype=np.float64)
        self.angular_velocity = np.array(b["ang_vel"], dtype=np.float64)


class GameStateStub:
    def __init__(self, dump):
        import numpy as np

        self.ball = BallData(dump["ball"])
        self.players = [PlayerData(p) for p in dump["players"]]
        self.boost_pads = np.array(
            [1 if pad["active"] else 0 for pad in dump["boost_pads"]], dtype=np.int64
        )


class BoostPadInfo:
    def __init__(self, pad):
        self.location = Vec(pad["loc"])
        self.is_full_boost = bool(pad["is_big"])


class FieldInfoStub:
    def __init__(self, dump):
        self.boost_pads = [BoostPadInfo(pad) for pad in dump["boost_pads"]]


def build_reference(dump, player_idx):
    """Runs the upstream NectoObsBuilder and applies the two substitutions."""
    import numpy as np

    sys.path.insert(0, str(NECTO_DIR))
    from necto.necto_obs import NectoObsBuilder

    builder = NectoObsBuilder(FieldInfoStub(dump), tick_skip=8)
    # demo_timers is a CLASS attribute in necto_obs.py, so it leaks between
    # instances. Reset it so each call starts clean.
    builder.demo_timers = Counter()

    state = GameStateStub(dump)
    prev_action = np.array(dump["prev_action"], dtype=np.float64)

    q, kv, _mask = builder.build_obs(state.players[player_idx], state, prev_action)
    q = np.array(q, dtype=np.float64)
    kv = np.array(kv, dtype=np.float64)

    n_players = len(dump["players"])
    for i, p in enumerate(dump["players"]):
        kv[0, 1 + i, 21] = float(p["demo_respawn_timer"]) / 10.0
    for j, pad in enumerate(dump["boost_pads"]):
        kv[0, 1 + n_players + j, 21] = float(pad["cooldown"]) / 10.0
    q[0, 0, 21] = float(dump["players"][player_idx]["demo_respawn_timer"]) / 10.0

    return q[0, 0], kv[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path, help="JSON from `DashBot necto-selftest`")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print("numpy is required; run this with libs/opponents/NectoFamily/.venv", file=sys.stderr)
        return 2

    dump = json.loads(args.dump.read_text())
    features = dump["features"]
    failures = 0

    for entry in dump["obs"]:
        idx = entry["player_idx"]
        team = "blue" if dump["players"][idx]["team"] == 0 else "orange"

        cpp_q = np.array(entry["q"], dtype=np.float64)
        cpp_kv = np.array(entry["kv"], dtype=np.float64)
        ref_q, ref_kv = build_reference(dump, idx)

        if cpp_kv.shape != ref_kv.shape:
            print(f"player {idx} ({team}): kv shape {cpp_kv.shape} vs {ref_kv.shape}")
            failures += 1
            continue

        dq = np.abs(cpp_q - ref_q)
        dkv = np.abs(cpp_kv - ref_kv)
        print(f"player {idx} ({team}): max |dq| = {dq.max():.3e}   max |dkv| = {dkv.max():.3e}")

        if dq.max() > args.tol:
            failures += 1
            for c in np.argsort(-dq)[:8]:
                if dq[c] > args.tol:
                    print(f"    q[{c}]: cpp {cpp_q[c]:+.8f}  ref {ref_q[c]:+.8f}")

        if dkv.max() > args.tol:
            failures += 1
            worst = np.dstack(np.unravel_index(np.argsort(-dkv, axis=None), dkv.shape))[0]
            for row, col in worst[:12]:
                if dkv[row, col] > args.tol:
                    kind = (
                        "ball"
                        if row == 0
                        else ("car" if row <= len(dump["players"]) else "pad")
                    )
                    print(
                        f"    kv[{row:2d}][{col:2d}] ({kind}): "
                        f"cpp {cpp_kv[row, col]:+.8f}  ref {ref_kv[row, col]:+.8f}"
                    )

    if failures:
        print(f"\nFAIL: {failures} mismatch group(s) above tolerance {args.tol}")
        return 1

    print(f"\nPASS: C++ and Python agree to {args.tol} across {features} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
