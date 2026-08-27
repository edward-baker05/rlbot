#!/usr/bin/env python3
"""Diff the C++ Nexto observation against the real Python NextoObsBuilder.

    bot/build/DashBot nexto-selftest /tmp/nexto_obs_cpp.json
    scripts/nexto_obs_check.py /tmp/nexto_obs_cpp.json

Same purpose as necto_obs_check.py, and worth more: Nexto's observation differs
from Necto's in three places, and every one of them fails silently. It rotates
the whole frame into the main car's yaw, it subtracts position but NOT velocity,
and its column 21 carries raw binary flags rather than timers. Get any of those
wrong and the model still runs, on a coherent-looking observation of a world
that is not the one being simulated.

The reference here is `NextoObsBuilder.batched_build_obs`, which is the function
that ran during Nexto's own training -- not the RLBot wrapper around it. It
takes rocket-learn's flat encoded-state array rather than a GameState, so this
script builds that array directly from the C++ dump. That is the point: both
sides observe identical inputs by construction.

ONE THING IS PATCHED OUT on the reference side. batched_build_obs reads car
orientation as a quaternion and calls its own _quats_to_rot_mtx; the C++ has the
rotation matrix already and never does that conversion. Rather than round-trip
through a quaternion and diff two floating-point paths to the same matrix, the
conversion is replaced with one that returns the dumped forward and up vectors
directly. Column 1 of that matrix (`right`) is never read by the builder, so
only the two columns that matter are supplied.

Nothing else is substituted. Normalization, the mate/opponent flags, the field
inversion, the token order, the relative pass with its yaw rotation, the q
layout and the appended previous action are all compared with no allowance.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NECTO_DIR = REPO / "libs" / "opponents" / "NectoFamily"

TOL = 1e-5

# Layout of rocket-learn's encoded state, from nexto_obs.py.
BALL_STATE_LENGTH = 18
PLAYER_CAR_STATE_LENGTH = 13
PLAYER_TERTIARY_INFO_LENGTH = 10
PLAYER_INFO_LENGTH = 2 + 2 * PLAYER_CAR_STATE_LENGTH + PLAYER_TERTIARY_INFO_LENGTH


class BoostPadInfo:
    def __init__(self, pad):
        class Loc:
            def __init__(self, xyz):
                self.x, self.y, self.z = (float(v) for v in xyz)

        self.location = Loc(pad["loc"])
        self.is_full_boost = bool(pad["is_big"])


class FieldInfoStub:
    def __init__(self, dump):
        self.boost_pads = [BoostPadInfo(pad) for pad in dump["boost_pads"]]


def encode_state(dump):
    """Builds the flat array batched_build_obs consumes, shaped (1, N).

    Only the fields the builder actually reads are filled; the inverted ball and
    inverted car blocks stay zero because it never touches them.
    """
    import numpy as np

    pads = dump["boost_pads"]
    players = dump["players"]

    ball_start = 3 + len(pads)
    players_start = ball_start + BALL_STATE_LENGTH
    total = players_start + len(players) * PLAYER_INFO_LENGTH

    enc = np.zeros((1, total), dtype=np.float64)

    # [0] unused, [1:3] scores.
    for j, pad in enumerate(pads):
        enc[0, 3 + j] = 1.0 if pad["active"] else 0.0

    ball = dump["ball"]
    enc[0, ball_start + 0 : ball_start + 3] = ball["pos"]
    enc[0, ball_start + 3 : ball_start + 6] = ball["vel"]
    enc[0, ball_start + 6 : ball_start + 9] = ball["ang_vel"]

    for i, p in enumerate(players):
        base = players_start + i * PLAYER_INFO_LENGTH
        enc[0, base + 0] = p["car_id"]
        enc[0, base + 1] = p["team"]
        enc[0, base + 2 : base + 5] = p["pos"]
        # base+5 .. base+8 is the quaternion. Nothing reads it except the
        # conversion this script replaces, so it carries the player index
        # instead -- that way the replacement returns the right car's basis
        # without depending on the order the builder happens to call it in.
        enc[0, base + 5] = i
        enc[0, base + 9 : base + 12] = p["vel"]
        enc[0, base + 12 : base + 15] = p["ang_vel"]
        # Tertiary block starts at base+28: five zeros, then the flags.
        enc[0, base + 33] = 1.0 if p["is_demoed"] else 0.0
        enc[0, base + 34] = 1.0 if p["on_ground"] else 0.0
        # base+35 is ball_touched, which the builder does not read.
        enc[0, base + 36] = 1.0 if p["has_flip"] else 0.0
        # rlgym_compat reports boost 0-1; the sim stores it 0-100.
        enc[0, base + 37] = float(p["boost_100"]) / 100.0

    return enc


def build_reference(dump):
    """Runs the upstream NextoObsBuilder and returns (q, kv) per player."""
    import numpy as np

    sys.path.insert(0, str(NECTO_DIR))
    from nexto.nexto_obs import NextoObsBuilder

    players = dump["players"]

    # The C++ carries the rotation matrix and never converts a quaternion, so
    # feed the dumped basis vectors straight in. Column 1 (`right`) is unread.
    forwards = np.array([p["forward"] for p in players], dtype=np.float64)
    ups = np.array([p["up"] for p in players], dtype=np.float64)

    def fake_quats_to_rot_mtx(quats):
        # Called once per car, with that car's "quaternion" -- see encode_state.
        idx = int(round(float(quats[0, 0])))
        mtx = np.zeros((quats.shape[0], 3, 3), dtype=np.float64)
        mtx[:, :, 0] = forwards[idx]
        mtx[:, :, 2] = ups[idx]
        return mtx

    builder = NextoObsBuilder(FieldInfoStub(dump), tick_skip=8)
    builder._quats_to_rot_mtx = staticmethod(fake_quats_to_rot_mtx)

    obs = builder.batched_build_obs(encode_state(dump))

    prev_action = np.array(dump["prev_action"], dtype=np.float64)
    for i in range(len(players)):
        builder.add_actions(obs, prev_action, player_index=i)

    return [(np.array(q[0, 0]), np.array(kv[0])) for q, kv, _m in obs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dump", type=Path, help="JSON from `DashBot nexto-selftest`")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    try:
        import numpy as np
    except ImportError:
        print(
            "numpy is required; run this with libs/opponents/NectoFamily/.venv",
            file=sys.stderr,
        )
        return 2

    dump = json.loads(args.dump.read_text())
    if dump.get("family") != "Nexto":
        print(
            f"dump is for {dump.get('family')!r}, not Nexto -- "
            "use `DashBot nexto-selftest`",
            file=sys.stderr,
        )
        return 2

    features = dump["features"]
    n_players = len(dump["players"])
    failures = 0

    # The action table first: the head emits an index into it, so a reordered
    # loop here would leave the observation perfect and every decision wrong.
    sys.path.insert(0, str(NECTO_DIR))
    from nexto.agent import make_lookup_table

    ref_table = np.array(make_lookup_table(), dtype=np.float64)
    cpp_table = np.array(dump["lookup_table"], dtype=np.float64)
    if cpp_table.shape != ref_table.shape:
        print(f"lookup table: {cpp_table.shape} vs {ref_table.shape}")
        failures += 1
    elif not np.array_equal(cpp_table, ref_table):
        bad = np.argwhere(cpp_table != ref_table)
        print(f"lookup table: {len(bad)} entries differ")
        for row, col in bad[:8]:
            print(
                f"    action[{row:2d}][{col}]: "
                f"cpp {cpp_table[row, col]:+.0f}  ref {ref_table[row, col]:+.0f}"
            )
        failures += 1
    else:
        print(f"lookup table: {len(ref_table)} actions match exactly")

    reference = build_reference(dump)

    for entry in dump["obs"]:
        idx = entry["player_idx"]
        team = "blue" if dump["players"][idx]["team"] == 0 else "orange"

        cpp_q = np.array(entry["q"], dtype=np.float64)
        cpp_kv = np.array(entry["kv"], dtype=np.float64)
        ref_q, ref_kv = reference[idx]

        if cpp_kv.shape != ref_kv.shape:
            print(f"player {idx} ({team}): kv shape {cpp_kv.shape} vs {ref_kv.shape}")
            failures += 1
            continue

        dq = np.abs(cpp_q - ref_q)
        dkv = np.abs(cpp_kv - ref_kv)
        print(
            f"player {idx} ({team}): max |dq| = {dq.max():.3e}   "
            f"max |dkv| = {dkv.max():.3e}"
        )

        if dq.max() > args.tol:
            failures += 1
            for c in np.argsort(-dq)[:8]:
                if dq[c] > args.tol:
                    print(f"    q[{c}]: cpp {cpp_q[c]:+.8f}  ref {ref_q[c]:+.8f}")

        if dkv.max() > args.tol:
            failures += 1
            worst = np.dstack(np.unravel_index(np.argsort(-dkv, axis=None), dkv.shape))[
                0
            ]
            for row, col in worst[:12]:
                # Nexto's token order: cars, then the ball, then the pads.
                if row < n_players:
                    kind = "car"
                elif row == n_players:
                    kind = "ball"
                else:
                    kind = "pad"
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
