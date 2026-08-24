# Hivemind

A Rocket League bot trained with [GigaLearn](https://github.com/ZealanL) on
RocketSim, deployed through [RLBot v5](https://wiki.rlbot.org/v5/).

This is a bare skeleton: it builds, trains a policy against a single `Goal`
reward, and can be deployed to play a live match. Reward design, curriculum,
observation and everything else that makes it actually good is not written
yet.

---

## Layout

```
bot/                    Our code
  src/
    Config.h            Every tunable, in one place
    main.cpp            Entry point: train | play
    env/                Obs, rewards, actions, state setters
    policy/              Policy: single-model inference wrapper, ModelShape
    rlbot/               RLBot v5 hivemind client and packet conversion
    train/               Training loop
  rlbot-config/          bot.toml, loadouts, match configs
  build/                 Build output (gitignored)
external/
  GigaLearnCPP-Leak/     GigaLearn + RLGymCPP + RocketSim
  cpp-interface/         RLBot v5 C++ interface
  RocketSimVis/          Live viewer
libs/                    libtorch, NCCL, NVSHMEM (gitignored; scripts/setup_libs.sh)
tools/                   Collision meshes RocketSim needs at runtime
scripts/                 build / train / play
```

---

## Quickstart

```bash
scripts/build.sh              # First build takes a while; later ones are fast
scripts/train.sh              # Start training. Press Q to save and quit.
```

Play against a checkpoint once you have one (needs `pip install --user --pre
rlbot` and the RLBotServer binary — `scripts/run_match.py` prints the
download command if it is missing):

```bash
scripts/play.sh
```
