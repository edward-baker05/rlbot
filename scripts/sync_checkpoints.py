#!/usr/bin/env python3
"""Sync the latest checkpoint of each test into the repo, alongside its config and config history.

Usage:
    python3 scripts/sync_checkpoints.py [--src bot/build/checkpoints] [--dest checkpoints]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Standard fallback configs for older tests that predated automated CONFIG.json generation.
#
# These are RECONSTRUCTIONS, not records. `addLayerNorm` was reconstructed as False for
# every pre-p13 run and is False in none of them: p9rel, p10touch, p11boost and p12goal
# were each verified on 2026-08-23 by loading their saved weights, as were the obs=Default
# entries p3strike..p8ref. ALL TEN require addLayerNorm=True. A wrong value here does not
# degrade a checkpoint, it makes it fail to load at all. Trust the weights, not this table.
FALLBACK_CONFIGS = {
    "main-smoke-metrics": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Default",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.00015,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.035,
            "entropyTarget": 0.0,
            "epochs": 1,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.00015,
            "tsPerItr": 100000,
        },
        "rewards": {
            "air": 0.0,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 0.1,
            "flipSpeed": 0.0,
            "goal": 30.0,
            "pickupBoost": 5.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 3.0,
            "touchAccelExponent": 1.0,
            "touchEdge": 0.0,
            "touchGoalAccel": 50.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.15,
            "trainAgainstOldVersions": False,
            "tsPerVersion": 5000000,
        },
    },
    "main-smoke-metrics2": "main-smoke-metrics",
    "main-smoke-rewards": "main-smoke-metrics",
    "main-p3strike": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Default",
            "spawn": "Curriculum",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0003,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.004,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0003,
            "tsPerItr": 100000,
        },
        "rewards": {
            "air": 0.04,
            "airTouch": 15.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 0.075,
            "flipSpeed": 0.0,
            "goal": 100.0,
            "pickupBoost": 5.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 0.5,
            "touchAccelExponent": 1.0,
            "touchEdge": 5.0,
            "touchGoalAccel": 75.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.05,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.15,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p4pbrs": "main-p3strike",
    "main-p5goalpot": "main-p3strike",
    "main-p6budget": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Default",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0003,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.004,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0003,
            "tsPerItr": 100000,
        },
        "rewards": {
            "air": 0.0,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 0.20,
            "flipSpeed": 0.0,
            "goal": 1.0,
            "pickupBoost": 0.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 0.30,
            "touchAccelExponent": 1.0,
            "touchEdge": 0.15,
            "touchGoalAccel": 0.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.10,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.15,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p7approach": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Default",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0003,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.01,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0003,
            "tsPerItr": 100000,
        },
        "rewards": {
            "air": 0.0,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 0.05,
            "flipSpeed": 0.0,
            "goal": 1.0,
            "pickupBoost": 0.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 0.50,
            "touchAccelExponent": 1.0,
            "touchEdge": 0.30,
            "touchGoalAccel": 0.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.10,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.15,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p8ref": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Default",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.002,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.513,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 3.42,
            "flipSpeed": 0.0,
            "goal": 0.0,
            "pickupBoost": 0.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 17.1,
            "touchAccelExponent": 1.0,
            "touchEdge": 1.0,
            "touchGoalAccel": 0.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p9rel": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.002,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.513,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 3.42,
            "flipSpeed": 0.0,
            "goal": 0.0,
            "pickupBoost": 0.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 17.1,
            "touchAccelExponent": 1.0,
            "touchEdge": 1.0,
            "touchGoalAccel": 0.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p10touch": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.002,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.513,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 3.42,
            "flipSpeed": 0.0,
            "goal": 0.0,
            "pickupBoost": 0.0,
            "save": 0.0,
            "saveBoost": 0.0,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 17.1,
            "touchAccelExponent": 1.0,
            "touchEdge": 0.25,
            "touchGoalAccel": 1.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p11boost": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.002,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.513,
            "airTouch": 0.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 3.42,
            "flipSpeed": 0.0,
            "goal": 0.0,
            "pickupBoost": 0.5,
            "save": 0.0,
            "saveBoost": 1.5,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 17.1,
            "touchAccelExponent": 1.0,
            "touchEdge": 0.25,
            "touchGoalAccel": 3.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p11boost-archived": "main-p11boost",
    "main-p12goal": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.0,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.0,
            "entropyScale": 0.002,
            "entropyTarget": 0.0,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.513,
            "airTouch": 2.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 3.42,
            "flipSpeed": 0.0,
            "goal": 10.0,
            "pickupBoost": 0.5,
            "save": 0.0,
            "saveBoost": 1.5,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 17.1,
            "touchAccelExponent": 1.0,
            "touchEdge": 0.25,
            "touchGoalAccel": 3.0,
            "touchGoalAccelOpponentScale": 0.0,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p12goal-archived": "main-p12goal",
    "main-p13strike": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.1,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.15,
            "entropyScale": 0.002,
            "entropyTarget": 0.40,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.88,
            "airTouch": 12.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 1.59,
            "flipSpeed": 2.5,
            "goal": 25.0,
            "pickupBoost": 0.5067,
            "save": 0.0,
            "saveBoost": 0.79,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 18.9,
            "touchAccelExponent": 2.0,
            "touchEdge": 0.1394,
            "touchGoalAccel": 45.0,
            "touchGoalAccelOpponentScale": 0.5,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p13strike-archived": "main-p13strike",
    "main-p13strike-attempt2": "main-p13strike",
    "main-p14cal": "main-p13strike",
    "main-p14aerial": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.1,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.15,
            "entropyScale": 0.002,
            "entropyTarget": 0.40,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 50000,
        },
        "rewards": {
            "air": 0.8736,
            "airTouch": 40.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 2.0,
            "faceBall": 1.585,
            "flipSpeed": 0.5,
            "goal": 40.0,
            "pickupBoost": 0.5067,
            "save": 0.0,
            "saveBoost": 0.785,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 0.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 14.51,
            "touchAccelExponent": 2.0,
            "touchEdge": 0.1394,
            "touchGoalAccel": 62.14,
            "touchGoalAccelOpponentScale": 0.5,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 0.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
    "main-p15manual": "main-p14aerial",
    "main-p15manual-archived": "main-p14aerial",
    "main-p16": {
        "env": {
            "actionDelay": 7,
            "infiniteBoostChance": 0.1,
            "maskActions": False,
            "maxPlayersPerTeam": 1,
            "noTouchTimeoutSeconds": 12.0,
            "numGames": 128,
            "obs": "Relative",
            "spawn": "Random",
            "teamSpirit": 0.0,
            "tickSkip": 8,
        },
        "model": {
            "addLayerNorm": True,
            "policyLayers": [512, 512, 512],
            "sharedHeadLayers": [512, 512],
        },
        "ppo": {
            "criticLR": 0.0002,
            "entropyAdjustRate": 0.15,
            "entropyScale": 0.003,
            "entropyTarget": 0.40,
            "epochs": 2,
            "gaeGamma": 0.99,
            "miniBatchSize": 25000,
            "policyLR": 0.0002,
            "tsPerItr": 100000,
        },
        "rewards": {
            "air": 1.9,
            "airTouch": 55.0,
            "airTouchDirectionExponent": 1.0,
            "airTouchHeightExponent": 1.0,
            "faceBall": 1.59,
            "flipSpeed": 2.5,
            "goal": 34.0,
            "pickupBoost": 0.5067,
            "save": 0.0,
            "saveBoost": 0.79,
            "saveOpponentScale": 1.0,
            "shotOnTarget": 22.0,
            "shotOnTargetOpponentScale": 1.0,
            "speedToBall": 18.9,
            "touchAccelExponent": 2.0,
            "touchEdge": 0.1394,
            "touchGoalAccel": 45.0,
            "touchGoalAccelOpponentScale": 0.8,
            "touchGoalAccelTeamSpirit": 0.0,
            "wrongSurface": 1.0,
        },
        "selfPlay": {
            "maxOldVersions": 32,
            "trackSkill": True,
            "trainAgainstOldChance": 0.20,
            "trainAgainstOldVersions": True,
            "tsPerVersion": 5000000,
        },
    },
}


def resolve_fallback_config(key: str) -> dict:
    val = FALLBACK_CONFIGS.get(key)
    while isinstance(val, str):
        val = FALLBACK_CONFIGS.get(val)
    return val


def sync_checkpoints(src_dir: Path, dest_dir: Path):
    if not src_dir.exists():
        print(f"Source directory '{src_dir}' does not exist.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    test_folders = [d for d in sorted(src_dir.iterdir()) if d.is_dir()]

    synced_count = 0
    for test_dir in test_folders:
        test_name = test_dir.name
        # Find numeric step folders
        step_dirs = [d for d in test_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if not step_dirs:
            continue

        latest_step_dir = max(step_dirs, key=lambda d: int(d.name))
        latest_step = int(latest_step_dir.name)

        target_test_dir = dest_dir / test_name
        target_test_dir.mkdir(parents=True, exist_ok=True)
        target_step_dir = target_test_dir / str(latest_step)

        # Copy the latest step directory if needed
        if not target_step_dir.exists():
            target_step_dir.mkdir(parents=True, exist_ok=True)
            for file_path in latest_step_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, target_step_dir / file_path.name)
            print(f"[{test_name}] Synced latest checkpoint: {latest_step}")
        else:
            # Update files in target_step_dir if newer
            for file_path in latest_step_dir.iterdir():
                if file_path.is_file():
                    dest_file = target_step_dir / file_path.name
                    if not dest_file.exists() or file_path.stat().st_mtime > dest_file.stat().st_mtime:
                        shutil.copy2(file_path, dest_file)
            print(f"[{test_name}] Checkpoint {latest_step} up to date.")

        # Clean up any older step folders in destination so only the latest remains in the repo
        for existing_dir in target_test_dir.iterdir():
            if existing_dir.is_dir() and existing_dir.name.isdigit() and int(existing_dir.name) != latest_step:
                shutil.rmtree(existing_dir)
                print(f"[{test_name}] Removed older checkpoint {existing_dir.name}")

        # Sync or generate CONFIG.json
        src_config = test_dir / "CONFIG.json"
        target_config = target_test_dir / "CONFIG.json"
        config_data = None
        if src_config.exists():
            shutil.copy2(src_config, target_config)
            with open(target_config) as f:
                config_data = json.load(f)
        else:
            config_data = resolve_fallback_config(test_name)
            if config_data:
                with open(target_config, "w") as f:
                    json.dump(config_data, f, indent=2)
                    f.write("\n")

        # Sync or generate CONFIG_HISTORY.json
        src_history = test_dir / "CONFIG_HISTORY.json"
        target_history = target_test_dir / "CONFIG_HISTORY.json"
        if src_history.exists():
            shutil.copy2(src_history, target_history)
        else:
            history_data = []
            if config_data:
                history_data = [
                    {
                        "changed": {},
                        "config": config_data,
                        "total_timesteps_at_start": 0,
                    }
                ]
            with open(target_history, "w") as f:
                json.dump(history_data, f, indent=2)
                f.write("\n")

        synced_count += 1

    print(f"\nSuccessfully synced {synced_count} test(s) into {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sync latest checkpoints and configs into repository")
    parser.add_argument("--src", type=Path, default=Path("bot/build/checkpoints"), help="Source checkpoints folder")
    parser.add_argument("--dest", type=Path, default=Path("checkpoints"), help="Destination checkpoints folder in repo")
    args = parser.parse_args()

    sync_checkpoints(args.src, args.dest)


if __name__ == "__main__":
    main()
