# Run log

One line per run that matters. Comparisons are only valid between labeled
runs recorded here. Append newest at the top.

Format: `date | label | config delta from previous entry | why | outcome`

| Date | Label | Config delta | Why | Outcome |
|---|---|---|---|---|
| 2026-08-17 | p1probe-b | + GroundedReward gate on VelPlayerToBall and FaceBall | Flip-spam farms velocity shaping; pay approach money only on wheels | Gate works: VP2B share 0.39->0.06. But VelBallToGoal absorbed it (share 0.67 -- mostly passive ball motion = zero-sum noise). Behavior still frozen at 30M (In Air 0.90, touch 0.0007) -> 30M too short to discriminate; next probe longer + VB2G cut |
| 2026-08-17 | p1probe-a | Rewards: VP2B 3->0.5, +Touch 5 (new term), VB2G 4->2, Goal 30->100, FaceBall 0.1->0.05 | Baseline paid 67% to VP2B, ~5% to outcomes -- rebalance toward outcomes | Shares moved as computed (VP2B 0.39, Goal 0.11, outcomes ~15%) but behavior unchanged at 30M: touch 0.0008, In Air 0.90 |
| 2026-08-17 | deploy-probe (as reward baseline) | old weights: VP2B 3, ST 50, VB2G 4, Goal 30, PB 5, FB 0.1 | First RewardShare telemetry | FARMING: VP2B share 0.67 flat, outcomes ~5%, In Air 0.89, touch 0.001 flat, Phase/Recover 0.84 -- flip-spam toward ball, never learns to drive |
| 2026-08-17 | (live match) | deploy-probe @30M, deterministic, CPU | First-ever live RLBot v5 match; parity check (plan Task 13) | Deployment VERIFIED: model loaded, pads mapped with no warnings, no console errors. Bot flip-spams and lands upside down in-game -- matches training telemetry exactly (In Air Ratio ~0.90 all run, touch ratio ~0.001), so behavior is faithful, just undertrained + a farming-shaped policy. Nexto probe: v4 bridge moot -- NectoFamily v5 port (libs/opponents/NectoFamily) runs natively, 2/2 agents spawned. Gotchas: RLBotServer cannot auto-launch RL under GE-Proton ("Could not find Proton installation") -- launch RL manually with `%command% -rlbot RLBot_ControllerURL=127.0.0.1:23233 RLBot_PacketSendRate=120 -nomovie`; HIVE_MODEL must be an absolute path (run.sh re-checks it from another cwd) |
| 2026-08-17 | deploy-probe | defaults, 128 games, 30M steps | Throwaway checkpoint for the first live RLBot match (plan Task 13) | Trained ~6 min; `verify` PASSED (4/4); live match not yet run |
| 2026-08-17 | throughput-* | 1v1 obs (89 wide), tsPerItr 100k; games swept 64-320 | Re-measure steps/sec at 1v1 width (Phase 0) | steps/sec: 64g=71.4k, 128g=80.6k, 192g=79.4k, 256g=78.1k, 320g=74.6k -- 128 games wins; old ~120k/s figure was multi-size (up to 6 cars/arena, more player-steps per sim step) |
