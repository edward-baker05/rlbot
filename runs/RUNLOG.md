# Run log

One line per run that matters. Comparisons are only valid between labeled
runs recorded here. Append newest at the top.

Format: `date | label | config delta from previous entry | why | outcome`

| Date | Label | Config delta | Why | Outcome |
|---|---|---|---|---|
| 2026-08-17 | deploy-probe | defaults, 128 games, 30M steps | Throwaway checkpoint for the first live RLBot match (plan Task 13) | Trained ~6 min; `verify` PASSED (4/4); live match not yet run |
| 2026-08-17 | throughput-* | 1v1 obs (89 wide), tsPerItr 100k; games swept 64-320 | Re-measure steps/sec at 1v1 width (Phase 0) | steps/sec: 64g=71.4k, 128g=80.6k, 192g=79.4k, 256g=78.1k, 320g=74.6k -- 128 games wins; old ~120k/s figure was multi-size (up to 6 cars/arena, more player-steps per sim step) |
