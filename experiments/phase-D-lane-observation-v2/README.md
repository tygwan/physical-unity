# Phase D v2: Lane Observation - Staggered Curriculum

## Status: PENDING

## Quick Reference

| Item | Value |
|------|-------|
| Run ID | `phase-D-v2` |
| Config | `python/configs/planning/vehicle_ppo_phase-D-v2.yaml` |
| Scene | PhaseB_DecisionLearning |
| Agent | E2EDrivingAgentBv2 |
| Observation | 254D (242D + 12D lane) |
| Max Steps | 10,000,000 |
| Init Path | None (fresh start) |

## Background

Phase D v1 achieved +406 reward (target met) but failed due to **curriculum shock** at step 4.68M when 3 parameters advanced simultaneously. See `../phase-D-lane-observation/ANALYSIS.md` for full failure analysis.

## v2 Key Change: Staggered Thresholds

v1 had all curriculum parameters checking the same reward threshold (~400), causing simultaneous transitions. v2 uses **different thresholds per parameter** so they advance one at a time:

| Order | Parameter | Threshold | Transition |
|-------|-----------|-----------|------------|
| 1st | num_active_npcs (1→2) | 200 | Lowest bar, advances first |
| 2nd | npc_speed_variation (0→0.3) | 300 | After NPC adaptation |
| 3rd | npc_speed_ratio (0.3→0.6) | 350 | After variation adaptation |
| 3rd | goal_distance (80→150) | 350 | Groups with ratio |
| 3rd | speed_zone_count (1→2) | 350 | Groups with ratio |
| 4th | num_active_npcs (2→3) | 300 | After moderate difficulty |
| 5th | Second group advances | 350 | ratio/goal/zones to final |
| 6th | num_active_npcs (3→4) | 300 | Final NPC count |

## Expected Progression

- Steps 0-2M: Learn basics with 1 NPC → reward reaches 200
- Steps 2-4M: Adapt to 2 NPCs → reward recovers to 300
- Steps 4-5M: Add speed variation → reward recovers to 350
- Steps 5-7M: Increase ratio/goal/zones → reward recovers to 300
- Steps 7-9M: Add 3rd NPC → reward recovers to 350
- Steps 9-10M: Final complexity (4 NPCs, fast, long goal, 4 zones)

## Success Criteria

| Metric | Target |
|--------|--------|
| Final Reward | > +300 |
| Curriculum | Reach Stage 3+ |
| Speed | > 8 m/s sustained |

## Related

- v1 results: `../phase-D-lane-observation/`
- v1 failure analysis: `../phase-D-lane-observation/ANALYSIS.md`
- v1 training log: `../phase-D-lane-observation/results/TRAINING-LOG.md`
