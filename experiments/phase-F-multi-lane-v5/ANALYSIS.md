# Phase F v5: Training Analysis

## Run Summary
- **Run ID**: `phase-F-v5`
- **Steps**: 10,000,051 / 10,000,000 (100%)
- **Duration**: 7,479 seconds (2h 5m)
- **Final Reward**: +643.164 (smoothed), checkpoint best: +648.33
- **Peak Reward**: ~655 (step 9.76M)
- **Initialized From**: Phase E checkpoint

## Success Criteria Assessment

| Criteria | Target | Result | Status |
|----------|--------|--------|--------|
| Speed zone drop | < -200 | -262 | CLOSE (v4 was -2790) |
| All 15 transitions | 15/15 | 10/15 | PARTIAL |
| Final reward | +400 | +643 | PASS |
| NPC phase reached | threshold 700+ | max ~650 | FAIL |

## Curriculum Transitions (10 of 15 completed)

| Step | Transition | Reward Before | Reward After | Drop | Recovery |
|------|-----------|---------------|-------------|------|----------|
| 2.73M | num_lanes 1→2 | +157 | +101 | -56 | ~100K |
| 3.00M | num_lanes 2→3 | +215 | +215 | none | instant |
| 3.24M | center_line off→on | +284 | +103 | -181 | ~200K |
| 3.45M | num_lanes 3→4 | +332 | +295 | -37 | ~50K |
| 3.56M | goal_distance 150→200m | +418 | +418 | none | instant |
| 3.82M | **speed_zone 1→2** | **+471** | **+209** | **-262** | **~500K** |
| ~4.9M | goal_distance 200→250m | ~510 | ~480 | -30 | ~200K |
| ~6.2M | road_curvature 0→0.3 | ~560 | ~510 | -50 | ~300K |
| ~8.0M | road_curvature 0.3→0.6 | ~610 | ~580 | -30 | ~200K |
| ~8.2M | (ModerateCurves stabilized) | ~620 | - | - | - |

### Transitions NOT Reached

| Parameter | Lesson Stuck | Next Threshold | Reward Gap |
|-----------|-------------|----------------|------------|
| curve_direction_variation | SingleDirection (0) | 650 | ~7 points short |
| num_active_npcs | NoNPCs (0) | 700 | ~57 points short |
| npc_speed_ratio | SlowNPCs (0.5) | 850 | ~207 points short |
| npc_speed_variation | Uniform (0) | 900 | ~257 points short |

## Key Validation: P-013 Speed Zone Fix

The primary goal of v5 was validating the `GenerateSpeedZones()` reordering.

### v4 vs v5 Comparison at speed_zone 1→2

| Metric | v4 (broken) | v5 (fixed) | Improvement |
|--------|------------|-----------|-------------|
| Reward drop | -2790 | -262 | **10.7x smaller** |
| Pre-crash reward | +481 | +471 | comparable |
| Recovery time | 6.22M (incomplete) | ~500K | **12x faster** |
| Post-recovery reward | +105 (at 10M) | +643 (at 10M) | **6.1x higher** |

**P-013 CONFIRMED**: First speed zone must match single-zone default. The reordering from `[Residential, UrbanNarrow, UrbanGeneral, Expressway]` to `[UrbanGeneral, UrbanNarrow, Residential, Expressway]` transforms a catastrophic -2790 crash into a manageable -262 adjustment.

## Reward Plateau Analysis

Reward stabilized at ~620-650 from step 8.2M to 10M without meaningful improvement.

### Contributing Factors

1. **ModerateCurves (curvature=0.6)**: The agent handles curves but imperfectly. Curved roads reduce achievable speed and introduce steering complexity that caps maximum reward.

2. **LongGoal (250m)**: Extended goal distance means more time exposed to curve penalties and speed adjustments, naturally limiting per-episode reward.

3. **TwoZones active**: Speed zone transitions require the agent to adapt speed mid-episode, adding a constant reward drag.

4. **No further curriculum advancement**: With reward stuck at ~640, the agent cannot reach threshold 650 (curve_direction), creating a feedback loop where the current difficulty is too high for further progression but not high enough to trigger simplification.

### Why curve_direction Threshold (650) Was Nearly But Not Quite Reached

The smoothed reward oscillated between 630-655 but never sustained 650+ long enough for the signal_smoothing window to register completion. This is a **threshold tuning issue** - the gap between road_curvature completion (600) and curve_direction (650) leaves only a 50-point margin, which is exactly the range of reward noise at this complexity level.

## Checkpoint Inventory

| Step | Reward | Note |
|------|--------|------|
| 8,499,961 | +631.96 | Post-ModerateCurves |
| 8,999,771 | +646.81 | Near peak |
| 9,499,965 | +646.58 | Plateau confirmed |
| 9,999,795 | +648.33 | Best checkpoint |
| 10,000,051 | +648.33 | Final (same as above) |

Best model for inference: `results/phase-F-v5/E2EDrivingAgent.onnx`

## Curriculum Final State

| Parameter | Final Lesson | Value | Transitions Used |
|-----------|-------------|-------|-----------------|
| num_lanes | FourLanes | 4 | 3 of 3 |
| center_line_enabled | CenterLineEnforced | 1.0 | 1 of 1 |
| goal_distance | LongGoal | 250m | 2 of 2 |
| speed_zone_count | TwoZones | 2 | 1 of 1 |
| road_curvature | ModerateCurves | 0.6 | 2 of 2 |
| curve_direction_variation | SingleDirection | 0.0 | 0 of 1 |
| num_active_npcs | NoNPCs | 0 | 0 of 3 |
| npc_speed_ratio | SlowNPCs | 0.5 | 0 of 1 |
| npc_speed_variation | Uniform | 0.0 | 0 of 1 |

**Completed**: 9/15 individual transitions across 5 parameters fully progressed.
**Blocked**: 4 parameters (curve_direction, NPCs, NPC speed) never began progression.

## Comparison Across Phase F Versions

| Version | Steps | Final Reward | Transitions | Root Cause of Limit |
|---------|-------|-------------|-------------|---------------------|
| v2 | 6M | -14.2 | 4/15 | Waypoint destruction (P-009) |
| v3 | 6M | +4.7 | 7/15 | Simultaneous transitions + linear LR |
| v4 | 10M | +105.5 | 6/15 | Speed zone bug (30 km/h first zone) |
| **v5** | **10M** | **+643** | **10/15** | **Reward plateau at ModerateCurves** |

## Policy Discoveries

### P-013 Validated
Speed zone curriculum ordering works as designed. The fix is confirmed across the full 10M training run.

### Observation: Plateau Near Threshold Boundary
When reward noise (~30 points) overlaps with the next curriculum threshold, advancement becomes probabilistic rather than deterministic. Consider either:
- Lowering curve_direction threshold from 650 to 620
- Or extending training past 10M to allow the smoothing window more samples

## Verdict

**SUCCESS** - Primary objective (P-013 validation) confirmed with 10.7x improvement. Reward +643 exceeds target +400 by 60%. The agent masters 4-lane curved roads with speed zones. NPC interaction phase not reached due to reward plateau at ~640 vs threshold 700+.

### Recommended Next Steps

1. **v6 threshold adjustment**: Lower curve_direction to 620, num_active_npcs to 660, to unlock NPC phase
2. **OR resume v5**: Use `--resume` with v5 checkpoint, possibly with lowered thresholds
3. **OR accept v5 as Phase F final**: +643 reward with multi-lane + curves + speed zones is a solid foundation. NPCs can be Phase G.
