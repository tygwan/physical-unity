# Phase E: Curved Roads

## Training Summary

| Item | Value |
|------|-------|
| Run ID | phase-E |
| Status | COMPLETED |
| Total Steps | 6,000,000 |
| Peak Reward | +938.2 at 4.5M steps |
| Final Reward | +892.6 at 6M steps |
| Training Time | ~15 minutes (6M steps at 20x) |
| Initialize From | Phase D v3 (--initialize-from=phase-D-v3-254d) |
| Observation Space | 254D (242D base + 12D lane) |
| Scene | PhaseE_CurvedRoads |

## Objective

Master curved road navigation while maintaining safe driving behaviors learned in previous phases.
Add road curvature (0 to 1.0 intensity) with direction variation as curriculum parameters.

## Parallel Training Environment

| Item | Value |
|------|-----|
| Training Areas | 16 (linear layout) |
| Each Area | Independent curved road + NPC |
| Area Spacing | 100m |
| Simultaneous Agents | 16 |

## Config File

`config/vehicle_ppo_v12_phaseE.yaml`

Active config: `python/configs/planning/vehicle_ppo_phase-E.yaml`

## Key Parameters

- **max_steps**: 6,000,000
- **batch_size**: 4096
- **buffer_size**: 40960
- **learning_rate**: 1.5e-4
- **Road curvature**: 0 -> 0.3 -> 0.6 -> 1.0 (curriculum)
- **parallel_envs**: 16 Training Areas
- **time_scale**: 20x

## Curriculum Parameters (7 total)

| Parameter | Stages | Final Value |
|-----------|--------|-------------|
| road_curvature | Straight -> Gentle(0.3) -> Moderate(0.6) -> Sharp(1.0) | 1.0 |
| curve_direction_variation | Single -> Mixed | 1.0 |
| num_active_npcs | 0 -> 1 -> 2 | 2 |
| npc_speed_ratio | Slow(0.4) -> Medium(0.7) | 0.7 |
| goal_distance | 100m -> 150m -> 200m | 200m |
| speed_zone_count | 1 -> 2 | 2 |
| npc_speed_variation | Uniform(0) -> Varied(0.2) | 0.2 |

All 7 parameters reached their final lesson.

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| E2EDrivingAgent-4499975.onnx | 4.5M | +938.2 | Peak reward |
| E2EDrivingAgent-4999956.onnx | 5.0M | +883.5 | Post-peak |
| E2EDrivingAgent-5499856.onnx | 5.5M | +917.6 | Recovery |
| E2EDrivingAgent-5999820.onnx | 6.0M | +892.6 | Final |
| E2EDrivingAgent-6000076.onnx | 6.0M | +892.6 | Final checkpoint |

## Critical Event: Curriculum Collapse and Recovery

At step 1.68M, 4 curriculum parameters transitioned simultaneously:
- npc_speed_ratio, goal_distance, speed_zone_count, npc_speed_variation

**Impact**: Reward crashed from +362 to -3,863 (P-002 violation)

**Recovery**: Agent recovered to positive rewards by 2.44M steps (~800K recovery period).
Unlike Phase D v1/v2, the Phase E agent successfully recovered from curriculum shock.

**Why recovery succeeded (unlike D v1/v2)**:
- Phase E already had a strong base policy from Phase D v3 (+895)
- Curvature parameters hadn't yet changed (still Straight), so only NPC/env complexity shifted
- Agent could fall back to known straight-road driving while adapting to new NPC conditions

## Curvature Progression (Post-Recovery)

| Step | Lesson | Curvature | Reward |
|------|--------|-----------|--------|
| 0-1.68M | Straight | 0.0 | +362 (rising) |
| 1.68M | COLLAPSE | 0.0 | -3,863 |
| 2.44M | Recovery | 0.0 | +0 (crossing zero) |
| 3.47M | GentleCurves | 0.3 | +800+ |
| 3.58M | Peak | 0.3-0.6 | +956 |
| 3.81M | ModerateCurves | 0.6 | +920-940 |
| 4.15M | SharpCurves | 1.0 | +938 |
| 4.5M-6M | Final | 1.0 | +880-938 (stable) |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-E.yaml \
  --run-id=phase-E \
  --initialize-from=phase-D-v3-254d
```

## TensorBoard

```bash
tensorboard --logdir=results/phase-E
```

## Success Criteria

| Criteria | Target | Achieved |
|----------|--------|----------|
| Mean reward | > +800 | +892.6 (final) |
| Sharp curve navigation | Mastered | road_curvature=1.0 completed |
| All curriculum stages | 7/7 | 7/7 completed |
| Recovery from collapse | N/A | YES (800K steps) |

## Policy Discoveries

- **P-002 reconfirmed**: 4 simultaneous transitions caused collapse (same pattern as Phase D v1)
- **Recovery capability**: Transfer learning from strong base (D v3) enables recovery that fresh-start (D v1/v2) couldn't achieve
- **Curvature learning order**: Straight -> Gentle -> Moderate -> Sharp progresses smoothly once NPC adaptation is complete

## Notes

- Phase D v1/v2 both failed with curriculum collapse and never recovered
- Phase E recovered because it inherited a robust driving policy from D v3
- The curriculum collapse at 1.68M validates P-002 (staggered thresholds) as a necessary design principle
- Phase E serves as the foundation for Phase F (multi-lane roads)
