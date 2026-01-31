# Phase F v4: Multi-Lane Roads - Strict Staggered Curriculum

## Experiment ID
- **Run ID**: `phase-F-v4`
- **Config**: `python/configs/planning/vehicle_ppo_phase-F-v4.yaml`
- **Scene**: `PhaseF_MultiLane`
- **Date**: 2026-01-31

## Motivation

Phase F v3 validated the WaypointManager fix (lane transitions no longer cause permanent collapse), but suffered a -2480 reward crash at step 4.38M when three curriculum parameters transitioned simultaneously.

### v3 Root Cause: Shared Thresholds
| Threshold | Parameters sharing it |
|-----------|---------------------|
| 350 | goal_distance L0, speed_zone_count L0, road_curvature L1, num_active_npcs L2 |
| 250 | center_line L0, num_lanes L1 |
| 400 | road_curvature L0, num_active_npcs L1 |
| 450 | curve_direction L0, npc_speed_variation L0 |
| 500 | num_active_npcs L0, npc_speed_ratio L0 |

Threshold 350 had **4 parameters** sharing it, causing the triple simultaneous transition.

### Additional v3 Issue: LR Decay
- `learning_rate_schedule: linear` decayed lr to ~0 when extended beyond original max_steps
- 2M extension at near-zero lr produced no meaningful learning

## v4 Design: Strict P-002 Compliance

### Key Changes from v3
1. **Every threshold is UNIQUE** - no two params share any threshold (15 unique values)
2. **50-point minimum gap** between consecutive thresholds
3. **`learning_rate_schedule: constant`** - no decay, consistent learning throughout
4. **3 macro phases** with clear separation:
   - Lane Phase (150-300): Core multi-lane capability
   - Env Phase (400-650): Goal, speed zones, curvature
   - NPC Phase (700-900): Most dangerous, transitions last
5. **Higher min_lesson_length** for lanes (1000) and NPCs (800)
6. **10M step budget** - generous for 15 curriculum transitions

### Complete Threshold Map

| Order | Threshold | Parameter | Lesson Transition | Macro Phase |
|-------|-----------|-----------|-------------------|-------------|
| 1 | 150 | num_lanes | 1 -> 2 lanes | Lane |
| 2 | 200 | num_lanes | 2 -> 3 lanes | Lane |
| 3 | 250 | center_line_enabled | off -> on | Lane |
| 4 | 300 | num_lanes | 3 -> 4 lanes | Lane |
| 5 | 400 | goal_distance | 150m -> 200m | Env 1 |
| 6 | 450 | speed_zone_count | 1 -> 2 zones | Env 1 |
| 7 | 500 | goal_distance | 200m -> 250m | Env 1 |
| 8 | 550 | road_curvature | 0 -> 0.3 | Env 2 |
| 9 | 600 | road_curvature | 0.3 -> 0.6 | Env 2 |
| 10 | 650 | curve_direction | 0 -> 1 | Env 2 |
| 11 | 700 | num_active_npcs | 0 -> 1 NPC | NPC |
| 12 | 750 | num_active_npcs | 1 -> 2 NPCs | NPC |
| 13 | 800 | num_active_npcs | 2 -> 3 NPCs | NPC |
| 14 | 850 | npc_speed_ratio | 0.5 -> 0.7 | NPC |
| 15 | 900 | npc_speed_variation | 0 -> 0.2 | NPC |

### Initialization
- **Source**: Phase E final checkpoint (`results/phase-E/E2EDrivingAgent/checkpoint.pt`)
- **Phase E reward**: +892.6 at step 6M
- **Method**: `--initialize-from=phase-E` (copies NN weights, resets training stats)

### Hyperparameters (unchanged from v3 except LR schedule)
| Parameter | Value | Note |
|-----------|-------|------|
| batch_size | 4096 | Same as all phases |
| buffer_size | 40960 | Same |
| learning_rate | 1.5e-4 | Same |
| learning_rate_schedule | **constant** | v3 was linear |
| hidden_units | 512 | Same |
| num_layers | 3 | Same |
| max_steps | 10,000,000 | v3 was 6M |

## Success Criteria
- All 15 curriculum transitions complete without simultaneous multi-param crashes
- No single transition causes >1M step recovery
- Final reward: +400 or higher with all params at final values
- NPC phase reached (threshold 700+)

## Policy References
- **P-002**: Staggered Curriculum (strict compliance - every threshold unique)
- **P-012 (NEW)**: No shared thresholds across curriculum parameters
