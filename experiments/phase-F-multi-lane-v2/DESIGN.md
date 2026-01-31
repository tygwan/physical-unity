# Phase F v2: Multi-Lane Roads - First Correct Scene Attempt

## Experiment ID
- **Run ID**: `phase-F-v2`
- **Config**: `python/configs/planning/vehicle_ppo_phase-F.yaml`
- **Scene**: `PhaseF_MultiLane`
- **Date**: 2026-01-30

## Background

Phase F v1 failed immediately (within 10K steps) because it used the wrong scene (`PhaseE` with 4.5m road width). v2 corrected this by using the `PhaseF_MultiLane` scene with 11.5m road width, sufficient for up to 4 lanes.

### v1 Failure Summary
- Scene: `PhaseE` (4.5m road, single-lane geometry)
- 2-lane waypoints required 7m road width
- Agent went off-road immediately -> collision -> -5 penalty loop
- Failed within 10K steps (obvious failure)

## v2 Design

### Observation Space
- **254D** (same as Phase E): includes 12D lane observations via `enableLaneObservation`
- Waypoint observations (30D), ego state, surrounding vehicles, speed policy

### Curriculum: 9 Parameters, Non-Unique Thresholds
| Parameter | Lessons | Thresholds |
|-----------|---------|------------|
| num_lanes | 1->2->3->4 | 300, 250, 200 |
| center_line_enabled | off->on | 250 |
| road_curvature | 0->0.3->0.6 | 400, 350 |
| curve_direction_variation | 0->1 | 450 |
| num_active_npcs | 0->1->2->3 | 500, 400, 350 |
| npc_speed_ratio | 0.5->0.7 | 500 |
| goal_distance | 150->200->250m | 350, 300 |
| speed_zone_count | 1->2 | 350 |
| npc_speed_variation | 0->0.2 | 450 |

**P-002 Compliance**: Partial. Thresholds were spread (250-500), but multiple params shared the same thresholds:
- 250: center_line L0, num_lanes L1
- 300: num_lanes L0, goal_distance L1
- 350: road_curvature L1, num_active_npcs L2, goal_distance L0, speed_zone_count L0
- 400: road_curvature L0, num_active_npcs L1
- 450: curve_direction L0, npc_speed_variation L0
- 500: num_active_npcs L0, npc_speed_ratio L0

### Initialization
- **Source**: Phase E final checkpoint via `--initialize-from=phase-E`
- **Phase E final reward**: +892.6

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| batch_size | 4096 |
| buffer_size | 40960 |
| learning_rate | 1.5e-4 |
| learning_rate_schedule | linear |
| hidden_units | 512 |
| num_layers | 3 |
| max_steps | 6,000,000 |
| time_horizon | 256 |

## Known Issues (Post-Mortem)
1. **WaypointManager.GenerateWaypoints()** destroys ALL waypoints on lane count change
2. **Shared thresholds** allow simultaneous multi-parameter transitions
3. **linear LR schedule** decays to 0 at max_steps, making extensions useless
