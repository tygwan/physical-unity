# Phase H v1: NPC Interaction in Intersections

## Experiment ID
- **Run ID**: `phase-H`
- **Config**: `python/configs/planning/vehicle_ppo_phase-H.yaml`
- **Scene**: `PhaseH_NPCIntersection`
- **Date**: 2026-01-31

## Motivation

Phase G v2 completed (+633 reward) mastering all intersection geometry (T/Cross/Y-junction, all turn directions) with 260D observation space. Phase H introduces NPC vehicles that follow waypoints through intersections, requiring the agent to handle dynamic traffic at intersections.

## Strategy

**WARM START** from Phase G v2 final checkpoint (reward 633, 260D obs, 5M steps).

### Two-Phase Curriculum
1. **Phase G params** (thresholds 50-130): Fast re-unlock of intersection geometry (already mastered)
2. **NPC params** (thresholds 550-700): Progressive NPC introduction after intersection skills confirmed

## Observation Space

**260D** (unchanged from Phase G v2):
- ego_state: 8D
- route_info: 30D
- surrounding vehicles: 8 x 5 = 40D
- lane_info: 112D
- intersection_info: 6D (type 4D one-hot + distance 1D + turn_direction 1D)
- speed_zone: 4D
- curvature: 60D

## Key Code Changes

1. **NPCVehicleController**: New waypoint-following mode for intersection navigation
2. **DrivingSceneManager**: Waypoint-index spawning for NPC placement at intersections
3. **PhaseH_NPCIntersection scene**: 16 areas x 3 NPCs = 48 NPC GameObjects

## Curriculum Design

### Phase G Quick-Unlock (thresholds 50-130)

| Order | Threshold | Parameter | Transition |
|-------|-----------|-----------|------------|
| 1 | 50 | intersection_type | None -> T-Junction |
| 2 | 60 | turn_direction | Straight -> Left |
| 3 | 70 | num_lanes | 1 -> 2 |
| 4 | 75 | intersection_type | T -> Cross |
| 5 | 85 | turn_direction | Left -> Right |
| 6 | 90 | center_line | off -> on |
| 7 | 100 | intersection_type | Cross -> Y |
| 8 | 110 | goal_distance | 150 -> 200m |
| 9 | 130 | goal_distance | 200 -> 230m |

### NPC Introduction (thresholds 550-700)

| Order | Threshold | Parameter | Transition |
|-------|-----------|-----------|------------|
| 10 | 550 | num_active_npcs | 0 -> 1 |
| 11 | 600 | npc_speed_ratio | 0.5 -> 0.7 |
| 12 | 620 | num_active_npcs | 1 -> 2 |
| 13 | 660 | npc_speed_ratio | 0.7 -> 0.85 |
| 14 | 680 | num_active_npcs | 2 -> 3 |
| 15 | 700 | npc_speed_variation | 0 -> 0.15 |

### Locked Parameters
- road_curvature: 0 (straight roads)
- curve_direction_variation: 0
- speed_zone_count: 1

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-H.yaml \
  --run-id=phase-H --force
```

## Results

### Outcome: Partial Success -> v2 Needed

- **Steps completed**: ~3.5M / 8M
- **Peak reward**: ~700 (with 3 NPCs, speed_ratio 0.85)
- **Failure point**: npc_speed_variation transition at threshold 700
  - Reward crashed from ~700 to ~550
  - Std deviation spiked to 300+
  - Single-step jump from 0.0 to 0.15 variation was too aggressive

### Curriculum Completion

| Parameter | Status | Steps |
|-----------|--------|-------|
| Phase G params (9 transitions) | Complete | ~500K |
| num_active_npcs (0->1->2->3) | Complete | ~2M |
| npc_speed_ratio (0.5->0.7->0.85) | Complete | ~2.5M |
| npc_speed_variation (0->0.15) | FAILED | Crashed at 3.5M |

### Key Lesson (P-016)
**Never introduce high variation in a single step.** npc_speed_variation should have been graduated: 0 -> 0.05 -> 0.10 -> 0.15. The sudden jump destabilized the learned NPC interaction policy.

## Handoff to v2
- v2 uses gradual speed_variation (0 -> 0.05 -> 0.10 -> 0.15)
- v2 switches to build-based multi-env training (3x speedup)
- v2 warm starts from Phase G v2 checkpoint (v1 3.5M deleted by keep_checkpoints)
