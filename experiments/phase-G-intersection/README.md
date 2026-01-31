# Phase G: Intersection Navigation

## Experiment ID
- **Run ID**: `phase-G`
- **Config**: `python/configs/planning/vehicle_ppo_phase-G.yaml`
- **Scene**: `PhaseG_Intersection`
- **Date**: 2026-01-31 (config prepared)

## Motivation

Phase F v5 completed (+643 reward) mastering 4-lane curved roads with speed zones.
Phase G introduces intersection navigation: T-junctions, cross-intersections, and Y-junctions with turn decisions.

## Observation Space Change

**254D → 260D (+6D intersection info)**:
- intersection_type: 4D one-hot [None, T, Cross, Y]
- distance_to_intersection: 1D normalized
- turn_direction: 1D (0=straight, 0.5=left, 1=right)

**NOTE**: 254D→260D dimension change means first-layer weight shape mismatch (`[512,254]` → `[512,260]`). Cannot transfer from Phase F v5 checkpoint. **Fresh start required**.

## Config Design

### Policy Compliance
- **P-002**: Staggered curriculum, all thresholds unique
- **P-012**: No shared thresholds (13 transitions, 13 unique threshold values)
- **P-013**: N/A (speed zones disabled for intersection training)
- **P-009**: 260D + intersection env combined (acceptable: fresh start, obs directly useful)

### Threshold Map (P-012 Compliant)

| Order | Threshold | Parameter | Transition | Macro Phase |
|-------|-----------|-----------|------------|-------------|
| 1 | 150 | intersection_type | None → T-Junction | Intersection |
| 2 | 200 | turn_direction | Straight → Left | Intersection |
| 3 | 250 | num_lanes | 1 → 2 | Environment |
| 4 | 300 | center_line | off → on | Environment |
| 5 | 350 | intersection_type | T → Cross | Intersection |
| 6 | 400 | turn_direction | Left → Right | Intersection |
| 7 | 450 | goal_distance | 120 → 150m | Environment |
| 8 | 500 | num_active_npcs | 0 → 1 | NPC |
| 9 | 550 | intersection_type | Cross → Y | Intersection |
| 10 | 600 | goal_distance | 150 → 200m | Environment |
| 11 | 700 | num_active_npcs | 1 → 2 | NPC |
| 12 | 750 | npc_speed_ratio | 0.5 → 0.7 | NPC |
| 13 | 800 | npc_speed_variation | 0 → 0.15 | NPC |

### Disabled Parameters
- road_curvature: 0 (straight roads, intersection focus)
- curve_direction_variation: 0
- speed_zone_count: 1 (single zone, 60 km/h)

### Hyperparameters
| Parameter | Value | Note |
|-----------|-------|------|
| batch_size | 4096 | Proven across all phases |
| buffer_size | 40960 | Same |
| learning_rate | 1.5e-4 | Same |
| learning_rate_schedule | **constant** | NEVER linear (Phase F v3 lesson) |
| hidden_units | 512 | Same |
| num_layers | 3 | Same |
| max_steps | 10,000,000 | Fresh start needs generous budget |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-G.yaml \
  --run-id=phase-G --force
```

**No `--initialize-from`**: 254D→260D dimension change prevents transfer.

## Pre-Training Checklist

1. **Scene verification (P-011)**:
   - Active scene must be `PhaseG_Intersection`
   - Verify via Unity Editor or `manage_scene(action="get_active")`

2. **Agent configuration**:
   - `enableIntersectionObservation = true` (all 16 agents)
   - `VectorObservationSize = 260` (BehaviorParameters)

3. **WaypointManager**:
   - `intersectionType`, `turnDirection` fields present
   - `SetIntersection()` method functional

4. **Console check**:
   - No compilation errors after scene load
   - `read_console` clean

## Success Criteria

1. All 13 curriculum transitions complete within 10M steps
2. No single transition causes >500 reward drop
3. Final reward: +500 or higher
4. Y-junction navigation with 2 NPCs demonstrated
5. Turn execution (left/right) with correct lane positioning

## Risk Assessment

- **Fresh start**: No checkpoint transfer. Phase D v3 achieved +895 from scratch in 5M steps with similar setup, so 10M should be sufficient.
- **Intersection complexity**: Turns require coordinated steering + speed control. May be harder than lane changes.
- **P-009 combined change**: 260D + intersection env together. Mitigated by fresh start (no existing policy to destabilize).
