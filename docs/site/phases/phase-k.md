---
layout: default
title: Phase K - Dense Urban
---

# Phase K: Dense Urban Integration

Curved roads + intersections + traffic signals -- the first comprehensive driving test

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-K |
| **Status** | IN PROGRESS |
| **Date** | 2026-02-02 |
| **Total Steps** | 5M (target) |
| **Final Reward** | *training* |
| **Peak Reward** | *training* |
| **Observation** | 268D (same as Phase J) |
| **Initialize From** | Phase J v5 5M checkpoint |
| **Training Mode** | Editor mode (visual) / Build + 3 envs |

---

## Objective

For the first time, combine ALL driving skills simultaneously:
- Curved approach road leading into intersection
- Traffic signal compliance at intersection
- NPC traffic with speed variation
- Multi-lane (2 lanes) with center line rules

Previously, curved roads and intersections were **mutually exclusive** in WaypointManager. Phase K adds `CollectCurvedIntersectionPositions` to support both simultaneously.

### New Capabilities
- Curved approach road + intersection turn (new waypoint generation path)
- Simultaneous curvature + signal compliance
- Dense urban driving generalization

---

## Code Changes

### WaypointManager.cs
Added `CollectCurvedIntersectionPositions()`:
1. **Curved approach**: Standard curved waypoints until 30m before intersection
2. **Straightening zone**: Smooth interpolation (ease-in-out) returning X to lane position and heading to 0
3. **Intersection maneuver**: Reuses existing turn logic (straight/left/right)
4. **Exit section**: Reuses existing exit logic

### DrivingSceneManager.cs
- Always set `roadCurvature` field (fixes stale value from previous episodes)
- Always call `SetIntersection()` to regenerate waypoints with both params
- Updated scene validation to accept PhaseK scene name

---

## Observation Space

**268D (identical to Phase J -- no new dimensions)**

| Component | Dimensions | Source |
|-----------|-----------|--------|
| ego_state | 8D | Phase 0 |
| ego_history | 40D (5x8D) | Phase 0 |
| surrounding vehicles | 160D (20x8) | Phase D |
| route_info | 30D (10x3) | Phase 0 |
| speed_info | 4D | Phase D |
| lane_info | 12D | Phase D |
| intersection_info | 6D | Phase G |
| traffic_signal | 8D | Phase J |
| **Total** | **268D** | |

---

## Curriculum Design

Single parameter curriculum (P-022 compliant): `road_curvature`

| Lesson | road_curvature | Threshold | Expected Reward |
|--------|---------------|-----------|-----------------|
| NoCurve | 0.0 | 490 | ~530-540 (same as J v5) |
| LightCurve | 0.3 | 430 | ~430-480 |
| MediumCurve | 0.5 | (final) | ~400-450 |

### Locked Parameters (J v5 final values)

| Parameter | Value | Notes |
|-----------|-------|-------|
| intersection_type | 3 (Y-junction) | From Phase G v2 |
| traffic_signal_enabled | 1 (ON) | From Phase J |
| signal_green_ratio | 0.4 | From Phase J v5 |
| num_active_npcs | 3 | From Phase H |
| npc_speed_ratio | 0.85 | Standard |
| npc_speed_variation | 0.15 | From Phase H v3 |
| num_lanes | 2 | From Phase F |
| center_line_enabled | 1 | From Phase F |
| curve_direction_variation | 0.5 | Random curves |
| goal_distance | 230m | Standard |

---

## Version History

| Metric | v1 |
|--------|-----|
| Init from | Phase J v5 (268D) |
| Steps | 5M (target) |
| Curriculum | road_curvature 0/0.3/0.5 |
| Peak Reward | *training* |
| Final Reward | *training* |
| Code changes | Curved+Intersection waypoints |

---

## Training Progress

*Will be updated as training completes*

---

## Architecture: Curved + Intersection Waypoints

```
      Curved Approach          Straighten    Intersection     Exit
  ~~~~~~~~~~~~~~~~~~~~~~~ ============== ======/======= -----------
  Z=-250m            Z=63m         Z=93m    Z=100m    Z=107m   Z=250m
                              (30m zone)   (turn arc)

  ~~~ = Curved waypoints (random heading changes)
  === = Smoothing zone (X returns to lane, heading returns to 0)
  /   = Turn arc (left/right/straight, reused from Phase G)
  --- = Straight exit
```

---

## Artifacts

- Config: `python/configs/planning/vehicle_ppo_phase-K.yaml`
- Editor Config: `python/configs/planning/vehicle_ppo_phase-K-editor.yaml`
- Scene: `Assets/Scenes/PhaseK_DenseUrban.unity`
- Build: `Builds/PhaseK/PhaseK.exe` (118 MB)
- Warm Start: `results/phase-J-v5/E2EDrivingAgent/E2EDrivingAgent-5000148.pt`

---

[Phase J](./phase-j) | [Home](../)
