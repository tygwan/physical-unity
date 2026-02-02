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
| **Run ID** | phase-K-v1 |
| **Status** | COMPLETE (3/3 curriculum) |
| **Date** | 2026-02-02 |
| **Total Steps** | 5M |
| **Training Time** | 25.3 min (Build + 3 envs) |
| **Final Reward** | **+590** |
| **Peak Reward** | **+703.2** (at 4.67M) |
| **Observation** | 268D (same as Phase J) |
| **Initialize From** | Phase J v5 5M checkpoint |
| **Training Mode** | Build + 3 envs (no_graphics) |

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
- Updated scene validation to accept PhaseK scene name (multi-lane + intersection)

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

| Lesson | road_curvature | Threshold | Actual Transition |
|--------|---------------|-----------|-------------------|
| NoCurve | 0.0 | 490 | ~780K (reward 599) |
| LightCurve | 0.3 | 430 | ~1.62M (reward 682) |
| MediumCurve | 0.5 | (final) | -- |

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
| Steps | 5M |
| Curriculum | road_curvature 0/0.3/0.5 |
| Peak Reward | **+703.2** |
| Final Reward | +590 |
| Code changes | Curved+Intersection waypoints |

---

## Training Progress

### Reward Curve

```
Reward
+703  |                                                          *703 (4.67M)
      |                                          ___/ \__/ \__/ \_/ \__
+680  |                                       __/                       \__
      |                    *682 (1.62M)     __/
+650  |                   / \___          _/
      |         *654    _/      \_      _/
+620  |        / \   __/         \_  __/
      |       /   \_/              \/
+600  | *599/                        576 (2.0M, curve shock)
      |  /
+550  | /
      |/
+530  |* (30K, warm start recovery)
      +--------------------------------------------------------------------
       0    0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0 M
                   ^              ^
                  0.3            0.5
```

### Curriculum Transition Timeline

| Step | Parameter | Transition | Threshold | Status |
|------|-----------|-----------|-----------|--------|
| ~780K | road_curvature | 0.0 -> 0.3 | 490 | DONE (reward 599) |
| ~1.62M | road_curvature | 0.3 -> 0.5 | 430 | DONE (reward 682) |

### Key Milestones

| Step | Reward | Event |
|------|--------|-------|
| 30K | 526 | Warm start recovery (J v5 baseline) |
| 780K | 599 | Curriculum -> curvature 0.3 |
| 810K | 654 | Quick adaptation to light curves |
| 1.62M | 682 | Curriculum -> curvature 0.5 |
| 1.66M | 696 | Quick adaptation to medium curves |
| 2.0M | 576 | Curve shock (temporary dip) |
| 4.67M | **703** | Peak reward |
| 5.0M | 590 | Final (high variance, std=425) |

---

## Phase Comparison

| | Phase J v5 | Phase K v1 | Delta |
|---|-----------|-----------|-------|
| Peak Reward | +605 | **+703** | **+98** |
| Final Reward | +537 | +590 | +53 |
| Conditions | Signals + Intersection | Signals + Intersection + **Curves** | +Curves |
| Curriculum | 5/5 green_ratio | 3/3 road_curvature | Both complete |

Despite adding curved roads, Phase K achieved **+98 higher peak** than Phase J v5. The warm start from J v5 enabled rapid skill combination.

---

## Key Findings

### 1. Skill Integration Works
Combining curved roads + intersections + signals + NPCs simultaneously yielded +703 peak, surpassing Phase J (+605) despite increased complexity.

### 2. Fast Curriculum Completion
Both transitions completed within 1.62M steps (total 5M). The agent adapted to curvature 0.3 within ~30K steps of transition.

### 3. High Variance at Full Complexity
At curvature=0.5, reward std reached 425 (final step), indicating some episodes crash while others succeed. The curved approach + intersection + signal combination creates challenging edge cases.

### 4. Editor Inference Works (P-025 Resolved)
ONNX model works correctly in editor inference mode. The initial failure was due to BehaviorType=1 (HeuristicOnly) being set instead of BehaviorType=2 (InferenceOnly). With correct setting, the agent drives at 14-17 m/s, navigates curves, and achieves reward 776.8 in editor — surpassing the training peak of 703.

### 5. Signal Compliance Partial
Agent sometimes runs red lights (RED_LIGHT_VIOLATION), sometimes stops correctly (Signal=0.00 in best episode). Signal compliance is learned but not 100% reliable.

---

## Lessons Learned

### P-025: BehaviorType Enum Values (RESOLVED)
ML-Agents BehaviorType enum: 0=Default, 1=HeuristicOnly, 2=InferenceOnly. Setting BehaviorType=1 uses keyboard/expert input (Heuristic fallback returns [0,0] actions), NOT the ONNX model. Always use BehaviorType=2 for ONNX inference.

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
- ONNX Model: `results/phase-K-v1/E2EDrivingAgent.onnx`
- Best Checkpoint: `results/phase-K-v1/E2EDrivingAgent/E2EDrivingAgent-5000083.pt`

---

[Phase J](./phase-j) | [Home](../)

*Last Updated: 2026-02-02 (Phase K v1 Complete, 3/3 curriculum)*
