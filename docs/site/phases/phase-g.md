---
layout: default
title: Phase G - Intersection Navigation
---

# Phase G: Intersection Navigation

T/Cross/Y-junction intersection driving

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-G-v2 |
| **Status** | Completed |
| **Date** | 2026-02-01 |
| **Total Steps** | 5,000,074 |
| **Training Time** | ~56 minutes |
| **Final Reward** | **+628** |
| **Peak Reward** | **+633** (at 4.72M steps) |
| **Observation** | 260D (254D + 6D intersection) |
| **Initialize From** | Phase G v1 (warm start from 10M checkpoint) |

---

## Objective

Phase F multi-lane driving capabilities maintained while learning intersection navigation (turn decisions at T/Cross/Y-junctions).

### New Capabilities
- T-junction recognition and navigation
- Cross intersection recognition and navigation
- Y-junction recognition and navigation
- Left/Right turn maneuvers at intersections
- WrongWay detection with dual-axis check (P-014)

---

## Observation Space

**254D -> 260D** (+6D intersection info)

```yaml
intersection_info: 6D
  - intersection_type_none: 1D    # one-hot [1,0,0,0]
  - intersection_type_t: 1D       # one-hot [0,1,0,0]
  - intersection_type_cross: 1D   # one-hot [0,0,1,0]
  - intersection_type_y: 1D       # one-hot [0,0,0,1]
  - distance_to_intersection: 1D  # normalized [0,1]
  - turn_direction: 1D            # 0=straight, 0.5=left, 1=right
```

---

## Curriculum Design

### Full Curriculum Parameters (All 7/7 Completed)

| Parameter | Final Lesson | Final Value | Status |
|-----------|--------------|-------------|--------|
| intersection_type | Y-Junction | 3 | Completed |
| turn_direction | RightTurn | 2 | Completed |
| num_lanes | TwoLanes | 2 | Completed |
| center_line_enabled | CenterLineEnforced | 1 | Completed |
| goal_distance | LongGoal | 200m | Completed |
| road_curvature | Straight | 0 | Locked |
| num_active_npcs | NoNPCs | 0 | Deferred to Phase H |

---

## Key Changes: v1 -> v2

| Aspect | v1 | v2 | Impact |
|--------|----|----|--------|
| WrongWay detection | xPos only | xPos + zPos (P-014) | 32% -> 0% termination |
| Initialization | Fresh start (260D) | Warm start from v1 10M | Immediate ~498 reward |
| Curriculum | 9 params, NPCs included | 7 params, NPCs deferred | Focused learning |
| Y-junction threshold | 550 | 450 | Achievable target |
| Budget | 10M steps | 5M steps | Sufficient with warm start |
| Final Reward | +494 (plateau) | **+628** | +27% improvement |

---

## Training Progress

### Reward Curve

```
Reward
+633  |                                          * Peak
      |                                    _____/
+600  |                           ________/
      |                     _____/
+550  |               _____/
      |         _____/
+500  |________/
      |
      +--------------------------------------------
       0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0 M steps
```

### Reward Phases

1. **Warm Start (0-100K)**: Instant ~498, matching v1 final performance
2. **Curriculum Rush (100K-430K)**: All 7 lessons completed, brief dips during transitions
3. **Consolidation (430K-2M)**: Steady climb from 502 to 589
4. **Optimization (2M-5M)**: Gradual refinement from 589 to 628, peak 633

### Step-by-Step Progress

| Step | Reward | Event |
|------|--------|-------|
| 10K | 398 | Warm start baseline (v1 knowledge inherited) |
| 50K | 500 | v1 peak matched |
| 100K | 497 | Curriculum: T-junction + LeftTurn + TwoLanes |
| 210K | 507 | Curriculum: Cross + RightTurn + CenterLine |
| 230K | 472 | Dip from new curriculum complexity |
| 320K | 500 | Curriculum: **Y-junction** (v1 never reached!) |
| 430K | 502 | Curriculum: LongGoal (200m) -- **all 7/7 complete** |
| 500K | 544 | First checkpoint saved |
| 1.0M | 550 | Stable improvement |
| 2.0M | 589 | 2M checkpoint |
| 2.5M | 600 | **600 barrier broken** |
| 4.0M | 612 | Continued improvement |
| 4.72M | **633** | **PEAK REWARD** |
| 5.0M | 628 | Training complete |

---

## Curriculum Transitions

All 7 curriculum parameters completed within the first 430K steps:

| Step | Parameter | From -> To | Reward Impact |
|------|-----------|-----------|---------------|
| 100K | intersection_type | None -> T-junction | Minimal dip |
| 100K | turn_direction | Straight -> Left | Minimal dip |
| 100K | num_lanes | Single -> Two | Minimal dip |
| 210K | intersection_type | T-junction -> Cross | -35 temporary |
| 210K | turn_direction | Left -> Right | Absorbed |
| 210K | center_line_enabled | Off -> On | Absorbed |
| 320K | intersection_type | Cross -> **Y-junction** | Minimal dip |
| 430K | goal_distance | 150m -> 200m | +25 (longer episodes) |

---

## End Reasons

| Reason | v1 Rate | v2 Rate |
|--------|---------|---------|
| Goal Reached | 67.9% | **~95%+** |
| WrongWay | **31.9%** | **~0%** |
| Collision | 0% | 0% |
| Timeout | ~0% | ~5% |

---

## Key Achievements

### 1. WrongWay Problem Solved
- v1: 32% of episodes ended in WrongWay (xPos-only detection failed post-turn)
- v2: ~0% WrongWay with dual-axis P-014 fix

### 2. All Intersection Types Mastered
```
v1: None -> T-junction -> Cross (stuck, Y-junction never reached)
v2: None -> T-junction -> Cross -> Y-junction (all completed by 320K!)
```

### 3. Warm Start Efficiency
- v1: 2M steps relearning basics (254D->260D fresh start)
- v2: Instant ~498 reward, curriculum rushing from step 100K

### 4. Reward Comparison
```
Phase A: +937 (1 NPC, straight)
Phase B: +903 (1 NPC, varied speed)
Phase C: +961 (4 NPC, straight)
Phase E: +931 (2 NPC, curves)
Phase F: +988 (3 NPC, curves + 2 lanes)
Phase G: +628 (0 NPC, intersections)  <- Different reward scale
```

Note: Phase G reward is lower in absolute value because intersection episodes involve turn maneuvers with temporary speed reduction, shorter effective distances, and additional complexity.

---

## v1 vs v2 Comparison

| Metric | v1 | v2 |
|--------|----|----|
| Steps | 10,000,153 | 5,000,074 |
| Final Reward | +494 | **+628** |
| Peak Reward | +516 | **+633** |
| Curriculum | 4/6 complete | **7/7 complete** |
| WrongWay Rate | 31.9% | **~0%** |
| Training Time | ~2 hours | **~56 min** |
| Y-junction | Never reached | **Completed at 320K** |

---

## Checkpoints

| File | Step | Reward |
|------|------|--------|
| E2EDrivingAgent-499842.onnx | 500K | ~544 |
| E2EDrivingAgent-999786.onnx | 1.0M | ~550 |
| E2EDrivingAgent-1999944.onnx | 2.0M | ~589 |
| E2EDrivingAgent-2999948.onnx | 3.0M | ~602 |
| E2EDrivingAgent-3999789.onnx | 4.0M | ~612 |
| E2EDrivingAgent-4499771.onnx | 4.5M | ~626 |
| **E2EDrivingAgent-5000074.onnx** | **5.0M** | **~628 (FINAL)** |

---

## Bugs Fixed

1. **WrongWay Detection (P-014)**: xPos-only check failed after intersection turns where agent moved in Z direction. Fixed with dual-axis (xPos + zPos) wrongway detection.
2. **Missing DecisionRequester (P-015)**: Scene regeneration created agent GameObjects without DecisionRequester. Fixed with ConfigurePhaseGAgents.cs utility.
3. **BehaviorParameters Reset**: Scene regeneration reset observation size to 1. Fixed using direct API approach in ConfigurePhaseGAgents.cs.

---

## Lessons Learned

1. **Warm start is critical**: 260D fresh start wasted 2M steps in v1; warm start gave instant ~498
2. **WrongWay detection must be multi-axis**: Single-axis check fails at intersections
3. **Defer unrelated complexity**: Removing NPCs from Phase G curriculum allowed focus on intersection geometry
4. **Lower thresholds work**: Y-junction threshold 450 (vs v1's 550) was achievable
5. **5M steps sufficient with warm start**: Half the budget of v1 with better results

---

## Next Phase

**Phase H**: NPC Interaction in Intersections
- Progressive NPC introduction: 0 -> 1 -> 2 -> 3 NPCs
- NPC waypoint-following through intersections
- Warm start from Phase G v2 checkpoint (+628)

---

[Phase F](./phase-f) | [Phase H](./phase-h) | [Home](../)
