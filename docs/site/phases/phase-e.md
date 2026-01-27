---
layout: default
title: Phase E - Curved Roads
---

# Phase E: Curved Roads Navigation

곡선 도로 주행 학습

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | v12_phaseE_v2 |
| **Status** | Completed |
| **Date** | 2026-01-27 |
| **Total Steps** | 6,000,090 |
| **Training Time** | ~70 minutes (4231 seconds) |
| **Final Reward** | **+931.1** |
| **Peak Reward** | **+931.1** (at 6M steps) |
| **Observation** | 254D |
| **Initialize From** | Phase D (v12_phaseD) |

---

## Objective

직선 도로에서 곡선 도로로 환경 확장

### Road Complexity
- 곡률 (Curvature): 0 -> 0.3 -> 0.6 -> 1.0
- 방향 변화: 단일 -> 혼합 (좌/우)
- NPC 수: 0 -> 1 -> 2 (안전을 위해 제한)

---

## Curriculum Design

### Curvature Curriculum

```
Lesson 0: Straight (0.0)
    │ threshold: 200
    ▼
Lesson 1: Gentle (0.3)
    │ threshold: 200
    ▼
Lesson 2: Moderate (0.6)
    │ threshold: 200
    ▼
Lesson 3: Sharp (1.0)
    (final)
```

### Full Curriculum Parameters (All Completed)

| Parameter | Final Lesson | Final Value | Status |
|-----------|--------------|-------------|--------|
| road_curvature | SharpCurves | 1.0 | Completed |
| curve_direction_variation | MixedDirections | 1.0 | Completed |
| num_active_npcs | TwoNPCs | 2 | Completed |
| npc_speed_ratio | MediumNPCs | 0.7 | Completed |
| goal_distance | LongGoal | 200m | Completed |
| speed_zone_count | TwoZones | 2 | Completed |
| npc_speed_variation | Varied | 0.2 | Completed |

---

## Training Progress

| Step | Reward | Curriculum State | Notes |
|------|--------|------------------|-------|
| 0K | - | Initialize | From Phase D checkpoint |
| 4.5M | +858 | Advancing | Learning curves |
| 5.0M | +876 | Advancing | Stable progress |
| 5.5M | +897 | SharpCurves | Good adaptation |
| **6.0M** | **+931** | **All Complete** | **Training finished** |

---

## Key Changes from Phase D

| Aspect | Phase D | Phase E |
|--------|---------|---------|
| Road Type | Straight only | **Curved roads** |
| Curvature | 0 | 0 -> 0.3 -> 0.6 -> **1.0** |
| Curve Direction | N/A | Single -> **Mixed** |
| Goal Distance | 80-230m | 100-**200m** (shorter for curves) |
| NPC Count | 1-4 | 0-**2** (safety on curves) |
| Final Reward | +332 | **+931** (+180% improvement) |

---

## Reward Curve

```
Reward
+931  │                                    ★ Final
      │                               ╱────
+897  │                          ____╱
      │                    _____╱
+876  │               ____╱
      │          ____╱
+858  │_________╱
      │
      └────────────────────────────────────────
       4.5   5.0   5.5   6.0 M steps
```

**Note**: Consistent upward trend with no major curriculum shocks!

---

## Key Achievements

### 1. All Curriculum Lessons Passed
- Sharp curves (curvature 1.0) mastered
- Mixed curve directions handled
- 2 NPCs on curved roads managed

### 2. Massive Reward Improvement
```
Phase D Final:  +332
Phase E Final:  +931  (+180% improvement!)
```

### 3. Long Goals on Curves
- 200m goal distance achieved on curved roads
- No increase in collision rate

---

## Success Criteria Verification

| Criterion | Target | Result |
|-----------|--------|--------|
| Navigate gentle curves (0.3) | No off-road | Passed |
| Navigate sharp curves (1.0) | Safe speed | Passed |
| Maintain reward > +200 on curves | With 1 NPC | **+931 with 2 NPCs!** |
| No collision rate increase | vs Phase D | Verified |

---

## Phase Comparison

| Phase | Reward | Environment | Observation | Key Learning |
|-------|--------|-------------|-------------|--------------|
| Phase A | +937 | 1 NPC @ 30% | 242D | Overtaking |
| Phase B | +903 | 1 NPC @ 30-90% | 242D | Decision making |
| Phase C | +961 | 4 NPC, 4 zones | 242D | Multi-NPC |
| Phase D | +332 | 2 NPC, lane obs | 254D | Lane awareness |
| **Phase E** | **+931** | **2 NPC, curves** | **254D** | **Curve navigation** |

---

## Checkpoints

| Checkpoint | Step | Reward |
|------------|------|--------|
| E2EDrivingAgent-4499924.onnx | 4.5M | +858 |
| E2EDrivingAgent-4999885.onnx | 5M | +876 |
| E2EDrivingAgent-5499938.onnx | 5.5M | +897 |
| E2EDrivingAgent-5999834.onnx | 6M | +931 |
| **E2EDrivingAgent.onnx** | Final | +931 |

---

## Lessons Learned

1. **Curved road curriculum works**: Progressive curvature (0->0.3->0.6->1.0) is effective
2. **Phase D initialization crucial**: Starting from Phase D enabled rapid learning
3. **NPC handling on curves**: Agent successfully manages NPCs even on sharp curves
4. **No curriculum shock**: Smooth reward progression throughout training
5. **Reduced NPC count helps**: Limiting to 2 NPCs allowed focus on curve learning

---

## Verified Behaviors

| Road Type | NPC Count | Expected | Observed |
|-----------|-----------|----------|----------|
| Straight | 0 | Maintain lane | Correct |
| Gentle curve (0.3) | 1 | Navigate safely | Correct |
| Moderate curve (0.6) | 1 | Adjust speed | Correct |
| Sharp curve (1.0) | 2 | Safe navigation + NPC avoidance | Correct |

---

## Curve Navigation Visualization

```
Sharp Curve (curvature=1.0)

    ●───●       <- NPCs
   ╱     ╲
  ╱       ╲
 ╱    ▲    ╲    <- Agent navigating
╱     │     ╲
      │
    Start
```

The agent learned to:
1. Reduce speed before entering curve
2. Maintain lane position through curve
3. Avoid NPCs while navigating

---

## Next Phase

**Phase F**: Multi-Lane Roads
- Lane count: 1 -> 2
- Center line enforcement
- Curved + multi-lane combination

---

[Phase C](./phase-c) | [Phase F](./phase-f) | [Home](../)

