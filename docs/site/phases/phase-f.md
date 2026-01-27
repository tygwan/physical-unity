---
layout: default
title: Phase F - Multi-Lane Roads
---

# Phase F: Multi-Lane Roads

다차선 도로 및 중앙선 규칙 학습

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-F |
| **Status** | Completed |
| **Date** | 2026-01-27 |
| **Total Steps** | 6,000,000 |
| **Training Time** | ~70 minutes |
| **Final Reward** | **+988** |
| **Peak Reward** | **+988** (at 6M steps) |
| **Observation** | 254D |
| **Initialize From** | Phase E (phase-E) |

---

## Objective

단일 차선에서 다차선 도로로 환경 확장

### Lane Complexity
- 차선 수: 1 -> 2
- 중앙선: 비활성 -> 활성 (규칙 적용)
- 곡률: 0 -> 0.3 -> 0.6 (곡선 + 다차선 복합)

---

## Curriculum Design

### Lane Curriculum

```
Lesson 0: SingleLane (1 lane)
    │ threshold: 300
    ▼
Lesson 1: TwoLanes (2 lanes)
    │ threshold: 300
    ▼
Lesson 2: CenterLineEnforced
    (final)
```

### Full Curriculum Parameters (All Completed)

| Parameter | Final Lesson | Final Value | Status |
|-----------|--------------|-------------|--------|
| num_lanes | TwoLanes | 2 | Completed |
| center_line_enabled | CenterLineEnforced | 1 | Completed |
| road_curvature | ModerateCurve | 0.6 | Completed |
| curve_direction_variation | MixedDirections | 1.0 | Completed |
| num_active_npcs | ThreeNPCs | 3 | Completed |
| npc_speed_ratio | MediumNPCs | 0.7 | Completed |
| goal_distance | LongGoal | 200m | Completed |

---

## Key Changes from Phase E

| Aspect | Phase E | Phase F |
|--------|---------|---------|
| Lanes | 1 | 1 -> **2** |
| Center Line | No | **Yes (enforced)** |
| Curvature | 0 -> 1.0 | 0 -> **0.6** |
| NPC Count | 0-2 | 0-**3** |
| Final Reward | +931 | **+988** (+6% improvement) |

---

## Training Progress

Phase F showed consistently high performance:

```
Reward
+988  │                              ★ Final
      │                        ╱─────
+950  │                  _____╱
      │            _____╱
+900  │_______╱____
      │
      └────────────────────────────────────────
       1.0   2.0   3.0   4.0   5.0   6.0 M steps
```

---

## Key Achievements

### 1. Multi-Lane Mastery
- 2-lane road navigation achieved
- Proper lane selection and maintenance

### 2. Center Line Compliance
- Learned to respect center line rules
- No illegal center line crossings

### 3. Complexity Stacking
- Curved roads + Multi-lanes + 3 NPCs handled simultaneously

### 4. Best Reward in v12 Series
```
Phase A: +937 (1 NPC, straight)
Phase B: +903 (1 NPC, varied speed)
Phase C: +961 (4 NPC, straight)
Phase E: +931 (2 NPC, curves)
Phase F: +988 (3 NPC, curves + 2 lanes) <- BEST!
```

---

## Phase Comparison

| Phase | Reward | Lanes | Curves | NPCs | Key Feature |
|-------|--------|-------|--------|------|-------------|
| Phase A | +937 | 1 | 0 | 1 | Overtaking |
| Phase B | +903 | 1 | 0 | 1 | Decision making |
| Phase C | +961 | 1 | 0 | 4 | Multi-NPC |
| Phase E | +931 | 1 | 0-1.0 | 2 | Curves |
| **Phase F** | **+988** | **2** | **0-0.6** | **3** | **Multi-lane** |

---

## Reward Evolution Chart

```
Reward
+1000 ─────────────────────────────────★ Phase F (+988)
      │                        ────────
+950  ─────────────────────────────────★ Phase C (+961)
      │                 ───────────────★ Phase A (+937)
+900  ─────────────────────────────────★ Phase E (+931)
      │                  ──────────────★ Phase B (+903)
      │
+850  ─────────────────────────────────────────────────
      │
      └────────────────────────────────────────────────
       Phase A    B    C    D    E    F
```

---

## Checkpoints

| Checkpoint | Step | Reward |
|------------|------|--------|
| E2EDrivingAgent-1999xxx.onnx | 2M | ~+900 |
| E2EDrivingAgent-3999xxx.onnx | 4M | ~+950 |
| E2EDrivingAgent-5999xxx.onnx | 6M | +988 |
| **E2EDrivingAgent.onnx** | Final | +988 |

---

## Lessons Learned

1. **Multi-lane curriculum works**: Progressive lane count increase is effective
2. **Center line integration smooth**: No major curriculum shock from center line rules
3. **Phase E init crucial**: Curve knowledge from Phase E transferred well
4. **Complexity stacking possible**: Curves + lanes + NPCs can be learned together
5. **Reduced curvature helps**: Limiting to 0.6 curvature allowed focus on lane learning

---

## Verified Behaviors

| Scenario | Expected | Observed |
|----------|----------|----------|
| 1-lane straight | Maintain lane | Correct |
| 2-lane straight | Choose appropriate lane | Correct |
| 2-lane with NPC | Overtake in adjacent lane | Correct |
| Curved 2-lane | Navigate safely | Correct |
| Center line active | No crossing | Correct |

---

## Multi-Lane Visualization

```
2-Lane Road with Center Line

    │  Lane 2  │  Lane 1  │
    │          │   ▲      │  <- Agent in Lane 1
    │   ●      │   │      │  <- NPC in Lane 2
    │          │   │      │
    ├──────────┼──────────┤  <- Center Line
    │          │          │
    │    ●     │          │  <- Oncoming NPC
    │          │          │
```

The agent learned to:
1. Stay in appropriate lane
2. Use Lane 2 for overtaking
3. Return to Lane 1 after passing
4. Respect center line boundary

---

## Next Phase

**Phase G**: Intersection Navigation
- Intersection types: None -> T-Junction -> Cross -> Y-Junction
- Turn directions: Straight -> Left -> Right
- Observation expansion: 254D -> 260D (+6D intersection info)

---

[Phase E](./phase-e) | [Phase G](./phase-g) | [Home](../)

