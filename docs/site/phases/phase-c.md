---
layout: default
title: Phase C - Multi-NPC Generalization
---

# Phase C: Multi-NPC Generalization

복잡한 다중 NPC 환경에서의 일반화 학습

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-C_242D |
| **Status** | Completed |
| **Date** | 2026-01-27 |
| **Total Steps** | 4,000,000 |
| **Training Time** | ~54 minutes (3269 seconds) |
| **Final Reward** | **+961.8** |
| **Peak Reward** | **+1086.0** (at 3.85M steps) |
| **Observation** | 242D |
| **Initialize From** | Phase B (phase-B) |

---

## Objective

Phase A/B에서 학습한 추월 능력을 복잡한 다중 NPC 환경으로 확장

### Environment Complexity
- NPC 수: 1 -> 2 -> 3 -> 4
- Speed Zone: 1 -> 2 -> 4
- Goal Distance: 100m -> 160m -> 230m
- NPC Speed Variation: 0.1 -> 0.2 -> 0.3

---

## Curriculum Design

### Multi-parameter Curriculum

| Parameter | Lesson 0 | Lesson 1 | Lesson 2 |
|-----------|----------|----------|----------|
| num_active_npcs | 1 | 2 | 4 |
| goal_distance | 100m | 160m | 230m |
| speed_zone_count | 1 | 2 | 4 |
| npc_speed_variation | 0.1 | 0.2 | 0.3 |

### Curriculum Thresholds
- Threshold for advancement: 300.0
- min_lesson_length: 50000 steps

---

## Training Progress

| Step | Reward | Std | Curriculum Event | Notes |
|------|--------|-----|------------------|-------|
| 0K | +546 | - | Start | Initialized from Phase B |
| 90K | +766 | - | Pre-transition | Peak before shock |
| 110K | **-814** | - | Curriculum shock | goal_distance->160m, zones->2 |
| 500K | -777 | - | Recovery | Learning new environment |
| 760K | +11 | - | Positive again | Adaptation complete |
| 960K | +207 | - | zones->4 | Another transition |
| 1.08M | +127 | - | goal_distance->230m | Final curriculum |
| 2M | +1000 | - | Stable | High performance |
| 3.85M | **+1086** | - | **Peak** | Best performance |
| **4M** | **+961** | 182 | **Final** | **Converged** |

---

## Curriculum Shock Analysis

### The Shock Pattern

```
Reward
+766  |       * Peak (90K)
      |      *
+500  |     *
      |    *
    0 |───*─────────────────────*───
      |                        * (760K)
-400  |
      |
-814  |            * Shock (110K)
      └────────────────────────────────
       0   0.1   0.3   0.5   0.8   1.0 M steps
```

### Recovery Timeline
- **Pre-shock peak**: +766 (at 90K)
- **Shock drop**: -814 (at 110K) - 1580 point drop!
- **Recovery start**: 500K steps
- **Positive again**: 760K steps (650K to recover)
- **Final performance**: +961 (exceeds pre-shock!)

**Key Insight**: Curriculum transitions cause temporary drops, but recovery is possible with sufficient training time.

---

## Key Observations

### 1. Phase B Knowledge Preserved
```
Phase B Final:   +903
Phase C Initial: +546 (60% preserved)
Phase C Final:   +961 (+6% improvement in harder environment)
```

### 2. Resilient Learning
- Multiple curriculum transitions handled
- 4 NPCs with 4 speed zones mastered
- 230m goal distance achieved

### 3. Curriculum Shock Recovery
- Agent recovered from -1329 to +1000 within 1M steps
- Final reward exceeds Phase B despite 4x complexity

---

## Phase Comparison

| Aspect | Phase A | Phase B | Phase C |
|--------|---------|---------|---------|
| NPC Count | 1 | 1 | 1-4 |
| NPC Speed | 30% | 30-90% | 30-100% varied |
| Speed Zones | 1 | 1 | 1-4 |
| Goal Distance | 150m | 150m | 100-230m |
| Final Reward | +937 | +903 | **+961** |
| Peak Reward | +937 | +994 | **+1086** |
| Training Time | 30 min | 16 min | **54 min** |

---

## Reward Curve

```
Reward
+1086 │                             ★ Peak (3.85M)
      │                           ╱╲
+1000 │_________________________╱  ╲___
      │
 +766 │ ★ (pre-shock)
      │
 +500 │    ╲
      │     ╲
    0 │──────────────────*─────────────
      │                 ↑ (760K recovery)
-400  │        ╲      ╱
      │         ╲    ╱
-814  │          ★ Shock
      └────────────────────────────────
       0   0.5   1.0   2.0   3.0   4.0 M steps
```

---

## Checkpoints

| Checkpoint | Step | Reward |
|------------|------|--------|
| E2EDrivingAgent-499806.onnx | 500K | ~-700 (recovery) |
| E2EDrivingAgent-999802.onnx | 1M | ~+100 |
| E2EDrivingAgent-2999929.onnx | 3M | ~+1000 |
| E2EDrivingAgent-4000054.onnx | 4M | +961 |
| **E2EDrivingAgent.onnx** | Final | +961 |

---

## Lessons Learned

1. **Curriculum shock is normal**: Temporary reward drops expected during transitions
2. **Recovery time varies**: NPC count increase took ~650K steps to recover
3. **Knowledge preservation works**: Phase B checkpoint provides strong foundation
4. **Final performance can exceed**: +961 > +903 despite 4x complexity increase
5. **Multi-parameter curriculum**: Multiple parameters can change simultaneously with careful thresholds

---

## Verified Behaviors

| Environment | Expected | Observed |
|-------------|----------|----------|
| 1 NPC @ 30% | Overtake | Overtake |
| 2 NPCs varied | Navigate safely | Navigate safely |
| 4 NPCs, 4 zones | Adapt to speed changes | Adapt successfully |
| 230m goal | Complete route | Complete route |

---

## Next Phase

**Phase D**: Lane Observation (254D)
- Observation space expansion: 242D -> 254D
- Lane features: centerline offset, lane type, curvature
- Foundation for lane-aware driving

---

[Phase B](./phase-b) | [Phase E](./phase-e) | [Home](../)

