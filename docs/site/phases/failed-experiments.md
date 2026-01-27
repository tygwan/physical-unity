---
layout: default
title: Failed Experiments
---

# Failed Experiments

실패한 접근 방식과 그 교훈

---

## Overview

| Experiment | Approach | Result | Key Lesson |
|------------|----------|--------|------------|
| v10g | Lane keeping + followingBonus | Plateau at +40 | followingBonus prevents overtaking |
| v11 | Sparse overtake reward | +51 (8M steps) | Sparse reward insufficient |
| HybridPolicy | Transfer learning + gradual unfreeze | Collapsed to -2171 | Don't unfreeze pretrained encoder |

---

## v10g: Lane Keeping + NPC Coexistence

### Intent
NPC 환경에서 안전한 주행 학습

### Approach
```yaml
headingAlignmentReward: 0.02
lateralDeviationPenalty: -0.02
followingBonus: 0.3       # <- Problem!
collisionPenalty: -5.0
```

### Results (8M steps)
| NPC Count | Reward | Notes |
|-----------|--------|-------|
| 0 | ~90-95 | Free driving OK |
| 4 | ~35-40 | **Plateau for 3.5M steps** |

### Why It Failed

```
Expected:
  Agent overtakes slow NPCs to maintain speed reward

Actual:
  Agent follows slow NPCs indefinitely
  followingBonus (+0.3) + no collision = optimal policy!
```

**Root Cause**: `followingBonus` rewarded "not crashing" which is already implicit in collision penalty. Agent learned that following is safer than overtaking.

### Lesson Learned
> "의도치 않은 꼼수가 가능한지 검토 필수"

---

## v11: Sparse Overtake Reward

### Intent
v10g 에이전트에 추월 보상 추가

### Approach
```yaml
overtakePassBonus: 3.0      # One-time reward on completion
overtakeSpeedBonus: 0.15    # Per-step bonus while beside NPC
overtakeSlowLeadThreshold: 0.7
```

### Results (8M steps)
| Metric | Value |
|--------|-------|
| Final Reward | ~51 |
| Improvement over v10g | +11 (marginal) |
| Training Time | 8M steps |

### Why It Failed

```
Overtake Process: ~100+ steps
Reward Signal:    1 time (at completion)

Problem: Credit Assignment
  - Which of 100 actions led to success?
  - Agent cannot learn which behavior was good
```

**Root Cause**: `overtakePassBonus (+3.0)` only awarded at completion. Agent couldn't associate intermediate actions with final reward.

### Lesson Learned
> "복잡한 행동은 전 과정에 걸쳐 보상해야 학습 가능"

---

## v12_HybridPolicy: Transfer Learning Attempt

### Intent
Phase B 지식 보존하면서 관측 공간 확장 (242D -> 254D)

### Approach

**Architecture**:
```
254D input → [Phase B Encoder (frozen)] → 512D features
           → [Lane Encoder (trainable)] → 32D features
           → [Combiner (trainable)]     → 512D fused
           → [Policy Head]              → 2D actions
```

**6-Stage Gradual Unfreezing**:
| Stage | Steps | Trainable | LR |
|-------|-------|-----------|-----|
| 0 | 0-200K | value_head only | 3e-4 |
| 1 | 200K-600K | + lane_encoder | 1.5e-4 |
| 2 | 600K-1M | + combiner | 1e-4 |
| 3 | 1M-1.5M | + policy_head | 5e-5 |
| 4 | 1.5M-2.25M | + fusion | 3e-5 |
| 5 | 2.25M-3M | + Phase B encoder | **3e-6** |

### Results

```
Reward
    0 ├───────────────────────────────────────────────
      │          ╱─╲
 -100 ├─────────╱───╲────────────────────────────────
      │        ╱     ╲    ★ Best (-82.7)
 -200 ├───────╱───────╲──────────────────────────────
      │      ╱         ╲
 -400 ├─────╱───────────╲────────────────────────────
      │    ╱             ╲
      │   ╱               ╲
-1000 ├──╱─────────────────╲─────────────────────────
      │ ╱                   ╲    Stage 5 begins
      │╱                     ╲      ↓
-2000 ├───────────────────────╲──────────────────────
      │                        ╲____★ Collapsed (-2171)
      └────────────────────────────────────────────────
       0.2M   0.6M   1.0M   1.5M   2.0M   2.5M   3.0M
```

| Step | Stage | Reward | Event |
|------|-------|--------|-------|
| 204K | 0→1 | -341.7 | Value warmup done |
| 614K | 1→2 | -159.3 | Lane encoder learning |
| 1.44M | 3→4 | **-82.7** | **BEST** |
| 2.25M | 4→5 | -334.5 | Pre-encoder unfreeze |
| 2.87M | 5 | **-2171.9** | **COLLAPSED** |

### Why It Failed

**Stage 5 Catastrophic Forgetting**:
```
Before Stage 5:
  Phase B encoder: frozen, preserving learned features
  Reward: -82.7 (best so far)

After Stage 5 (encoder unfrozen):
  Even with 0.1x learning rate (3e-6)
  Encoder weights destabilized
  Learned features corrupted
  Reward: -2171 (26x worse!)
```

### Lesson Learned
> "사전학습 encoder는 unfreeze하지 말 것. 차라리 처음부터 재학습"

### Additional Issues

**ONNX Export Failures**:
1. Wrong input name ("obs" instead of "obs_0")
2. Missing version_number/memory_size outputs
3. Caused Unity "Reloading Domain" hang

**Root Causes**:
- HybridDrivingPolicy architecture differs from ML-Agents format
- ML-Agents requires specific output naming convention

---

## Comparison: Failed vs Successful

| Aspect | Failed (v10g/v11) | Successful (v12) |
|--------|-------------------|------------------|
| Reward Type | Sparse | **Dense** |
| targetSpeed | leadSpeed | **speedLimit always** |
| followingBonus | Yes | **Removed** |
| Overtake Reward | +3.0 (one-time) | **5-phase dense** |
| Best Reward | +51 | **+988** |

---

## Summary of Lessons

### Reward Design
```yaml
DO:
  - Dense reward (과정 전체에 보상)
  - Progressive penalty (점진적 패널티)
  - targetSpeed = constant (not dynamic)

DON'T:
  - Sparse reward only
  - Bonus that encourages wrong behavior
  - Dynamic targetSpeed based on NPC
```

### Transfer Learning
```yaml
DO:
  - Train from scratch with curriculum
  - Use checkpoint initialization for same architecture
  - Early stopping when performance degrades

DON'T:
  - Unfreeze pretrained encoder
  - Continue training after collapse detected
  - Trust gradual unfreezing to work for core encoder
```

---

## Recommended Approach

Based on these failures, the successful pattern is:

1. **Start simple**: Single NPC, slow speed, short distance
2. **Dense rewards**: Reward every step of complex behavior
3. **targetSpeed = speedLimit**: Never reduce based on NPC speed
4. **Remove followingBonus**: Don't reward "not crashing"
5. **Curriculum learning**: Gradually increase complexity
6. **Don't transfer encoder**: Train from scratch with new observation space

---

[← Phases](./index) | [Lessons Learned](../lessons-learned) | [Home](../)

