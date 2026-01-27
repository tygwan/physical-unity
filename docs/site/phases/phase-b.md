---
layout: default
title: Phase B - Overtake Decision
---

# Phase B: Overtake vs Follow Decision

NPC 속도에 따른 추월/따라가기 판단 학습

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-B |
| **Status** | ✅ Completed |
| **Date** | 2026-01-25 |
| **Total Steps** | 2,000,150 |
| **Training Time** | ~16 minutes (950 seconds) |
| **Final Reward** | **+903.3** |
| **Peak Reward** | **+994.5** (at 1.64M steps) |
| **Observation** | 242D |
| **Initialize From** | Phase A (phase-A_fixed) |

---

## Objective

Phase A에서 학습한 추월 능력을 확장하여, NPC 속도에 따른 의사결정 학습

### Decision Policy
- **NPC < 70% speedLimit**: 추월 (overtake)
- **NPC > 85% speedLimit**: 따라가기 (follow)
- **70-85%**: 상황에 따라 판단

---

## Curriculum Design

### NPC Speed Curriculum

```
Lesson 1: VerySlow (30%)
    │ threshold: reward > 50
    ▼
Lesson 2: Slow (50%)
    │ threshold: reward > 45
    ▼
Lesson 3: Medium (70%)
    │ threshold: reward > 40
    ▼
Lesson 4: Fast (90%)
    (final)
```

| Lesson | NPC Speed | Threshold | Expected Behavior |
|--------|-----------|-----------|-------------------|
| VerySlow | 30% | 50.0 | Always overtake |
| Slow | 50% | 45.0 | Mostly overtake |
| Medium | 70% | 40.0 | Decision boundary |
| Fast | 90% | - | Mostly follow |

---

## Training Progress

| Step | Reward | Std | Curriculum Lesson |
|------|--------|-----|-------------------|
| 680K | +630 | 11 | Fast (all lessons passed) |
| 800K | +834 | 190 | Fast |
| 1.0M | +850 | 220 | Fast |
| 1.2M | +870 | 230 | Fast |
| 1.4M | +880 | 235 | Fast |
| **1.64M** | **+994** | 193 | **Peak** |
| 1.8M | +886 | 228 | Stabilizing |
| **2.0M** | **+903** | 225 | **Converged** |

---

## Key Observations

### Curriculum Completion
- 모든 4개 lesson이 680K step 이전에 완료
- Phase A checkpoint에서 시작하여 빠르게 적응

### Reward Improvement
```
Phase A Final:  +714
Phase B Final:  +903  (+26% improvement)
Phase B Peak:   +994  (+39% over Phase A)
```

---

## Phase A vs Phase B Comparison

| Aspect | Phase A | Phase B |
|--------|---------|---------|
| NPC Speed | Fixed at 30% | Varies 30-90% |
| Decision | Always overtake | Conditional |
| Goal | Learn overtaking | Learn decision-making |
| Final Reward | +714 | **+903** (+26%) |
| Peak Reward | +937 | **+994** (+6%) |
| Training Time | 30 min | 16 min |

---

## Reward Curve

```
Reward
 +994 │                    ★ Peak
      │                   ╱╲
 +900 │__________________╱  ╲___
      │
 +800 │     ╱────────────
      │    ╱
 +700 │   ╱
      │  ╱
 +600 │_╱ (start from Phase A)
      └────────────────────────────
       0   0.5   1.0   1.5   2.0 M steps
```

---

## Checkpoints

| Checkpoint | Step | Reward |
|------------|------|--------|
| E2EDrivingAgent-999988.onnx | 1M | ~+850 |
| E2EDrivingAgent-1499899.onnx | 1.5M | ~+900 |
| E2EDrivingAgent-1999894.onnx | 2M | +903 |
| **E2EDrivingAgent.onnx** | Final | +903 |

---

## Lessons Learned

1. **Phase transition works**: Phase A → Phase B checkpoint 초기화 성공
2. **Curriculum completed quickly**: 모든 lesson이 680K 전에 완료
3. **Decision policy emerged**: NPC 속도 인식 및 조건부 행동 학습
4. **BehaviorType issue**: 모든 Agent가 BehaviorType=0 (Default)여야 학습 가능

---

## Verified Behaviors

| NPC Speed | Expected | Observed |
|-----------|----------|----------|
| 30% | Overtake | ✅ Overtake |
| 50% | Overtake | ✅ Overtake |
| 70% | Mixed | ✅ Context-dependent |
| 90% | Follow | ✅ Follow/Pass safely |

---

## Next Phase

**Phase C**: Multi-NPC Generalization
- NPC 수 증가: 1 → 2 → 3 → 4
- 속도 변동 증가
- 목표 거리 증가

---

[← Phase A](./phase-a) | [Phase C →](./phase-c) | [Home](../)
