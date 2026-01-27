---
layout: default
title: Training Phases
---

# Training Phases

단계별 학습 과정 기록

---

## Phase Overview

```
Foundation ──► Phase A ──► Phase B ──► Phase C ──► Phase E ──► Phase F ──► Phase G ──► ...
 (v10-v11)     (추월)      (판단)      (일반화)     (곡선)      (다차선)    (교차로)
   +40~51      +937        +994        +1086       +931        +988        🔄
```

---

## Completed Phases

### [Phase A: Dense Overtaking](./phase-a)
- **Goal**: 느린 NPC 추월 기동 학습
- **Result**: +937 reward
- **Key**: Dense 5-phase reward 설계

### [Phase B: Overtake Decision](./phase-b)
- **Goal**: 추월 vs 따라가기 판단
- **Result**: +994 reward
- **Key**: NPC 속도 커리큘럼 (30%→90%)

### [Phase C: Multi-NPC](./phase-c)
- **Goal**: 4대 NPC 환경 일반화
- **Result**: +1086 reward
- **Key**: 점진적 복잡도 증가

### [Phase E: Curved Roads](./phase-e)
- **Goal**: 곡선 도로 주행
- **Result**: +931 reward
- **Key**: 곡률 커리큘럼 (0→1.0)

### [Phase F: Multi-Lane](./phase-f)
- **Goal**: 다중 차선 + 중앙선 규칙
- **Result**: +988 reward
- **Key**: 차선 수 커리큘럼 (1→2)

---

## In Progress

### [Phase G: Intersection](./phase-g) 🔄
- **Goal**: 교차로 (T자/십자/Y자) 주행
- **Current**: +492 reward (750K steps)
- **Target**: 8M steps

---

## Planned Phases

| Phase | Focus | Observation | Status |
|-------|-------|-------------|--------|
| H | 신호등 + 정지선 | +8D | 📋 Planned |
| I | U턴 + 특수 기동 | +4D | 📋 Planned |
| J | 횡단보도 + 보행자 | +12D | 📋 Planned |
| K | 장애물 + 긴급 상황 | +10D | 📋 Planned |
| L | 복합 시나리오 통합 | ~320D | 📋 Planned |

---

## Failed Experiments

### [v11: Sparse Reward](./failed/v11-sparse)
- **Problem**: Sparse reward로는 추월 학습 불가
- **Lesson**: Dense reward 필수

### [HybridPolicy: Encoder Fine-tuning](./failed/hybrid-policy)
- **Problem**: Catastrophic forgetting 발생
- **Lesson**: 사전학습 encoder는 freeze 유지

---

[← Back to Home](../)
