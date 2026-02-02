---
layout: default
title: Training Phases
---

# Training Phases

단계별 학습 과정 기록

---

## Phase Overview

```
Foundation --> Phase A --> Phase B --> Phase C --> Phase E --> Phase F --> Phase G --> Phase H --> Phase I --> Phase J
 (v10-v11)     (추월)      (판단)      (일반화)     (곡선)      (다차선)    (교차로)    (NPC교차로)  (곡선+NPC)   (신호등)
   +40~51      +937        +994        +1086       +931        +988        +628        +701        +770       +497*
```
*Phase J v4 with signals ON + green_ratio=0.5, 3/4 green_ratio curriculum

---

## Completed Phases

### [Phase A: Dense Overtaking](./phase-a)
- **Goal**: 느린 NPC 추월 기동 학습
- **Result**: +937 reward
- **Key**: Dense 5-phase reward 설계

### [Phase B: Overtake Decision](./phase-b)
- **Goal**: 추월 vs 따라가기 판단
- **Result**: +994 reward
- **Key**: NPC 속도 커리큘럼 (30%->90%)

### [Phase C: Multi-NPC](./phase-c)
- **Goal**: 4대 NPC 환경 일반화
- **Result**: +1086 reward
- **Key**: 점진적 복잡도 증가

### [Phase E: Curved Roads](./phase-e)
- **Goal**: 곡선 도로 주행
- **Result**: +931 reward
- **Key**: 곡률 커리큘럼 (0->1.0)

### [Phase F: Multi-Lane](./phase-f)
- **Goal**: 다중 차선 + 중앙선 규칙
- **Result**: +988 reward
- **Key**: 차선 수 커리큘럼 (1->2)

### [Phase G: Intersection](./phase-g)
- **Goal**: 교차로 (T자/십자/Y자) 주행
- **Result**: +628 reward (v2, 7/7 curriculum complete)
- **Key**: Warm start + WrongWay fix (32%->0%)

---

### [Phase H: NPC Intersection](./phase-h)
- **Goal**: 교차로에서 NPC 상호작용 (1->2->3 NPCs)
- **Result**: +701 reward (v3, 11/11 curriculum complete)
- **Key**: Build training + 점진적 speed_variation (v1 crash -> v2 stuck -> v3 완료)

### [Phase I: Curved Roads + NPC](./phase-i)
- **Goal**: 곡선 도로 + NPC 트래픽 통합
- **Result**: +770 reward (v2, 17/17 curriculum complete, 프로젝트 최고 기록)
- **Key**: Triple-param crash 회복 (v1: 724->-40->623, v2: 623->770)

---

## In Progress

### [Phase J: Traffic Signals](./phase-j)
- **Goal**: 신호등 인식 + 정지선 준수 + 268D observation
- **Result (v4)**: +616 peak, +497 final (3/4 green_ratio, signal-first)
- **Key**: P-022 fix validated (no signal crash), green_ratio 0.8->0.5 completed
- **Issue**: Plateau at ~490-500 with green_ratio=0.5, threshold 540 unreachable (P-023)

---

## Planned Phases

| Phase | Focus | Observation | Status |
|-------|-------|-------------|--------|
| J | 신호등 + 정지선 | +8D (268D) | 🔄 In Progress (v4 done 3/4, v5 or Phase K next) |
| K | U턴 + 특수 기동 | +4D | 📋 Planned |
| L | 횡단보도 + 보행자 | +12D | 📋 Planned |
| M | 장애물 + 긴급 상황 | +10D | 📋 Planned |
| N | 복합 시나리오 통합 | ~320D | 📋 Planned |

---

## Failed Experiments

See [Failed Experiments](./failed-experiments) for detailed analysis.

### v10g/v11: Sparse Reward
- **Problem**: Sparse reward + followingBonus로는 추월 학습 불가
- **Result**: +40~51 (8M steps, plateau)
- **Lesson**: Dense reward 필수, followingBonus 제거

### HybridPolicy: Encoder Fine-tuning
- **Problem**: Stage 5에서 Catastrophic forgetting 발생
- **Result**: -82.7 -> -2171 (collapsed)
- **Lesson**: 사전학습 encoder는 unfreeze하지 말 것

### Phase G v1: Fresh Start
- **Problem**: 254D->260D fresh start로 2M steps 낭비, WrongWay 32%
- **Result**: +494 (10M steps, plateau at ~500)
- **Lesson**: Warm start 필수, WrongWay detection multi-axis 필요

### Phase H v1: Abrupt Speed Variation
- **Problem**: speed_variation 0->0.15 급격한 전환으로 reward crash (700->550)
- **Result**: Training halted (catastrophic instability)
- **Lesson**: 커리큘럼 전환은 점진적이어야 함

### Phase H v2: Unreachable Thresholds
- **Problem**: speed_variation thresholds (710/720)이 variation 활성 상태에서 도달 불가
- **Result**: +681 (9/11 curriculum, variation stuck at 0.05)
- **Lesson**: Threshold는 목표 조건 하에서 달성 가능해야 함

### Phase I v1: Triple-Param Crash
- **Problem**: Curve thresholds 700/702/705 너무 촘촘 -> 3개 파라미터 동시 전환
- **Result**: +623 (17/17 curriculum complete, but reward crashed 724->-40 then recovered)
- **Lesson**: Threshold 간격 >= 15 포인트 유지 필수 (P-018)

### Phase J v1: Observation Dimension Mismatch
- **Problem**: 260D checkpoint -> 268D observation, Adam optimizer tensor crash
- **Result**: ~40K steps (immediate crash)
- **Lesson**: Observation 차원 변경 시 warm start 불가, fresh start 필수 (P-020)

### Phase J v3: Signal Ordering Conflict
- **Problem**: signal_green_ratio threshold < signal_enabled threshold -> green_ratio changed before signals ON
- **Result**: +477 (12/13 curriculum, signal crash 647->470 never recovered)
- **Lesson**: 독립 커리큘럼 파라미터 간 순서 보장 불가, 단일 파라미터 커리큘럼 필요 (P-022)

### Phase J v4: Green Ratio Plateau
- **Problem**: Reward compression at green_ratio=0.5, threshold 540 unreachable (plateau ~490-500)
- **Result**: +497 (3/4 green_ratio, missed 0.5->0.4)
- **Lesson**: 신호 대기 시간 증가로 보상 범위 축소, threshold 또는 보상 구조 조정 필요 (P-023)

---

*Last Updated: 2026-02-02 (Phase J v4 3/4 green_ratio)*

[<- Back to Home](../)
