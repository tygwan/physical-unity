---
layout: default
title: Phase G - Intersection Navigation
---

# Phase G: Intersection Navigation

교차로 (T자/십자/Y자) 주행 학습

---

## Overview

| Item | Value |
|------|-------|
| **Status** | 🔄 In Progress |
| **Start Date** | 2026-01-27 |
| **Target Steps** | 8,000,000 |
| **Current Steps** | ~750,000 (9.4%) |
| **Current Reward** | +492 |
| **Initialize From** | Phase F (+988) |

---

## Objective

Phase F에서 학습한 다차선 주행 능력을 유지하면서, 교차로에서의 방향 전환(직진/좌회전/우회전)을 학습합니다.

### New Capabilities
- T자 교차로 인식 및 통과
- 십자 교차로 인식 및 통과
- Y자 분기점 인식 및 통과
- 좌회전/우회전 기동

---

## Observation Space

**254D → 260D** (+6D intersection info)

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

### Intersection Type Curriculum

```
Stage 1: NoIntersection (직선 도로만)
    │ threshold: reward > 800
    ▼
Stage 2: T-Junction (T자 교차로)
    │ threshold: reward > 600
    ▼
Stage 3: Cross (십자 교차로)
    │ threshold: reward > 500
    ▼
Stage 4: Y-Junction (Y자 분기점)
```

### Turn Direction Curriculum

```
Stage 1: Straight Only (직진만)
    │ threshold: reward > 700
    ▼
Stage 2: Left Turn (좌회전 추가)
    │ threshold: reward > 500
    ▼
Stage 3: Right Turn (우회전 추가)
```

---

## Training Progress

### Reward Curve

![Phase G Reward Curve](../gallery/charts/phase-g-reward.png)

*이미지 준비 중*

### Step-by-Step Progress

| Step | Reward | Std | Curriculum State |
|------|--------|-----|------------------|
| 10K | +423 | 14 | NoIntersection, Straight |
| 100K | +439 | 5 | NoIntersection, Straight |
| 200K | +442 | 6 | NoIntersection, Straight |
| 300K | +456 | 8 | NoIntersection, Straight |
| 400K | +467 | 6 | NoIntersection, Straight |
| 500K | +480 | 15 | NoIntersection, Straight |
| 600K | +496 | 16 | NoIntersection, Straight |
| 700K | +474 | 94 | NoIntersection, Straight |
| **750K** | **+492** | - | **Current** |

---

## Screenshots

### NoIntersection Stage (현재)

![NoIntersection](../gallery/screenshots/phase-g-no-intersection.png)

*스크린샷 준비 중*

### T-Junction Stage (예정)

| 진입 전 | 교차로 내 | 통과 후 |
|---------|----------|---------|
| ![](../gallery/screenshots/phase-g-t-approach.png) | ![](../gallery/screenshots/phase-g-t-inside.png) | ![](../gallery/screenshots/phase-g-t-exit.png) |

*스크린샷 준비 중*

---

## Environment Setup

### Simplified Environment

Phase G에서는 교차로 학습에 집중하기 위해 환경을 단순화했습니다:

| Parameter | Phase F | Phase G | Reason |
|-----------|---------|---------|--------|
| road_curvature | 0~0.6 | **0** | 교차로 집중 |
| num_npcs | 0~3 | **0~2** | 복잡도 제한 |
| goal_distance | 200m | **120~200m** | 짧은 에피소드 |

---

## Expected Milestones

| Milestone | Expected Step | Condition |
|-----------|---------------|-----------|
| T-Junction 도입 | ~1-1.5M | reward > 800 |
| T-Junction 마스터 | ~2M | reward > 600 |
| Cross 도입 | ~2-3M | T-Junction 완료 |
| Y-Junction 도입 | ~4-5M | Cross 완료 |
| 좌회전 학습 | ~3-4M | turn curriculum |
| 우회전 학습 | ~5-6M | turn curriculum |
| **Phase G 완료** | ~8M | 모든 curriculum 완료 |

---

## Notes

- Phase F checkpoint에서 초기화하여 기존 능력 (차선 유지, 추월 등) 유지
- 현재 보상 +492는 threshold 800까지 약 300 gap 있음
- 커리큘럼 전환 시 일시적인 보상 하락 예상 (curriculum shock)

---

[← Back to Phases](./index) | [Home](../)
