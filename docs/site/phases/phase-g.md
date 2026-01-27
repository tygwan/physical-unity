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
| **Current Steps** | ~3,560,000 (44.5%) |
| **Current Reward** | **+792** (peak: +882 at 3.19M) |
| **Initialize From** | Phase F (+988) |
| **Current Curriculum** | **CrossIntersection** |

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

| Step | Reward | Std | Curriculum State | Notes |
|------|--------|-----|------------------|-------|
| 10K | +423 | 14 | NoIntersection, Straight | Start |
| 500K | +480 | 15 | NoIntersection, Straight | Checkpoint saved |
| 800K | +521 | 30 | NoIntersection, Straight | - |
| 1.0M | +615 | 91 | NoIntersection, Straight | Checkpoint saved |
| 1.08M | +750 | 140 | **Curriculum transition** | **TwoLanes, CenterLine enabled** |
| 1.25M | +722 | 11 | TwoLanes, CenterLine | Stable |
| 1.33M | +720 | 15 | **Turn curriculum** | **LeftTurn, OneNPC** |
| 1.44M | +683 | 195 | **Turn curriculum** | **RightTurn, TwoNPCs** |
| 2.0M | +683 | 159 | RightTurn, TwoNPCs | Checkpoint saved |
| 2.15M | +734 | 17 | RightTurn, TwoNPCs | Peak (no intersection) |
| 2.78M | +750 | 172 | RightTurn, TwoNPCs | Rising |
| 3.0M | +792 | 141 | RightTurn, TwoNPCs | Checkpoint saved |
| 3.11M | +855 | 218 | RightTurn, TwoNPCs | Peak |
| 3.19M | **+882** | 208 | RightTurn, TwoNPCs | **PEAK** |
| 3.20M | +837 | 280 | **Curriculum transition** | **T-Junction entered** |
| 3.40M | +763 | 193 | **Curriculum transition** | **CrossIntersection entered** |
| **3.56M** | **+792** | 221 | **CrossIntersection** | **Current** |

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

## Milestones (Actual vs Expected)

| Milestone | Expected Step | Actual Step | Status |
|-----------|---------------|-------------|--------|
| TwoLanes transition | - | 1.08M | Completed |
| LeftTurn introduced | - | 1.33M | Completed |
| RightTurn introduced | - | 1.44M | Completed |
| Pre-intersection peak | - | 3.19M (+882) | Completed |
| **T-Junction 도입** | ~1-1.5M | **3.20M** | **Completed** |
| **Cross 도입** | ~2-3M | **3.40M** | **Completed** |
| Y-Junction 도입 | ~4-5M | TBD | Pending |
| **Phase G 완료** | ~8M | TBD | In Progress |

---

## Notes

- Phase F checkpoint에서 초기화하여 기존 능력 (차선 유지, 추월 등) 유지
- T-Junction과 Cross 교차로 진입 완료! (3.2M~3.4M steps)
- 커리큘럼 전환 시에도 reward +700~800 유지 (curriculum shock 최소화)
- Y-Junction 진입 전까지 Cross 교차로 안정화 중

---

[← Back to Phases](./index) | [Home](../)
