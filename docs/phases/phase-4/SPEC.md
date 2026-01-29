# Phase 4: Prediction Models (Simplified)

> ⏸️ **STATUS: SUSPENDED** - Using Constant Velocity baseline instead of learned prediction
>
> **Reason**: Planning-focused strategy prioritizes RL/IL motion planning. Prediction uses simple CV model for faster iteration and reliable baselines.

## Overview

Planning 집중 전략에 따라 Prediction은 기존 baseline 모델을 활용합니다. nuPlan baseline predictor 또는 간단한 Constant Velocity 모델 사용을 우선합니다.

## Goals

1. **Baseline Predictor 설정**: nuPlan-devkit baseline 활용
2. **Constant Velocity Model**: 간단한 예측 모델 구현
3. **Planning 연동**: Prediction 출력을 Planning 입력으로 연결
4. **(Optional) Custom Predictor**: Transformer 기반 예측기

## Strategy

> **Planning 집중 전략**: Prediction 자체 개발을 최소화하고, nuPlan baseline 활용

### Approach Options

| Option | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| A | nuPlan Baseline | 검증됨, 빠른 적용 | 제한된 커스터마이징 | Primary |
| B | Constant Velocity | 매우 간단 | 비현실적 예측 | Primary |
| C | Custom Transformer | 최적화 가능 | 개발 시간 | Optional |

## Scope

### In Scope
- nuPlan baseline predictor 설정
- Constant velocity baseline 구현
- Prediction → Planning 인터페이스 정의
- 예측 성능 평가 (ADE, FDE)

### Out of Scope
- Transformer 기반 예측기 자체 개발 (Optional로 남김)
- Multi-modal 예측
- 행동 인식 (Intention Prediction)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 4: Prediction (Simplified)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      INPUT (from Perception)                     ││
│  │  ┌──────────────────┐         ┌──────────────────────────────┐  ││
│  │  │  Agent Tracks    │         │       Map Information        │  ││
│  │  │  (History)       │         │   Lanes, Roads, Signals      │  ││
│  │  └────────┬─────────┘         └─────────────┬────────────────┘  ││
│  └───────────┼───────────────────────────────────┼─────────────────┘│
│              │                                   │                   │
│              └───────────────┬───────────────────┘                   │
│                              ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     PREDICTION MODULES                         │  │
│  │  ┌──────────────────┐    ┌────────────────────────────────┐   │  │
│  │  │  Constant        │    │      nuPlan Baseline           │   │  │
│  │  │  Velocity        │    │      Predictor                 │   │  │
│  │  │  (Fallback)      │    │      (Primary)                 │   │  │
│  │  └────────┬─────────┘    └──────────────┬─────────────────┘   │  │
│  └───────────┼──────────────────────────────┼────────────────────┘  │
│              │                              │                        │
│              └──────────────┬───────────────┘                        │
│                             ▼                                        │
│              ┌────────────────────────────────┐                      │
│              │      Prediction Output         │                      │
│              │  - Future Trajectories         │                      │
│              │  - 5 seconds horizon           │                      │
│              │  - 50 timesteps (0.1s each)    │                      │
│              │  - Multi-modal (optional)      │                      │
│              └────────────────┬───────────────┘                      │
│                               │                                      │
│                               ▼                                      │
│              ┌────────────────────────────────┐                      │
│              │    → Planning Module (Phase 5) │                      │
│              └────────────────────────────────┘                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Task Breakdown

| ID | Task | Priority | Est. Time |
|----|------|----------|-----------|
| P4-01 | nuPlan-devkit 설치 및 설정 | High | 2일 |
| P4-02 | nuPlan baseline predictor 테스트 | High | 2일 |
| P4-03 | Constant Velocity 모델 구현 | High | 1일 |
| P4-04 | Prediction 출력 인터페이스 정의 | High | 1일 |
| P4-05 | Planning 연동 테스트 | High | 2일 |
| P4-06 | 예측 성능 평가 (ADE/FDE) | Medium | 2일 |
| P4-07 | (Optional) Custom predictor 설계 | Low | TBD |

## Prediction Output Interface

```python
# python/src/models/prediction/interface.py

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class PredictedTrajectory:
    """단일 에이전트의 예측 궤적"""
    agent_id: int
    trajectory: np.ndarray  # [T, 2] (x, y) positions
    timestamps: np.ndarray  # [T] time from current
    confidence: float  # 예측 신뢰도

@dataclass
class PredictionOutput:
    """Prediction 모듈 출력"""
    current_time: float
    horizon: float  # 예측 horizon (default: 5.0s)
    predictions: Dict[int, List[PredictedTrajectory]]  # agent_id -> trajectories
    # Multi-modal prediction: 각 에이전트에 대해 여러 궤적 가능

# 예시 사용
def create_constant_velocity_prediction(
    agent_state: np.ndarray,  # [x, y, vx, vy]
    horizon: float = 5.0,
    dt: float = 0.1
) -> np.ndarray:
    """Constant Velocity 기반 예측"""
    n_steps = int(horizon / dt)
    trajectory = np.zeros((n_steps, 2))

    x, y, vx, vy = agent_state
    for t in range(n_steps):
        trajectory[t, 0] = x + vx * (t + 1) * dt
        trajectory[t, 1] = y + vy * (t + 1) * dt

    return trajectory
```

## Evaluation Metrics

| Metric | Description | Formula | Target |
|--------|-------------|---------|--------|
| ADE | Average Displacement Error | mean(‖pred - gt‖) | < 2.0m |
| FDE | Final Displacement Error | ‖pred[-1] - gt[-1]‖ | < 4.0m |
| Miss Rate | 최종 위치 오차 > 2m 비율 | - | < 30% |

## Success Criteria

- [ ] nuPlan baseline predictor 동작 확인
- [x] Constant Velocity 모델 구현 완료
- [x] Prediction → Planning 인터페이스 동작
- [ ] ADE < 2.0m on validation set
- [x] 처리 시간 < 20ms per agent

**Status: ⏸️ 보류** - Planning 집중 전략에 따라 Constant Velocity 방식으로 진행 (2026-01-22)

## Timeline

**예상 소요**: 3-4주

## Dependencies

- Phase 3 완료 (Perception)
- nuPlan-devkit 설치
- Phase 2 완료 (데이터 파이프라인)

## nuPlan Baseline Models

| Model | Type | ADE@5s | FDE@5s | Priority |
|-------|------|--------|--------|----------|
| IDM | Rule-based | 3.2m | 6.5m | Baseline |
| MLP | Learning | 1.8m | 3.9m | Primary |
| Transformer | Attention | 1.4m | 3.1m | Optional |

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| nuPlan 버전 호환성 | Medium | Medium | 버전 고정 |
| 예측 품질 부족 | Medium | Low | CV 모델로 fallback |
| 추론 속도 | Low | Low | 단순 모델 사용 |

## Deliverables

1. **Constant Velocity Model**: 간단한 baseline 예측기
2. **nuPlan Predictor Wrapper**: nuPlan baseline 래퍼
3. **Prediction Interface**: Planning과의 표준 인터페이스
4. **Evaluation Scripts**: ADE/FDE 측정 스크립트
5. **Documentation**: 사용 가이드

---

## 📚 Phase 완료 시: Obsidian 지식화

### 지식화 대상
Phase 4 완료 후 다음 내용을 Obsidian vault에 정리합니다:

| 카테고리 | 내용 |
|----------|------|
| **예측 모델** | CV 모델, nuPlan baseline 구조 및 원리 |
| **평가 메트릭** | ADE/FDE/Miss Rate 계산 방법, 해석 |
| **nuPlan-devkit** | 설치, 설정, 사용법 |
| **인터페이스 설계** | Prediction → Planning 데이터 흐름 |
| **성능 분석** | 시나리오별 예측 성능 분석 |

### 실행 방법
```bash
/obsidian sync --phase=4
```

### 생성될 노트 구조
```
Obsidian Vault/
├── Projects/
│   └── AD-ML-Platform/
│       ├── Phase-4-Prediction/
│       │   ├── Constant-Velocity-모델.md
│       │   ├── nuPlan-Baseline-가이드.md
│       │   ├── 평가-메트릭-해석.md
│       │   ├── Prediction-Planning-인터페이스.md
│       │   └── 트러블슈팅-로그.md
│       └── ...
