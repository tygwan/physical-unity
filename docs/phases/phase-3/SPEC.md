# Phase 3: Perception Models (Simplified)

## Overview

Planning 집중 전략에 따라 Perception은 간소화하여 구현합니다. Pre-trained 모델 활용 또는 Ground Truth 사용을 우선합니다.

## Goals

1. **Ground Truth 추출**: 시뮬레이션에서 직접 객체 정보 획득
2. **Pre-trained 모델 연동**: MMDetection3D 또는 OpenPCDet 활용
3. **BEV Representation**: Bird's Eye View 표현 생성
4. **Planning 연동**: Perception 출력을 Planning 입력으로 연결

## Strategy

> **Planning 집중 전략**: Perception 자체 개발을 최소화하고, 기존 도구 활용

### Approach Options

| Option | Description | Pros | Cons | Priority |
|--------|-------------|------|------|----------|
| A | Ground Truth (Simulation) | 정확, 빠름 | 실제 환경 불가 | Primary |
| B | Pre-trained Model | 실제 적용 가능 | 설정 복잡 | Secondary |
| C | Custom BEV Encoder | 최적화 가능 | 개발 시간 | Optional |

## Scope

### In Scope
- 시뮬레이션 Ground Truth 추출 시스템
- Pre-trained 3D detection 모델 테스트 (MMDetection3D)
- 간단한 BEV representation 생성
- Perception-Planning 인터페이스 정의

### Out of Scope
- 3D Detection 모델 자체 학습
- LiDAR 세그멘테이션
- Multi-sensor Fusion 모델 개발

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Phase 3: Perception (Simplified)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      INPUT SOURCES                               ││
│  │  ┌──────────────────┐         ┌──────────────────────────────┐  ││
│  │  │   Simulation     │         │        Real Sensors          │  ││
│  │  │   Ground Truth   │         │   LiDAR   │   Camera         │  ││
│  │  └────────┬─────────┘         └─────┬─────┴──────┬───────────┘  ││
│  └───────────┼─────────────────────────┼────────────┼──────────────┘│
│              │                         │            │                │
│              ▼                         ▼            ▼                │
│  ┌───────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Ground Truth Parser  │    │      Pre-trained Detector        │  │
│  │  (Primary Mode)       │    │  (MMDetection3D / OpenPCDet)     │  │
│  └───────────┬───────────┘    └───────────────┬──────────────────┘  │
│              │                                │                      │
│              └────────────────┬───────────────┘                      │
│                               ▼                                      │
│              ┌────────────────────────────────┐                      │
│              │      Perception Output         │                      │
│              │  - Detected Objects            │                      │
│              │  - Position, Velocity, Size    │                      │
│              │  - Object Class                │                      │
│              │  - BEV Features (optional)     │                      │
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
| P3-01 | Unity Ground Truth 추출 시스템 | High | 2일 |
| P3-02 | Perception 출력 인터페이스 정의 | High | 1일 |
| P3-03 | MMDetection3D 테스트 환경 구축 | Medium | 2일 |
| P3-04 | Pre-trained 모델 추론 테스트 | Medium | 2일 |
| P3-05 | BEV representation 생성 | Medium | 3일 |
| P3-06 | Planning 연동 인터페이스 구현 | High | 2일 |

## Perception Output Interface

```python
# python/src/models/perception/interface.py

from dataclasses import dataclass
from typing import List
import numpy as np

@dataclass
class DetectedObject:
    """Planning 모듈에 전달되는 객체 정보"""
    object_id: int
    object_class: str  # vehicle, pedestrian, cyclist
    position: np.ndarray  # [x, y, z] in ego frame
    velocity: np.ndarray  # [vx, vy, vz]
    dimensions: np.ndarray  # [length, width, height]
    heading: float  # radians
    confidence: float  # 0.0 - 1.0

@dataclass
class PerceptionOutput:
    """Perception 모듈 출력"""
    timestamp: float
    objects: List[DetectedObject]
    ego_position: np.ndarray
    ego_velocity: np.ndarray
    bev_features: Optional[np.ndarray] = None  # [H, W, C]
```

## Success Criteria

- [ ] Unity에서 Ground Truth 객체 정보 추출 가능
- [ ] Pre-trained 모델 추론 동작 확인
- [ ] Perception → Planning 인터페이스 동작
- [ ] BEV representation 생성 가능
- [ ] 처리 시간 < 50ms per frame

## Timeline

**예상 소요**: 2-3주

## Dependencies

- Phase 1 완료 (Unity 환경)
- Phase 2 완료 (데이터 파이프라인)
- MMDetection3D 설치
- PyTorch 2.0+

## Pre-trained Models (참고)

| Model | Dataset | mAP | Inference | Priority |
|-------|---------|-----|-----------|----------|
| PointPillars | nuScenes | 40.1 | 20ms | High |
| CenterPoint | Waymo | 66.8 | 35ms | Medium |
| BEVFusion | nuScenes | 72.9 | 50ms | Low |

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Pre-trained 모델 호환성 | Medium | Medium | Ground Truth 우선 |
| 추론 속도 부족 | Low | Low | 더 가벼운 모델 선택 |
| BEV 품질 문제 | Medium | Medium | 단순 표현 사용 |

## Deliverables

1. **Ground Truth Extractor**: Unity에서 객체 정보 추출
2. **Perception Interface**: Planning과의 표준 인터페이스
3. **Pre-trained Model Wrapper**: MMDetection3D 래퍼
4. **BEV Generator**: Bird's Eye View 생성 모듈
5. **Documentation**: 사용 가이드 및 API 문서
