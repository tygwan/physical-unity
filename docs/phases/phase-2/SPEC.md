# Phase 2: Data Infrastructure

## Overview

자율주행 ML 플랫폼을 위한 데이터셋 확보 및 전처리 파이프라인 구축 단계입니다.

## Goals

1. **데이터셋 확보**: nuPlan, Waymo, highD, INTERACTION 다운로드
2. **통합 데이터 포맷**: 여러 데이터셋을 위한 통합 스키마 설계
3. **전처리 파이프라인**: 시나리오 추출, 정규화, 증강
4. **시각화 도구**: 데이터 탐색 및 품질 검증

## Scope

### In Scope
- 데이터셋 다운로드 및 검증
- 통합 데이터 포맷 설계
- 시나리오 추출 파이프라인 구현
- 데이터 증강 전략 구현
- 시각화 및 탐색 도구 개발
- Train/Val/Test 분할

### Out of Scope
- ML 모델 학습 (Phase 3-5)
- 실시간 데이터 스트리밍

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Phase 2: Data Infrastructure                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      RAW DATA SOURCES                            ││
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   ││
│  │  │  nuPlan  │ │  Waymo   │ │  highD   │ │   INTERACTION    │   ││
│  │  │  50GB+   │ │  100GB+  │ │   5GB    │ │      2GB         │   ││
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────────┬─────────┘   ││
│  └───────┼────────────┼────────────┼────────────────┼──────────────┘│
│          │            │            │                │               │
│          └────────────┴────────────┴────────────────┘               │
│                              │                                       │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                    PROCESSING PIPELINE                         │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │  Loader  │→ │ Parser   │→ │ Filter   │→ │  Augment     │  │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                              │                                       │
│  ┌───────────────────────────▼───────────────────────────────────┐  │
│  │                    UNIFIED FORMAT                              │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │ Scenarios  │  │  Features  │  │   Splits (train/val)   │  │  │
│  │  │ (Parquet)  │  │  (Parquet) │  │   test)                │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Datasets

| Dataset | Size | Use Case | Priority | License |
|---------|------|----------|----------|---------|
| nuPlan (mini) | ~50GB | IL, Planning | Primary | CC BY-NC-SA 4.0 |
| Waymo Motion | ~100GB | Trajectory | Primary | Waymo License |
| highD | ~5GB | Highway | Secondary | Research only |
| INTERACTION | ~2GB | Intersection | Secondary | CC BY-NC-SA 4.0 |

## Task Breakdown

| ID | Task | Priority | Est. Time |
|----|------|----------|-----------|
| P2-01 | nuPlan mini 데이터셋 다운로드 | High | 1일 |
| P2-02 | Waymo Open Motion 다운로드 | High | 2일 |
| P2-03 | highD/INTERACTION 다운로드 | Medium | 0.5일 |
| P2-04 | 통합 데이터 스키마 설계 | High | 2일 |
| P2-05 | nuPlan 파서 구현 | High | 3일 |
| P2-06 | Waymo 파서 구현 | High | 3일 |
| P2-07 | 시나리오 필터링 구현 | Medium | 2일 |
| P2-08 | 데이터 증강 구현 | Medium | 2일 |
| P2-09 | 시각화 도구 개발 | Medium | 2일 |
| P2-10 | Train/Val/Test 분할 | High | 1일 |

## Unified Data Schema

```python
# 통합 시나리오 스키마
@dataclass
class Scenario:
    scenario_id: str
    source: str  # nuplan, waymo, highd
    duration: float  # seconds

    # Ego vehicle trajectory
    ego_trajectory: np.ndarray  # [T, 7] x, y, vx, vy, ax, ay, heading

    # Other agents
    agents: List[AgentTrack]

    # Map information
    map_features: MapFeatures

    # Traffic lights
    traffic_lights: List[TrafficLightState]

@dataclass
class AgentTrack:
    agent_id: str
    agent_type: str  # vehicle, pedestrian, cyclist
    trajectory: np.ndarray  # [T, 7]
    dimensions: Tuple[float, float, float]  # length, width, height
```

## Success Criteria

- [ ] nuPlan mini 데이터셋 로딩 가능
- [ ] Waymo Motion 데이터셋 로딩 가능
- [ ] 통합 포맷으로 10,000+ 시나리오 추출
- [ ] 데이터 증강 (노이즈, 변환) 동작
- [ ] 시각화 도구로 시나리오 확인 가능
- [ ] Train/Val/Test 분할 완료 (80/10/10)

## Timeline

**예상 소요**: 3-4주

## Dependencies

- Phase 1 완료 (Python 환경)
- 충분한 저장 공간 (500GB+)
- nuPlan 라이센스 동의
- Waymo 데이터셋 접근 권한

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| 다운로드 시간 | Medium | High | Mini 버전 우선 |
| 저장 공간 부족 | High | Medium | 점진적 다운로드 |
| 데이터 포맷 불일치 | Medium | Medium | 유연한 파서 설계 |
| 라이센스 제한 | Medium | Low | 연구 목적 명시 |

## Deliverables

1. **Dataset Loaders**: nuPlan, Waymo, highD, INTERACTION
2. **Unified Schema**: 통합 데이터 포맷 정의
3. **Processing Pipeline**: 전처리 파이프라인
4. **Augmentation Tools**: 데이터 증강 모듈
5. **Visualization Tools**: 시나리오 시각화
6. **Data Splits**: train/val/test 분할 파일
