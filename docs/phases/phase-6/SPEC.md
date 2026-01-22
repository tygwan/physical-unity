# Phase 6: Integration & Evaluation

## Overview

Perception → Prediction → Planning 파이프라인을 통합하고, 벤치마크 평가를 수행합니다.

## Goals

1. **End-to-end 통합**: 전체 파이프라인 연결
2. **벤치마크 평가**: nuPlan closed-loop 평가
3. **성능 최적화**: 레이턴시 및 처리량 최적화
4. **시각화 및 분석**: 결과 시각화 도구

## Scope

### In Scope
- Perception-Prediction-Planning 통합
- End-to-end latency 측정 및 최적화
- nuPlan benchmark 평가
- 다양한 시나리오 테스트
- 결과 시각화 및 분석
- 모델 배포 준비

### Out of Scope
- 실차 배포
- Hardware-in-the-loop 테스트
- V2X 통합

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase 6: Integration & Evaluation                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     END-TO-END PIPELINE                                 │ │
│  │                                                                         │ │
│  │  ┌──────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────┐    │ │
│  │  │  Sensors │ → │  Perception  │ → │  Prediction  │ → │ Planning │    │ │
│  │  │ (AWSIM)  │   │ (Phase 3)    │   │  (Phase 4)   │   │(Phase 5) │    │ │
│  │  │  ~10ms   │   │   ~30ms      │   │   ~20ms      │   │  ~50ms   │    │ │
│  │  └──────────┘   └──────────────┘   └──────────────┘   └────┬─────┘    │ │
│  │                                                             │          │ │
│  │                                                             ▼          │ │
│  │                                                        ┌──────────┐    │ │
│  │                                                        │ Control  │    │ │
│  │                                                        │  ~10ms   │    │ │
│  │                                                        └────┬─────┘    │ │
│  │                                                             │          │ │
│  │                                                             ▼          │ │
│  │                                                        ┌──────────┐    │ │
│  │                                                        │  Unity   │    │ │
│  │                                                        │Simulator │    │ │
│  │                                                        └──────────┘    │ │
│  │                                                                         │ │
│  │  Total Latency Target: < 200ms (120ms end-to-end + margin)             │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                     EVALUATION FRAMEWORK                                │ │
│  │                                                                         │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │ │
│  │  │  Closed-loop    │  │  Open-loop      │  │     Visualization      │ │ │
│  │  │  Simulation     │  │  Evaluation     │  │     & Analysis         │ │ │
│  │  │  (nuPlan)       │  │  (Metrics)      │  │                        │ │ │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────────────┘ │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Task Breakdown

| ID | Task | Priority | Est. Time |
|----|------|----------|-----------|
| **Integration** |
| P6-01 | Perception-Prediction 연결 | High | 2일 |
| P6-02 | Prediction-Planning 연결 | High | 2일 |
| P6-03 | Planning-Control 연결 | High | 2일 |
| P6-04 | End-to-end 테스트 | High | 3일 |
| **Optimization** |
| P6-05 | Latency 프로파일링 | High | 2일 |
| P6-06 | 병목 최적화 | High | 3일 |
| P6-07 | 배치 처리 최적화 | Medium | 2일 |
| **Evaluation** |
| P6-08 | nuPlan benchmark 설정 | High | 2일 |
| P6-09 | Closed-loop 평가 실행 | High | 3일 |
| P6-10 | 시나리오별 분석 | Medium | 3일 |
| **Visualization** |
| P6-11 | 결과 시각화 도구 | Medium | 3일 |
| P6-12 | 비디오 생성 | Low | 2일 |

## Evaluation Metrics

### Safety Metrics

| Metric | Description | Target | Weight |
|--------|-------------|--------|--------|
| Collision Rate | 충돌 발생 비율 | < 5% | 0.30 |
| TTC Violation | TTC < 2s 발생 비율 | < 10% | 0.15 |
| Off-road Rate | 도로 이탈 비율 | < 3% | 0.10 |

### Progress Metrics

| Metric | Description | Target | Weight |
|--------|-------------|--------|--------|
| Route Completion | 경로 완주 비율 | > 85% | 0.20 |
| Goal Reached | 목표 도달 비율 | > 80% | 0.10 |

### Comfort Metrics

| Metric | Description | Target | Weight |
|--------|-------------|--------|--------|
| Jerk | 가속도 변화율 | < 2 m/s³ | 0.10 |
| Lateral Acceleration | 횡가속도 | < 3 m/s² | 0.05 |

### Efficiency Metrics

| Metric | Description | Target | Weight |
|--------|-------------|--------|--------|
| End-to-end Latency | 전체 처리 시간 | < 200ms | - |
| Inference Time | 모델 추론 시간 | < 50ms | - |

## Test Scenarios

### Urban Scenarios
- [ ] Intersection crossing
- [ ] Traffic light compliance
- [ ] Pedestrian crossing
- [ ] Lane changing
- [ ] U-turn

### Highway Scenarios
- [ ] Lane keeping
- [ ] Lane changing at speed
- [ ] Merging/exiting
- [ ] Following distance

### Edge Cases
- [ ] Sudden obstacle appearance
- [ ] Aggressive drivers
- [ ] Construction zones
- [ ] Adverse weather (simulation)

## Success Criteria

- [ ] End-to-end pipeline 동작 확인
- [ ] Total latency < 200ms
- [ ] Collision rate < 5%
- [ ] Route completion > 85%
- [ ] nuPlan closed-loop score > 60
- [ ] 모든 주요 시나리오 통과

## Timeline

**예상 소요**: 4-6주

## Dependencies

- Phase 1-5 완료
- nuPlan evaluation toolkit
- 충분한 테스트 시나리오

## Performance Optimization Strategies

### Latency Reduction
1. **Model Quantization**: FP32 → FP16/INT8
2. **TensorRT Optimization**: NVIDIA 최적화
3. **Batch Processing**: 여러 프레임 동시 처리
4. **Async Pipeline**: 비동기 처리

### Throughput Improvement
1. **Multi-threading**: 병렬 처리
2. **GPU Utilization**: CUDA 스트림 최적화
3. **Memory Management**: 메모리 풀링

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| 통합 시 호환성 문제 | High | Medium | 인터페이스 표준화 |
| 레이턴시 목표 미달 | High | Medium | 최적화 우선순위화 |
| 평가 재현성 문제 | Medium | Low | 시드 고정, 환경 기록 |

## Deliverables

1. **Integrated Pipeline**: 전체 E2E 시스템
2. **Evaluation Scripts**: 벤치마크 평가 스크립트
3. **Performance Report**: 성능 분석 보고서
4. **Scenario Test Results**: 시나리오별 결과
5. **Visualization Tools**: 결과 시각화 도구
6. **Optimized Models**: 최적화된 모델 파일
7. **Deployment Package**: 배포 패키지
