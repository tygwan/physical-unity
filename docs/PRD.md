# Product Requirements Document (PRD)

## Autonomous Driving ML Platform

**Version**: 1.0
**Last Updated**: 2026-01-22
**Status**: Active Development

---

## 1. Executive Summary

### 1.1 Product Vision
Unity 시뮬레이션과 ROS2를 활용하여 자율주행을 위한 ML 모델(특히 Planning)을 개발하고 검증하는 통합 플랫폼 구축

### 1.2 Primary Focus
- **Planning (RL/IL)**: 강화학습 및 모방학습 기반 모션 플래닝
- Perception/Prediction은 기존 모델 활용 또는 간소화

### 1.3 Target Users
- ML 엔지니어/연구자
- 자율주행 시스템 개발자
- 로보틱스 엔지니어

---

## 2. Problem Statement

### 2.1 Current Challenges
1. **End-to-end 시스템 복잡성**: Perception → Prediction → Planning 파이프라인 구축 어려움
2. **실험 환경 부재**: 안전하고 재현 가능한 시뮬레이션 환경 필요
3. **데이터 부족**: 다양한 시나리오에 대한 학습 데이터 확보
4. **평가 기준 부재**: 표준화된 벤치마크 및 평가 메트릭

### 2.2 Solution
Unity + ROS2 + ML-Agents 통합 플랫폼으로 시뮬레이션 기반 자율주행 ML 모델 개발

---

## 3. Goals & Success Criteria

### 3.1 Primary Goals
| Goal | Description | Metric |
|------|-------------|--------|
| G1 | 안전한 자율주행 Planning | Collision Rate < 5% |
| G2 | 승차감 있는 주행 | Jerk < 2 m/s³ |
| G3 | 목표 지점 도달 | Route Completion > 85% |
| G4 | 실시간 추론 | Latency < 200ms |

### 3.2 Secondary Goals
- nuPlan Closed-loop Score > 60
- 다양한 시나리오 일반화 (Urban, Highway, Intersection)
- Sim-to-Real Transfer 가능성 검증

---

## 4. User Stories

### 4.1 ML 엔지니어
```
AS A ML 엔지니어
I WANT TO 시뮬레이션 환경에서 RL 에이전트를 학습시키고
SO THAT 안전한 모션 플래닝 모델을 개발할 수 있다
```

**Acceptance Criteria**:
- [ ] Unity 환경에서 차량 제어 가능
- [ ] PPO/SAC 알고리즘으로 학습 가능
- [ ] TensorBoard에서 학습 모니터링 가능
- [ ] 학습된 모델을 ONNX로 내보내기 가능

### 4.2 데이터 사이언티스트
```
AS A 데이터 사이언티스트
I WANT TO nuPlan/Waymo 데이터셋으로 모방학습을 수행하고
SO THAT Expert 수준의 주행 행동을 모방할 수 있다
```

**Acceptance Criteria**:
- [ ] 데이터셋 로딩 및 전처리 파이프라인
- [ ] Behavioral Cloning 학습 가능
- [ ] GAIL 학습 가능
- [ ] Expert 궤적과 비교 평가 가능

### 4.3 시스템 개발자
```
AS A 시스템 개발자
I WANT TO Perception-Prediction-Planning 파이프라인을 통합하고
SO THAT End-to-end 자율주행 시스템을 테스트할 수 있다
```

**Acceptance Criteria**:
- [ ] 센서 데이터 수집 (LiDAR, Camera)
- [ ] 3D 객체 인식 모델 연동
- [ ] 경로 예측 모델 연동
- [ ] Planning 모델 연동

---

## 5. Functional Requirements

### 5.1 Simulation Environment (Unity)

| ID | Requirement | Priority |
|----|-------------|----------|
| SIM-01 | 도시 환경 시뮬레이션 (도로, 교차로, 신호등) | P0 |
| SIM-02 | 고속도로 환경 시뮬레이션 | P1 |
| SIM-03 | 동적 장애물 (차량, 보행자) | P0 |
| SIM-04 | 날씨/조명 변화 | P2 |
| SIM-05 | 물리 엔진 기반 차량 동역학 | P0 |

### 5.2 Sensor Simulation (AWSIM)

| ID | Requirement | Priority |
|----|-------------|----------|
| SEN-01 | LiDAR 시뮬레이션 (64채널) | P0 |
| SEN-02 | Camera 시뮬레이션 (RGB/Depth) | P0 |
| SEN-03 | Radar 시뮬레이션 | P2 |
| SEN-04 | IMU/GPS 시뮬레이션 | P1 |

### 5.3 ML Training

| ID | Requirement | Priority |
|----|-------------|----------|
| ML-01 | PPO 알고리즘 지원 | P0 |
| ML-02 | SAC 알고리즘 지원 | P0 |
| ML-03 | Behavioral Cloning 지원 | P0 |
| ML-04 | GAIL 지원 | P1 |
| ML-05 | Hybrid RL+IL (CIMRL) 지원 | P1 |
| ML-06 | Multi-GPU 학습 | P2 |

### 5.4 Data Pipeline

| ID | Requirement | Priority |
|----|-------------|----------|
| DAT-01 | nuPlan 데이터셋 로딩 | P0 |
| DAT-02 | Waymo Open Motion 로딩 | P1 |
| DAT-03 | 데이터 증강 (노이즈, 변환) | P1 |
| DAT-04 | 시나리오 필터링/샘플링 | P1 |

### 5.5 Evaluation

| ID | Requirement | Priority |
|----|-------------|----------|
| EVL-01 | Collision Rate 측정 | P0 |
| EVL-02 | Route Completion 측정 | P0 |
| EVL-03 | Comfort Metrics (Jerk, Lat Acc) | P0 |
| EVL-04 | nuPlan Benchmark 호환 | P1 |
| EVL-05 | 시각화 도구 | P1 |

---

## 6. Non-Functional Requirements

### 6.1 Performance
- **Training**: 1M steps/day (24GB VRAM)
- **Inference**: < 50ms per step
- **Memory**: < 20GB (Training), < 4GB (Inference)

### 6.2 Scalability
- 최대 32개 병렬 환경 지원
- 다중 시나리오 동시 학습 가능

### 6.3 Compatibility
- Unity 2023.2+
- ROS2 Humble
- Python 3.10+
- PyTorch 2.0+
- ONNX Runtime

### 6.4 Reproducibility
- 랜덤 시드 고정 가능
- 실험 설정 버전 관리
- 모델 체크포인트 저장

---

## 7. Technical Constraints

### 7.1 Hardware Requirements
| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3080 (10GB) | RTX 4090 (24GB) |
| RAM | 32GB | 128GB |
| Storage | 500GB SSD | 4TB SSD |

### 7.2 Software Dependencies
- Windows 10/11 (Native development)
- Unity 2023.2+ (LTS preferred)
- ROS2 Humble (Windows build)
- CUDA 12.x / cuDNN 8.x

---

## 8. Assumptions & Dependencies

### 8.1 Assumptions
- Unity-ROS2 연동이 안정적으로 동작 (ros2-for-unity or Unity Robotics Hub)
- nuPlan 데이터셋 접근 가능
- RTX 4090 GPU 사용 가능

### 8.2 External Dependencies
- [nuPlan-devkit](https://github.com/motional/nuplan-devkit)
- [Waymo Open Dataset](https://waymo.com/open/)
- [ros2-for-unity](https://github.com/RobotecAI/ros2-for-unity)
- [AWSIM](https://github.com/tier4/AWSIM)

---

## 9. Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Unity-ROS2 연동 불안정 | High | Medium | 두 방식 비교 후 선택, fallback 준비 |
| 데이터셋 다운로드 실패 | Medium | Low | Mini 버전부터 시작 |
| RL 학습 불안정 | High | Medium | IL 초기화로 안정성 확보 |
| Sim-to-Real Gap | Medium | High | Domain randomization 적용 |

---

## 10. Out of Scope

- 실제 차량 배포 (Sim-to-Real Transfer는 검증만)
- V2X (Vehicle-to-Everything) 통신
- HD Map 자체 생성 (기존 맵 데이터 사용)
- 3D Perception 모델 자체 개발 (Pre-trained 사용)

---

## 11. Timeline Overview

| Phase | Duration | Key Deliverable |
|-------|----------|-----------------|
| Phase 1: Foundation | 2-3주 | Unity-ROS2 연동 |
| Phase 2: Data | 3-4주 | 데이터 파이프라인 |
| Phase 3: Perception | 2-3주 | Pre-trained 모델 연동 |
| Phase 4: Prediction | 3-4주 | Baseline Predictor |
| Phase 5: Planning | 6-8주 | RL/IL Motion Planner |
| Phase 6: Integration | 4-6주 | E2E 시스템 |
| Phase 7: Advanced | Ongoing | 최신 기술 연구 |

**Total Estimated Duration**: 20-28주 (5-7개월)

---

## 12. References

1. [nuPlan: A closed-loop ML-based planning benchmark](https://arxiv.org/abs/2106.11810)
2. [CIMRL: Combining Imitation and Reinforcement Learning for Safe Autonomous Driving](https://medium.com/nuro/cimrl-combining-imitation-and-reinforcement-learning-for-safe-autonomous-driving-13148ac99527)
3. [ML-Agents Documentation](https://github.com/Unity-Technologies/ml-agents)
4. [AWSIM Documentation](https://tier4.github.io/AWSIM/)

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| PPO | Proximal Policy Optimization, On-policy RL 알고리즘 |
| SAC | Soft Actor-Critic, Off-policy RL 알고리즘 |
| GAIL | Generative Adversarial Imitation Learning |
| BC | Behavioral Cloning, Supervised IL |
| BEV | Bird's Eye View, 조감도 표현 |
| TTC | Time-To-Collision, 충돌까지 남은 시간 |
