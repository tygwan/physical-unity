# Progress Tracker

## Project: Autonomous Driving ML Platform

**Focus**: Planning (RL/IL 모션 플래닝)

---

## Current Status

| Metric | Value |
|--------|-------|
| **Current Phase** | Phase 1 - Foundation & Architecture |
| **Sprint** | Sprint 1 (환경 구축) |
| **Overall Progress** | 5% |
| **Estimated Completion** | Week 33 |

---

## Phase Overview (7 Phases)

| Phase | Name | Duration | Status | Progress | Deliverable |
|-------|------|----------|--------|----------|-------------|
| **Phase 1** | Foundation & Architecture | 2-3주 | 🔄 In Progress | 10% | Unity-ROS2 연동 |
| **Phase 2** | Data Infrastructure | 3-4주 | ⏳ Pending | 0% | 데이터 파이프라인 |
| **Phase 3** | Perception Models | 2-3주 | ⏳ Pending | 0% | Pre-trained 모델 연동 |
| **Phase 4** | Prediction Models | 3-4주 | ⏳ Pending | 0% | Baseline Predictor |
| **Phase 5** | Planning Models ⭐ | 6-8주 | ⏳ Pending | 0% | RL/IL Planner |
| **Phase 6** | Integration & Evaluation | 4-6주 | ⏳ Pending | 0% | E2E 시스템 |
| **Phase 7** | Advanced Topics | Ongoing | ⏳ Pending | 0% | 최신 기술 연구 |

---

## Milestone Tracker

| Milestone | Target | Status | Actual |
|-----------|--------|--------|--------|
| M1: 환경 완료 | Week 3 | 🔄 In Progress | - |
| M2: 데이터 파이프라인 | Week 7 | ⏳ Pending | - |
| M3: Perception MVP | Week 13 | ⏳ Pending | - |
| M4: Prediction MVP | Week 19 | ⏳ Pending | - |
| M5: Planning MVP | Week 27 | ⏳ Pending | - |
| M6: 통합 시스템 | Week 33 | ⏳ Pending | - |

---

## Phase 1: Foundation & Architecture (Current)

### Objectives
- Windows 네이티브 환경 구축
- Unity-ROS2 연동 확립
- ML-Agents RL 학습 환경 구축
- 기본 주행 환경 Scene 생성

### Task Breakdown

| ID | Task | Priority | Status | Notes |
|----|------|----------|--------|-------|
| P1-01 | Windows에 ROS2 Humble 설치 | High | ⏳ Pending | |
| P1-02 | Unity Robotics Hub 테스트 | High | ⏳ Pending | |
| P1-03 | ros2-for-unity 테스트 | High | ⏳ Pending | |
| P1-04 | 두 방식 성능 비교 후 선택 | High | ⏳ Pending | |
| P1-05 | ML-Agents 3.0 RL 환경 구축 | High | ⏳ Pending | |
| P1-06 | 기본 주행 환경 Scene 생성 | High | ⏳ Pending | |
| P1-07 | AWSIM 센서 통합 (LiDAR/Camera) | Medium | ⏳ Pending | |
| P1-08 | 실험 추적 설정 (MLflow/W&B) | Medium | ⏳ Pending | |

### Completed ✅
- [x] 프로젝트 저장소 초기화
- [x] 디렉토리 구조 설계 (AD 플랫폼 용도로 재구성)
- [x] README.md 작성
- [x] 기술 문서 작성 (PRD, TECH-SPEC)
- [x] cc-initializer 설정 업데이트

### In Progress 🔄
- [ ] P1-01: ROS2 Humble 설치 (Windows)
- [ ] P1-05: ML-Agents 환경 구축

### Blocked 🚧
*현재 블로커 없음*

---

## Phase 2: Data Infrastructure (Upcoming)

### Objectives
- 데이터셋 확보 및 전처리 파이프라인 구축

### Planned Datasets

| Dataset | Size | Use Case | Priority |
|---------|------|----------|----------|
| nuPlan (mini) | ~50GB | Imitation Learning, Planning | Primary |
| Waymo Motion | ~100GB | Trajectory Prediction | Primary |
| highD | ~5GB | Highway Behavior | Secondary |
| INTERACTION | ~2GB | Intersection Scenarios | Secondary |

### Key Tasks
- 데이터셋 다운로드 및 통합 포맷 설계
- 시나리오 추출 파이프라인 구현
- 데이터 증강 전략 구현
- 시각화 도구 개발

---

## Phase 3: Perception Models (Simplified)

### Strategy
> Planning 집중을 위해 Perception은 간소화

### Approach Options
1. **Option A**: Ground Truth 직접 사용 (시뮬레이션)
2. **Option B**: Pre-trained 모델 활용 (MMDetection3D)
3. **Option C**: 간단한 BEV 인코더만 구현

### Key Tasks
- P3-01: 시뮬레이션 Ground Truth 추출
- P3-02: Pre-trained 3D detection 모델 테스트
- P3-03: BEV representation 생성

---

## Phase 4: Prediction Models (Simplified)

### Strategy
> nuPlan baseline predictor 활용

### Approach Options
1. **Primary**: nuPlan-devkit baseline predictor
2. **Secondary**: Constant velocity model
3. **Optional**: Custom Transformer predictor

### Key Tasks
- P4-01: nuPlan baseline predictor 설정
- P4-02: Constant velocity baseline 구현
- P4-03: Planning과 prediction 연동

---

## Phase 5: Planning Models ⭐ (Primary Focus)

### Objectives
- RL/IL 기반 모션 플래닝 개발
- **이 Phase가 프로젝트의 핵심**

### Experiment Roadmap

```
1. Behavioral Cloning (BC)
   └─ nuPlan 데이터로 Expert 모방 → Baseline

2. Pure RL (PPO/SAC)
   └─ 보상 함수 설계 및 튜닝

3. GAIL
   └─ 보상 없이 모방 학습

4. Hybrid (BC → RL fine-tuning)
   └─ CIMRL 방식, 최종 모델

5. Ablation Studies
   └─ 각 요소별 기여도 분석
```

### Algorithm Comparison

| Algorithm | Type | Pros | Cons | Priority |
|-----------|------|------|------|----------|
| PPO | RL (On-policy) | 안정적, 구현 쉬움 | 샘플 비효율 | High |
| SAC | RL (Off-policy) | 샘플 효율, 연속 행동 | 복잡도 높음 | High |
| BC | IL (Supervised) | 간단, 빠른 학습 | Covariate shift | High |
| GAIL | IL (GAN-based) | 보상 불필요, 분포 학습 | 불안정 가능 | High |
| DAgger | IL (Interactive) | Covariate shift 해결 | Expert 필요 | Medium |

### Success Criteria

| Metric | Target |
|--------|--------|
| Collision Rate | < 5% |
| Progress Score | > 80% |
| Comfort Score | > 70% (jerk < 2 m/s³) |
| nuPlan Score | > 60 |

---

## Phase 6: Integration & Evaluation

### Objectives
- End-to-end 통합
- 벤치마크 평가

### Integration Pipeline
```
Perception → Prediction → Planning → Control → Simulation
```

### Evaluation Metrics

| Category | Metric | Target |
|----------|--------|--------|
| Safety | Collision Rate | < 5% |
| Comfort | Jerk | < 2 m/s³ |
| Progress | Route Completion | > 85% |
| Latency | End-to-end | < 200ms |

---

## Phase 7: Advanced Topics

### Research Areas
- World Model for Driving
- LLM-based Planning (DriveGPT)
- VLA Framework Integration
- Sim-to-Real Transfer

---

## Recent Activity Log

| Date | Activity | Status |
|------|----------|--------|
| 2026-01-22 | AD 플랫폼으로 프로젝트 재구성 | ✅ |
| 2026-01-22 | 7-Phase 시스템 설계 | ✅ |
| 2026-01-22 | PRD, TECH-SPEC 문서 작성 | ✅ |
| 2026-01-21 | 프로젝트 초기화, 구조 설계 | ✅ |
| 2026-01-21 | cc-initializer 연동 | ✅ |

---

## Notes & Decisions

### Key Decisions
1. **ROS2 Bridge**: ros2-for-unity vs Unity Robotics Hub → Phase 1에서 비교 후 선택
2. **Perception**: Pre-trained 모델 우선, 필요시 직접 개발
3. **Prediction**: nuPlan baseline 사용
4. **Planning Focus**: RL/IL에 집중 (PPO, SAC, BC, GAIL)

### Blockers & Risks
*현재 확인된 블로커 없음*

### Next Actions
1. ROS2 Humble Windows 설치
2. Unity-ROS2 연동 테스트
3. ML-Agents 환경 구축
4. 기본 주행 Scene 생성
