# Phase 5: Planning Models (PRIMARY FOCUS)

## Overview

**프로젝트의 핵심 Phase**. 강화학습(RL)과 모방학습(IL) 기반 모션 플래닝 모델을 개발합니다.

## Goals

1. **RL 실험**: PPO, SAC 알고리즘으로 모션 플래닝
2. **IL 실험**: Behavioral Cloning, GAIL로 Expert 모방
3. **Hybrid 모델**: IL 초기화 + RL Fine-tuning (CIMRL)
4. **Ablation Study**: 각 요소별 기여도 분석

## Strategy

```
실험 순서:
1. Behavioral Cloning (BC) → Baseline 확립
2. Pure RL (PPO/SAC) → RL 한계 확인
3. GAIL → 보상 없는 모방 학습
4. Hybrid (BC → RL) → 최종 모델
5. Ablation Studies → 분석
```

## Scope

### In Scope
- PPO 알고리즘 구현 및 학습
- SAC 알고리즘 구현 및 학습
- Behavioral Cloning 구현 및 학습
- GAIL 구현 및 학습
- Hybrid RL+IL (CIMRL) 구현
- 보상 함수 설계 및 튜닝
- Observation/Action space 설계
- 성능 평가 및 비교

### Out of Scope
- 실차 적용 (Sim-to-Real)
- Model-based RL
- World Model

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Phase 5: Planning Models (PRIMARY FOCUS)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         EXPERIMENT PIPELINE                             │ │
│  │                                                                         │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │   │                    STAGE 1: Behavioral Cloning                   │  │ │
│  │   │  nuPlan Expert Data → Supervised Learning → BC Baseline          │  │ │
│  │   └─────────────────────────────────┬───────────────────────────────┘  │ │
│  │                                     │                                   │ │
│  │   ┌─────────────────────────────────▼───────────────────────────────┐  │ │
│  │   │                    STAGE 2: Pure RL (PPO/SAC)                    │  │ │
│  │   │  Random Init → Reward Shaping → Policy Optimization              │  │ │
│  │   └─────────────────────────────────┬───────────────────────────────┘  │ │
│  │                                     │                                   │ │
│  │   ┌─────────────────────────────────▼───────────────────────────────┐  │ │
│  │   │                    STAGE 3: GAIL                                 │  │ │
│  │   │  Expert Data → Discriminator → Policy w/o Reward                 │  │ │
│  │   └─────────────────────────────────┬───────────────────────────────┘  │ │
│  │                                     │                                   │ │
│  │   ┌─────────────────────────────────▼───────────────────────────────┐  │ │
│  │   │                    STAGE 4: Hybrid (CIMRL)                       │  │ │
│  │   │  BC Initialization → RL Fine-tuning → Final Model                │  │ │
│  │   └─────────────────────────────────┬───────────────────────────────┘  │ │
│  │                                     │                                   │ │
│  │   ┌─────────────────────────────────▼───────────────────────────────┐  │ │
│  │   │                    STAGE 5: Ablation Studies                     │  │ │
│  │   │  Reward Components, Network Architecture, Hyperparameters        │  │ │
│  │   └─────────────────────────────────────────────────────────────────┘  │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         MODEL ARCHITECTURE                              │ │
│  │                                                                         │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │  │                    Observation Encoder                           │   │ │
│  │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐                    │   │ │
│  │  │  │ Ego State │  │Route Info │  │Surrounding│  → Fusion → 256D   │   │ │
│  │  │  │    8D     │  │   30D     │  │   40D     │                    │   │ │
│  │  │  └───────────┘  └───────────┘  └───────────┘                    │   │ │
│  │  └─────────────────────────────────┬───────────────────────────────┘   │ │
│  │                                    │                                    │ │
│  │  ┌─────────────────────────────────▼───────────────────────────────┐   │ │
│  │  │                    Policy Network (Actor)                        │   │ │
│  │  │  256D → 256 → 256 → [μ_acc, μ_steer] / [σ_acc, σ_steer]         │   │ │
│  │  └─────────────────────────────────┬───────────────────────────────┘   │ │
│  │                                    │                                    │ │
│  │  ┌─────────────────────────────────▼───────────────────────────────┐   │ │
│  │  │                    Value Network (Critic)                        │   │ │
│  │  │  256D → 256 → 256 → V(s)                                         │   │ │
│  │  └─────────────────────────────────────────────────────────────────┘   │ │
│  │                                                                         │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Observation Space Design

```yaml
# Total: ~140 dimensions (without BEV) or ~200D (with BEV)

ego_state: 8D
  - position: [x, y]           # 2D
  - velocity: [vx, vy]         # 2D
  - heading: [sin, cos]        # 2D
  - acceleration: [ax, ay]     # 2D

route_info: 30D
  - waypoints: 10 x [x, y]     # 20D
  - distances: 10              # 10D

surrounding: 40D
  - vehicles: 8 x [x, y, vx, vy, heading]  # 40D

bev_features: 64D (optional)
  - encoded BEV representation
```

## Action Space Design

```yaml
# Continuous Action Space
continuous:
  acceleration:
    range: [-4.0, 2.0]    # m/s² (braking to acceleration)
    description: "Longitudinal control"

  steering:
    range: [-0.5, 0.5]    # rad (left to right)
    description: "Lateral control"

# Alternative: Discrete (for initial experiments)
discrete:
  acceleration: [-4, -2, 0, 1, 2]  # 5 levels
  steering: [-0.3, -0.15, 0, 0.15, 0.3]  # 5 levels
```

## Reward Function Design

```yaml
# Composite Reward Function

# === Base Rewards ===
progress: +1.0
  description: "Progress towards goal (normalized)"
  formula: "dot(velocity, goal_direction) / max_speed"

goal_reached: +10.0
  description: "Bonus for reaching goal"
  condition: "distance_to_goal < 2.0m"

# === Safety Penalties ===
collision: -10.0
  description: "Collision with any object"
  terminates: true

near_collision: -0.5
  description: "Time-to-Collision < threshold"
  formula: "-0.5 * (2.0 - TTC) / 2.0 if TTC < 2.0"

off_road: -5.0
  description: "Vehicle leaves drivable area"

# === Comfort Rewards ===
jerk: -0.1
  description: "Penalize sudden acceleration changes"
  formula: "-0.1 * |d(acceleration)/dt|"

lateral_acc: -0.05
  description: "Penalize high lateral acceleration"
  formula: "-0.05 * |lateral_acceleration|"

steering_rate: -0.02
  description: "Penalize rapid steering"
  formula: "-0.02 * |steering - prev_steering| / dt"

# === Traffic Rules ===
lane_keeping: +0.5
  description: "Bonus for staying in lane"
  condition: "center_offset < 0.5m"

speed_limit: -0.5
  description: "Penalty for exceeding speed limit"
  condition: "speed > speed_limit"

traffic_light: -5.0
  description: "Penalty for running red light"
```

## Task Breakdown

| ID | Task | Priority | Est. Time |
|----|------|----------|-----------|
| **Stage 1: Behavioral Cloning** |
| P5-01 | BC 데이터 로더 구현 | High | 2일 |
| P5-02 | BC 네트워크 구현 | High | 2일 |
| P5-03 | BC 학습 파이프라인 | High | 3일 |
| P5-04 | BC 평가 및 튜닝 | High | 3일 |
| **Stage 2: Pure RL** |
| P5-05 | PPO 구현 | High | 3일 |
| P5-06 | SAC 구현 | High | 3일 |
| P5-07 | 보상 함수 구현 | High | 2일 |
| P5-08 | Unity 환경 연동 | High | 3일 |
| P5-09 | RL 학습 및 튜닝 | High | 5일 |
| **Stage 3: GAIL** |
| P5-10 | Discriminator 구현 | Medium | 2일 |
| P5-11 | GAIL 학습 파이프라인 | Medium | 3일 |
| P5-12 | GAIL 평가 | Medium | 2일 |
| **Stage 4: Hybrid** |
| P5-13 | BC → RL Fine-tuning | High | 3일 |
| P5-14 | CIMRL 구현 | High | 3일 |
| P5-15 | Hybrid 평가 | High | 2일 |
| **Stage 5: Ablation** |
| P5-16 | 보상 요소별 분석 | Medium | 3일 |
| P5-17 | 아키텍처 분석 | Medium | 2일 |
| P5-18 | 하이퍼파라미터 분석 | Medium | 2일 |

## Algorithm Comparison

| Algorithm | Type | Pros | Cons | Sample Efficiency |
|-----------|------|------|------|-------------------|
| PPO | RL (On-policy) | 안정적, 구현 쉬움 | 샘플 비효율 | Low |
| SAC | RL (Off-policy) | 샘플 효율, 연속 행동 | 복잡도 높음 | High |
| TD3 | RL (Off-policy) | SAC보다 안정적 | - | High |
| BC | IL (Supervised) | 간단, 빠른 학습 | Covariate shift | N/A |
| GAIL | IL (GAN-based) | 보상 불필요 | 불안정 가능 | Medium |
| DAgger | IL (Interactive) | Shift 해결 | Expert 필요 | N/A |
| CIMRL | Hybrid | 두 장점 결합 | 복잡한 튜닝 | Medium |

## Hyperparameters

### PPO
```yaml
clip_ratio: 0.2
vf_coef: 0.5
entropy_coef: 0.01
learning_rate: 3e-4
batch_size: 2048
minibatch_size: 64
epochs_per_update: 10
gamma: 0.99
gae_lambda: 0.95
```

### SAC
```yaml
learning_rate: 3e-4
tau: 0.005
gamma: 0.99
buffer_size: 1_000_000
batch_size: 256
auto_entropy: true
```

### BC
```yaml
learning_rate: 1e-4
batch_size: 256
loss: mse  # or nll
epochs: 100
```

## Success Criteria

| Metric | Target | Notes |
|--------|--------|-------|
| Collision Rate | < 5% | Safety-critical |
| Progress Score | > 80% | Route following |
| Comfort Score | > 70% | jerk < 2 m/s³ |
| nuPlan Score | > 60 | Closed-loop benchmark |
| Inference Time | < 50ms | Real-time capable |

## Timeline

**예상 소요**: 6-8주

## Dependencies

- Phase 1-4 완료
- nuPlan 데이터셋 접근
- 충분한 GPU 리소스 (RTX 4090)

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| RL 학습 불안정 | High | Medium | BC 초기화 |
| 보상 함수 튜닝 어려움 | High | High | GAIL 대안 |
| 긴 학습 시간 | Medium | High | 병렬 환경 |
| Sim-to-Real Gap | Medium | High | Domain Randomization |

## Experiment Tracking

```yaml
# MLflow/W&B Logging

metrics:
  - episode_reward
  - collision_rate
  - progress_score
  - comfort_score (jerk)
  - policy_loss
  - value_loss

artifacts:
  - model checkpoints
  - training curves
  - evaluation videos
  - hyperparameter configs
```

## Deliverables

1. **BC Model**: Expert 데이터로 학습된 baseline
2. **PPO/SAC Models**: RL로 학습된 정책
3. **GAIL Model**: 보상 없이 모방 학습된 정책
4. **Hybrid Model**: BC + RL fine-tuning 최종 모델
5. **Reward Function**: 검증된 보상 함수 구현
6. **Evaluation Scripts**: 벤치마크 평가 스크립트
7. **Ablation Report**: 실험 분석 보고서
8. **ONNX Models**: Unity 추론용 모델
