# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Overview

**Autonomous Driving ML Platform** - Unity + ROS2 + ML-Agents + PyTorch 기반 자율주행 ML 모델 개발 플랫폼

### Project Focus
- **Primary**: Planning (RL/IL 모션 플래닝) - 강화학습/모방학습 집중
- **Secondary**: Perception (3D Detection), Prediction (Trajectory/Behavior)

### Tech Stack
- **Simulation**: Unity 2023.2+ (Windows)
- **Middleware**: ROS2 Humble (Windows Native)
- **ML Framework**: ML-Agents 3.0, PyTorch 2.0+
- **Sensors**: AWSIM (LiDAR, Camera, Radar)
- **Language**: C# (Unity), Python (ML)

### Hardware Environment
| Component | Spec | Notes |
|-----------|------|-------|
| GPU | RTX 4090 (24GB VRAM) | 대규모 모델 학습 가능 |
| RAM | 128GB | 대용량 데이터 처리 |
| Storage | 4TB SSD | 전체 데이터셋 저장 가능 |

### Architecture
```
┌─────────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS DRIVING ML PLATFORM                     │
├─────────────────────────────────────────────────────────────────────┤
│                        Windows Native                                │
│  ┌────────────────────┐   ┌────────────────────┐   ┌─────────────┐  │
│  │   Unity 2023.2+    │   │    Python 3.10+    │   │  ROS2 Humble│  │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │   │  (Windows)  │  │
│  │  │AWSIM Sensors │  │◄─►│  │ PyTorch 2.0+ │  │◄─►│  Topics     │  │
│  │  │LiDAR/Cam/Rad │  │   │  │ ML Training  │  │   │  & Services │  │
│  │  └──────────────┘  │   │  └──────────────┘  │   └─────────────┘  │
│  │  ros2-for-unity    │   │  RL/IL Models      │                    │
│  └────────────────────┘   └────────────────────┘                    │
├─────────────────────────────────────────────────────────────────────┤
│                         DATA LAYER                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────────┐  │
│  │   nuPlan    │ │   Waymo     │ │   highD     │ │  INTERACTION  │  │
│  │  1282 hrs   │ │ Open Motion │ │  Highway    │ │  Interaction  │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                       MODEL PIPELINE                                 │
│  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────────┐   │
│  │ Perception  │ → │ Prediction  │ → │       Planning           │   │
│  │ 3D Detection│   │ Trajectory  │   │  RL (PPO/SAC) + IL (GAIL)│   │
│  │ BEV Encoder │   │ Behavior    │   │  Hybrid Motion Planner   │   │
│  └─────────────┘   └─────────────┘   └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
physical-unity/
├── .claude/                 # cc-initializer 설정
├── docs/                    # 프로젝트 문서
│   ├── PRD.md              # 제품 요구사항
│   ├── TECH-SPEC.md        # 기술 설계서
│   ├── PROGRESS.md         # 진행 상황
│   └── phases/             # Phase별 문서 (7 phases)
│       ├── phase-1/        # Foundation & Architecture
│       ├── phase-2/        # Data Infrastructure
│       ├── phase-3/        # Perception Models
│       ├── phase-4/        # Prediction Models
│       ├── phase-5/        # Planning Models (PRIMARY FOCUS)
│       ├── phase-6/        # Integration & Evaluation
│       └── phase-7/        # Advanced Topics
│
├── python/                  # Python ML 코드
│   ├── src/
│   │   ├── data/           # 데이터 로딩, 전처리
│   │   ├── models/         # ML 모델
│   │   │   ├── perception/
│   │   │   ├── prediction/
│   │   │   └── planning/   # RL/IL 모델 (핵심)
│   │   ├── training/       # 학습 스크립트
│   │   └── evaluation/     # 평가 스크립트
│   └── configs/
│       ├── perception/
│       ├── prediction/
│       └── planning/       # PPO/SAC/GAIL 설정
│
├── Assets/                  # Unity 프로젝트
│   ├── Scripts/Agents/     # AD 에이전트
│   ├── Scripts/Sensors/    # 센서 구현
│   └── Scripts/ROS2/       # ROS2 통합
│
├── datasets/               # 데이터셋 저장
│   ├── raw/               # 원본 데이터
│   ├── processed/         # 전처리된 데이터
│   └── splits/            # train/val/test 분할
│
├── experiments/            # 실험 추적
│   ├── configs/           # 실험 설정
│   ├── logs/              # 실험 로그
│   └── checkpoints/       # 체크포인트
│
├── models/                 # 학습된 모델
│   ├── perception/        # 3D Detection 모델
│   ├── prediction/        # Trajectory 예측 모델
│   └── planning/          # RL/IL Planner (.onnx)
│
└── scripts/                # 유틸리티 스크립트
```

## Key Commands

| Command | Purpose |
|---------|---------|
| `/experiment` | 실험 생성, 실행, 비교 |
| `/dataset` | 데이터셋 작업 |
| `/train` | 학습 시작/모니터링 |
| `/evaluate` | 벤치마크 실행 |
| `/phase status` | Phase 진행 상황 확인 |

## Development Workflow

### 학습 실행
```bash
# Python 환경 활성화
source .venv/bin/activate

# RL 학습 (PPO/SAC)
python python/src/training/train_rl.py --config python/configs/planning/ppo.yaml

# IL 학습 (Behavioral Cloning / GAIL)
python python/src/training/train_il.py --config python/configs/planning/gail.yaml

# Unity 환경과 연동 학습
mlagents-learn python/configs/planning/trainer_config.yaml --run-id=planning_v1
```

### TensorBoard / MLflow 모니터링
```bash
tensorboard --logdir=experiments/logs
mlflow ui --port 5000
```

## Domain Knowledge

### Planning Algorithms (Primary Focus)

#### Reinforcement Learning
- **PPO**: 안정적, On-policy, 기본 선택
- **SAC**: 샘플 효율, Off-policy, 연속 행동 공간
- **TD3**: SAC 대안, Twin Delayed DDPG

#### Imitation Learning
- **Behavioral Cloning**: Expert 데이터 직접 학습
- **GAIL**: GAN 기반 모방, 보상 함수 불필요
- **DAgger**: Interactive IL, Covariate shift 해결

#### Hybrid RL+IL (최종 목표)
- **CIMRL**: IL로 초기화 → RL로 fine-tuning (Waymo 방식)

### Reward Design
```yaml
progress: +1.0          # 전진 진행
goal_reached: +10.0     # 목표 도달
collision: -10.0        # 충돌 패널티
near_collision: -0.5    # TTC < 2초
off_road: -5.0          # 도로 이탈
jerk: -0.1              # 급가속/급감속
```

### Observation Space (~140D)
- ego_state: position, velocity, heading, acceleration (8D)
- route_info: waypoints, distances (30D)
- surrounding: 8 vehicles × 5 features (40D)
- BEV_features: optional (64D)

### Action Space (Continuous)
- acceleration: [-4.0, 2.0] m/s²
- steering: [-0.5, 0.5] rad

## Project Phases

1. **Phase 1**: Foundation & Architecture (2-3주)
2. **Phase 2**: Data Infrastructure (3-4주)
3. **Phase 3**: Perception Models (2-3주) - Simplified
4. **Phase 4**: Prediction Models (3-4주) - Simplified
5. **Phase 5**: Planning Models (6-8주) ⭐ PRIMARY FOCUS
6. **Phase 6**: Integration & Evaluation (4-6주)
7. **Phase 7**: Advanced Topics (Ongoing)

## Milestones

| Milestone | Target | Deliverable |
|-----------|--------|-------------|
| M1 | Week 3 | ROS2+Unity 연동 |
| M2 | Week 7 | 데이터셋 파이프라인 |
| M3 | Week 13 | Perception MVP |
| M4 | Week 19 | Prediction MVP |
| M5 | Week 27 | Planning MVP (RL Planner) |
| M6 | Week 33 | E2E 통합 시스템 |

## Success Criteria

| Category | Metric | Target |
|----------|--------|--------|
| Safety | Collision Rate | < 5% |
| Comfort | Jerk | < 2 m/s³ |
| Progress | Route Completion | > 85% |
| Latency | End-to-end | < 200ms |

## Important Notes

- Unity-ROS2 통합: ros2-for-unity 또는 Unity Robotics Hub 검토 후 선택
- Planning 집중: Perception/Prediction은 기존 모델 활용 또는 간소화
- 실험 추적: MLflow/W&B 사용하여 체계적 실험 관리
- `.onnx` 모델 파일은 Unity에서 직접 사용 가능
