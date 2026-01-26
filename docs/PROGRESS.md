# Progress Tracker

## Project: Autonomous Driving ML Platform

**Focus**: Planning (RL/IL 모션 플래닝)

---

## Current Status

| Metric | Value |
|--------|-------|
| **Current Phase** | Phase 5 - Reinforcement Learning |
| **Current Training** | v12 Phase B (Overtake vs Follow Decision) |
| **Overall Progress** | 65% |
| **Architecture** | Tesla-style E2E (Camera → Neural Net → Control) |
| **Last Updated** | 2026-01-25 |

### v12 Overtaking Training Progress 🔄

#### Phase A: 기본 추월 학습 ✅ COMPLETED
- **Config**: `vehicle_ppo_v12_phaseA.yaml`
- **Steps**: 2M (완료)
- **Best Reward**: **+937** (1.37M step)
- **Final Reward**: +714.7
- **핵심 성과**: Speed penalty 버그 수정 후 완벽한 추월 행동 학습
- **Model**: `results/v12_phaseA_fixed/E2EDrivingAgent-1999953.onnx`

#### Phase B: 추월 vs 따라가기 판단 🔄 IN PROGRESS
- **Config**: `vehicle_ppo_v12_phaseB.yaml`
- **Initialize From**: Phase A best model
- **Goal**: NPC 속도에 따라 추월/따라가기 결정 학습
- **NPC Speed Curriculum**: 0.3 → 0.5 → 0.7 → 0.9
- **Expected Steps**: 2M
- **Status**: 학습 시작됨 (Unity Play 대기중)

#### Phase C: 복잡 환경 일반화 (계획)
- **Config**: `vehicle_ppo_v12_phaseC.yaml`
- **NPC Count**: 2-4대
- **Speed Variation**: 0.15-0.3
- **Goal Distance**: 100m → 160m → 230m
- **Expected Steps**: 4M

### 연구 기반 개선 계획 (RESEARCH-TRENDS-2024-2026)

| 개선 사항 | 출처 | Phase | 우선순위 |
|-----------|------|-------|----------|
| TTC Observation 추가 | Safe RL | 5.5 | HIGH |
| Network 512→1024 | Quick Win | 5.5 | MEDIUM |
| Teacher-Student Distillation | CuRLA | 6 | HIGH |
| GAIL 통합 | IL Research | 6 | HIGH |
| CMDP/Safe RL | LSTC | 6 | MEDIUM |
| Diffusion Planning | ICLR 2025 | 7 | LOW |

### Stage 4 Reward 재조정 (v7→v8 변경사항)

| 항목 | v7 (이전) | v8 (현재) | 변경 이유 |
|------|-----------|-----------|-----------|
| collision | -10.0 | **-5.0** | PPO gradient instability 완화 |
| nearCollision | -0.5/frame | **-1.5/sec** (×deltaTime) | 프레임 비독립 → rate-independent |
| off_road | -5.0/sec (누적) | **-5.0 + EndEpisode** | -200 누적 방지, 즉시 종료 |
| NPC curriculum | 0→2→4→6 | **0→1→2→4** (점진적) | 급격한 난이도 증가 방지 |
| NPC threshold | -2.0 (goal과 동일) | **-1.5, -1.0, 0.0** (개별) | 동시 진행 방지 |
| goal_distance | 50→120→230 | **50→100→160→230** | 중간 단계 추가 |
| max_steps | 5,000,000 | **8,000,000** | 수렴 시간 확보 |
| timeout_wait | 300s | **600s** | Unity 응답 시간 확보 |

### Experiment Pipeline (Revised 2026-01-24)

```
Phase 5A: Vector-based RL (벡터 관측 기반)
  ✅ Stage 1: BC Baseline
  ✅ Stage 2: PPO Single Area (950K, Reward ~700)
  ✅ Stage 3: 16x Parallel PPO (1.66M, Reward ~750 수렴)
  🔄 Stage 4: 속도 정책 (v8, 3.07M/8M steps)          ← 현재 진행 중
  ⏳ Stage 5: Multi-Lane + 차선 정책
  ⏳ Stage 6: 도로 네트워크 + 교차로 + 경로 추종 ★

Phase 5B: Vision-based RL (카메라 입력)
  ⏳ Stage 7: Camera Visual Observation (nature_cnn)
  ⏳ Stage 8: Euro NCAP 평가 (ELK/LKA) + 신호등 인식

Phase 5C: Hybrid & Advanced
  ⏳ Stage 9: Expert 녹화 → GAIL/Hybrid
  ⏳ Stage 10: Full E2E (BEV + Temporal)
```

### Stage 4 완료 후 계획
1. **32 Training Areas 확장** - 벡터 전용 관측 시 32개 병렬 학습 가능 (VRAM ~8GB)
2. **SAC 비교 실험** - 동일 환경에서 PPO vs SAC 알고리즘 검증
3. **카메라/LiDAR 병렬 제한** - 센서 추가 시 8-16 Areas로 축소 필요 (VRAM 제약)

### 향후 병렬 Training Areas 계획
```
현재: 16 Areas (Vector-only) → GPU 11%, VRAM 3.6GB
다음: 32 Areas (Vector-only) → GPU ~22%, VRAM ~8GB  ← Stage 4 완료 후 적용
카메라: 8-16 Areas (Camera 84x84) → VRAM 8-12GB     ← Stage 7에서 적용
LiDAR:  8 Areas (LiDAR+Camera) → VRAM 12-16GB       ← Phase 6에서 검토
```

### 신호등 인식 학습 계획 (Phase 6 범위)
- **현재 Stage 4**: 속도 제한만 (표지판 = ground truth observation)
- **Stage 7-8**: 카메라 입력 + CNN encoder로 시각적 인식
- **Phase 6**: 신호등 인식 (적/녹/황) + 정지/출발 정책
  - Camera observation으로 신호등 색상 인식
  - traffic_light reward: 적색 정지(+0.5), 적색 위반(-5.0)
  - 교차로 + 신호체계 통합 (Stage 6 도로 네트워크 이후)

### 포트 충돌 해결법
이전 학습 프로세스가 포트 5004를 점유 중일 수 있음:
```bash
netstat -ano | findstr 5004    # PID 확인
taskkill /PID <PID> /F         # 프로세스 종료 (cmd에서 실행)
```
또는 Unity Play 모드 중지 → 재시작으로 해결

---

## Phase Overview (7 Phases) - Tesla-style E2E Architecture

| Phase | Name | Duration | Status | Progress | Deliverable |
|-------|------|----------|--------|----------|-------------|
| **Phase 1** | Foundation & Architecture | 2-3주 | ✅ Complete | 100% | Unity + ML-Agents + ROS2 + Sensors |
| **Phase 2** | Data Infrastructure | 3-4주 | ✅ Complete | 90% | nuPlan 파이프라인 (loader/processor/augmentation) |
| **Phase 3** | E2E Model Architecture | 3-4주 | ✅ Complete | 100% | PyTorch E2E 모델 (RegNet→Planning) |
| **Phase 4** | Imitation Learning | 3-4주 | ✅ Complete | 100% | BC/GAIL 학습 파이프라인 |
| **Phase 5** | Reinforcement Learning ⭐ | 6-10주 | 🔄 In Progress | 40% | Stage 4 진행 중 (v8 3.07M/8M, NPC+속도정책 학습) |
| **Phase 6** | Hybrid & Deployment | 3-4주 | ⏳ Pending | 0% | IL→RL + Sentis + ROS2 배포 |
| **Phase 7** | Advanced Topics | Ongoing | ⏳ Pending | 0% | World Model, Sim-to-Real |

### Architecture Decision: Tesla-style E2E

```
기존 계획 (Modular):
  Perception → Prediction → Planning → Control (각각 독립)

새 계획 (E2E):
  Camera Images → [Single Neural Network] → steering + acceleration
  (Tesla FSD v12+ 방식: 모든 모듈이 하나의 네트워크로 통합)
```

**근거**: Tesla FSD 조사 결과, E2E 접근법이 모듈식 대비:
- 코드 복잡도 250x 감소 (500k→2k lines)
- 에러 전파 제거 (72.7% → unified optimization)
- 학습 속도 향상 (단일 backprop)
- 일반화 성능 우수

---

## Milestone Tracker

| Milestone | Target | Status | Actual | Deliverable |
|-----------|--------|--------|--------|-------------|
| M1: 환경 완료 | Week 3 | ✅ Complete | Week 4 | Unity+ML-Agents+ROS2 |
| M2: 데이터 파이프라인 | Week 7 | ✅ Complete | Week 5 | nuPlan loader+processor |
| M3: E2E 모델 구현 | Week 11 | ✅ Complete | Week 8 | PyTorch E2E model (.pt) |
| M4: IL 학습 완료 | Week 15 | ✅ Complete | Week 8 | BC/GAIL training pipeline |
| M5: RL 학습 완료 | Week 27 | 🔄 In Progress | - | Stage 3/10 완료. 경로추종+카메라+E2E 진행 중 |
| M6: 배포 완료 | Week 25 | ⏳ Pending | - | Sentis .onnx + ROS2 |

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
| P1-01 | ROS2 Humble 설치 | High | ✅ Complete | WSL2 (초기 설정, 현재 ML-Agents 사용) |
| P1-02 | Unity Robotics Hub 테스트 | High | ✅ Complete | ROS-TCP-Connector 연동 |
| P1-03 | ros2-for-unity 테스트 | High | ⏳ Pending | (선택사항) |
| P1-04 | ROS2-Unity 연결 확인 | High | ✅ Complete | TCP 통신 성공 |
| P1-05 | ML-Agents 4.0 RL 환경 구축 | High | ✅ Complete | Unity 6 + ML-Agents 4.0.1 |
| P1-06 | 기본 주행 환경 Scene 생성 | High | ✅ Complete | DrivingScene.unity |
| P1-07 | 센서 통합 (LiDAR/Camera) | Medium | ✅ Complete | CameraSensor, LiDARSensor |
| P1-08 | 실험 추적 설정 (MLflow/W&B) | Medium | ✅ Complete | experiment_tracker.py |

### Completed ✅
- [x] 프로젝트 저장소 초기화
- [x] 디렉토리 구조 설계 (AD 플랫폼 용도로 재구성)
- [x] README.md 작성
- [x] 기술 문서 작성 (PRD, TECH-SPEC)
- [x] cc-initializer 설정 업데이트
- [x] **Unity 6 (6000.3.4f1) 환경 구축**
- [x] **ML-Agents 4.0.1 설치 및 설정**
- [x] **Unity Sentis 2.4.1 설치**
- [x] **Python mlagents 1.1.0 + PyTorch 2.3.1 설치**
- [x] **3DBall 학습 테스트 성공 (500K steps, Reward 100)**
- [x] **ONNX 모델 Export 및 Inference 확인**
- [x] **WSL2 + ROS2 Humble 설치** *(초기 설정, 현재 ML-Agents 직접 통신 사용)*
- [x] **ROS-TCP-Endpoint 빌드**
- [x] **Unity ROS-TCP-Connector 설치**
- [x] **ROS2 ↔ Unity 연결 테스트 성공**
- [x] **기본 주행 Scene (DrivingScene) 생성**
- [x] **VehicleAgent.cs 생성** (ML-Agents 기반)
- [x] **SimpleVehicleController.cs 생성** (키보드 테스트용)
- [x] **키보드 주행 테스트 성공** (W/A/S/D)
- [x] **VehicleROSBridge.cs 생성** (ROS2 Pub/Sub)
- [x] **ROS2 Topic 구현** (/vehicle/odom, /vehicle/pose, /vehicle/cmd_vel)
- [x] **CameraSensor.cs 생성** (VLA용 이미지 캡처)
- [x] **LiDARSensor.cs 생성** (포인트클라우드 레이캐스팅)
- [x] **센서 ROS2 Publish** (/vehicle/camera/image_raw, /vehicle/lidar/points)
- [x] **실험 추적 설정** (MLflow/W&B integration)

### In Progress 🔄
*Phase 1 완료 - Phase 2로 진행*

### Blocked 🚧
*현재 블로커 없음*

---

## Phase 2: Data Infrastructure ✅

### Status: 90% Complete

### Completed Tasks
| ID | Task | Status | File |
|----|------|--------|------|
| P2-01 | nuPlan 환경 설정 스크립트 | ✅ | `python/scripts/setup_nuplan.py` |
| P2-02 | nuPlan 데이터 로더 | ✅ | `python/src/data/nuplan_loader.py` |
| P2-03 | 시나리오 전처리기 (PlanningProcessor) | ✅ | `python/src/data/processor.py` |
| P2-04 | 데이터 증강 (6 techniques) | ✅ | `python/src/data/augmentation.py` |
| P2-05 | 시각화 도구 (BEV plot) | ✅ | `python/src/data/visualizer.py` |
| P2-06 | Train/Val/Test 분할 | ✅ | `python/src/data/splitter.py` |
| P2-07 | Waymo Motion 로더 | ⏳ Deferred | 우선순위 낮음 |

### Data Format (Unified Scenario)
```python
Scenario:
  ego_trajectory: [T, 7]  # x, y, heading, vx, vy, ax, ay
  agents: List[AgentTrack]  # 주변 차량
  map_features: Dict        # 도로 정보
  traffic_lights: List      # 신호등 상태

Observation (238D):
  ego_state: 8D + ego_history: 40D + agents: 160D + route: 30D

Action (2D):
  acceleration: [-4.0, +2.0] m/s²
  steering: [-0.5, +0.5] rad
```

---

## Phase 3: E2E Model Architecture 🔄 (Current)

### Objectives
Tesla FSD v12+ 스타일 E2E 신경망 구현

### Architecture Overview
```
Input Layer
├── Camera Images: [B, 8, 3, 768, 576] (8 cameras)
├── Ego State: [B, 8] (position, velocity, heading, acceleration)
├── Route Info: [B, 30] (waypoints)
└── Temporal History: [B, 10, ...] (10 frames)
        |
        v
┌─────────────────────────────────────────────┐
│ 3-1. Backbone: RegNet                        │
│      Multi-scale features P1-P5              │
│      Output: [B, 512, 10, 8]                │
├─────────────────────────────────────────────┤
│ 3-2. Neck: BiFPN                             │
│      Cross-scale fusion (6 layers)           │
│      Output: [B, 256, 40, 30]               │
├─────────────────────────────────────────────┤
│ 3-3. Occupancy Network                       │
│      2D features → 3D voxel grid            │
│      Output: [B, 100, 100, 4] (occupied?)   │
├─────────────────────────────────────────────┤
│ 3-4. BEV Former                              │
│      Multi-camera → BEV (Transformer)        │
│      Output: [B, 100, 100, 256]             │
├─────────────────────────────────────────────┤
│ 3-5. Temporal Fusion                         │
│      LSTM + Transformer (10 frames)          │
│      Output: [B, 256] context vector         │
├─────────────────────────────────────────────┤
│ 3-6. Planning Network                        │
│      Features → Trajectory → Control         │
│      Output: steering, acceleration          │
├─────────────────────────────────────────────┤
│ 3-7. E2E Integration                         │
│      All modules unified, single forward()   │
│      Export: .pt → .onnx → Sentis            │
└─────────────────────────────────────────────┘
```

### Task Breakdown

| ID | Task | Priority | Status | File Path |
|----|------|----------|--------|-----------|
| P3-01 | RegNet Backbone | High | ⏳ | `python/src/models/backbone/regnet.py` |
| P3-02 | BiFPN Neck | High | ⏳ | `python/src/models/neck/bifpn.py` |
| P3-03 | Occupancy Network | High | ⏳ | `python/src/models/perception/occupancy.py` |
| P3-04 | BEV Former | High | ⏳ | `python/src/models/perception/bev_former.py` |
| P3-05 | Temporal Fusion | Medium | ⏳ | `python/src/models/temporal/fusion.py` |
| P3-06 | Planning Network | High | ⏳ | `python/src/models/planning/planner.py` |
| P3-07 | E2E 통합 모델 | High | ⏳ | `python/src/models/e2e_model.py` |
| P3-08 | 모델 단위 테스트 | Medium | ⏳ | `python/tests/test_models.py` |
| P3-09 | ONNX Export 검증 | Medium | ⏳ | `python/scripts/export_onnx.py` |

### Module Specifications

#### P3-01: RegNet Backbone
```python
Input:  [B, 3, H, W] per camera (or [B, 8, 3, H, W] concatenated)
Output: Multi-scale features
  P1: [B, 32, H/4, W/4]    # High-res details
  P2: [B, 64, H/8, W/8]
  P3: [B, 128, H/16, W/16]
  P4: [B, 256, H/32, W/32]
  P5: [B, 512, H/64, W/64]  # Global semantic

Config:
  depth: 50 (ResNet-50 scale, 확장 가능)
  width_multiplier: 1.0
  group_width: 32
```

#### P3-02: BiFPN Neck
```python
Input:  P1-P5 multi-scale features
Output: Enhanced P3 features [B, 256, H/16, W/16]

Config:
  num_layers: 3-6
  channels: 256
  attention_type: "fast_attention"
```

#### P3-03: Occupancy Network
```python
Input:  BEV features [B, 256, 100, 100]
Output:
  occupancy: [B, 100, 100, 4]  # 100m x 100m x 4 height levels
  flow: [B, 100, 100, 2]       # vx, vy per cell

Resolution: 1m per cell
Height levels: 0-1m, 1-2m, 2-3m, 3-4m
```

#### P3-04: BEV Former
```python
Input:  8 camera features + camera matrices
Output: BEV features [B, 100, 100, 256]

Config:
  num_queries: 10000 (100x100 grid)
  num_heads: 8
  num_layers: 6
  d_model: 256
```

#### P3-05: Temporal Fusion
```python
Input:  Feature sequence [T=10, B, 256]
Output: Temporal context [B, 256]

Config:
  method: "transformer"  # or "lstm", "conv3d"
  num_frames: 10
  num_layers: 3
```

#### P3-06: Planning Network
```python
Input:
  occupancy: [B, 100, 100, 4]
  bev_features: [B, 100, 100, 256]
  temporal_context: [B, 256]
  ego_state: [B, 8]
  route_info: [B, 30]

Output:
  steering: [-0.5, +0.5] rad
  acceleration: [-4.0, +2.0] m/s^2
  trajectory: [B, 30, 3]  # 30 timesteps x (x, y, heading)
  confidence: [B, 6]      # 6 trajectory candidates

Architecture:
  - Multi-head attention fusion
  - 6 trajectory candidates generation
  - Confidence scoring + weighted selection
  - Final control output
```

#### P3-07: E2E Integration
```python
class E2EDrivingModel(nn.Module):
    """
    Tesla-style End-to-End Driving Model
    Input: cameras + ego_state + route
    Output: steering + acceleration
    """
    def __init__(self, config):
        self.backbone = RegNet(config.backbone)
        self.neck = BiFPN(config.neck)
        self.occupancy = OccupancyNetwork(config.occupancy)
        self.bev_former = BEVFormer(config.bev_former)
        self.temporal = TemporalFusion(config.temporal)
        self.planner = PlanningNetwork(config.planning)

    def forward(self, cameras, ego_state, route, camera_matrices):
        features = self.backbone(cameras)
        features = self.neck(features)
        bev = self.bev_former(features, camera_matrices)
        occ = self.occupancy(bev)
        temporal = self.temporal(bev)
        return self.planner(occ, bev, temporal, ego_state, route)
```

### Simplified Start Strategy
> 처음부터 전체 모델을 구현하면 디버깅이 어려움.
> 단계적으로 복잡도를 높이는 전략:

```
Level 1 (MVP): MLP Planner only
  Input: 238D observation vector (ego+agents+route)
  Network: MLP (256→256→256→2)
  Output: steering, acceleration
  → 빠르게 IL/RL 파이프라인 검증

Level 2: + Backbone
  Input: Camera images
  Network: ResNet-18 → MLP → control
  → 이미지 입력 검증

Level 3: + BEV + Occupancy
  Network: RegNet → BiFPN → BEV → Planner
  → 공간 이해 추가

Level 4: + Temporal
  Network: Full E2E (RegNet→BiFPN→BEV→Temporal→Planner)
  → 시간 정보 추가 (최종 모델)
```

---

## Phase 4: Imitation Learning (IL)

### Objectives
nuPlan Expert 데이터를 사용하여 E2E 모델 학습

### Task Breakdown

| ID | Task | Priority | Status |
|----|------|----------|--------|
| P4-01 | Behavioral Cloning 학습기 | High | ⏳ |
| P4-02 | BC Loss 설계 (MSE + auxiliary) | High | ⏳ |
| P4-03 | DataLoader 연동 (PlanningDataset) | High | ⏳ |
| P4-04 | 학습 스크립트 (train_il.py) | High | ⏳ |
| P4-05 | 검증 스크립트 (validate_il.py) | Medium | ⏳ |
| P4-06 | GAIL 구현 | Medium | ⏳ |
| P4-07 | DAgger 구현 | Low | ⏳ |
| P4-08 | 성능 평가 및 비교 | Medium | ⏳ |

### Training Configuration
```yaml
# Behavioral Cloning
optimizer: AdamW
lr: 3e-4
lr_scheduler: CosineAnnealing
batch_size: 256
epochs: 100
loss:
  steering_weight: 0.4
  acceleration_weight: 0.3
  trajectory_weight: 0.2
  auxiliary_weight: 0.1  # occupancy, lane prediction
early_stopping: patience=10
```

### IL Loss Design
```python
L_total = (
    w_steer * MSE(pred_steer, expert_steer) +
    w_accel * MSE(pred_accel, expert_accel) +
    w_traj  * MSE(pred_trajectory, expert_trajectory) +
    w_aux   * (BCE(pred_occupancy, gt_occupancy) +
               CE(pred_lanes, gt_lanes))
)
```

### Success Criteria (IL Phase)
| Metric | Target |
|--------|--------|
| Steering MSE | < 0.01 rad |
| Acceleration MSE | < 0.5 m/s^2 |
| Trajectory ADE | < 2.0m (6sec) |
| Collision Rate (sim) | < 15% |

---

## Phase 5: Reinforcement Learning (RL) ⭐

### Objectives
Unity ML-Agents 환경에서 RL로 Planner 최적화 → 경로 추종 → 카메라 입력 → E2E 통합

### Stage Progress

| Stage | Name | Status | Result |
|-------|------|--------|--------|
| 1 | BC Baseline | ✅ | Expert 데이터 학습 완료 |
| 2 | PPO Single Area | ✅ | 950K steps, Reward ~700 |
| 3 | 16x Parallel PPO | ✅ | 1.66M steps, Reward ~750 수렴 |
| 4 | 속도 정책 | 🔄 | v8: 3.07M/8M, NPC+속도, Best +2.46 |
| 5 | Multi-Lane + 차선 | ⏳ | 5종 차선 마킹 + 정책 |
| 6 | 도로 네트워크 + Navigation | ⏳ | 교차로 + 경로 추종 ★ |
| 7 | Camera Visual Obs | ⏳ | CameraSensor + CNN encoder |
| 8 | Euro NCAP 평가 + 신호등 | ⏳ | ELK/LKA + 신호 인식 |
| 9 | GAIL + Hybrid | ⏳ | IL→RL 결합 |
| 10 | Full E2E + Ablation | ⏳ | BEV + Temporal 통합 |

### Training Progress (PPO v1) - 87K Steps (구버전, 물리 버그 있음)
```
Step 5K:  Reward -6.061 (initial random policy)
Step 87K: Reward -4.932 (차량 움직이지 않음 - 물리 마찰 버그)
원인: PhysX 마찰력이 구동력보다 커서 차량 정지 상태
```

### Training Progress (Curriculum PPO v5) - 950K+ Steps ✅ 성공!
```
=== Lesson 0: No Traffic, 50m Goal ===
Step   5K: Reward -18.3 (random exploration, 차량 움직임 확인!)
Step  45K: Reward  -9.7 (빠른 개선)
Step  90K: Reward +22.0 → [Curriculum Advanced!]

=== Lesson 1: 2 NPCs, 120m Goal ===
Step 110K: Reward +35.0 (교통 회피 학습)
Step 170K: Reward +137  → [Curriculum Advanced!]

=== Lesson 2: 4 NPCs, 230m Goal ===
Step 250K: Reward +223 (Near Mastery)
Step 395K: Reward +445 (Peak) → [Curriculum Advanced!]

=== Lesson 3: 6 NPCs, 230m Goal (Full Traffic) ===
Step 465K: Reward +454 (6 NPCs 적응 시작)
Step 545K: Reward +591 (안정적 Std=29)
Step 675K: Reward +656 (Std=27, 극도로 안정적)
Step 695K: Reward +689 (Std=14, 최고 안정성)
Step 840K: Reward +696 (고수준 유지)
Step 925K: Reward +697 (Plateau 도달)
Step 950K: Reward +618 (학습 진행 중...)

500K Checkpoint: E2EDrivingAgent-499836.onnx
289K Checkpoint: models/planning/E2EDrivingAgent_curriculum_v5_289k.onnx
```

### Training Progress (16x Parallel PPO v6) - 1.66M Steps ✅ 수렴!
```
Config: vehicle_ppo_curriculum_parallel.yaml
  batch=4096, buffer=40960, threaded=false, device=cuda
  16 Training Areas, 92K steps/min (4.6x 가속)

=== Lesson 0→3 전환: 2.5분 만에 Full Traffic 진입 ===
Step   10K: Reward  -19.2 (random)
Step  160K: Reward   +1.4 → [Lesson1: 2 NPCs, 120m]
Step  230K: Reward  +34.9 → [Lesson2: 4 NPCs, 230m]
Step  310K: Reward  +77.2 → [Lesson3: 6 NPCs, 230m]

=== Lesson 3 수렴 과정 ===
Step  500K: Reward +147   (Checkpoint saved)
Step 1000K: Reward +463   (Checkpoint saved)
Step 1300K: Reward +697   (안정화)
Step 1500K: Reward +706   (Checkpoint saved)
Step 1520K: Reward +748.6 ★ Peak (Std=10.7, 극도로 안정)
Step 1660K: Reward +716   (학습 중단 - 수렴 확인)

Checkpoints: results/curriculum_v6_parallel/E2EDrivingAgent/
  - E2EDrivingAgent-499849.onnx
  - E2EDrivingAgent-999809.onnx
  - E2EDrivingAgent-1499993.onnx

핵심 수정사항:
- 물리엔진: ForceMode → 내부 속도 추적 + rb.linearVelocity 직접 설정
- 중력/마찰 제거: useGravity=false, PhysicsMaterial(friction=0)
- FreezePositionY: 차량 Y축 고정
- SimpleVehicleController 비활성화
```

### Experiment Pipeline (Revised 2026-01-24)
```
Phase 5A: Vector-based RL (정책 검증)
  1. ✅ Behavioral Cloning (BC)        → python/src/training/train_il.py
  2. ✅ Pure RL (PPO Single)           → curriculum_v5: 950K steps, Reward ~700
  3. ✅ 16x Parallel PPO               → curriculum_v6_parallel: 1.66M, Reward ~750 수렴
  4. 🔄 속도 정책 (v8)                 → 3.07M/8M, NPC+속도, Best +2.46
  5. ⏳ Multi-Lane + 차선 정책          → 5종 마킹, Raycast 감지, 위반 패널티
  6. ⏳ 도로 네트워크 + Navigation ★    → 교차로(T/십자) + 경로 추종 + A* planner

Phase 5B: Vision-based RL (카메라 추가)
  7. ⏳ Camera Visual Observation       → CameraSensor + nature_cnn/resnet encoder
  8. ⏳ Euro NCAP 평가 + 신호등 인식    → ELK/LKA + 카메라 신호체계 학습

Phase 5C: Hybrid & Advanced (E2E 통합)
  9. ⏳ Expert 녹화 → GAIL/Hybrid      → 카메라 기반 시연, vehicle_gail.yaml
  10. ⏳ Full E2E + Ablation            → BEV + Temporal + Planning 통합

원칙: 벡터 기반 정책 검증 → 카메라 → E2E (비전/정책 문제 분리)
Stage 4 이후: 32 Areas 확장 적용 (Vector-only), 센서 추가 시 8-16 Areas
```

### 병렬 Training Areas 계획 (GPU 활용 극대화)
```
=== 현재 (Stage 4): 16 Training Areas ===
  - GPU: RTX 4090, ~11% 활용, 3.6GB VRAM
  - 속도: ~92K steps/min
  - 학습 시간: 1M steps 약 10분 (NPC+속도 포함)

=== 다음 (Stage 4 완료 후): 32 Training Areas ===
  - 적용 조건: Vector-only observation (242D)
  - 예상 VRAM: ~8GB / 24GB
  - 예상 속도: ~160K steps/min (1.7x 추가 가속)
  - batch_size: 8192, buffer_size: 81920
  - 레이아웃: 8×4 그리드, 100m 간격

=== Stage 7 (Camera 추가): 8-16 Training Areas ===
  - Camera 84x84 RGB → 렌더링 부하 증가
  - 예상 VRAM: 8-12GB (16 areas) / 12-16GB (8 areas + LiDAR)
  - batch_size: 2048-4096 (메모리 제약)
  - 효과: 카메라 렌더링이 병목 → GPU 활용률 자연 증가

=== 하드웨어 제약 분석 (RTX 4090) ===
  ┌──────────────────┬───────────┬────────────┬───────────────┐
  │ Config           │ Areas     │ VRAM       │ 학습 속도     │
  ├──────────────────┼───────────┼────────────┼───────────────┤
  │ Vector-only 현재 │ 16        │ ~3.6 GB    │ 92K steps/min │
  │ Vector-only 확장 │ 32        │ ~8 GB      │ ~160K/min     │
  │ Camera (84x84)   │ 16        │ ~10 GB     │ ~60K/min      │
  │ Camera (84x84)   │ 8         │ ~6 GB      │ ~35K/min      │
  │ Camera + LiDAR   │ 8         │ ~14 GB     │ ~25K/min      │
  └──────────────────┴───────────┴────────────┴───────────────┘
```

### Euro NCAP LSS 차선 정책 계획
```
=== 차선 마킹 유형 및 정책 ===
┌────────────────┬──────────────────────┬──────────────┐
│ 마킹 유형       │ 의미                  │ Reward 설계  │
├────────────────┼──────────────────────┼──────────────┤
│ 백색 점선       │ 차선 변경 가능         │ 허용 (0)     │
│ 백색 실선       │ 차선 변경 금지         │ -2.0         │
│ 황색 점선       │ 중앙선 (추월 가능)     │ -3.0         │
│ 황색 실선       │ 중앙선 (추월 금지)     │ -5.0         │
│ 이중 황색 실선   │ 절대 넘지 않음         │ -10.0 (치명) │
└────────────────┴──────────────────────┴──────────────┘

=== Euro NCAP ELK 테스트 시나리오 (구현 대상) ===
1. ELK Solid Line Left/Right  - 실선 이탈 시 자동 교정
2. ELK Road Edge              - 도로 끝 이탈 시 교정
3. ELK Oncoming Vehicle       - 중앙선 침범 시 역주행 충돌 방지
4. ELK Overtaking (Unintentional) - 비의도적 차선 변경 충돌 방지
5. ELK Overtaking (Intentional)   - 의도적 추월 시 충돌 방지

=== LKA 테스트 시나리오 ===
1. LKA Dashed Line  - 점선 이탈 시 경고+교정
2. LKA Solid Line   - 실선 이탈 시 교정 (DTLE ≤ -0.3m 이내)

=== Observation 확장 (차선 인식용) ===
  lane_info: 12D (추가)
    - left_lane_dist: 1D      # 좌측 차선까지 거리
    - right_lane_dist: 1D     # 우측 차선까지 거리
    - left_lane_type: 4D      # [점선, 백실선, 황점선, 황실선] one-hot
    - right_lane_type: 4D     # one-hot
    - center_offset: 1D       # 차선 중앙까지 offset
    - heading_error: 1D       # 차선 방향 대비 heading 오차

=== 속도 구간 정책 (한국 도로교통법 기반) ===
┌────────────────┬──────────────────────┬──────────────┐
│ 도로 구간       │ 제한속도              │ m/s          │
├────────────────┼──────────────────────┼──────────────┤
│ 주거/스쿨존     │ 30 km/h              │ 8.3          │
│ 시가지 이면도로  │ 50 km/h              │ 13.9         │
│ 일반도로 (도시) │ 60 km/h              │ 16.7         │
│ 자동차전용도로   │ 80 km/h              │ 22.2         │
│ 고속도로        │ 100-110 km/h         │ 27.8-30.6    │
└────────────────┴──────────────────────┴──────────────┘

  속도 위반 패널티 구조:
    10km/h 초과: -0.5 (범칙금 3만원급)
    20km/h 초과: -1.0 (범칙금 6만원급)
    40km/h 초과: -2.0 (범칙금 9만원 + 벌점)
    60km/h 초과: -3.0 (면허정지급)

  적정 속도 보상:
    0.8*limit ≤ speed ≤ limit: +0.3
    구간 전환 smooth 감속: +0.2
    저속 교통방해: -0.1

=== Observation 확장 (속도 인식용) ===
  speed_info: 4D (추가)
    - current_speed_norm: 1D     # 현재속도/max_speed
    - speed_limit_norm: 1D       # 구간제한속도/max_speed
    - speed_ratio: 1D            # 현재속도/제한속도 (1.0 적정)
    - next_speed_limit_norm: 1D  # 다음 구간 제한속도

=== 구현 순서 (속도→차선→도로네트워크→카메라) ===
Stage 4: 속도 정책
  1. 속도 구간 시스템 (WaypointManager 태그)
  2. Observation +4D speed_info 추가
  3. 속도 위반 점진적 패널티 Reward
  4. 16 Areas 학습 (Curriculum: 단일속도 → 다구간)

Stage 5: 차선 정책
  5. 다차선 도로 환경 구축 (2차선 + 중앙선)
  6. 차선 마킹 오브젝트 (Layer/Tag 구분)
  7. Raycast 차선 감지 + Observation +12D lane_info
  8. 차선 위반 Reward + 통합 학습

Stage 6: 도로 네트워크 + 경로 추종 ★
  9. 도로 그래프 시스템 (IntersectionNode + RoadEdge)
  10. 교차로 프리팹 (T자/십자)
  11. Route Planner (A* 경로 탐색)
  12. Navigation Command + Observation +10D
  13. 경로 추종 Reward (correct_turn, wrong_turn)
  14. Curriculum (직선→T자→십자→복합)

Stage 7: 카메라 입력
  15. CameraSensorComponent 추가 (84x84 front)
  16. ML-Agents visual encoder (nature_cnn)
  17. Vector+Visual 복합 학습
  18. Camera-dominant 학습 (vector 축소)

Stage 8-10: Euro NCAP + GAIL + E2E
  19. ELK/LKA 벤치마크
  20. Expert 시연 녹화 (카메라 포함)
  21. GAIL → Hybrid BC→RL
  22. Full E2E (BEV + Temporal)
```

### 학습 실행 명령어 정리
```bash
# 1. Curriculum PPO (추천 - 가장 빠른 수렴)
mlagents-learn python/configs/planning/vehicle_ppo_curriculum.yaml --run-id=driving_ppo_curriculum_v1 --force

# 2. 기본 PPO (이어서 학습)
mlagents-learn python/configs/planning/vehicle_ppo.yaml --run-id=driving_ppo_v2 --force

# 3. SAC (샘플 효율적)
mlagents-learn python/configs/planning/vehicle_sac.yaml --run-id=driving_sac_v1 --force

# 4. GAIL (시연 녹화 후)
mlagents-learn python/configs/planning/vehicle_gail.yaml --run-id=driving_gail_v1 --force

# 5. Hybrid BC→RL (시연 녹화 후)
mlagents-learn python/configs/planning/vehicle_hybrid.yaml --run-id=driving_hybrid_v1 --force
```

### Expert 시연 녹화 절차
```
1. Vehicle에 ExpertDriverController 컴포넌트 추가
2. BehaviorParameters → Behavior Type = "Heuristic Only"
3. DemonstrationRecorder 자동 추가됨 (autoRecord=true)
4. Play 모드 → 자동으로 waypoint 추종하며 시연 녹화
5. 50 에피소드 후 자동 정지
6. 결과: Assets/Demonstrations/expert_driving.demo
7. 녹화 후 BehaviorParameters → "Default" (학습용)로 복원
```

### Reward Function Design (v8 - 현재 적용 중)
```python
def compute_reward(state, action, next_state):
    # Progress reward (목표 방향 이동, 속도제한 기준 정규화)
    r_progress = +1.0 * progress_along_route

    # Safety reward (충돌/근접 회피)
    r_safety = -5.0 * collision          # v8: -10→-5 (gradient 안정화)
    r_safety += -1.5 * deltaTime * (ttc < 2.0)  # v8: rate-independent
    # off_road: -5.0 + EndEpisode (누적 방지, 즉시 종료)

    # Comfort reward (급가속/급조향 패널티)
    r_comfort = -0.1 * abs(jerk)
    r_comfort += -0.05 * abs(steering_rate)

    # Speed compliance (속도 정책)
    r_speed = +0.3 if 0.8*speed_limit <= speed <= speed_limit
    r_speed += -0.5 ~ -3.0 * speed_over_ratio  # 점진적 초과 패널티

    # Goal reward
    r_goal = +10.0 * reached_destination

    return r_progress + r_safety + r_comfort + r_speed + r_goal
    # collision/off_road 시 EndEpisode (에피소드 종료)
```

### v7→v8 Reward 변경 교훈
```
문제 1: collision=-10 → PPO gradient explosion (Std 40+)
해결: collision=-5로 감소, 충분한 회피 학습 유도

문제 2: nearCollision=-0.5/frame → 100프레임(2초)에 -150 누적
해결: ×Time.fixedDeltaTime 적용 (-1.5/초, rate-independent)

문제 3: offRoad=-5/sec → 40초 off-road 시 -200 누적
해결: EndEpisode() 즉시 종료 (1회 -5 패널티)

문제 4: NPC 0→2 + goal 50→120m 동시 진행
해결: NPC/goal 임계값 분리, NPC 0→1→2→4 점진적 도입
```

### RL Training Configuration
```yaml
# PPO Config
algorithm: PPO
policy:
  hidden_layers: [512, 512, 256]
  activation: tanh
hyperparameters:
  batch_size: 2048
  buffer_size: 20480
  learning_rate: 3e-4
  num_epoch: 3
  epsilon: 0.2  # clip ratio
  gamma: 0.99
  lambda: 0.95
  beta: 0.005  # entropy bonus
max_steps: 10_000_000
curriculum:
  - lesson_0: straight road only
  - lesson_1: + gentle curves
  - lesson_2: + intersections
  - lesson_3: + traffic vehicles
  - lesson_4: + pedestrians + complex scenarios
```

### Unity Environment Requirements
```csharp
// DrivingAgent observation space for E2E
ObservationSpec:
  - CameraImages: 8 x [3, 84, 84] (downscaled for RL)
    OR
  - VectorObservation: 238D (ego+agents+route)

ActionSpec:
  - ContinuousActions: 2 (steering, acceleration)

Rewards:
  - Per-step reward (composite)
  - Episode termination on collision/goal
```

### Success Criteria (RL Phase)
| Metric | Target |
|--------|--------|
| Collision Rate | < 5% |
| Route Completion | > 85% |
| Comfort (jerk < 2m/s^3) | > 80% episodes |
| Average Reward | > 500/episode |

---

## Phase 6: Hybrid Training & Deployment

### Objectives
IL + RL 결합 (CIMRL) 및 Unity/ROS2 실시간 배포

### Task Breakdown

| ID | Task | Priority | Status |
|----|------|----------|--------|
| P6-01 | CIMRL 구현 (IL 초기화 → RL fine-tuning) | High | ⏳ |
| P6-02 | ONNX Export 파이프라인 | High | ⏳ |
| P6-03 | Unity Sentis 추론 통합 | High | ⏳ |
| P6-04 | ROS2 Control Publisher | Medium | ⏳ |
| P6-05 | 실시간 성능 최적화 (<50ms) | High | ⏳ |
| P6-06 | 벤치마크 평가 (nuPlan scoring) | High | ⏳ |
| P6-07 | Ablation Study | Medium | ⏳ |
| P6-08 | 최종 모델 선정 및 문서화 | Medium | ⏳ |

### Hybrid Training Strategy (CIMRL)
```
Step 1: Pre-train with IL (nuPlan expert data)
  → model_il.pt (good initialization, safe driving)

Step 2: Fine-tune with RL (Unity sim)
  → model_rl.pt (optimized for reward, exceeds expert)

Step 3: Evaluate & Select
  → Compare IL-only, RL-only, Hybrid
  → Select best model by composite score

Step 4: Export & Deploy
  → torch.onnx.export(best_model) → model.onnx
  → Unity Sentis로 로드 → 실시간 추론
  → ROS2 /vehicle/control 토픽으로 publish
```

### Deployment Pipeline
```
PyTorch (.pt)
  → torch.onnx.export()
  → model.onnx (ONNX format)
  → Unity Sentis ModelLoader.Load()
  → Worker.Schedule(input_tensor)
  → output: [steering, acceleration]
  → vehicle.ApplyControl()
  → ROS2 publish (optional)
```

### Final Success Criteria

| Category | Metric | Target | Weight |
|----------|--------|--------|--------|
| Safety | Collision Rate | < 5% | 30% |
| Comfort | Jerk | < 2 m/s^3 | 20% |
| Progress | Route Completion | > 85% | 25% |
| Latency | Inference time | < 50ms | 15% |
| Generalization | Unseen scenarios | > 70% success | 10% |

---

## Phase 7: Advanced Topics

### Research Areas
- World Model for Driving (GAIA-1 style)
- Vision-Language-Action (VLA) Integration
- Sim-to-Real Transfer Techniques
- Multi-agent Cooperative Driving
- Adversarial Robustness Testing

---

## Recent Activity Log

| Date | Activity | Status |
|------|----------|--------|
| 2026-01-24 | **Stage 4 v8 학습 진행 중** (3.07M/8M, NPC+속도정책, Best +2.46) | 🔄 |
| 2026-01-24 | **Reward 재조정** (collision=-5, nearCollision rate-independent, off-road termination) | ✅ |
| 2026-01-24 | **Curriculum 재설계** (NPC 0→1→2→4 점진적, threshold 분리) | ✅ |
| 2026-01-24 | **32 Areas 확장 계획** (Stage 4 완료 후 적용) | ⏳ |
| 2026-01-24 | **카메라/LiDAR 병렬 제한 검토** (8-16 Areas) | ✅ |
| 2026-01-24 | **신호등 인식 학습 계획 확인** (Phase 6 범위) | ✅ |
| 2026-01-23 | **Tesla-style E2E 아키텍처 결정** | ✅ |
| 2026-01-23 | **개발 계획 전면 업데이트 (E2E 반영)** | ✅ |
| 2026-01-23 | **Tesla FSD 기술 조사** (docs/knowledge/tesla-fsd-technology.md) | ✅ |
| 2026-01-23 | **AD 기업/연구소 조사** (docs/knowledge/AD-research-landscape.md) | ✅ |
| 2026-01-23 | **Phase 2 구현**: nuPlan loader, processor, augmentation, visualizer, splitter | ✅ |
| 2026-01-23 | **Phase 1 완료** | ✅ |
| 2026-01-23 | **실험 추적 설정** (MLflow/W&B) | ✅ |
| 2026-01-23 | **LiDARSensor 구현** (16x360 레이캐스팅) | ✅ |
| 2026-01-23 | **CameraSensor 구현** (640x480 RGB, ROS Publish) | ✅ |
| 2026-01-23 | **VehicleROSBridge 구현** (Pub/Sub) | ✅ |
| 2026-01-23 | **키보드 주행 테스트 성공** | ✅ |
| 2026-01-23 | **VehicleAgent + SimpleVehicleController 생성** | ✅ |
| 2026-01-22 | **ROS2 ↔ Unity 연결 테스트 성공** | ✅ |
| 2026-01-22 | **기본 주행 Scene (DrivingScene) 생성** | ✅ |
| 2026-01-22 | **Unity ROS-TCP-Connector 설치** | ✅ |
| 2026-01-22 | **WSL2 + ROS2 Humble + ROS-TCP-Endpoint 설치** | ✅ |
| 2026-01-22 | **Phase 1-7 Obsidian 지식화 섹션 추가** | ✅ |
| 2026-01-22 | **ML-Agents 4.0.1 + Unity 6 환경 구축** | ✅ |
| 2026-01-22 | **Sentis 2.4.1 설치 (Barracuda 대체)** | ✅ |
| 2026-01-22 | **3DBall PPO 학습 성공 (5분, Reward 100)** | ✅ |
| 2026-01-22 | **ONNX 모델 Export 완료** | ✅ |
| 2026-01-22 | **문서 업데이트 (TECH-SPEC, PRD, CLAUDE.md)** | ✅ |
| 2026-01-22 | AD 플랫폼으로 프로젝트 재구성 | ✅ |
| 2026-01-22 | 7-Phase 시스템 설계 | ✅ |
| 2026-01-22 | PRD, TECH-SPEC 문서 작성 | ✅ |
| 2026-01-21 | 프로젝트 초기화, 구조 설계 | ✅ |
| 2026-01-21 | cc-initializer 연동 | ✅ |

---

## Notes & Decisions

### Current Environment (2026-01-27)
| Component | Version | Notes |
|-----------|---------|-------|
| OS | Windows 11 | Native (WSL 미사용) |
| Unity | 6000.3.4f1 (Unity 6) | LTS |
| ML-Agents | 4.0.1 | Unity Package |
| Sentis | 2.4.1 | ONNX Inference |
| Python | 3.10.11 | Windows Native |
| PyTorch | 2.3.1 | CUDA 12.x |
| mlagents | 1.2.0 | Python Package |

> **Note**: 초기에 ROS2 (WSL2)를 설정했으나, ML-Agents 직접 통신이 더 효율적이어서 현재는 ML-Agents 기반으로 학습 진행 중

### Key Decisions
1. **ROS2 Bridge**: ✅ Unity Robotics Hub (ROS-TCP-Connector) 선택 *(초기 설정, 현재 미사용)*
2. **ROS2 환경**: ✅ WSL2 Ubuntu 22.04 + ROS2 Humble *(초기 설정, 현재 ML-Agents 사용)*
3. **Architecture**: ✅ **Tesla-style E2E** (modular 대신 unified neural net)
4. **Sensor**: ✅ **Camera-only** (Vision-only, no LiDAR for planning)
5. **Learning**: IL (nuPlan) → RL (Unity) → Hybrid (CIMRL)
6. **Inference**: PyTorch → ONNX → Unity Sentis
7. **Action Space**: steering [-0.5, +0.5] rad, acceleration [-4.0, +2.0] m/s^2
8. **No separate Control layer**: Planning Network이 직접 제어 출력

### Architecture Decision Record (ADR)
```
ADR-001: Tesla-style E2E vs Modular Pipeline
  Decision: E2E (End-to-End)
  Date: 2026-01-23
  Reason:
    - Unified gradient flow (joint optimization)
    - Error cascade 제거
    - 코드 복잡도 감소
    - Tesla FSD v12+ 검증된 접근법
  Trade-off:
    - 디버깅 어려움 (intermediate output 없으면)
    - 해결: Auxiliary losses (occupancy, lane prediction)

ADR-002: Simplified Start (Level 1 → Level 4)
  Decision: MLP Planner부터 시작, 점진적 확장
  Reason:
    - 전체 모델 한번에 구현 시 디버깅 불가능
    - 각 Level에서 파이프라인 검증
    - Level 1 (MLP) → Level 2 (+ ResNet) → Level 3 (+ BEV) → Level 4 (Full E2E)

ADR-003: Modular Encoder Architecture for Incremental Learning
  Decision: Implement modular encoder with freeze/unfreeze capability
  Date: 2026-01-25
  Problem:
    - Observation space changes (242D → 254D for lane info) cause full training restart
    - ML-Agents initialize_from requires matching dimensions
    - Phase B training (+903 reward) was lost when adding lane observation
  Solution:
    - Named encoder modules (ego, history, agents, route, speed, lane)
    - Freeze/unfreeze capability per module
    - Dynamic encoder addition with fusion weight transfer
    - Two-phase training: new encoder (500K) → fine-tune all (1.5M)
  Research References:
    - Progressive Neural Networks (freeze columns approach)
    - UPGD (ICLR 2024) - Utility-based Perturbed Gradient Descent
    - EWC (Elastic Weight Consolidation)
  Expected Benefit:
    - Phase B knowledge preserved (~+700 reward at start of C-1)
    - Future observation additions: add encoder + partial train (vs full restart)
  Implementation:
    - python/src/models/modular_encoder.py
    - python/src/models/modular_policy.py
    - python/src/training/train_modular_rl.py
    - python/configs/planning/modular_ppo_phaseC1.yaml
```

### Next Actions
1. **Phase 3 시작**: E2E 모델 아키텍처 구현
2. Level 1 (MLP Planner) 먼저 구현 → IL 파이프라인 검증
3. Level 2-4 점진적 확장
