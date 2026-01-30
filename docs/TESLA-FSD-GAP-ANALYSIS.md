# Tesla FSD vs Physical Unity: Gap Analysis & Implementation Roadmap

**Document Version**: 1.0
**Created**: 2026-01-30
**Status**: Active Analysis
**Purpose**: Tesla FSD 12/13의 아키텍처 분석 및 본 프로젝트의 현실적 구현 로드맵 제시

---

## 1. Executive Summary

### 1.1 본 프로젝트와 Tesla FSD의 근본적 차이 3줄 요약

1. **데이터 규모**: Tesla는 400만대 이상 플릿의 Shadow Mode 데이터를 활용, 본 프로젝트는 Unity 시뮬레이션 + 공개 데이터셋(nuPlan 1282시간) 활용
2. **컴퓨팅 자원**: Tesla는 Dojo Supercomputer(1.1 ExaFLOPS, D1 칩 수천개), 본 프로젝트는 RTX 4090 1대(82.6 TFLOPS FP32)
3. **아키텍처 복잡도**: Tesla는 Occupancy Network 기반 4D 공간 추론 → MCTS Planner, 본 프로젝트는 Ground Truth Vector → RL Policy

### 1.2 현실적 목표 설정

**RTX 4090 1대로 할 수 있는 것의 한계**:

| 측면 | Tesla FSD | 본 프로젝트 현실 |
|------|-----------|----------------|
| 입력 | 8x 1280x960 Camera (Multi-view) | 242D Ground Truth Vector 또는 단일 84x84 Camera |
| Perception | Occupancy Network (3D Voxel Grid) | Pre-trained 모델 fine-tune 또는 GT 직접 사용 |
| Planning | MCTS + Neural Evaluator (20 candidates) | PPO/SAC Direct Control (2D output) |
| 학습 데이터 | 400만대 Fleet + Auto-Label | nuPlan 10k scenarios + Self-play |
| 학습 시간 | Dojo에서 수일 (PB 규모 데이터) | RTX 4090에서 수주-수개월 (GB-TB 규모) |
| 목표 | Level 2-3 상용화 (No geo-fence) | 시뮬레이션 환경 내 알고리즘 검증 |

**핵심 차이점**: Tesla FSD는 **상용 제품**, 본 프로젝트는 **연구/검증 플랫폼**이다. 동일 수준 달성은 불가능하며 불필요하다.

**합리적 목표**: 학술 연구 수준의 E2E Pipeline 구축 + 주요 알고리즘 검증 + Phase별 점진적 구현

---

## 2. Tesla FSD 아키텍처 전체 구조

### 2.1 Vision Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        TESLA FSD 12/13 VISION PIPELINE                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 1: Multi-Camera Input                            │    │
│  │  8 Cameras: Front(3x), Left(2x), Right(2x), Rear(1x)               │    │
│  │  Resolution: 1280x960 @ 36 Hz                                      │    │
│  │  Total: ~88M pixels/frame                                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 2: RegNet Backbone (Multi-scale)                 │    │
│  │  Input: 8x [1280, 960, 3]                                          │    │
│  │  Backbone: RegNetY-120GF (120B FLOPS)                              │    │
│  │  Output: Multi-scale features [C1, C2, C3, C4, C5]                 │    │
│  │    C5: 1/32 resolution, 2048 channels (semantic features)          │    │
│  │    C3: 1/8 resolution, 512 channels (detail features)              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 3: BiFPN (Bi-directional FPN)                    │    │
│  │  Purpose: 다중 스케일 정보 융합                                      │    │
│  │  Method: Top-down + Bottom-up bidirectional feature flow           │    │
│  │  Output: Fused multi-scale features                                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 4: Transformer Cross-Attention Fusion             │    │
│  │  Input: 8 camera features (각각 다른 viewpoint)                     │    │
│  │  Method: Query-based attention (DETR-style)                        │    │
│  │    Queries: BEV grid positions (200x200x8 = 320k queries)          │    │
│  │    Keys/Values: Camera features (with camera intrinsics)           │    │
│  │  Output: Fused BEV feature map [200, 200, 256]                     │    │
│  │    - 200x200 grid = 100m x 100m @ 0.5m resolution                  │    │
│  │    - 256 channels = semantic + geometric features                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 5: Occupancy Network (3D Voxel Grid)             │    │
│  │  Input: BEV features [200, 200, 256]                               │    │
│  │  3D Decoder: Sparse 3D CNN or NeRF-style MLP                       │    │
│  │  Output: Occupancy Grid [200, 200, 16, C]                          │    │
│  │    - 16 height bins (0-8m, 0.5m steps)                             │    │
│  │    - C classes: road, vehicle, pedestrian, static, ...             │    │
│  │  Supervision: Auto-labeled 3D boxes + LiDAR pseudo-GT              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 6: Occupancy Flow (Motion Prediction)            │    │
│  │  Input: Occupancy Grid @ t-1, t, t+1 (3 frames)                    │    │
│  │  Method: 3D ConvGRU or Transformer temporal encoder                │    │
│  │  Output: Future Occupancy @ t+2, ..., t+10 (2s ahead)              │    │
│  │    - Per-voxel flow vectors [dx, dy, dz]                           │    │
│  │    - Uncertainty estimates (entropy map)                           │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**핵심 기술**:
- **Multi-view Fusion**: 8개 카메라의 정보를 Transformer로 BEV 공간에 통합
- **Occupancy Network**: 전통적인 Object Detection 대신 밀집 공간 표현 사용
- **Temporal Modeling**: 과거-현재-미래 프레임의 시간적 일관성 학습

**모델 크기 추정**:
- RegNet Backbone: ~250M parameters
- Transformer Fusion: ~100M parameters
- Occupancy Decoder: ~50M parameters
- **Total**: ~400M parameters, ~120GB VRAM (FP32), ~30GB (FP16 Mixed)

### 2.2 Planning Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        TESLA FSD PLANNING PIPELINE                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Input: Occupancy + Lane + Route                       │    │
│  │  - Occupancy Grid: [200, 200, 16] (future 2s)                      │    │
│  │  - Lane Graph: [N_lanes x waypoints x 3] (x, y, type)              │    │
│  │  - Route: [M_waypoints x 3] (GPS → local map)                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 1: Monte Carlo Tree Search (Discrete)            │    │
│  │  Search Space: Discrete maneuvers                                  │    │
│  │    - Lane keep                                                     │    │
│  │    - Lane change left/right                                        │    │
│  │    - Speed up/down/maintain                                        │    │
│  │    - Merge/exit                                                    │    │
│  │  Method: MCTS with UCB1 selection                                  │    │
│  │    Simulation depth: 5-10 steps (2-4 seconds)                      │    │
│  │    Rollouts: 50-100 per decision cycle                             │    │
│  │  Output: ~20 Candidate Maneuver Sequences                          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 2: Neural Network Evaluator (Continuous)         │    │
│  │  Input: 20 Candidate Sequences → Continuous Trajectory             │    │
│  │  Method: Trajectory Optimizer Network                              │    │
│  │    - Input: Maneuver + Occupancy + Lane                            │    │
│  │    - Output: [N_steps x 5] (x, y, v, heading, accel)               │    │
│  │  Optimization: Gradient descent on smoothness + feasibility        │    │
│  │    - Kinematic constraints (curvature, accel limits)               │    │
│  │    - Comfort (jerk, lateral accel)                                 │    │
│  │    - Collision checking (Occupancy grid query)                     │    │
│  │  Output: 20 Optimized Trajectories                                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 3: Cost Function Scoring                         │    │
│  │  Components:                                                       │    │
│  │    w_collision * C_collision(traj, occupancy)                      │    │
│  │      - SDF distance to occupied voxels                             │    │
│  │      - Penalty: exp(-dist/sigma)                                   │    │
│  │    w_comfort * C_comfort(jerk, lat_accel)                          │    │
│  │      - Jerk^2 + Lateral_accel^2 integrated                         │    │
│  │    w_intervention * C_intervention(traj)                           │    │
│  │      - Learned from driver takeover data                           │    │
│  │      - Neural network: traj → P(intervention)                      │    │
│  │    w_human * C_human_likeness(traj)                                │    │
│  │      - GAN Discriminator: real driver vs model                     │    │
│  │      - Pre-trained on Fleet data                                   │    │
│  │  Total Cost: Σ w_i * C_i(traj)                                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 4: Best Trajectory Selection                     │    │
│  │  Method: argmin(cost) over 20 candidates                           │    │
│  │  Fallback: If all costs > threshold, select "safe stop"            │    │
│  │  Output: Selected Trajectory [40 steps x 5] (2s @ 20Hz)            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 5: Direct Neural Network Control                 │    │
│  │  Input: Selected Trajectory + Current State                        │    │
│  │  Method: MPC-style tracking controller (Neural Network)            │    │
│  │    - Receding horizon: 0.5s                                        │    │
│  │    - Control output: steering_angle, accel, brake                  │    │
│  │  Update Rate: 40ms (25 Hz)                                         │    │
│  │  Output: [steering, accel, brake] → CAN Bus                        │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 6: Replanning (every 40ms)                       │    │
│  │  Condition: New Occupancy prediction available                     │    │
│  │  Action: Return to Step 1 with updated world state                 │    │
│  │  Continuity: Warm-start MCTS from previous search tree             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**Planning 계산 복잡도**:
- MCTS: 50 rollouts × 10 steps × 0.1ms = 50ms
- Neural Evaluator: 20 trajectories × 2ms = 40ms
- Cost Scoring: 20 trajectories × 1ms = 20ms
- **Total**: ~110ms per cycle (25Hz 달성 가능)

### 2.3 Data Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        TESLA FSD DATA PIPELINE                              │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 1: Shadow Mode Data Collection                   │    │
│  │  Fleet Size: 4M+ vehicles (2024 기준)                              │    │
│  │  Trigger: FSD vs Human Driver disagreement detection               │    │
│  │    - FSD wants: lane change left                                   │    │
│  │    - Human does: lane keep                                         │    │
│  │    → Record 10s clip (5s before + 5s after)                        │    │
│  │  Data Rate: ~1% of driving time (edge cases)                       │    │
│  │  Storage: ~1M clips/day → ~10 PB/year                              │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 2: Hard Clip Selection                           │    │
│  │  Method: Criticality Score = f(TTC, Intervention, Speed)           │    │
│  │  Filter:                                                           │    │
│  │    - Intervention occurred: High priority                          │    │
│  │    - TTC < 3s: Medium priority                                     │    │
│  │    - Novel scenario (low cluster density): High priority           │    │
│  │  Selection Rate: 1% of collected clips (top 10k/day)               │    │
│  │  Diversity: K-means clustering on scenario features                │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 3: Auto-Labeling (4D Reconstruction)             │    │
│  │  Input: 8 camera streams + IMU + GPS                               │    │
│  │  Method:                                                           │    │
│  │    1. SLAM: Camera pose estimation                                 │    │
│  │    2. Multi-view Stereo: Depth reconstruction                      │    │
│  │    3. 3D Object Detection: Pre-trained model                       │    │
│  │    4. 3D Tracking: Kalman Filter + Hungarian matching              │    │
│  │    5. Occupancy Grid: Voxel fusion from depth maps                 │    │
│  │  Output: Pseudo Ground Truth                                       │    │
│  │    - 3D boxes: [x, y, z, l, w, h, heading, class, track_id]        │    │
│  │    - Occupancy: [200, 200, 16] per frame                           │    │
│  │  Accuracy: ~95% (verified by human on sample)                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 4: Human Verification (Sample)                   │    │
│  │  Sample Rate: 10% of auto-labeled clips (1k/day)                   │    │
│  │  Task: Annotators verify/correct 3D boxes and occupancy            │    │
│  │  Tool: Custom 3D labeling interface (Blender-like)                 │    │
│  │  Metrics:                                                          │    │
│  │    - 3D IoU > 0.7: Accept                                          │    │
│  │    - 3D IoU < 0.5: Reject & manual label                           │    │
│  │  Feedback Loop: Auto-labeler fine-tuning quarterly                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 5: Training on Dojo                              │    │
│  │  Cluster: 1.1 ExaFLOPS (2024)                                      │    │
│  │    - D1 chips: Custom 7nm ASIC for ML training                     │    │
│  │    - Nodes: 3,000+ (each with 25 D1 chips)                         │    │
│  │  Training Config:                                                  │    │
│  │    - Batch size: 2048 (distributed across nodes)                   │    │
│  │    - Model size: 400M params (~1.6GB FP16)                         │    │
│  │    - Dataset: ~100M clips (10 PB)                                  │    │
│  │    - Duration: 3-5 days per major version                          │    │
│  │  Output: FSD Model vX.Y.Z (.onnx format for HW3)                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 6: Policy Distillation (Teacher→Student)         │    │
│  │  Teacher: Large model trained on Dojo (400M params)                │    │
│  │  Student: Compact model for HW3 inference (200M params)            │    │
│  │  Method: Knowledge Distillation                                    │    │
│  │    - Soft labels from teacher (logits)                             │    │
│  │    - Mimicry loss: KL(student || teacher)                          │    │
│  │  Performance: 95-98% of teacher accuracy                           │    │
│  │  Latency: 40ms/frame (HW3 chip: 144 TOPS INT8)                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                  │                                          │
│                                  ▼                                          │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │              Step 7: OTA Update to Fleet                           │    │
│  │  Distribution: 4M vehicles via cellular OTA                        │    │
│  │  Rollout Strategy: Gradual (0.1% → 1% → 10% → 100%)                │    │
│  │  Monitoring: Real-time intervention metrics                        │    │
│  │    - Intervention rate increase > 20% → rollback                   │    │
│  │    - Critical safety events → immediate disable                    │    │
│  │  Cycle Time: 2-4 weeks per release                                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

**데이터 파이프라인 규모**:
- 일일 수집: ~1M clips × 10s × 8 cameras × 1280×960 = ~100 TB/day
- 저장 비용: 연간 ~10 PB → AWS S3 기준 $250k/month
- 라벨링 비용: 수동 10% → $1M-5M/month (추정)

---

## 3. Gap Analysis: 컴포넌트별 비교

### 3.1 7개 핵심 컴포넌트 비교표

| 컴포넌트 | Tesla FSD | 본 프로젝트 현재 | Gap Level | 구현 난이도 | RTX 4090 가능성 |
|----------|-----------|--------------|-----------|-----------|----------------|
| **1. Vision Perception** | HydraNet + RegNet-120GF + Occupancy Network (400M params) | Ground Truth 벡터 (242D) | **CRITICAL** | Very High | 제한적 (경량 모델만) |
| **2. BEV Representation** | Transformer-based 8-camera fusion (200x200x256) | 없음 | **CRITICAL** | High | 가능 (단일 카메라) |
| **3. Trajectory Prediction** | Occupancy Flow (3D ConvGRU, 2s horizon) | Constant Velocity 가정 | **MAJOR** | Medium | 가능 (LSTM/GNN) |
| **4. Trajectory Planning** | MCTS + Neural Evaluator (20 candidates) | 없음 (reactive control only) | **CRITICAL** | High | 가능 (단순화) |
| **5. Route Planning** | GPS → local lane-level graph | 고정 Waypoint 시스템 | **MAJOR** | Medium | 가능 (A*) |
| **6. Vehicle Control** | Direct neural network (40ms replan) | RL Policy → steering/accel | **MODERATE** | Low | 이미 달성 |
| **7. Data Pipeline** | Shadow Mode + Auto-Label + Dojo | ML-Agents self-play | **MAJOR** | High | 불가 (규모 차이) |

### 3.2 각 Gap 상세 분석

#### Gap 1: Vision Perception (CRITICAL)

**Tesla FSD**:
- 8개 카메라로 360도 시야 확보
- RegNet-120GF Backbone으로 Multi-scale feature extraction
- Transformer Cross-Attention으로 BEV 공간에 fusion
- Occupancy Network로 밀집 3D 공간 표현 (200×200×16 voxels)
- 학습 데이터: 수백만 자동 라벨링된 클립

**본 프로젝트 현재**:
- Unity 시뮬레이터에서 Ground Truth 벡터 직접 제공
- 242D observation: ego state(8D) + route(30D) + NPCs(160D) + lane(12D) + goal(12D) + history(20D)
- 카메라 입력 없음 (ML-Agents CameraSensor 지원은 하지만 미사용)

**Gap의 본질**:
1. **데이터 규모**: Tesla는 수백만 실제 주행 이미지, 본 프로젝트는 시뮬레이션
2. **모델 크기**: 400M params는 RTX 4090 VRAM(24GB)에 FP16으로 겨우 적재 가능, 학습은 불가능 (Gradient + Optimizer state 포함 시 ~100GB 필요)
3. **추론 속도**: RegNet-120GF는 단일 forward pass에 ~200ms (RTX 4090 기준), FSD는 HW3 칩에 최적화되어 40ms

**기술적 차이**:
- Tesla의 Occupancy Network는 전통적 Object Detection의 한계(Occlusion, Long-tail objects) 극복
- BEV representation은 lane-level planning에 필수적 (차선 중심선 표현, lateral offset 정확도)
- 본 프로젝트의 GT Vector는 정확하지만 실제 센서의 불확실성 학습 불가

#### Gap 2: BEV Representation (CRITICAL)

**Tesla FSD**:
- Transformer Query-based fusion (DETR 스타일)
- 200×200 grid @ 0.5m resolution = 100m×100m 커버
- 256 channels: semantic (road, vehicle, pedestrian) + geometric (height, depth)

**본 프로젝트 현재**:
- BEV 표현 없음
- Surrounding vehicles는 local coordinate (ego-centric) 벡터로 표현
- Lane marking도 ego vehicle 기준 left/right distance만 제공

**Gap의 본질**:
- BEV는 **spatial reasoning**에 필수: 복잡한 도로 기하학(곡선, 교차로) 이해
- Transformer fusion은 계산 비용 높음: 320k queries × 8 cameras → ~100M FLOPs
- 단일 RTX 4090으로는 학습 가능하나, 실시간 추론(< 50ms)은 경량화 필수

**학술 참조**:
- BEVFormer (2022): Temporal BEV representation, 50ms @ V100
- LSS (Lift-Splat-Shoot, 2020): Simpler BEV projection, 35ms @ V100
- 본 프로젝트는 LSS 스타일 경량 BEV가 현실적

#### Gap 3: Trajectory Prediction (MAJOR)

**Tesla FSD**:
- Occupancy Flow: 과거 3 frames → 미래 2s (10 frames @ 5Hz)
- Per-voxel motion vectors + uncertainty
- 3D ConvGRU 또는 Transformer temporal model

**본 프로젝트 현재**:
- Constant Velocity Model: `future_pos = current_pos + velocity * dt`
- NPC 행동 예측 없음 (lane change, braking 예측 불가)

**Gap의 본질**:
- CV model은 직선 주행만 정확, 차선 변경 시 완전히 실패
- Occupancy Flow는 dense prediction → 신규 객체(갑자기 나타난 차) 자연스럽게 처리
- 본 프로젝트는 Object-centric prediction (per-vehicle LSTM/GNN)이 현실적

**학술 참조**:
- Trajectron++ (2020): Graph-based multi-agent prediction, 15ms @ RTX 2080
- nuPlan Baseline Predictor: Simple LSTM, 5ms 추론
- 본 프로젝트는 nuPlan Baseline 수준 목표

#### Gap 4: Trajectory Planning (CRITICAL)

**Tesla FSD**:
- 2-stage planning: MCTS (discrete maneuver) → Neural Evaluator (continuous trajectory)
- 20 candidates 생성 후 cost function으로 선택
- Cost function에 human-likeness (GAN discriminator) 포함

**본 프로젝트 현재**:
- Direct control: RL policy가 [steering, accel] 2D output 직접 생성
- Trajectory 개념 없음 (waypoint 없이 즉시 제어 명령)
- Human-likeness 학습 없음 (reward function만)

**Gap의 본질**:
- Tesla의 2-stage는 **interpretability** + **safety** 장점: 궤적을 먼저 생성하므로 검증 가능
- Direct control은 black-box, 왜 그 steering을 선택했는지 설명 불가
- MCTS는 계산 비용 높지만 exploration 우수
- RL direct control은 빠르지만 sample efficiency 낮음

**구현 가능성**:
- Sampling-based planner (CEM: Cross-Entropy Method) 사용 가능
- 본 프로젝트의 RL을 trajectory output으로 변경 → PID/MPC low-level controller 추가
- 계산 복잡도: CEM 100 samples × 20 steps → ~5ms (행렬 연산으로 병렬화 가능)

#### Gap 5: Route Planning (MAJOR)

**Tesla FSD**:
- GPS 경로 → HD Map → Lane-level graph
- A* on lane graph with dynamic cost (traffic, road closure)
- Rerouting every 10s or on significant deviation

**본 프로젝트 현재**:
- 고정 waypoints (Unity Inspector에서 수동 배치)
- 동적 경로 변경 없음
- Lane-level routing 없음 (단순 position target)

**Gap의 본질**:
- 고정 waypoint는 단순 트랙 학습에만 유효
- 실제 환경: 교차로에서 좌/우회전 선택, 우회로 탐색 필요
- Lane graph 구조 없음 → lane change decision 불가능

**구현 가능성**:
- Unity 내 Road graph 생성 (nodes + edges)
- A* pathfinding 구현 (C# 또는 Python)
- Lane connectivity 정의 (left/right adjacent lanes)
- 복잡도: O(E log V), 실시간 가능 (수백 노드 기준 < 1ms)

#### Gap 6: Vehicle Control (MODERATE)

**Tesla FSD**:
- Direct Neural Network: trajectory → [steering, accel, brake]
- MPC-style receding horizon (0.5s)
- 40ms update rate (25Hz)

**본 프로젝트 현재**:
- RL Policy: observation → [steering, accel]
- Update rate: 50ms (20Hz, Unity FixedUpdate)
- Unity 물리 엔진 통합 (Rigidbody)

**Gap의 본질**:
- 이미 유사한 수준 달성
- 차이점: Tesla는 trajectory tracking, 본 프로젝트는 direct control
- Latency는 50ms로 충분 (human reaction time 200-300ms)

**개선 방향**:
- Trajectory output 추가 시 PID/MPC 필요
- 현재 구조로는 큰 문제 없음

#### Gap 7: Data Pipeline (MAJOR)

**Tesla FSD**:
- 400만대 플릿 → 일일 100TB 수집
- Shadow Mode: FSD vs Human disagreement 자동 감지
- Auto-labeling: 4D reconstruction + 3D tracking
- Dojo Supercomputer: 1.1 ExaFLOPS

**본 프로젝트 현재**:
- Unity 시뮬레이션 self-play
- ML-Agents 병렬 환경 (16 areas)
- RTX 4090 단일 GPU: ~10M steps/day

**Gap의 본질**:
- **규모 차이**: 10,000배 이상 (100TB vs 10GB)
- **데이터 품질**: 실제 주행 vs 시뮬레이션 (Sim-to-Real gap)
- **컴퓨팅**: Dojo(1.1 ExaFLOPS) vs RTX 4090(0.082 TFLOPS FP32) = 13,000배

**완전히 극복 불가능한 Gap**:
- Shadow Mode fleet 데이터는 대체 불가
- 시뮬레이션 데이터로 보완 가능하지만 generalization 한계 존재
- 본 프로젝트의 목표는 **알고리즘 검증**이므로, 규모는 불필요

---

## 4. 현실적 구현 로드맵 (RTX 4090 단일 GPU)

### 4.1 하드웨어 제약 분석

**RTX 4090 스펙**:
- CUDA Cores: 16,384
- Tensor Cores: 512 (4세대)
- VRAM: 24GB GDDR6X
- Memory Bandwidth: 1,008 GB/s
- FP32 Performance: 82.6 TFLOPS
- FP16 Performance: 165.2 TFLOPS (Tensor Core)
- INT8 Performance: 330.3 TOPS (Tensor Core)
- TDP: 450W

**학습 시 VRAM 사용량 추정**:
```
Model Parameters (FP32):     P * 4 bytes
Gradients (FP32):            P * 4 bytes
Optimizer State (Adam):      P * 8 bytes (2 moments)
Activations (batch=B):       ~P * B * 2 bytes (추정)

Total (FP32): P * (4 + 4 + 8 + 2B) = P * (16 + 2B)

Example: P=100M, B=32
  → 100M * (16 + 64) = 8GB

Maximum model size @ VRAM=24GB, B=32:
  P_max = 24GB / 80 bytes ≈ 300M params (FP32)
  P_max = 24GB / 44 bytes ≈ 545M params (FP16 Mixed Precision)
```

**추론 시 제약**:
- Unity Sentis 사용 시: ~2GB VRAM 예약 (Unity 자체)
- 실시간 추론(< 50ms) 목표: 모델 크기 < 50M params

**단일 GPU 학습 속도 제약**:
```
Throughput (samples/sec) = GPU_FLOPS / (FLOPs_per_sample)

Example: RegNet-120GF
  FLOPs_per_sample = 120B
  GPU_FLOPS = 165 TFLOPS (FP16)
  Throughput = 165T / 120B = 1,375 samples/sec

Batch=32 → 43 batches/sec
PPO epoch (10 minibatches, 2048 samples) → 1.5 sec/epoch

현실: Data loading, CPU overhead 고려 → 실제 0.3-0.5x
  → PPO update 3-5 sec/epoch
```

### 4.2 구현 가능한 것 vs 불가능한 것

| 컴포넌트 | 구현 가능 여부 | 대안 / 이유 | VRAM | 학습 시간 |
|----------|------------|-----------|------|---------|
| **Camera-based Perception** | 제한적 가능 | ML-Agents CameraSensor 84x84 (small CNN) | 2-4GB | 수일-수주 |
| **Multi-camera BEV** | 불가 | 단일 카메라 BEV로 축소 (LSS 방식) | 8-12GB | 수주 |
| **Occupancy Network** | 불가 (학습 데이터 부족) | GT-based occupancy grid from Unity | - | - |
| **Trajectory Prediction** | 가능 | LSTM/GNN 경량 모델 (nuPlan baseline 수준) | 1-2GB | 수일 |
| **MCTS Planning** | 불가 (계산 비용) | CEM (Cross-Entropy Method) sampling-based | - | - |
| **Neural Planner** | 가능 | RL policy output = trajectory waypoints | 3-5GB | 수주 |
| **Route Planning** | 가능 | A* on Unity road graph | - | 수시간 (구현) |
| **GAIL/IL** | 가능 | ML-Agents 내장 기능 사용 | 4-6GB | 수일 |
| **World Model** | 제한적 가능 | Dreamer-v3 소규모 (2D simplified) | 6-10GB | 수주 |
| **Shadow Mode** | 불가 | 시뮬레이션 self-play로 대체 | - | - |
| **400M Param Model** | 불가 (학습 불가) | 50-100M param 모델로 축소 | 20-24GB | 수개월 |

**결론**:
- **가능**: Prediction (LSTM), Planning (CEM/RL), Route (A*), IL (GAIL)
- **제한적**: BEV (단일 카메라), World Model (2D)
- **불가능**: Multi-camera Occupancy, MCTS, Fleet data

### 4.3 단계별 구현 계획

Phase 5 내부에서의 확장 (Stages 0-L):

```
현재 상태 (2026-01-30):
  Stage 0: Foundation (Lane Keeping)              [✅ 완료]
  Stage A: Dense Overtaking                       [✅ 완료]
  Stage B v2: Decision Making                     [✅ 완료]
  Stage C: Multi-NPC Generalization               [📋 설계 완료]
  Stage D v2: Lane Observation (254D)             [🔄 진행중]
  Stage E-G: Curved Roads, Multi-lane, Intersection [📋 계획]

확장 계획 (Tesla FSD 기능 추가):
  Stage 5A: Reactive RL Control (2D output)       [✅ 현재]
  Stage 5B: Trajectory Output (N waypoints)       [📋 계획]
    - Action space 변경: 2D → [N x 2] waypoints
    - Reward에 trajectory smoothness 추가
    - 예상: 2-4M steps, 1-2주

  Stage 5C: Prediction Integration (LSTM)         [📋 계획]
    - NPC trajectory predictor 학습 (offline)
    - Predicted trajectories → observation 추가
    - 예상: 1M steps, 3-5일

  Stage 5D: Camera Perception (Single camera)     [📋 계획]
    - ML-Agents CameraSensor 추가 (84x84)
    - CNN encoder (nature_cnn) 학습
    - Vector obs + Image obs fusion
    - 예상: 5-10M steps, 2-4주

  Stage 5E: BEV Representation (LSS-style)        [📋 계획]
    - 단일 카메라 → BEV grid (50x50)
    - Lift-Splat-Shoot 방식 구현
    - BEV features → RL policy
    - 예상: 10-15M steps, 4-8주

  Stage 5F: Full E2E Pipeline (Camera → Trajectory) [📋 계획]
    - Camera → BEV → Prediction → Planning
    - End-to-end gradient flow
    - nuPlan benchmark 테스트
    - 예상: 20-30M steps, 8-12주
```

**우선순위 (현실적 순서)**:
1. **Stage 5B: Trajectory Output** (가장 중요, 즉시 구현 가능)
2. **Stage 5C: Prediction Integration** (Constant Velocity 대체)
3. **Route Planning 추가** (WaypointManager 확장)
4. **Stage 5D: Camera Perception** (실험적)
5. **Stage 5E: BEV** (장기 연구)

### 4.4 TECH-SPEC.md 매핑 및 현실적 수정

**TECH-SPEC.md에서 정의되었지만 미구현**:

| TECH-SPEC 컴포넌트 | 정의 위치 | 구현 상태 | Tesla 대응 | 현실적 구현 방법 |
|------------------|---------|---------|-----------|--------------|
| **BEVEncoder** | 3.4 Modular Encoder | 미구현 | RegNet + BiFPN | LSS (Lift-Splat-Shoot) 경량 버전, 단일 카메라 |
| **TrajectoryPredictor** | 3.5 (언급만) | 미구현 | Occupancy Flow | nuPlan Baseline LSTM (5 agents × 2s) |
| **ObservationEncoder Level 2-4** | 3.3.1 | Level 1만 구현 | Multi-modal Fusion | Level 2: CNN 추가, Level 3: BEV 추가, Level 4: Temporal LSTM |
| **GAIL Discriminator** | 3.6, 4.2 | 설정만 존재 | Human-likeness GAN | ML-Agents GAIL 구현 활용, nuPlan expert demo 필요 |
| **Trajectory Output** | 없음 | 없음 (2D control) | MCTS + Neural Evaluator | Action space → [N waypoints × 2], N=10 (2s @ 5Hz) |

**현실적 수정 제안**:

```python
# python/src/models/planning/encoder.py (수정안)

class ObservationEncoder(nn.Module):
    """
    Level 1: Vector only (현재)
    Level 2: Vector + CNN (Camera 84x84)
    Level 3: Vector + BEV (단일 카메라 LSS)
    Level 4: Vector + BEV + Temporal (LSTM)
    """
    def __init__(self, level=1, ...):
        super().__init__()
        self.level = level

        # Level 1: Vector encoders (기존)
        self.ego_encoder = nn.Sequential(...)
        self.route_encoder = nn.Sequential(...)
        self.surr_encoder = nn.Sequential(...)

        if level >= 2:
            # Level 2: Camera encoder
            self.camera_encoder = NatureCNN(
                input_shape=(84, 84, 3),
                output_dim=128
            )

        if level >= 3:
            # Level 3: BEV encoder (LSS-style)
            self.bev_encoder = LSS_BEV(
                camera_features_dim=128,
                bev_grid_size=(50, 50),
                bev_feature_dim=64
            )

        if level >= 4:
            # Level 4: Temporal encoder
            self.temporal_encoder = nn.LSTM(
                input_size=encoded_dim,
                hidden_size=256,
                num_layers=2
            )
```

**VRAM 요구량**:
- Level 1 (현재): ~3GB
- Level 2 (+CNN): ~4GB
- Level 3 (+BEV): ~8GB
- Level 4 (+LSTM): ~10GB

---

## 5. 핵심 기술 구현 상세

### 5.1 Trajectory Planning 추가 (가장 중요한 Gap)

**현재**: RL → [steering, acceleration] (2D output)
**목표**: RL → [trajectory waypoints] → PID/MPC → [steering, acceleration]

#### 5.1.1 Action Space 변경

```python
# python/src/models/planning/policy.py (수정)

class TrajectoryPlanningPolicy(nn.Module):
    """
    Output: Trajectory waypoints instead of direct control
    """
    def __init__(self, obs_dim=256, num_waypoints=10, hidden_dim=256):
        super().__init__()

        # Encoder (기존과 동일)
        self.encoder = ObservationEncoder()

        # Actor: Trajectory decoder
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_waypoints * 2)  # N waypoints × (x, y)
        )

        # 또는 RNN decoder (더 smooth)
        self.rnn_decoder = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=2
        )
        self.waypoint_head = nn.Linear(hidden_dim, 2)

    def forward(self, obs):
        encoded = self.encoder(*obs)

        # Option 1: MLP decoder (parallel generation)
        trajectory_flat = self.trajectory_decoder(encoded)
        trajectory = trajectory_flat.view(-1, self.num_waypoints, 2)

        # Option 2: RNN decoder (sequential, auto-regressive)
        hidden = encoded.unsqueeze(0).repeat(2, 1, 1)  # 2 layers
        waypoints = []
        for t in range(self.num_waypoints):
            out, hidden = self.rnn_decoder(encoded.unsqueeze(0), hidden)
            wp = self.waypoint_head(out.squeeze(0))
            waypoints.append(wp)
        trajectory = torch.stack(waypoints, dim=1)

        return trajectory
```

#### 5.1.2 Low-level Controller (PID/MPC)

```csharp
// Assets/Scripts/Controllers/TrajectoryTracker.cs (신규)

using UnityEngine;

public class TrajectoryTracker : MonoBehaviour
{
    // PID gains for steering
    public float Kp_steer = 1.5f;
    public float Ki_steer = 0.01f;
    public float Kd_steer = 0.3f;

    // PID gains for speed
    public float Kp_speed = 2.0f;
    public float Ki_speed = 0.05f;
    public float Kd_speed = 0.5f;

    private float integral_steer = 0f;
    private float prev_error_steer = 0f;
    private float integral_speed = 0f;
    private float prev_error_speed = 0f;

    public (float steering, float accel) TrackTrajectory(
        Vector2[] trajectory,  // N waypoints in local coordinates
        Vector3 currentPos,
        Quaternion currentRot,
        float currentSpeed,
        float dt = 0.02f
    )
    {
        // 1. Find closest waypoint on trajectory (lookahead)
        int lookahead_index = Mathf.Min(5, trajectory.Length - 1);  // 0.5s ahead
        Vector2 target_local = trajectory[lookahead_index];

        // 2. Steering control (Pure Pursuit variant)
        float crosstrack_error = target_local.x;  // lateral offset
        float heading_error = Mathf.Atan2(target_local.y, target_local.x);

        // PID for steering
        integral_steer += crosstrack_error * dt;
        float derivative_steer = (crosstrack_error - prev_error_steer) / dt;
        float steering = Kp_steer * crosstrack_error
                       + Ki_steer * integral_steer
                       + Kd_steer * derivative_steer;
        steering = Mathf.Clamp(steering, -0.5f, 0.5f);  // rad

        prev_error_steer = crosstrack_error;

        // 3. Speed control
        float target_speed = CalculateTargetSpeed(trajectory, lookahead_index);
        float speed_error = target_speed - currentSpeed;

        integral_speed += speed_error * dt;
        float derivative_speed = (speed_error - prev_error_speed) / dt;
        float accel = Kp_speed * speed_error
                    + Ki_speed * integral_speed
                    + Kd_speed * derivative_speed;
        accel = Mathf.Clamp(accel, -4.0f, 2.0f);  // m/s^2

        prev_error_speed = speed_error;

        return (steering, accel);
    }

    private float CalculateTargetSpeed(Vector2[] trajectory, int lookahead_index)
    {
        // Calculate curvature from trajectory
        if (lookahead_index < 2) return 15f;  // default

        Vector2 p0 = trajectory[lookahead_index - 2];
        Vector2 p1 = trajectory[lookahead_index - 1];
        Vector2 p2 = trajectory[lookahead_index];

        // Menger curvature: k = 4 * Area(triangle) / (|p0-p1| * |p1-p2| * |p2-p0|)
        float area = Mathf.Abs((p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y)) / 2f;
        float d01 = Vector2.Distance(p0, p1);
        float d12 = Vector2.Distance(p1, p2);
        float d20 = Vector2.Distance(p2, p0);
        float curvature = 4f * area / (d01 * d12 * d20 + 1e-6f);

        // Target speed based on curvature (v = sqrt(a_lat_max / k))
        float max_lat_accel = 3.0f;  // m/s^2
        float target_speed = Mathf.Sqrt(max_lat_accel / (curvature + 1e-6f));
        return Mathf.Clamp(target_speed, 5f, 20f);
    }
}
```

#### 5.1.3 Reward 수정

```python
# python/src/models/planning/reward.py (수정)

class TrajectoryReward:
    def __init__(self, config):
        self.weights = config.get('reward_weights', {
            'progress': 1.0,
            'trajectory_smoothness': 0.5,     # NEW
            'trajectory_feasibility': 0.3,    # NEW
            'tracking_error': -0.2,           # NEW
            'collision': -5.0,
            # ... (기존 항목)
        })

    def compute(self, state, action_trajectory, next_state, info, dt=0.02):
        reward = 0.0

        # 1. Trajectory smoothness (jerk along trajectory)
        trajectory = action_trajectory  # [N, 2]
        if len(trajectory) >= 3:
            # Numerical 2nd derivative
            accel = (trajectory[2:] - 2*trajectory[1:-1] + trajectory[:-2]) / (dt**2)
            jerk = torch.mean(torch.abs(accel))
            reward += self.weights['trajectory_smoothness'] * torch.exp(-jerk / 5.0)

        # 2. Trajectory feasibility (within kinematic constraints)
        max_curvature = 0.3  # 1/R, R_min ~ 3m
        for i in range(len(trajectory) - 2):
            p0, p1, p2 = trajectory[i], trajectory[i+1], trajectory[i+2]
            # Menger curvature
            area = torch.abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1])) / 2
            d01 = torch.norm(p1 - p0)
            d12 = torch.norm(p2 - p1)
            d20 = torch.norm(p0 - p2)
            k = 4 * area / (d01 * d12 * d20 + 1e-6)

            if k > max_curvature:
                reward += self.weights['trajectory_feasibility'] * (max_curvature - k)

        # 3. Tracking error (actual vs planned)
        if 'planned_waypoint' in info and 'actual_position' in info:
            tracking_error = torch.norm(info['actual_position'] - info['planned_waypoint'])
            reward += self.weights['tracking_error'] * tracking_error

        # ... (기존 reward 항목)

        return reward, done
```

#### 5.1.4 학습 파라미터

```yaml
# python/configs/planning/trajectory_ppo.yaml (신규)

algorithm: PPO
environment:
  name: ADPlanningTrajectory-v0
  num_envs: 8
  max_episode_steps: 1000

  action_space:
    type: continuous
    shape: [10, 2]  # 10 waypoints × (x, y)
    low: [-5.0, 0.0]  # local coordinates
    high: [5.0, 20.0]

network:
  encoder:
    level: 1  # Vector only initially
    ego_dim: 8
    route_dim: 30
    surr_dim: 40
    hidden_dim: 256

  trajectory_decoder:
    type: rnn  # 'mlp' or 'rnn'
    num_waypoints: 10
    hidden_dim: 256
    num_layers: 2

ppo:
  learning_rate: 3e-4
  batch_size: 2048
  minibatch_size: 128  # larger for trajectory output
  epochs_per_update: 10
  clip_ratio: 0.2

training:
  total_steps: 5_000_000
  eval_interval: 50_000
  checkpoint_interval: 100_000

  # Curriculum (optional)
  curriculum:
    num_waypoints:
      start: 5
      end: 10
      steps: 1_000_000
```

**예상 학습 시간**: 5M steps @ RTX 4090
- Throughput: ~5,000 steps/sec (action space 증가로 속도 감소)
- Total: 5M / 5k = 1,000초 = ~16분 (낙관적)
- 현실: overhead 고려 → 2-4시간

### 5.2 Prediction Module 추가

**현재**: Constant Velocity
**목표**: LSTM-based trajectory prediction (nuPlan baseline 수준)

#### 5.2.1 모델 아키텍처

```python
# python/src/models/prediction/lstm_predictor.py (신규)

import torch
import torch.nn as nn

class LSTMTrajectoryPredictor(nn.Module):
    """
    Predict future trajectories of surrounding agents

    Input: Past trajectories (T_past frames)
    Output: Future trajectories (T_future frames)
    """
    def __init__(
        self,
        input_dim=5,        # [x, y, vx, vy, heading]
        hidden_dim=128,
        num_layers=2,
        T_past=10,          # 1s @ 10Hz
        T_future=20,        # 2s @ 10Hz
        max_agents=8
    ):
        super().__init__()
        self.T_past = T_past
        self.T_future = T_future
        self.max_agents = max_agents

        # Per-agent encoder
        self.agent_encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Social context (optional, for multi-agent interaction)
        self.social_encoder = nn.Sequential(
            nn.Linear(hidden_dim * max_agents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Decoder (future trajectory)
        self.decoder = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.output_head = nn.Linear(hidden_dim, 2)  # (x, y) per timestep

    def forward(self, past_trajectories, agent_mask=None):
        """
        Args:
            past_trajectories: [batch, max_agents, T_past, input_dim]
            agent_mask: [batch, max_agents] (1 if agent exists, 0 otherwise)

        Returns:
            future_trajectories: [batch, max_agents, T_future, 2]
        """
        batch_size = past_trajectories.shape[0]

        # Encode past trajectories for each agent
        agent_features = []
        for i in range(self.max_agents):
            agent_past = past_trajectories[:, i, :, :]  # [batch, T_past, input_dim]
            _, (hidden, cell) = self.agent_encoder(agent_past)
            agent_features.append(hidden[-1])  # [batch, hidden_dim]

        agent_features = torch.stack(agent_features, dim=1)  # [batch, max_agents, hidden_dim]

        # Social context (global interaction)
        if agent_mask is not None:
            agent_features = agent_features * agent_mask.unsqueeze(-1)
        social_context = self.social_encoder(agent_features.flatten(1))  # [batch, hidden_dim]

        # Decode future trajectories
        future_trajectories = []
        decoder_input = social_context.unsqueeze(1)  # [batch, 1, hidden_dim]
        hidden_state = None

        for t in range(self.T_future):
            out, hidden_state = self.decoder(decoder_input, hidden_state)
            waypoint = self.output_head(out)  # [batch, 1, 2]
            future_trajectories.append(waypoint)
            decoder_input = out  # Auto-regressive

        future_trajectories = torch.cat(future_trajectories, dim=1)  # [batch, T_future, 2]

        # Broadcast to all agents (simplified, 실제로는 per-agent prediction)
        future_trajectories = future_trajectories.unsqueeze(1).repeat(1, self.max_agents, 1, 1)

        return future_trajectories
```

#### 5.2.2 데이터 수집 (Unity)

```csharp
// Assets/Scripts/Data/TrajectoryRecorder.cs (신규)

using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class TrajectoryRecorder : MonoBehaviour
{
    public List<GameObject> npcs;
    public float recordRate = 10f;  // 10 Hz
    public int maxFrames = 30;      // 3s total (1s past + 2s future)

    private List<List<Vector3>> trajectories;
    private float timer = 0f;
    private string outputPath = "datasets/trajectories/";

    void Start()
    {
        trajectories = new List<List<Vector3>>();
        foreach (var npc in npcs)
        {
            trajectories.Add(new List<Vector3>());
        }

        Directory.CreateDirectory(outputPath);
    }

    void Update()
    {
        timer += Time.deltaTime;

        if (timer >= 1f / recordRate)
        {
            for (int i = 0; i < npcs.Count; i++)
            {
                if (npcs[i] != null)
                {
                    trajectories[i].Add(npcs[i].transform.position);

                    if (trajectories[i].Count > maxFrames)
                    {
                        trajectories[i].RemoveAt(0);
                    }
                }
            }
            timer = 0f;
        }
    }

    public void SaveTrajectories(string filename)
    {
        using (StreamWriter writer = new StreamWriter(outputPath + filename))
        {
            writer.WriteLine("agent_id,frame,x,y,z,vx,vy,heading");

            for (int i = 0; i < trajectories.Count; i++)
            {
                for (int t = 0; t < trajectories[i].Count; t++)
                {
                    Vector3 pos = trajectories[i][t];
                    Vector3 vel = (t > 0) ? (trajectories[i][t] - trajectories[i][t-1]) * recordRate : Vector3.zero;
                    float heading = Mathf.Atan2(vel.z, vel.x);

                    writer.WriteLine($"{i},{t},{pos.x},{pos.y},{pos.z},{vel.x},{vel.z},{heading}");
                }
            }
        }
    }
}
```

#### 5.2.3 학습 스크립트

```python
# python/src/training/train_prediction.py (신규)

import torch
from torch.utils.data import Dataset, DataLoader
from models.prediction.lstm_predictor import LSTMTrajectoryPredictor
import pandas as pd

class TrajectoryDataset(Dataset):
    def __init__(self, csv_path, T_past=10, T_future=20):
        self.data = pd.read_csv(csv_path)
        self.T_past = T_past
        self.T_future = T_future

        # Group by agent_id and extract sequences
        self.sequences = []
        for agent_id in self.data['agent_id'].unique():
            agent_data = self.data[self.data['agent_id'] == agent_id].sort_values('frame')

            for i in range(len(agent_data) - T_past - T_future):
                past = agent_data.iloc[i:i+T_past][['x', 'y', 'vx', 'vy', 'heading']].values
                future = agent_data.iloc[i+T_past:i+T_past+T_future][['x', 'y']].values
                self.sequences.append((past, future))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        past, future = self.sequences[idx]
        return torch.FloatTensor(past), torch.FloatTensor(future)

# Training loop
model = LSTMTrajectoryPredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.MSELoss()

dataset = TrajectoryDataset('datasets/trajectories/train.csv')
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

for epoch in range(50):
    for past, future_gt in dataloader:
        past = past.unsqueeze(1)  # [batch, 1 agent, T_past, 5]

        future_pred = model(past)[:, 0, :, :]  # [batch, T_future, 2]

        loss = criterion(future_pred, future_gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: Loss {loss.item():.4f}")

torch.save(model.state_dict(), 'models/prediction/lstm_predictor.pth')
```

**학습 시간**: RTX 4090
- 데이터셋: 10k sequences (nuPlan mini subset)
- Batch size: 128
- Epochs: 50
- 예상: ~30분

#### 5.2.4 Unity 통합

```csharp
// Assets/Scripts/Inference/PredictionInference.cs (신규)

using Unity.Sentis;
using UnityEngine;

public class PredictionInference : MonoBehaviour
{
    public ModelAsset predictionModel;
    private Worker worker;

    void Start()
    {
        var runtimeModel = ModelLoader.Load(predictionModel);
        worker = new Worker(runtimeModel, BackendType.GPUCompute);
    }

    public Vector2[] PredictTrajectory(
        Vector3[] pastPositions,  // 10 frames
        Vector3[] pastVelocities
    )
    {
        // Prepare input tensor [1, 1, 10, 5]
        float[] inputData = new float[1 * 1 * 10 * 5];
        for (int i = 0; i < 10; i++)
        {
            inputData[i*5 + 0] = pastPositions[i].x;
            inputData[i*5 + 1] = pastPositions[i].z;
            inputData[i*5 + 2] = pastVelocities[i].x;
            inputData[i*5 + 3] = pastVelocities[i].z;
            inputData[i*5 + 4] = Mathf.Atan2(pastVelocities[i].z, pastVelocities[i].x);
        }

        using var inputTensor = new Tensor<float>(new TensorShape(1, 1, 10, 5), inputData);
        worker.SetInput("past_trajectories", inputTensor);
        worker.Schedule();

        var output = worker.PeekOutput("future_trajectories") as Tensor<float>;
        output.CompleteOperationsAndDownload();

        // Extract [1, 1, 20, 2] → Vector2[20]
        Vector2[] futureTrajectory = new Vector2[20];
        for (int i = 0; i < 20; i++)
        {
            futureTrajectory[i] = new Vector2(output[i*2], output[i*2+1]);
        }

        return futureTrajectory;
    }
}
```

### 5.3 Camera Perception 추가

**현재**: Ground Truth Vector (242D)
**목표**: Camera → CNN → feature vector → RL

#### 5.3.1 ML-Agents CameraSensor 설정

```csharp
// Assets/Scripts/Agents/E2EDrivingAgentCamera.cs (신규)

using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;

public class E2EDrivingAgentCamera : Agent
{
    public Camera frontCamera;

    public override void CollectObservations(VectorSensor sensor)
    {
        // 기존 vector observations (242D)
        sensor.AddObservation(transform.position);  // 3D
        sensor.AddObservation(GetComponent<Rigidbody>().velocity);  // 3D
        // ... (총 242D)
    }

    public override void Initialize()
    {
        // Add Camera Sensor (84x84 grayscale)
        var cameraSensorComponent = gameObject.AddComponent<CameraSensorComponent>();
        cameraSensorComponent.Camera = frontCamera;
        cameraSensorComponent.SensorName = "FrontCamera";
        cameraSensorComponent.Width = 84;
        cameraSensorComponent.Height = 84;
        cameraSensorComponent.Grayscale = true;  // 1 channel
        cameraSensorComponent.ObservationType = ObservationType.Default;
    }
}
```

**Camera 설정**:
- Resolution: 84×84 (Atari 표준, DQN/PPO에서 검증됨)
- Grayscale: 1 channel (RGB는 3배 VRAM)
- FOV: 90도 (front camera)
- Position: Ego vehicle 전면 1m, 높이 1.5m

#### 5.3.2 CNN Encoder (Nature CNN)

```python
# python/src/models/planning/encoder.py (Level 2 추가)

class NatureCNN(nn.Module):
    """
    Nature DQN CNN architecture
    Input: [batch, 84, 84, 1] (grayscale)
    Output: [batch, 512]
    """
    def __init__(self, input_shape=(84, 84, 1), output_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # 84 → 20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20 → 9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9 → 7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        # x: [batch, 1, 84, 84]
        return self.conv(x)

# Fusion with vector observations
class MultimodalEncoder(nn.Module):
    def __init__(self, vector_dim=242, image_shape=(84, 84, 1), hidden_dim=256):
        super().__init__()

        # Vector encoder
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Image encoder
        self.image_encoder = NatureCNN(image_shape, output_dim=128)

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, vector_obs, image_obs):
        vector_feat = self.vector_encoder(vector_obs)
        image_feat = self.image_encoder(image_obs)

        combined = torch.cat([vector_feat, image_feat], dim=-1)
        return self.fusion(combined)
```

**VRAM 추정**:
- Input: 84×84×1 = 7,056 floats = 28 KB (무시 가능)
- CNN params: (32×8×8 + 64×4×4×32 + 64×3×3×64 + 512×3136 + 128×512) × 4 bytes ≈ 7 MB
- Activations (batch=32): 32 × (20×20×32 + 9×9×64 + 7×7×64 + 512) × 4 bytes ≈ 2 MB
- **Total**: ~10 MB (negligible)

**학습 속도 저하**:
- Vector-only: ~5,000 steps/sec
- Vector+CNN: ~2,000 steps/sec (CNN forward pass 추가)
- 학습 시간: 2-3배 증가

### 5.4 Route Planning 추가

**현재**: 고정 waypoints
**목표**: 동적 경로 계획 (A* on road graph)

#### 5.4.1 Road Graph 구조

```csharp
// Assets/Scripts/Navigation/RoadGraph.cs (신규)

using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class RoadNode
{
    public int id;
    public Vector3 position;
    public List<int> neighbors;  // Adjacent node IDs
    public float speedLimit;
    public string laneType;  // "straight", "left_turn", "right_turn"
}

[System.Serializable]
public class RoadEdge
{
    public int fromNodeId;
    public int toNodeId;
    public float cost;  // Distance or travel time
}

public class RoadGraph : MonoBehaviour
{
    public List<RoadNode> nodes;
    public List<RoadEdge> edges;

    private Dictionary<int, RoadNode> nodeDict;
    private Dictionary<int, List<RoadEdge>> adjacencyList;

    void Start()
    {
        BuildGraph();
    }

    void BuildGraph()
    {
        nodeDict = new Dictionary<int, RoadNode>();
        adjacencyList = new Dictionary<int, List<RoadEdge>>();

        foreach (var node in nodes)
        {
            nodeDict[node.id] = node;
            adjacencyList[node.id] = new List<RoadEdge>();
        }

        foreach (var edge in edges)
        {
            adjacencyList[edge.fromNodeId].Add(edge);
        }
    }

    public List<Vector3> FindPath(Vector3 start, Vector3 goal)
    {
        // Find nearest nodes
        int startNodeId = FindNearestNode(start);
        int goalNodeId = FindNearestNode(goal);

        // A* search
        var path = AStar(startNodeId, goalNodeId);

        // Convert node IDs to positions
        List<Vector3> waypoints = new List<Vector3>();
        foreach (int nodeId in path)
        {
            waypoints.Add(nodeDict[nodeId].position);
        }

        return waypoints;
    }

    private int FindNearestNode(Vector3 position)
    {
        int nearestId = -1;
        float minDist = float.MaxValue;

        foreach (var node in nodes)
        {
            float dist = Vector3.Distance(node.position, position);
            if (dist < minDist)
            {
                minDist = dist;
                nearestId = node.id;
            }
        }

        return nearestId;
    }

    private List<int> AStar(int startId, int goalId)
    {
        var openSet = new HashSet<int> { startId };
        var cameFrom = new Dictionary<int, int>();
        var gScore = new Dictionary<int, float> { [startId] = 0f };
        var fScore = new Dictionary<int, float> { [startId] = Heuristic(startId, goalId) };

        while (openSet.Count > 0)
        {
            int current = GetLowestFScore(openSet, fScore);

            if (current == goalId)
            {
                return ReconstructPath(cameFrom, current);
            }

            openSet.Remove(current);

            foreach (var edge in adjacencyList[current])
            {
                int neighbor = edge.toNodeId;
                float tentativeGScore = gScore[current] + edge.cost;

                if (!gScore.ContainsKey(neighbor) || tentativeGScore < gScore[neighbor])
                {
                    cameFrom[neighbor] = current;
                    gScore[neighbor] = tentativeGScore;
                    fScore[neighbor] = tentativeGScore + Heuristic(neighbor, goalId);

                    if (!openSet.Contains(neighbor))
                    {
                        openSet.Add(neighbor);
                    }
                }
            }
        }

        return new List<int>();  // No path found
    }

    private float Heuristic(int nodeId, int goalId)
    {
        return Vector3.Distance(nodeDict[nodeId].position, nodeDict[goalId].position);
    }

    private int GetLowestFScore(HashSet<int> openSet, Dictionary<int, float> fScore)
    {
        int lowest = -1;
        float minScore = float.MaxValue;

        foreach (int nodeId in openSet)
        {
            float score = fScore.ContainsKey(nodeId) ? fScore[nodeId] : float.MaxValue;
            if (score < minScore)
            {
                minScore = score;
                lowest = nodeId;
            }
        }

        return lowest;
    }

    private List<int> ReconstructPath(Dictionary<int, int> cameFrom, int current)
    {
        var path = new List<int> { current };

        while (cameFrom.ContainsKey(current))
        {
            current = cameFrom[current];
            path.Insert(0, current);
        }

        return path;
    }
}
```

#### 5.4.2 Unity Editor Tool (Graph 생성)

```csharp
// Assets/Scripts/Editor/RoadGraphEditor.cs (신규)

using UnityEngine;
using UnityEditor;

[CustomEditor(typeof(RoadGraph))]
public class RoadGraphEditor : Editor
{
    public override void OnInspectorGUI()
    {
        DrawDefaultInspector();

        RoadGraph graph = (RoadGraph)target;

        if (GUILayout.Button("Auto-Generate from Scene"))
        {
            AutoGenerateGraph(graph);
        }

        if (GUILayout.Button("Visualize Graph"))
        {
            VisualizeGraph(graph);
        }
    }

    void AutoGenerateGraph(RoadGraph graph)
    {
        // Find all road segments in scene
        var roadSegments = GameObject.FindGameObjectsWithTag("RoadSegment");

        graph.nodes = new List<RoadNode>();
        graph.edges = new List<RoadEdge>();

        int nodeId = 0;
        foreach (var segment in roadSegments)
        {
            // Extract waypoints from road mesh or spline
            var waypoints = segment.GetComponent<RoadWaypoints>();

            foreach (var wp in waypoints.points)
            {
                graph.nodes.Add(new RoadNode
                {
                    id = nodeId++,
                    position = wp,
                    neighbors = new List<int>(),
                    speedLimit = 15f,
                    laneType = "straight"
                });
            }
        }

        // Connect sequential nodes
        for (int i = 0; i < graph.nodes.Count - 1; i++)
        {
            float dist = Vector3.Distance(graph.nodes[i].position, graph.nodes[i+1].position);

            graph.edges.Add(new RoadEdge
            {
                fromNodeId = i,
                toNodeId = i + 1,
                cost = dist
            });

            graph.nodes[i].neighbors.Add(i + 1);
        }

        EditorUtility.SetDirty(graph);
    }

    void VisualizeGraph(RoadGraph graph)
    {
        foreach (var edge in graph.edges)
        {
            var from = graph.nodes.Find(n => n.id == edge.fromNodeId);
            var to = graph.nodes.Find(n => n.id == edge.toNodeId);

            Debug.DrawLine(from.position, to.position, Color.green, 5f);
        }
    }
}
```

**사용법**:
1. 도로 오브젝트에 "RoadSegment" 태그 추가
2. RoadGraph 컴포넌트 추가 후 "Auto-Generate from Scene" 클릭
3. A* pathfinding 자동 동작

**성능**: 수백 노드 기준 < 1ms (C# 구현)

---

## 6. TECH-SPEC.md 매핑

### 6.1 현재 TECH-SPEC.md에서 정의했지만 구현되지 않은 항목

| 컴포넌트 | TECH-SPEC 섹션 | 정의 내용 | Tesla FSD 대응 | 현실적 구현 |
|----------|--------------|----------|--------------|-----------|
| **BEVEncoder** | 3.4 Modular Encoder | "BEV representation for spatial reasoning" | RegNet + Transformer → BEV 200×200 | LSS 단일 카메라 → BEV 50×50 |
| **TrajectoryPredictor** | 3.2 Prediction Module | "Constant Velocity or nuPlan Baseline" | Occupancy Flow (3D ConvGRU) | LSTM per-agent predictor |
| **ObservationEncoder Level 2** | 3.3.1 | "CNN for camera input" | RegNet-120GF Multi-scale | NatureCNN (84×84) |
| **ObservationEncoder Level 3** | 3.3.1 | "BEV features" | Transformer fusion 8 cameras | LSS 단일 카메라 |
| **ObservationEncoder Level 4** | 3.3.1 | "Temporal LSTM" | Occupancy Flow temporal | GRU on past observations |
| **GAIL Discriminator** | 3.3.3, 4.2 | "Human-likeness discriminator" | Fleet data GAN | nuPlan expert demo GAIL |
| **Trajectory Output** | 없음 | 없음 (직접 control만) | MCTS + Neural Evaluator | Action space → waypoints |

### 6.2 각 항목의 TECH-SPEC 정의 vs Tesla 방식 vs 현실적 구현

#### BEVEncoder (Section 3.4)

**TECH-SPEC 정의**:
```python
# 언급만 있고 구현 없음
# "BEV features: optional (64D)"
```

**Tesla 방식**:
- 8 cameras → RegNet features → Transformer Cross-Attention → BEV 200×200×256
- 모델 크기: ~300M params
- 추론 시간: ~60ms (HW3 최적화)

**현실적 구현** (RTX 4090):
```python
class LSS_BEV(nn.Module):
    """
    Lift-Splat-Shoot: 단일 카메라 → BEV
    Simplified from Tesla's 8-camera fusion
    """
    def __init__(self, camera_features_dim=128, bev_grid_size=(50, 50), bev_feature_dim=64):
        super().__init__()

        # Depth distribution predictor
        self.depth_net = nn.Sequential(
            nn.Linear(camera_features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)  # 64 depth bins
        )

        # BEV grid projection
        self.bev_grid_size = bev_grid_size
        self.bev_conv = nn.Sequential(
            nn.Conv2d(camera_features_dim, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, bev_feature_dim, 3, padding=1)
        )

    def forward(self, camera_features, camera_intrinsics):
        # camera_features: [batch, 128, 21, 21] (from CNN)
        # Output: [batch, 64, 50, 50] BEV features

        # 1. Predict depth distribution
        depth_dist = self.depth_net(camera_features.flatten(2).permute(0, 2, 1))
        depth_dist = torch.softmax(depth_dist, dim=-1)  # [batch, H*W, 64]

        # 2. Lift to 3D (frustum grid)
        # Simplified: use depth bins to create pseudo-3D
        # (생략: 복잡한 3D projection 로직)

        # 3. Splat to BEV grid
        bev_features = self.bev_conv(camera_features)

        # 4. Resize to target grid size
        bev_features = F.interpolate(bev_features, size=self.bev_grid_size, mode='bilinear')

        return bev_features
```

**VRAM**: ~2GB (단일 카메라)
**추론**: ~15ms @ RTX 4090
**학습 시간**: 5-10M steps, 2-4주

#### TrajectoryPredictor (Section 3.2)

**TECH-SPEC 정의**:
```python
# python/src/models/prediction/predictor.py
class PredictionModule:
    def __init__(self, mode: str = "constant_velocity"):
        # Constant Velocity 또는 nuPlan Baseline 언급
```

**Tesla 방식**:
- Occupancy Flow: past 3 frames → future 2s
- 3D ConvGRU, per-voxel motion
- 모델 크기: ~50M params

**현실적 구현**:
- LSTM per-agent: 8 agents × 2s horizon
- 모델 크기: ~5M params
- 추론: ~5ms @ RTX 4090
- (Section 5.2에서 상세 설명)

#### GAIL Discriminator (Section 4.2)

**TECH-SPEC 정의**:
```yaml
# python/configs/planning/gail.yaml
gail:
  discriminator:
    hidden_layers: [256, 256]
```

**Tesla 방식**:
- Human-likeness GAN: real driver trajectories vs model
- Discriminator input: trajectory sequence (2s)
- 학습 데이터: Fleet Shadow Mode clips

**현실적 구현**:
```python
# python/src/models/planning/gail_discriminator.py

class GAILDiscriminator(nn.Module):
    """
    Discriminate expert (nuPlan) vs policy trajectories
    """
    def __init__(self, trajectory_dim=20, hidden_dim=256):
        super().__init__()

        # Trajectory encoder (RNN)
        self.encoder = nn.GRU(
            input_size=5,  # [x, y, v, heading, accel]
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # Binary classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, trajectory):
        # trajectory: [batch, T, 5]
        _, hidden = self.encoder(trajectory)
        logits = self.classifier(hidden[-1])
        return logits  # P(expert)
```

**데이터 소스**:
- Expert: nuPlan 시나리오 (10k clips)
- Policy: RL 학습 중 생성된 trajectory

**학습**:
- ML-Agents GAIL 구현 활용
- Discriminator update: 매 PPO epoch마다 2회
- 예상 학습: 3-5M steps, 1-2주

---

## 7. 학술 E2E 접근법 참조

### 7.1 UniAD (2023)

**논문**: "Planning-oriented Autonomous Driving" (CVPR 2023)

**아키텍처**:
```
Camera → Backbone → Query-based Multi-task Head
  ├─ Detection (DETR-style)
  ├─ Tracking (Track Queries)
  ├─ Mapping (Lane Queries)
  ├─ Motion (Trajectory Queries)
  └─ Planning (Ego Query)
```

**핵심 기술**:
- **Unified Query**: 단일 Transformer로 모든 Task 처리
- **Planning Query**: Ego vehicle future를 다른 객체와 동일하게 예측
- **End-to-end Loss**: Detection + Tracking + Planning joint training

**본 프로젝트 적용**:
- Query-based multi-task 구조는 RTX 4090에서 실현 가능
- Task별 independent head 대신 shared Transformer 사용
- 단, camera 입력은 단일로 축소 (8 cameras → 1 camera)

**구현 난이도**: High
**VRAM**: ~12GB (단순화 버전)
**학습 시간**: 10-20M steps, 4-8주

### 7.2 VAD (2024)

**논문**: "Vectorized Scene Representation for Autonomous Driving" (CVPR 2024)

**아키텍처**:
```
Camera → CNN → BEV → Vectorization
  ├─ Lane: Polyline representation
  ├─ Agents: Vector (position, velocity, size)
  └─ Planning: Bezier curve waypoints
```

**핵심 기술**:
- **Vectorized Scene**: Raster(픽셀) 대신 Vector(기하) 표현
- **Efficiency**: Vector는 memory-efficient (sparse)
- **Interpretability**: Waypoints가 명시적 (visualizable)

**본 프로젝트 적용**:
- 현재 observation이 이미 vector (242D) → VAD와 철학적으로 유사
- BEV raster → vector conversion 추가 가능
- Planning output을 Bezier curve로 표현

**구현 난이도**: Medium
**본 프로젝트 현재 상태와 호환성**: Very High

**예시 구현**:
```python
class BezierTrajectoryCurve:
    """
    Bezier curve representation for smooth trajectory
    """
    def __init__(self, control_points):
        self.control_points = control_points  # [N, 2]

    def get_waypoints(self, num_samples=20):
        # Bezier interpolation
        t = torch.linspace(0, 1, num_samples)
        waypoints = []

        for t_i in t:
            # De Casteljau algorithm
            points = self.control_points.clone()
            while len(points) > 1:
                points = (1 - t_i) * points[:-1] + t_i * points[1:]
            waypoints.append(points[0])

        return torch.stack(waypoints)

# RL policy output: control points
action_space = [4, 2]  # 4 control points × (x, y)
```

### 7.3 World Model (GAIA-1, DriveDreamer)

**GAIA-1** (Waymo, 2023):
- Video generation model: 과거 프레임 → 미래 프레임 생성
- 9B parameters (Transformer-based)
- 용도: Planning safety verification (미래 시뮬레이션)

**DriveDreamer** (2024):
- NeRF-based world model: Controllable scene generation
- Action-conditioned: steering → 미래 장면 변화

**본 프로젝트 적용**:
- World Model은 Phase 7 (Advanced Topics)에 적합
- RTX 4090으로는 소규모 버전만 가능 (Dreamer-v3 스타일)
- 2D simplified: BEV grid future prediction

**구현 가능성**:
- **불가능**: GAIA-1 규모 (9B params, video generation)
- **제한적 가능**: Dreamer-v3 (2D BEV prediction, 50M params)

**Dreamer-v3 간략 구조**:
```
Encoder: observation → latent state (z)
Dynamics: z_t, action → z_{t+1} (RNN)
Decoder: z → reconstructed observation
Reward: z → predicted reward
Actor: z → action (RL policy)
```

**VRAM**: ~6GB
**학습**: 5-10M steps, 2-4주
**용도**: Model-based RL, 샘플 효율 향상

---

## 8. 결론 및 우선순위

### 8.1 즉시 구현 가능 (Phase 5 내, 1-2주)

#### 1. Trajectory Output (가장 중요)
- **목표**: Action space를 2D control → trajectory waypoints로 변경
- **이유**: Tesla FSD와의 가장 큰 gap이며, safety/interpretability 향상
- **구현**: Section 5.1 참조
- **우선순위**: **P0 (최우선)**
- **예상 시간**: 2-4시간 구현 + 2-4M steps 학습 = 1-2주
- **파일**:
  - `Assets/Scripts/Agents/E2EDrivingAgentTrajectory.cs` (Unity)
  - `Assets/Scripts/Controllers/TrajectoryTracker.cs` (PID controller)
  - `python/src/models/planning/trajectory_policy.py`
  - `python/configs/planning/trajectory_ppo.yaml`

#### 2. Route Planning (WaypointManager 확장)
- **목표**: 고정 waypoints → 동적 A* pathfinding
- **이유**: 교차로, lane change decision에 필수
- **구현**: Section 5.4 참조
- **우선순위**: **P0**
- **예상 시간**: 1-2일 구현 + 테스트
- **파일**:
  - `Assets/Scripts/Navigation/RoadGraph.cs`
  - `Assets/Scripts/Navigation/AStarPathfinder.cs`
  - `Assets/Scripts/Editor/RoadGraphEditor.cs`

### 8.2 중기 구현 (Phase 6, 2-4주)

#### 3. Prediction Module (LSTM)
- **목표**: Constant Velocity → LSTM trajectory prediction
- **이유**: 차선 변경, braking 예측 가능
- **구현**: Section 5.2 참조
- **우선순위**: **P1**
- **예상 시간**: 3-5일 구현 + 데이터 수집 + 학습 = 2주
- **파일**:
  - `Assets/Scripts/Data/TrajectoryRecorder.cs` (Unity 데이터 수집)
  - `python/src/models/prediction/lstm_predictor.py`
  - `python/src/training/train_prediction.py`
  - `Assets/Scripts/Inference/PredictionInference.cs`

#### 4. Camera Perception (CNN Encoder)
- **목표**: Ground Truth → Camera (84×84) input
- **이유**: 실제 센서 모방, Sim-to-Real 준비
- **구현**: Section 5.3 참조
- **우선순위**: **P1**
- **예상 시간**: 1-2일 구현 + 5-10M steps = 2-4주
- **파일**:
  - `Assets/Scripts/Agents/E2EDrivingAgentCamera.cs`
  - `python/src/models/planning/encoder.py` (Level 2 추가)
  - `python/configs/planning/camera_ppo.yaml`

### 8.3 장기 구현 (Phase 7, 4-8주+)

#### 5. BEV Representation
- **목표**: 단일 카메라 → BEV grid (50×50)
- **이유**: Spatial reasoning, 곡선/교차로 이해
- **구현**: LSS (Lift-Splat-Shoot) 방식
- **우선순위**: **P2 (연구)**
- **예상 시간**: 1-2주 구현 + 10-15M steps = 4-8주
- **파일**:
  - `python/src/models/perception/lss_bev.py`
  - `python/src/training/train_bev.py`

#### 6. World Model (Dreamer-v3)
- **목표**: Model-based RL, 샘플 효율 향상
- **이유**: 학습 속도 2-3배 향상 (이론적)
- **우선순위**: **P2 (실험적)**
- **예상 시간**: 2-3주 구현 + 학습
- **참고**: DreamerV3 PyTorch 구현 활용

#### 7. GAIL/Hybrid RL+IL
- **목표**: nuPlan expert demo로 초기화 → RL fine-tuning
- **이유**: 학습 안정성, human-likeness
- **우선순위**: **P1-P2**
- **예상 시간**: 1주 데이터 준비 + 3-5M steps = 2-3주
- **파일**:
  - `python/configs/planning/gail.yaml` (기존 활용)
  - `python/src/data/nuplan_expert.py` (expert demo 추출)

### 8.4 구현 순서 권장 (Gantt Chart)

```
Week 1-2:   [Trajectory Output] + [Route Planning]
            └─> Stage 5B 완료, A* pathfinding 통합

Week 3-4:   [Prediction Module]
            └─> LSTM predictor 학습, Unity 통합

Week 5-8:   [Camera Perception] (optional, parallel)
            └─> CNN encoder 추가, Level 2 학습

Week 9-12:  [BEV Representation] (research phase)
            └─> LSS 구현, BEV grid 학습

Week 13+:   [World Model] / [GAIL] (advanced)
            └─> Dreamer-v3 또는 GAIL 실험
```

### 8.5 최종 권장 사항

**단기 (1-2주)**:
1. **Trajectory Output 추가** → 즉시 시작 (P0)
2. **Route Planning 추가** → 병렬 진행 (P0)

**중기 (2-4주)**:
3. **Prediction Module** → Trajectory 이후 (P1)
4. **Camera Perception** → 선택적 (P1, 실험적)

**장기 (2-3개월)**:
5. **BEV Representation** → Phase 7 연구 (P2)
6. **GAIL/World Model** → 고급 주제 (P2)

**현실적 목표**:
- Tesla FSD 수준 달성은 불가능 (하드웨어/데이터 gap)
- **학술 연구 수준의 E2E Pipeline** 구축은 가능
- nuPlan benchmark에서 **상위 30-50% 성능** 목표
- RTX 4090 활용 극대화: **50-100M params** 모델까지

**성공 기준**:
- Trajectory planning 성공률: > 80%
- nuPlan closed-loop score: > 50
- Collision rate: < 5%
- Inference latency: < 100ms

---

**Last Updated**: 2026-01-30
**Next Action**: Trajectory Output 구현 시작 (Section 5.1 코드 작성)
**Document Status**: Complete - Ready for Implementation
