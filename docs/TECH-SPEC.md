# Technical Specification

## Autonomous Driving ML Platform

**Version**: 1.0
**Last Updated**: 2026-01-22
**Author**: Physical AI Team

---

## 1. System Architecture

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         AUTONOMOUS DRIVING ML PLATFORM                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                        SIMULATION LAYER (Unity)                           │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │  Ego Vehicle│  │  Traffic    │  │  Sensors    │  │   Environment   │  │   │
│  │  │  Controller │  │  Simulation │  │  (AWSIM)    │  │   (Roads/Maps)  │  │   │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘  │   │
│  │         │                │                │                   │           │   │
│  │         └────────────────┴────────────────┴───────────────────┘           │   │
│  │                                   │                                       │   │
│  │                          [ML-Agents Interface]                            │   │
│  │                                   │                                       │   │
│  └───────────────────────────────────┼───────────────────────────────────────┘   │
│                                      │                                           │
│                        ┌─────────────┼─────────────┐                            │
│                        │  ROS2 Bridge (Optional)   │                            │
│                        │   TCP/UDP Communication   │                            │
│                        └─────────────┼─────────────┘                            │
│                                      │                                           │
│  ┌───────────────────────────────────┼───────────────────────────────────────┐   │
│  │                        ML TRAINING LAYER (Python)                         │   │
│  │                                   │                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────────────┐ │   │
│  │  │                         MODEL PIPELINE                               │ │   │
│  │  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────────────────┐   │ │   │
│  │  │  │ Perception  │ → │ Prediction  │ → │       Planning          │   │ │   │
│  │  │  │             │   │             │   │   (PRIMARY FOCUS)       │   │ │   │
│  │  │  │ - 3D Detect │   │ - Trajectory│   │ - PPO / SAC (RL)        │   │ │   │
│  │  │  │ - BEV Enc.  │   │ - Behavior  │   │ - BC / GAIL (IL)        │   │ │   │
│  │  │  │ (Pre-train) │   │ (Baseline)  │   │ - Hybrid (CIMRL)        │   │ │   │
│  │  │  └─────────────┘   └─────────────┘   └─────────────────────────┘   │ │   │
│  │  └─────────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                           │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │   │
│  │  │  PyTorch     │  │  ML-Agents   │  │  MLflow/W&B  │  │  TensorBoard │  │   │
│  │  │  2.0+        │  │  Python API  │  │  Tracking    │  │  Monitoring  │  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ┌───────────────────────────────────────────────────────────────────────────┐   │
│  │                           DATA LAYER                                       │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │   │
│  │  │   nuPlan    │  │   Waymo     │  │   highD     │  │   Simulation    │  │   │
│  │  │  (Primary)  │  │   Motion    │  │   Highway   │  │   Generated     │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘  │   │
│  └───────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Overview

| Component | Technology | Purpose |
|-----------|------------|---------|
| Simulation | Unity 2023.2+ | 물리 기반 시뮬레이션 환경 |
| Sensors | AWSIM | LiDAR, Camera, Radar 시뮬레이션 |
| Bridge | ros2-for-unity | Unity-ROS2 통신 |
| Training | PyTorch 2.0+ | 딥러닝 모델 학습 |
| RL Framework | ML-Agents 3.0 | Unity RL 학습 인터페이스 |
| Experiment | MLflow / W&B | 실험 추적 및 관리 |

---

## 2. Simulation Environment

### 2.1 Unity Scene Structure

```
Scenes/
├── Urban/
│   ├── CityBlock_01.unity      # 도심 블록
│   ├── Intersection_01.unity   # 교차로
│   └── Parking_01.unity        # 주차장
├── Highway/
│   ├── Straight_01.unity       # 직선 고속도로
│   ├── Merge_01.unity          # 합류 구간
│   └── Exit_01.unity           # 출구 구간
└── Training/
    ├── SimpleTrack.unity       # 단순 트랙 (RL 초기 학습)
    └── ComplexTrack.unity      # 복잡 트랙 (고급 학습)
```

### 2.2 Vehicle Model

```csharp
// Assets/Scripts/Agents/VehicleAgent.cs
public class VehicleAgent : Agent
{
    // Vehicle Parameters
    [Header("Vehicle Specs")]
    public float maxSpeed = 30f;           // m/s (108 km/h)
    public float maxAcceleration = 4f;     // m/s²
    public float maxDeceleration = 8f;     // m/s²
    public float maxSteeringAngle = 30f;   // degrees

    // Observation Space (~140 dimensions)
    // - Ego state: 8D
    // - Route info: 30D
    // - Surrounding vehicles: 40D
    // - (Optional) BEV features: 64D

    // Action Space (Continuous)
    // - Acceleration: [-4.0, 2.0] m/s²
    // - Steering: [-0.5, 0.5] rad
}
```

### 2.3 Sensor Configuration

```yaml
# AWSIM Sensor Config
sensors:
  lidar:
    type: VLP-32C
    channels: 32
    range: 120m
    points_per_second: 1.2M
    rotation_rate: 20Hz

  camera:
    front:
      resolution: [1920, 1080]
      fov: 90
      fps: 30
    surround:
      count: 4
      resolution: [1280, 720]
      fov: 120

  radar:
    type: Continental_ARS548
    range: 300m
    fov: 120
```

---

## 3. ML Model Architecture

### 3.1 Perception Module (Simplified)

> **전략**: Planning 집중을 위해 Pre-trained 모델 활용 또는 Ground Truth 직접 사용

```python
# python/src/models/perception/detector.py

class PerceptionModule:
    """
    Options:
    1. Ground Truth Mode (시뮬레이션)
    2. Pre-trained Model (MMDetection3D, OpenPCDet)
    3. Simple BEV Encoder (직접 구현)
    """

    def __init__(self, mode: str = "ground_truth"):
        self.mode = mode
        if mode == "pretrained":
            self.detector = self._load_pretrained()

    def detect(self, lidar_points, camera_images=None):
        """
        Returns:
            List[DetectedObject]: position, velocity, size, class
        """
        if self.mode == "ground_truth":
            return self._get_ground_truth()
        return self.detector(lidar_points)
```

### 3.2 Prediction Module (Simplified)

> **전략**: nuPlan baseline predictor 또는 Constant Velocity Model 사용

```python
# python/src/models/prediction/predictor.py

class PredictionModule:
    """
    Options:
    1. Constant Velocity Model (Baseline)
    2. nuPlan Baseline Predictor
    3. (Optional) Transformer-based Predictor
    """

    def __init__(self, mode: str = "constant_velocity"):
        self.mode = mode
        self.horizon = 5.0  # seconds
        self.dt = 0.1       # 100ms steps

    def predict(self, detected_objects, ego_state):
        """
        Returns:
            Dict[object_id, np.ndarray]: future trajectories (50 timesteps x 2)
        """
        trajectories = {}
        for obj in detected_objects:
            if self.mode == "constant_velocity":
                traj = self._constant_velocity(obj)
            trajectories[obj.id] = traj
        return trajectories
```

### 3.3 Planning Module (PRIMARY FOCUS)

#### 3.3.1 Observation Encoder

```python
# python/src/models/planning/encoder.py

import torch
import torch.nn as nn

class ObservationEncoder(nn.Module):
    """
    Encodes multi-modal observations for Planning

    Input:
        - ego_state: [batch, 8] (x, y, vx, vy, cos_h, sin_h, ax, ay)
        - route_info: [batch, 30] (10 waypoints x 3: x, y, dist)
        - surrounding: [batch, 8, 5] (8 vehicles x 5 features)

    Output:
        - encoded: [batch, 256] (latent representation)
    """

    def __init__(self, ego_dim=8, route_dim=30, surr_dim=40, hidden_dim=256):
        super().__init__()

        # Ego state encoder
        self.ego_encoder = nn.Sequential(
            nn.Linear(ego_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Route encoder
        self.route_encoder = nn.Sequential(
            nn.Linear(route_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )

        # Surrounding vehicles encoder (with attention)
        self.surr_encoder = nn.Sequential(
            nn.Linear(surr_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(64 + 64 + 128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, ego_state, route_info, surrounding):
        ego_feat = self.ego_encoder(ego_state)
        route_feat = self.route_encoder(route_info)
        surr_feat = self.surr_encoder(surrounding.flatten(1))

        combined = torch.cat([ego_feat, route_feat, surr_feat], dim=-1)
        return self.fusion(combined)
```

#### 3.3.2 Policy Network (Actor-Critic)

```python
# python/src/models/planning/policy.py

class PlanningPolicy(nn.Module):
    """
    Actor-Critic Policy for Motion Planning

    Actor Output: [acceleration, steering] (continuous)
    Critic Output: state value V(s)
    """

    def __init__(self, obs_dim=256, action_dim=2, hidden_dim=256):
        super().__init__()

        # Shared encoder
        self.encoder = ObservationEncoder()

        # Actor (Policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic (Value)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs):
        encoded = self.encoder(*obs)

        actor_hidden = self.actor(encoded)
        action_mean = self.action_mean(actor_hidden)
        action_std = self.action_log_std.exp()

        value = self.critic(encoded)

        return action_mean, action_std, value
```

#### 3.3.3 Reward Function

```python
# python/src/models/planning/reward.py

class RewardFunction:
    """
    Composite Reward Function for Autonomous Driving
    """

    def __init__(self, config):
        self.weights = config.get('reward_weights', {
            'progress': 1.0,
            'goal_reached': 10.0,
            'collision': -10.0,
            'near_collision': -0.5,
            'off_road': -5.0,
            'jerk': -0.1,
            'lateral_acc': -0.05,
            'steering_rate': -0.02,
            'lane_keeping': 0.5,
            'speed_limit': -0.5,
            'traffic_light': -5.0
        })

    def compute(self, state, action, next_state, info):
        reward = 0.0

        # Progress reward (moving towards goal)
        reward += self.weights['progress'] * info['progress']

        # Goal reached bonus
        if info['goal_reached']:
            reward += self.weights['goal_reached']

        # Safety penalties
        if info['collision']:
            reward += self.weights['collision']
            return reward, True  # Episode done

        if info['ttc'] < 2.0:  # Time-to-collision < 2 seconds
            reward += self.weights['near_collision']

        if info['off_road']:
            reward += self.weights['off_road']

        # Comfort rewards
        jerk = abs(next_state['acceleration'] - state['acceleration']) / 0.1
        reward += self.weights['jerk'] * jerk

        reward += self.weights['lateral_acc'] * abs(info['lateral_acceleration'])
        reward += self.weights['steering_rate'] * abs(action[1] - state['prev_steering'])

        # Traffic rules
        if info['in_lane']:
            reward += self.weights['lane_keeping']

        if info['speed'] > info['speed_limit']:
            reward += self.weights['speed_limit']

        if info['traffic_violation']:
            reward += self.weights['traffic_light']

        return reward, False
```

---

## 4. Training Pipeline

### 4.1 RL Training (PPO/SAC)

```yaml
# python/configs/planning/ppo.yaml

algorithm: PPO
environment:
  name: ADPlanning-v0
  num_envs: 8
  max_episode_steps: 1000

network:
  encoder:
    ego_dim: 8
    route_dim: 30
    surr_dim: 40
    hidden_dim: 256
  policy:
    hidden_layers: [256, 256]
    activation: relu

ppo:
  clip_ratio: 0.2
  vf_coef: 0.5
  entropy_coef: 0.01
  learning_rate: 3e-4
  batch_size: 2048
  minibatch_size: 64
  epochs_per_update: 10
  gamma: 0.99
  gae_lambda: 0.95

training:
  total_steps: 10_000_000
  eval_interval: 50_000
  checkpoint_interval: 100_000
  log_interval: 10_000
```

```yaml
# python/configs/planning/sac.yaml

algorithm: SAC
environment:
  name: ADPlanning-v0
  num_envs: 1  # Off-policy can use single env
  max_episode_steps: 1000

network:
  encoder:
    ego_dim: 8
    route_dim: 30
    surr_dim: 40
    hidden_dim: 256
  actor:
    hidden_layers: [256, 256]
  critic:
    hidden_layers: [256, 256]

sac:
  learning_rate_actor: 3e-4
  learning_rate_critic: 3e-4
  learning_rate_alpha: 3e-4
  tau: 0.005
  gamma: 0.99
  buffer_size: 1_000_000
  batch_size: 256
  gradient_steps: 1
  auto_entropy: true

training:
  total_steps: 5_000_000
  eval_interval: 50_000
  checkpoint_interval: 100_000
```

### 4.2 IL Training (BC/GAIL)

```yaml
# python/configs/planning/bc.yaml

algorithm: BehavioralCloning
dataset:
  name: nuplan
  split: train
  scenario_filter:
    - urban
    - highway
  max_scenarios: 10000

network:
  encoder:
    ego_dim: 8
    route_dim: 30
    surr_dim: 40
    hidden_dim: 256
  policy:
    hidden_layers: [256, 256]

bc:
  learning_rate: 1e-4
  batch_size: 256
  loss: mse  # or nll for gaussian policy

training:
  epochs: 100
  eval_interval: 5
  checkpoint_interval: 10
```

```yaml
# python/configs/planning/gail.yaml

algorithm: GAIL
dataset:
  name: nuplan
  split: train
  max_scenarios: 10000

network:
  encoder:
    ego_dim: 8
    route_dim: 30
    surr_dim: 40
    hidden_dim: 256
  policy:
    hidden_layers: [256, 256]
  discriminator:
    hidden_layers: [256, 256]

gail:
  disc_learning_rate: 1e-4
  policy_learning_rate: 3e-4
  disc_updates_per_step: 2
  use_ppo: true  # Use PPO for policy update

training:
  total_steps: 5_000_000
  expert_batch_size: 256
```

### 4.3 Hybrid Training (CIMRL)

```yaml
# python/configs/planning/cimrl.yaml

algorithm: CIMRL
# Phase 1: Imitation Learning Warmup
phase_1:
  algorithm: BehavioralCloning
  epochs: 50
  checkpoint: models/planning/bc_warmup.pt

# Phase 2: RL Fine-tuning
phase_2:
  algorithm: PPO
  init_from: phase_1_checkpoint
  total_steps: 5_000_000
  # Modified reward to include IL regularization
  imitation_weight: 0.1  # Decreases over time

training:
  phase_1_epochs: 50
  phase_2_steps: 5_000_000
  imitation_decay: 0.99  # Per 100k steps
```

---

## 5. Data Pipeline

### 5.1 Dataset Structure

```
datasets/
├── raw/
│   ├── nuplan/
│   │   ├── mini/           # 50GB (testing)
│   │   └── full/           # 1TB+ (production)
│   ├── waymo_motion/
│   │   └── v1.2/
│   └── highd/
│
├── processed/
│   ├── scenarios/          # Extracted driving scenarios
│   │   ├── urban/
│   │   ├── highway/
│   │   └── intersection/
│   └── features/           # Pre-computed features
│       ├── ego_states.parquet
│       └── predictions.parquet
│
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

### 5.2 Data Processing Pipeline

```python
# python/src/data/processor.py

class DataProcessor:
    """
    Unified data processing for multiple datasets
    """

    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config

        if dataset_name == "nuplan":
            self.loader = NuPlanLoader(config)
        elif dataset_name == "waymo":
            self.loader = WaymoLoader(config)

    def process_scenario(self, scenario_id):
        """
        Extract:
        - Ego trajectory
        - Surrounding agent trajectories
        - Map information
        - Traffic light states
        """
        raw_data = self.loader.load(scenario_id)

        processed = {
            'ego_trajectory': self._process_ego(raw_data),
            'agents': self._process_agents(raw_data),
            'map': self._process_map(raw_data),
            'traffic_lights': self._process_traffic(raw_data)
        }

        return processed

    def create_training_samples(self, scenario):
        """
        Create (observation, action) pairs for IL
        """
        samples = []
        for t in range(len(scenario['ego_trajectory']) - 1):
            obs = self._create_observation(scenario, t)
            action = self._create_action(scenario, t)
            samples.append((obs, action))
        return samples
```

---

## 6. Evaluation Framework

### 6.1 Metrics

| Category | Metric | Formula | Target |
|----------|--------|---------|--------|
| Safety | Collision Rate | collisions / total_episodes | < 5% |
| Safety | Time-to-Collision | min(TTC) over episode | > 2s |
| Progress | Route Completion | distance_traveled / total_distance | > 85% |
| Progress | Goal Reached Rate | goals_reached / total_episodes | > 80% |
| Comfort | Jerk | d(acceleration)/dt | < 2 m/s³ |
| Comfort | Lateral Acceleration | max(lat_acc) | < 3 m/s² |
| Efficiency | Travel Time | actual_time / optimal_time | < 1.2x |
| Latency | Inference Time | model_forward_pass | < 50ms |

### 6.2 Benchmark Integration

```python
# python/src/evaluation/benchmark.py

class NuPlanBenchmark:
    """
    nuPlan Closed-loop Simulation Benchmark

    Metrics:
    - No at-fault collision (ADE)
    - Drivable area compliance
    - Progress along route
    - Comfort metrics
    """

    def __init__(self, model, scenarios):
        self.model = model
        self.scenarios = scenarios
        self.simulator = NuPlanSimulator()

    def run_closed_loop(self, scenario_id):
        """
        Run closed-loop simulation where model controls the ego vehicle
        """
        scenario = self.scenarios[scenario_id]
        state = self.simulator.reset(scenario)

        history = []
        done = False

        while not done:
            obs = self._create_observation(state)
            action = self.model.predict(obs)
            state, reward, done, info = self.simulator.step(action)
            history.append({'state': state, 'action': action, 'info': info})

        return self._compute_metrics(history)

    def _compute_metrics(self, history):
        return {
            'collision_rate': self._collision_rate(history),
            'progress': self._route_progress(history),
            'comfort': self._comfort_score(history),
            'drivable_compliance': self._drivable_compliance(history)
        }
```

---

## 7. Unity-ROS2 Integration

### 7.1 Option A: ros2-for-unity

```yaml
# Pros:
#   - Native ROS2 DDS communication
#   - Low latency (~1ms)
#   - AWSIM compatible
# Cons:
#   - Complex initial setup
#   - Platform-specific builds

# Installation
ros2_for_unity:
  ros2_distro: humble
  platforms:
    - windows-x86_64
  packages:
    - std_msgs
    - sensor_msgs
    - geometry_msgs
    - nav_msgs
    - autoware_auto_msgs  # For AWSIM compatibility
```

### 7.2 Option B: Unity Robotics Hub

```yaml
# Pros:
#   - Official Unity support
#   - Easy setup
#   - Good documentation
# Cons:
#   - TCP-based (higher latency ~5-10ms)
#   - Separate ROS2 node required

# Components
unity_robotics:
  ros_tcp_connector: 0.7.0
  ros_tcp_endpoint: 0.7.0
  urdf_importer: 0.5.0
```

### 7.3 Communication Topics

```yaml
# ROS2 Topics for AD Platform

publishers:
  # Sensor data (Unity → Python)
  /sensor/lidar/points: sensor_msgs/PointCloud2
  /sensor/camera/front: sensor_msgs/Image
  /sensor/radar/targets: sensor_msgs/PointCloud2
  /vehicle/state: geometry_msgs/PoseStamped

subscribers:
  # Control commands (Python → Unity)
  /vehicle/cmd: geometry_msgs/Twist
  /planning/trajectory: nav_msgs/Path

services:
  /simulation/reset: std_srvs/Empty
  /simulation/pause: std_srvs/SetBool
```

---

## 8. Experiment Tracking

### 8.1 MLflow Configuration

```python
# python/src/training/experiment.py

import mlflow

class ExperimentTracker:
    def __init__(self, experiment_name: str):
        mlflow.set_tracking_uri("file:./experiments/mlruns")
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str, config: dict):
        self.run = mlflow.start_run(run_name=run_name)
        mlflow.log_params(flatten_dict(config))

    def log_metrics(self, metrics: dict, step: int):
        mlflow.log_metrics(metrics, step=step)

    def log_model(self, model, artifact_path: str):
        mlflow.pytorch.log_model(model, artifact_path)

    def end_run(self):
        mlflow.end_run()
```

### 8.2 Experiment Structure

```
experiments/
├── mlruns/                    # MLflow tracking
│   ├── 0/                     # Default experiment
│   └── 1/                     # Planning experiments
│       ├── meta.yaml
│       └── runs/
│           ├── run_001/
│           │   ├── params/
│           │   ├── metrics/
│           │   └── artifacts/
│           └── run_002/
│
├── configs/                   # Experiment configurations
│   ├── baseline_ppo.yaml
│   ├── baseline_sac.yaml
│   └── ablation_reward.yaml
│
├── logs/                      # TensorBoard logs
│   └── planning_v1/
│
└── checkpoints/              # Model checkpoints
    └── planning/
        ├── ppo_best.pt
        └── gail_final.pt
```

---

## 9. Deployment

### 9.1 ONNX Export

```python
# python/src/models/planning/export.py

def export_to_onnx(model, output_path: str):
    """
    Export PyTorch model to ONNX for Unity inference
    """
    model.eval()

    # Dummy inputs
    ego_state = torch.randn(1, 8)
    route_info = torch.randn(1, 30)
    surrounding = torch.randn(1, 8, 5)

    torch.onnx.export(
        model,
        (ego_state, route_info, surrounding),
        output_path,
        input_names=['ego_state', 'route_info', 'surrounding'],
        output_names=['action_mean', 'action_std'],
        dynamic_axes={
            'ego_state': {0: 'batch'},
            'route_info': {0: 'batch'},
            'surrounding': {0: 'batch'},
            'action_mean': {0: 'batch'},
            'action_std': {0: 'batch'}
        },
        opset_version=17
    )
```

### 9.2 Unity Inference

```csharp
// Assets/Scripts/Inference/PlanningInference.cs

using Unity.Barracuda;

public class PlanningInference : MonoBehaviour
{
    public NNModel planningModel;
    private IWorker worker;

    void Start()
    {
        var model = ModelLoader.Load(planningModel);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
    }

    public Vector2 GetAction(float[] egoState, float[] routeInfo, float[] surrounding)
    {
        var egoTensor = new Tensor(1, 8, egoState);
        var routeTensor = new Tensor(1, 30, routeInfo);
        var surrTensor = new Tensor(1, 40, surrounding);

        worker.Execute(new Dictionary<string, Tensor> {
            {"ego_state", egoTensor},
            {"route_info", routeTensor},
            {"surrounding", surrTensor}
        });

        var output = worker.PeekOutput("action_mean");
        return new Vector2(output[0], output[1]);  // acceleration, steering
    }
}
```

---

## 10. Performance Optimization

### 10.1 Training Optimization

| Technique | Expected Speedup | Implementation |
|-----------|------------------|----------------|
| Multi-GPU | 2-4x | PyTorch DDP |
| Mixed Precision | 1.5-2x | torch.cuda.amp |
| Vectorized Envs | 4-8x | ML-Agents parallel |
| Async Data Loading | 1.2x | DataLoader workers |

### 10.2 Inference Optimization

| Technique | Target Latency | Notes |
|-----------|---------------|-------|
| TensorRT | < 10ms | NVIDIA optimized |
| ONNX Runtime | < 30ms | Cross-platform |
| Barracuda (Unity) | < 50ms | Unity native |

---

## Appendix A: API Reference

### A.1 Environment API

```python
class ADPlanningEnv:
    """
    Gymnasium-compatible AD Planning Environment
    """

    def reset(self) -> Observation:
        """Reset environment to initial state"""

    def step(self, action: Action) -> Tuple[Observation, float, bool, dict]:
        """Execute action and return (obs, reward, done, info)"""

    def render(self, mode: str = "human"):
        """Render current state"""
```

### A.2 Model API

```python
class PlanningModel:
    """
    Base class for Planning models
    """

    def predict(self, obs: Observation) -> Action:
        """Predict action from observation"""

    def train(self, batch: Batch):
        """Update model parameters"""

    def save(self, path: str):
        """Save model checkpoint"""

    def load(self, path: str):
        """Load model checkpoint"""
```
