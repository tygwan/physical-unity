---
name: experiment
description: ML 실험 관리 스킬. 실험 생성, 실행, 비교, 추적을 지원합니다. "/experiment" 명령으로 호출.
---

# ML Experiment Management Skill

자율주행 ML 실험의 전체 라이프사이클을 관리하는 스킬입니다.

## Commands

### /experiment create [name]
새 실험 생성

```bash
# Example
/experiment create planning_ppo_v2
```

### /experiment list
실험 목록 확인

### /experiment status [name]
실험 상태 확인

### /experiment compare [exp1] [exp2]
실험 비교

## Workflow

### 1. Create Experiment
```bash
# Create experiment config
experiments/configs/planning_ppo_v2.yaml

# Content:
experiment:
  name: planning_ppo_v2
  description: "PPO with modified reward"

algorithm: PPO
hyperparameters:
  learning_rate: 3e-4
  batch_size: 2048
```

### 2. Start Training
```bash
python python/src/training/train_rl.py \
  --config experiments/configs/planning_ppo_v2.yaml \
  --run-id planning_ppo_v2
```

### 3. Monitor Progress
```bash
# MLflow UI
mlflow ui --port 5000

# TensorBoard
tensorboard --logdir=experiments/logs/
```

### 4. Compare Results
```bash
mlflow runs compare \
  --experiment-name planning \
  --metric collision_rate \
  --metric progress_score
```

## Experiment Structure

```
experiments/
├── configs/
│   └── planning_ppo_v2.yaml
├── logs/
│   └── planning_ppo_v2/
│       ├── events.out.tfevents.*
│       └── train.log
├── checkpoints/
│   └── planning_ppo_v2/
│       ├── checkpoint_100000.pt
│       ├── checkpoint_200000.pt
│       └── best.pt
└── mlruns/
    └── 1/
        └── runs/
```

## Metrics Tracked

| Metric | Description | Target |
|--------|-------------|--------|
| collision_rate | Collision frequency | < 5% |
| progress_score | Route completion | > 85% |
| mean_reward | Episode reward | Maximize |
| policy_loss | Policy network loss | Minimize |
| value_loss | Value network loss | Minimize |

## Integration

- **ad-experiment-manager agent**: Detailed experiment management
- **model-trainer agent**: Training execution
- **benchmark-evaluator agent**: Evaluation
