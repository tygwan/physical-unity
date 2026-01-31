---
name: train
description: 모델 학습 스킬. RL/IL 학습 시작 및 모니터링을 지원합니다. "/train" 명령으로 호출.
---

# Model Training Skill

자율주행 Planning 모델의 학습을 관리하는 스킬입니다.

## Commands

### /train ppo [options]
PPO 알고리즘으로 학습

```bash
/train ppo --run-id ppo_v1
/train ppo --config custom.yaml
```

### /train sac [options]
SAC 알고리즘으로 학습

### /train bc [options]
Behavioral Cloning 학습

### /train gail [options]
GAIL 학습

### /train status
현재 학습 상태 확인

## Algorithms

| Algorithm | Type | Best For | Command |
|-----------|------|----------|---------|
| PPO | RL | Stable training | /train ppo |
| SAC | RL | Sample efficiency | /train sac |
| BC | IL | Expert imitation | /train bc |
| GAIL | IL | No reward needed | /train gail |
| CIMRL | Hybrid | Best of both | /train hybrid |

## Workflow

### 1. Prepare Config
```yaml
# python/configs/planning/ppo.yaml
algorithm: PPO
environment:
  name: ADPlanning-v0
  num_envs: 8

ppo:
  learning_rate: 3e-4
  batch_size: 2048
  gamma: 0.99

training:
  total_steps: 10_000_000
```

### 2. Start Training

**PPO/SAC (RL)**
```bash
python python/src/training/train_rl.py \
  --config python/configs/planning/ppo.yaml \
  --run-id ppo_v1
```

**BC (IL)**
```bash
python python/src/training/train_bc.py \
  --config python/configs/planning/bc.yaml \
  --data datasets/processed/nuplan/ \
  --run-id bc_v1
```

**GAIL**
```bash
python python/src/training/train_gail.py \
  --config python/configs/planning/gail.yaml \
  --expert-data datasets/processed/nuplan/ \
  --run-id gail_v1
```

**Hybrid (CIMRL)**
```bash
# Step 1: BC warmup
python python/src/training/train_bc.py \
  --epochs 50 --output experiments/checkpoints/cimrl_bc/

# Step 2: RL fine-tuning
python python/src/training/train_rl.py \
  --init-from experiments/checkpoints/cimrl_bc/best.pt \
  --run-id cimrl_v1
```

### 3. Monitor
```bash
# GPU usage
nvidia-smi -l 1

# TensorBoard
tensorboard --logdir=experiments/logs/

# MLflow
mlflow ui --port 5000
```

## Training Status

```markdown
## Active Training Runs

| Run | Algorithm | Step | Reward | GPU | ETA |
|-----|-----------|------|--------|-----|-----|
| ppo_v1 | PPO | 350K | 5.2 | 78% | 8h |
| sac_v1 | SAC | 120K | 3.1 | 65% | 12h |

### System
- GPU: RTX 4090 - 78% usage
- VRAM: 18GB / 24GB
- RAM: 45GB / 128GB
```

## Integration

- **model-trainer agent**: Detailed training management
- **ad-experiment-manager agent**: Experiment tracking
