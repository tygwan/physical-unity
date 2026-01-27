# Phase A: Dense Overtaking

## Training Summary

| Item | Value |
|------|-------|
| Run ID | v12_phaseA |
| Status | Needs Retraining |
| Total Steps | 2,000,000 |
| Target Reward | +900 |
| Training Time | ~30 minutes |
| Initialize From | None (fresh start) |

## Objective

Train the agent to navigate dense traffic scenarios with slow NPC vehicles, learning basic overtaking maneuvers.

## Parallel Training Environment

| 항목 | 값 |
|------|-----|
| Training Areas | **16개** (일렬 배치) |
| 각 Area | 독립적 도로 + 1 NPC |
| Area 간격 | 100m (X축) |
| 동시 학습 | 16 에이전트 |

## Config File

`config/vehicle_ppo_v12_phaseA.yaml`

## Key Parameters

- **max_steps**: 2,000,000
- **batch_size**: 4096
- **buffer_size**: 40960
- **learning_rate**: 3e-4
- **parallel_envs**: 16 Training Areas
- **time_scale**: 20x

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Pending retraining* | - | - | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseA.yaml \
  --run-id=v12_phaseA_recovery --force
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +800
- Collision rate < 10%
- Successful overtakes > 70%
