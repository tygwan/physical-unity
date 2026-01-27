# Phase B: Decision Learning

## Training Summary

| Item | Value |
|------|-------|
| Run ID | phase-B |
| Status | Needs Retraining |
| Total Steps | 2,000,000 |
| Target Reward | +700 |
| Training Time | ~20 minutes |
| Initialize From | Phase A checkpoint |

## Objective

Build on Phase A to learn decision-making in complex traffic situations - when to overtake, when to wait, lane selection.

## Parallel Training Environment

| 항목 | 값 |
|------|-----|
| Training Areas | **16개** (일렬 배치) |
| 각 Area | 독립적 도로 + NPC |
| Area 간격 | 100m |
| 동시 학습 | 16 에이전트 |

## Config File

`config/vehicle_ppo_phase-B.yaml`

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
mlagents-learn python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B_recovery --force \
  --initialize-from=results/phase-A_recovery/E2EDrivingAgent
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +600
- Decision accuracy > 80%
- Safe lane changes > 90%
