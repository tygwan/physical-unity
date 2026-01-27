# Phase G: Intersection Navigation

## Training Summary

| Item | Value |
|------|-------|
| Run ID | phase-G |
| Status | In Progress (recovered from 4M) |
| Total Steps | 6,000,000 |
| Target Reward | +1300 |
| Training Time | ~1.5 hours |
| Initialize From | Phase F checkpoint |

## Objective

Navigate intersections safely including turns, yields, and traffic management.

## Parallel Training Environment

| 항목 | 값 |
|------|-----|
| Training Areas | **16개** (일렬 배치) |
| 각 Area | 독립적 교차로 + NPC |
| Area 간격 | 100m |
| 동시 학습 | 16 에이전트 |

## Config File

`config/vehicle_ppo_phase-G.yaml`

## Key Parameters

- **max_steps**: 6,000,000
- **batch_size**: 4096
- **buffer_size**: 40960
- **learning_rate**: 3e-4
- **Intersection types**: T-junction, 4-way, roundabout
- **parallel_envs**: 16 Training Areas
- **time_scale**: 20x

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Checkpoints from 4M onwards will be saved* | - | - | Previous logs lost to git clean |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-G.yaml \
  --run-id=phase-G_recovery --force \
  --initialize-from=results/phase-F_recovery/E2EDrivingAgent
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +1200
- Intersection navigation success > 90%
- Turn execution safety > 95%
- Yield compliance > 98%

## Recovery Notes

TensorBoard logs from 0-3.5M steps were lost due to `git clean -fd`.
Training continues from the last available checkpoint.
