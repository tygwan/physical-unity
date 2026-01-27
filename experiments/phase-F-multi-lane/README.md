# Phase F: Multi-Lane Navigation

## Training Summary

| Item | Value |
|------|-------|
| Run ID | phase-F |
| Status | Needs Retraining |
| Total Steps | 6,000,000 |
| Target Reward | +1200 |
| Training Time | ~1 hour |
| Initialize From | Phase E checkpoint |

## Objective

Navigate multi-lane roads with strategic lane selection, merging, and highway-style driving.

## Parallel Training Environment

| 항목 | 값 |
|------|-----|
| Training Areas | **16개** (일렬 배치) |
| 각 Area | 독립적 다차선 도로 + NPC |
| Area 간격 | 100m |
| 동시 학습 | 16 에이전트 |

## Config File

`config/vehicle_ppo_phase-F.yaml`

## Key Parameters

- **max_steps**: 6,000,000
- **batch_size**: 4096
- **buffer_size**: 40960
- **learning_rate**: 3e-4
- **Lane count**: 2-4 lanes
- **parallel_envs**: 16 Training Areas
- **time_scale**: 20x

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Pending retraining* | - | - | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-F.yaml \
  --run-id=phase-F_recovery --force \
  --initialize-from=results/phase-E_recovery/E2EDrivingAgent
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +1100
- Lane change safety > 95%
- Merge success rate > 90%
- Highway driving score > 85%
