# Phase E: Curved Roads

## Training Summary

| Item | Value |
|------|-------|
| Run ID | v12_phaseE |
| Status | Needs Retraining |
| Total Steps | 6,000,000 |
| Target Reward | +1100 |
| Training Time | ~1 hour |
| Initialize From | Phase C checkpoint |

## Objective

Master curved road navigation while maintaining safe driving behaviors learned in previous phases.

## Parallel Training Environment

| 항목 | 값 |
|------|-----|
| Training Areas | **16개** (일렬 배치) |
| 각 Area | 독립적 곡선 도로 + NPC |
| Area 간격 | 100m |
| 동시 학습 | 16 에이전트 |

## Config File

`config/vehicle_ppo_v12_phaseE.yaml`

## Key Parameters

- **max_steps**: 6,000,000
- **batch_size**: 4096
- **buffer_size**: 40960
- **learning_rate**: 3e-4
- **Road curvature**: Variable (gentle to sharp)
- **parallel_envs**: 16 Training Areas
- **time_scale**: 20x

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Pending retraining* | - | - | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseE.yaml \
  --run-id=v12_phaseE_recovery --force \
  --initialize-from=results/v12_phaseC_recovery/E2EDrivingAgent
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +1000
- Curve navigation success > 95%
- Lane keeping on curves > 90%

## Notes

Phase D was skipped - its features were merged into Phase E.
