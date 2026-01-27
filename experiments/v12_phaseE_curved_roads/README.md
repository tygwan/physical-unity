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

## Config File

`config/vehicle_ppo_v12_phaseE.yaml`

## Key Parameters

- **max_steps**: 6,000,000
- **batch_size**: 2048
- **buffer_size**: 20480
- **learning_rate**: 3e-4
- **Road curvature**: Variable (gentle to sharp)

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
