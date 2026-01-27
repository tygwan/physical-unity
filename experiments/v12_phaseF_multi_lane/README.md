# Phase F: Multi-Lane Navigation

## Training Summary

| Item | Value |
|------|-------|
| Run ID | v12_phaseF |
| Status | Needs Retraining |
| Total Steps | 6,000,000 |
| Target Reward | +1200 |
| Training Time | ~1 hour |
| Initialize From | Phase E checkpoint |

## Objective

Navigate multi-lane roads with strategic lane selection, merging, and highway-style driving.

## Config File

`config/vehicle_ppo_v12_phaseF.yaml`

## Key Parameters

- **max_steps**: 6,000,000
- **batch_size**: 2048
- **buffer_size**: 20480
- **learning_rate**: 3e-4
- **Lane count**: 2-4 lanes

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Pending retraining* | - | - | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseF.yaml \
  --run-id=v12_phaseF_recovery --force \
  --initialize-from=results/v12_phaseE_recovery/E2EDrivingAgent
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
