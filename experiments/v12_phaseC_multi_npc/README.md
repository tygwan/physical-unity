# Phase C: Multi-NPC Interaction

## Training Summary

| Item | Value |
|------|-------|
| Run ID | v12_phaseC |
| Status | Needs Retraining |
| Total Steps | 4,000,000 |
| Target Reward | +850 |
| Training Time | ~1 hour |
| Initialize From | Phase B checkpoint |

## Objective

Handle complex multi-vehicle scenarios with varying NPC behaviors and traffic densities.

## Config File

`config/vehicle_ppo_v12_phaseC.yaml`

## Key Parameters

- **max_steps**: 4,000,000
- **batch_size**: 2048
- **buffer_size**: 20480
- **learning_rate**: 3e-4
- **NPC count**: 3-5 vehicles

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Pending retraining* | - | - | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseC.yaml \
  --run-id=v12_phaseC_recovery --force \
  --initialize-from=results/v12_phaseB_recovery/E2EDrivingAgent
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +750
- Multi-vehicle collision rate < 5%
- Traffic flow adaptation > 85%
