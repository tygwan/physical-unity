# Phase B: Decision Learning

## Training Summary

| Item | Value |
|------|-------|
| Run ID | v12_phaseB |
| Status | Needs Retraining |
| Total Steps | 2,000,000 |
| Target Reward | +700 |
| Training Time | ~20 minutes |
| Initialize From | Phase A checkpoint |

## Objective

Build on Phase A to learn decision-making in complex traffic situations - when to overtake, when to wait, lane selection.

## Config File

`config/vehicle_ppo_v12_phaseB.yaml`

## Key Parameters

- **max_steps**: 2,000,000
- **batch_size**: 2048
- **buffer_size**: 20480
- **learning_rate**: 3e-4

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| *Pending retraining* | - | - | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseB.yaml \
  --run-id=v12_phaseB_recovery --force \
  --initialize-from=results/v12_phaseA_recovery/E2EDrivingAgent
```

## TensorBoard

```bash
tensorboard --logdir=logs/
```

## Success Criteria

- Mean reward > +600
- Decision accuracy > 80%
- Safe lane changes > 90%
