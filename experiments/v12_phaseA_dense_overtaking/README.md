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

Train the agent to navigate dense traffic scenarios with multiple NPC vehicles, learning basic overtaking maneuvers.

## Config File

`config/vehicle_ppo_v12_phaseA.yaml`

## Key Parameters

- **max_steps**: 2,000,000
- **batch_size**: 2048
- **buffer_size**: 20480
- **learning_rate**: 3e-4
- **num_envs**: 36 (6x6 training areas)

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
