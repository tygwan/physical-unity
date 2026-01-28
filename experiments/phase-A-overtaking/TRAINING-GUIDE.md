# Phase A: Dense Overtaking - Training Guide

## Quick Start

```bash
cd /c/Users/user/Desktop/dev/physical-unity

# 1. Start Unity (with PhaseA_DenseOvertaking scene, 16 areas)
# 2. Run training from Phase 0 checkpoint
mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml \
  --run-id=phase-A-overtaking \
  --initialize-from=Phase 0_lane_keeping

# 3. Monitor (separate terminal)
tensorboard --logdir=experiments/phase-A-overtaking/logs/
```

---

## Pre-Training Checklist

- [ ] Unity 2023.2+ running
- [ ] PhaseA_DenseOvertaking scene loaded
- [ ] 16 training areas spawned
- [ ] ROS2 bridge on port 5004
- [ ] Phase 0 checkpoint available in experiments/phase-0-foundation/

---

## Training Timeline

| Phase | Steps | Time | Expected Reward |
|-------|-------|------|-----------------|
| Stage 1 | 0-700K | 0-13 min | 250→400 |
| Stage 2 | 700K-1.8M | 13-33 min | 400→600 |
| Stage 3 | 1.8M-2.5M | 33-45 min | 600→950 |
| **Total** | **2.5M** | **45-60 min** | **→950** |

---

## Key Metrics to Monitor

### Critical (Stop if Violated)
- Collision rate exceeds 10%
- Mean reward < 650 by 1.5M steps
- Reward variance > 200 (instability)

### Target (Success if Met)
- Overtaking success > 70%
- Collision rate < 5%
- Final reward ≥ 900
- Goal completion > 90%

---

## Checkpoints

- Auto-saves every 500K steps
- Location: `results/E2EDrivingAgent/`
- Resume: `mlagents-learn ... --resume`

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Won't overtake | Check overtaking reward (+3.0), NPC speed (0.3) |
| Collision rate high | Load checkpoint, increase thresholds |
| Reward not converging | Reduce learning rate (3e-4→2e-4) |
| Connection lost | Restart Unity, training auto-resumes |
| Out of memory | Reduce batch_size (4096→2048) |

---

**Status**: Ready | **Duration**: 45-60 min | **Hardware**: RTX 4090
