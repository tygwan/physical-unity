# Training Experiments Archive

This directory contains all training experiments for the E2E Driving Agent project.

## Directory Structure

```
experiments/
├── early_experiments/      # Initial experiments (3DBot, curriculum v1-v9)
├── v10g_lane_keeping/      # v10g: Lane keeping focus
├── v11_sparse_overtake/    # v11: Sparse overtaking
├── v12_phaseA_dense_overtaking/  # Phase A: Dense traffic overtaking
├── v12_phaseB_decision_learning/ # Phase B: Decision learning
├── v12_phaseC_multi_npc/   # Phase C: Multi-NPC interaction
├── v12_phaseE_curved_roads/ # Phase E: Curved road handling
├── v12_phaseF_multi_lane/  # Phase F: Multi-lane navigation
├── v12_phaseG_intersection/ # Phase G: Intersection navigation
└── failed_experiments/     # Archived failed experiments
```

## Training Flow

```
Phase A (2M) ──> Phase B (2M) ──> Phase C (4M) ──> Phase E (6M) ──> Phase F (6M) ──> Phase G (6M)
   │                │                 │                │               │               │
   └─ Dense traffic └─ Decisions     └─ Multi-NPC    └─ Curves      └─ Lanes       └─ Intersections
```

## Quick Commands

### Start Training
```bash
# Phase A (start from scratch)
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseA.yaml --run-id=v12_phaseA

# Subsequent phases (initialize from previous)
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseB.yaml \
  --run-id=v12_phaseB --initialize-from=results/v12_phaseA/E2EDrivingAgent
```

### Monitor Training
```bash
tensorboard --logdir=experiments/
```

## Backup Policy

- **NEVER run `git clean -fd`** in this repository
- Commit checkpoints after each phase completion
- Use `git lfs` for large .onnx files if needed
