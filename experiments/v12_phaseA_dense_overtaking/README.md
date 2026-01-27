# Phase A: Dense Overtaking - Comprehensive Experiment Plan

## Executive Summary

Building on the v10g foundation success (+1018.43 reward, perfect safety), Phase A introduces overtaking capability while maintaining lane-keeping and collision avoidance skills. This phase validates the hypothesis that overtaking can be learned through progressive NPC density and explicit overtaking rewards.

**Status**: Ready for Retraining from v10g Checkpoint
**Checkpoint**: v10g_lane_keeping (8M steps, final reward 1018.43)

---

## Experiment Metadata

| Item | Value |
|------|-------|
| **Run ID** | v12_phaseA_dense_overtaking |
| **Base Configuration** | `vehicle_ppo_v12_phaseA.yaml` |
| **Initialize From** | v10g_lane_keeping (checkpoint transfer) |
| **Total Planned Steps** | 2,500,000 (additional) |
| **Estimated Training Time** | 45-60 minutes |
| **Target Reward** | +950 (accounting for overtaking challenge) |
| **Primary Objective** | Learn overtaking maneuvers in dense traffic |
| **Secondary Objectives** | Maintain v10g safety; Increase NPC density; Improve lane-switching |

---

## Design Rationale

### Why Overtaking After Lane-Keeping?

1. **Foundation First**: v10g established robust lane-keeping (93.6% speed tracking) and collision avoidance (0% collision rate)
2. **Skill Progression**: Overtaking builds naturally on lane-keeping - agent knows how to control vehicle safely
3. **Curriculum Learning**: Progressive NPC density allows agent to learn overtaking before handling complex traffic
4. **Transfer Learning**: Checkpoint initialization accelerates learning of new skill (typical 2-3M steps vs 8M for foundation)

### Key Assumptions

- Agent has learned vehicle dynamics control from v10g
- Collision avoidance patterns can transfer with minimal fine-tuning
- NPC density increase (4→8) is manageable with curriculum
- Overtaking reward (+3.0/successful overtake) provides clear learning signal
- Lane-keeping skills won't degrade with reward restructuring

---

## Curriculum Design: 3-Stage Progressive Density

### Curriculum Philosophy
- **Stage 1 (Easy)**: Single slow NPC, practice basic overtaking mechanics
- **Stage 2 (Medium)**: Multiple NPCs at varying speeds, learn to identify overtaking opportunities
- **Stage 3 (Hard)**: Dense traffic with multiple speed zones, master overtaking in complex scenarios

### Stage Breakdown

#### Stage 1: Skill Acquisition (0-700K steps)
- **NPC Count**: 1-2 vehicles (fixed)
- **NPC Speed**: 30-40% of speed limit (very slow, forces overtaking)
- **Goal Distance**: 80-100m (short routes, repeated practice)
- **Speed Zones**: 1 (single zone)
- **Objective**: Learn basic overtaking maneuver
- **Expected Behavior**: First overtakes by 300K-500K steps, stable by 700K

#### Stage 2: Density Introduction (700K-1.8M steps)
- **NPC Count**: 3-4 vehicles (increasing, threshold 50%)
- **NPC Speed**: 50-70% of speed limit (mixed speeds)
- **Goal Distance**: 120-160m (medium routes)
- **Speed Zones**: 2-3 (multiple speed regions)
- **Objective**: Handle multiple NPCs, identify overtaking opportunities
- **Expected Behavior**: Traffic flow awareness, overtaking sequences, close-proximity maneuvers

#### Stage 3: Complex Scenarios (1.8M-2.5M steps)
- **NPC Count**: 5-6 vehicles (challenging density)
- **NPC Speed**: 40-80% of speed limit (high variation)
- **Goal Distance**: 180-230m (long routes)
- **Speed Zones**: 3-4 (varying speed regions)
- **Objective**: Master overtaking in complex, dynamic traffic
- **Expected Behavior**: Chains of overtakes, handles faster vehicles, plans ahead

---

## Reward Structure

### Components (Updated from v10g)

| Component | v10g | Phase A | Change |
|-----------|------|---------|--------|
| Progress Reward | +1.0/m | +1.0/m | Unchanged |
| **Overtaking Bonus** | N/A | +3.0 | **NEW** |
| Lane-Keeping | +0.02 | +0.01 | Reduced |
| Following Bonus | +0.3 | Removed | Encourages overtaking |
| Collision | -10.0 | -10.0 | Maintained |
| Off-Road | -5.0 | -5.0 | Maintained |

### Expected Reward by Stage

```
Stage 1: 250-350/episode (single NPC practice)
Stage 2: 350-500/episode (multiple NPCs)
Stage 3: 450-650/episode (dense traffic)
Target Final: +950 (accounting for increased complexity)
```

---

## Configuration Details

### Preserved from v10g
- Network: 512 hidden units × 3 layers
- Hyperparameters: batch_size=4096, learning_rate=3e-4
- Observation space: 242D (ego, route, NPCs, environment)
- Action space: 2D continuous (acceleration, steering)

### Training Protocol

```bash
# Train from v10g checkpoint
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseA.yaml \
  --run-id=v12_phaseA_dense_overtaking \
  --initialize-from=v10g_lane_keeping \
  --force

# Monitor
tensorboard --logdir=experiments/v12_phaseA_dense_overtaking/logs/
```

---

## Success Criteria

| Metric | v10g | Target | Notes |
|--------|------|--------|-------|
| Mean Reward | 1018 | 950 | Slight decrease acceptable |
| Collision Rate | 0.0% | <5% | Safety maintained |
| Goal Completion | 100% | >90% | High success in dense traffic |
| Overtake Success | N/A | >70% | Primary new capability |
| Avg Speed | 93.6% | 85-90% | Acceptable while overtaking |

### Failure Criteria (Rollback)
- Collision rate > 10%
- Mean reward < 650 by 1.5M steps
- Reward variance > 200
- Goal completion < 70%

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| Skill degradation | Checkpoint init, strong penalties, monitor lane-keeping |
| Curriculum too aggressive | Fixed 1-2 NPCs in Stage 1, extended thresholds |
| Reward gaming | Limit overtakes per target, require position confirmation |
| Training instability | Preserve v10g hyperparams, signal_smoothing, long min_lesson_length |

---

## Timeline

| Phase | Steps | Time | Target Reward | Milestone |
|-------|-------|------|---------------|-----------|
| Stage 1 | 0-700K | 0-13 min | 250→400 | First overtakes |
| Stage 2 | 700K-1.8M | 13-33 min | 400→600 | Multi-NPC handling |
| Stage 3 | 1.8M-2.5M | 33-45 min | 600→950 | Dense traffic mastery |
| **Total** | **2.5M** | **45-60 min** | **→950** | Production Ready |

---

## Deliverables

Upon completion:

```
experiments/v12_phaseA_dense_overtaking/
├── README.md (this document)
├── HYPOTHESIS.md (testable claims)
├── results/
│   ├── E2EDrivingAgent-2500000.onnx (final model)
│   ├── checkpoints/ (500K, 1M, 1.5M, 2M, 2.5M)
│   └── run_logs/ (TensorBoard events)
├── ANALYSIS.md (detailed metrics)
└── config/vehicle_ppo_v12_phaseA.yaml
```

---

**Version**: 1.0 | **Status**: Ready for Training | **Owner**: ML Team
