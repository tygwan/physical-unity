# Phase A: Dense Overtaking - Design Summary

## Status: COMPLETE

Phase A comprehensive design is ready for training, built on v10g foundation success.

---

## v10g Foundation (Baseline)

| Metric | Result | Status |
|--------|--------|--------|
| Reward | +1018.43 | A+ Excellent |
| Collision | 0.0% | Perfect safety |
| Goal Completion | 100% | Complete |
| Status | Production Ready | ✓ Ready for Phase A |

---

## Phase A Design

### Objectives
1. **Primary**: Learn overtaking capability
2. **Secondary**: Maintain v10g safety (<5% collision)
3. **Tertiary**: Handle increased NPC density (4→6)

### Timeline: 3-Stage Progressive Curriculum

| Stage | Steps | NPCs | Speed | Goal | Expected Reward |
|-------|-------|------|-------|------|-----------------|
| 1 | 0-700K | 1-2 | 30-40% | 80m | 250→400 |
| 2 | 700K-1.8M | 3-4 | 50-60% | 120-140m | 400→600 |
| 3 | 1.8M-2.5M | 5-6 | 40-80% | 230m | 600→950 |

### Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Initialize From | v10g checkpoint | Transfer learning |
| Total Steps | 2.5M | 45-60 minutes |
| Network | 512x3 layers | Preserved (stable) |
| Learning Rate | 3e-4 | Linear decay |
| Overtaking Reward | +3.0 | Explicit signal |
| Target Reward | +950 | Account for difficulty |
| Success Rate | >70% | Overtaking success |

### Success Criteria

- Overtaking success: >70%
- Collision rate: <5%
- Goal completion: >90%
- Final reward: 900+

---

## Deliverables

### Documentation (3 files)
1. **README.md** (186 lines) - Comprehensive plan with curriculum design
2. **HYPOTHESIS.md** (177 lines) - Testable hypotheses & validation
3. **TRAINING-GUIDE.md** (76 lines) - Execution instructions

### Configuration
- **vehicle_ppo_phase-A.yaml** - ML-Agents config with 3-stage curriculum
- **Config:** max_steps=2.5M, initialize from v10g

### Post-Training (Auto-Generated)
- Model: E2EDrivingAgent-2500000.pt + .onnx
- Logs: TensorBoard events
- Analysis: ANALYSIS.md with detailed results

---

## Curriculum Design Overview

### Stage 1: Skill Acquisition (0-700K)
- 1 slow NPC (30% speed)
- Short routes (80m)
- Goal: Learn basic overtaking maneuver
- Milestone: First overtakes by 300-400K

### Stage 2: Density Introduction (700K-1.8M)
- Progressive: 1→2→3→4 NPCs
- Mixed speeds (30-60%)
- Medium routes (100-140m)
- Goal: Multi-vehicle coordination

### Stage 3: Complex Scenarios (1.8M-2.5M)
- Dense: 5-6 NPCs
- Speed variation (40-80%)
- Long routes (230m)
- Goal: Complex overtaking chains

---

## Reward Structure

| Component | Value | Change | Purpose |
|-----------|-------|--------|---------|
| Progress | +1.0/m | Same | Movement incentive |
| Overtaking | +3.0 | NEW | Explicit skill signal |
| Lane-keeping | +0.01 | Reduced | Control signal |
| Following | Removed | Changed | Encourage passing |
| Collision | -10 | Same | Safety |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Skill degradation | v10g checkpoint init, strong penalties |
| Curriculum too fast | Extended thresholds, monitoring |
| Reward gaming | Limit overtakes, position confirmation |
| Instability | Preserve hyperparams, signal smoothing |

---

## Files & Locations

```
experiments/phase-A-overtaking/
├── README.md                    # Experiment plan
├── HYPOTHESIS.md                # Validation protocol
├── TRAINING-GUIDE.md            # How to train
├── DESIGN-SUMMARY.md            # This document
├── config/
│   └── vehicle_ppo_phase-A.yaml
├── results/ (generated)
│   ├── E2EDrivingAgent-2500000.pt
│   ├── E2EDrivingAgent-2500000.onnx
│   └── ANALYSIS.md
└── logs/ (generated)
    └── TensorBoard events
```

Config: `python/configs/planning/vehicle_ppo_phase-A.yaml`

---

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml \
  --run-id=phase-A-overtaking \
  --initialize-from=v10g_lane_keeping
```

Monitor: `tensorboard --logdir=experiments/phase-A-overtaking/logs/`

---

## Validation Checkpoints

| Step | Expected | Validation |
|------|----------|-----------|
| 300K | First overtakes | Check logs |
| 700K | Stage 1 plateau | Reward 350-400 |
| 1.2M | 4-NPC handling | Collision <5% |
| 1.8M | Stage 3 start | Smooth transition |
| 2.5M | Convergence | Reward 900+ |

---

## Expected Outcomes

**Success** (Most Likely):
- Smooth 3-stage progression
- Overtaking success 0%→70%
- Reward +950 ± 50
- Ready for Phase B

**Partial Success** (Recovery):
- Stages 1-2 complete
- Stage 3 challenges
- Final reward +800-850
- Design revision needed

**Failure** (Rare):
- Early detection
- Load checkpoint
- Adjust curriculum
- Recover in 100-300K steps

---

## Next Phases

- **Phase B**: Decision Learning (2-3M steps)
- **Phase C**: Multi-NPC Coordination (2-3M steps)
- **Integration**: E2E System Validation
- **Deployment**: Production Readiness

---

**Status**: Ready for Training
**Duration**: 45-60 minutes
**Hardware**: RTX 4090, 128GB RAM
**Created**: 2026-01-27
