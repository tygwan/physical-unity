# Phase B: Decision Learning - Design Complete

**Date**: 2026-01-28
**Status**: All Design Documents Generated - Ready for Approval
**Location**: `/experiments/phase-B-decision/`

---

## What Was Delivered

### 6 Comprehensive Design Documents

1. **README.md** (3.6 KB)
   - Quick overview and key design decisions
   - Quick start guide
   - Success metrics

2. **HYPOTHESIS.md** (5.2 KB)
   - Root cause analysis: Why 0 overtaking events in Phase A?
   - 3 competing hypotheses with probabilities
   - Testing strategy
   - Success/failure criteria
   - 4 contingency plans (A-D)

3. **DESIGN-SUMMARY.md** (6.8 KB)
   - Executive overview
   - Reward structure detail with rationale
   - Curriculum design (4 stages)
   - Expected timeline and milestones
   - Risk analysis
   - Pre-training checklist

4. **TRAINING-GUIDE.md** (6.7 KB)
   - Pre-training checklist
   - Training commands (3 options)
   - TensorBoard monitoring strategy
   - Comprehensive troubleshooting guide
   - Checkpoint management
   - Post-training evaluation steps

5. **COMPARISON.md** (7.5 KB)
   - Phase A vs Phase B side-by-side comparison
   - Detailed risk analysis
   - Rollback and recovery strategies
   - Decision tree for progression
   - Contingency quick reference
   - Approval checklist

### 1 ML-Agents Configuration File

6. **vehicle_ppo_phase-B.yaml** (7.1 KB in `/python/configs/planning/`)
   - Complete PPO hyperparameters (proven from Phase A)
   - 4-stage curriculum definition
   - Reward structure specification
   - Phase metadata with rationale
   - Rollback conditions and contingency plans

---

## Key Design Decisions

### 1. Root Cause: Speed Reward Dominates (93.9% of Phase A reward)
- Agent learned to go faster (+2113.75 reward)
- But did NOT learn to switch lanes explicitly (0 overtaking events detected)
- Overtaking bonus (+3.0) negligible vs speed accumulation
- **Fix**: Reduce speed weight (0.5 → 0.3), increase overtaking bonus (3.0 → 5.0), add competing objectives

### 2. Initialization: Phase 0 (Not Phase A)
- Phase A faster (2x) but carries speed-only bias
- Phase 0 slower but clean slate for decision-learning
- **Rationale**: Correctness > speed for learning NEW behavior

### 3. Reward Rebalancing
```
Speed reward:           0.5 → 0.3 (40% reduction)
Lane center:            — → 0.2 (new, encourage positioning)
Following penalty:      — → -0.5 (new, create urgency)
Overtaking bonus:       3.0 → 5.0 (5x increase)
Collision/safety:       Unchanged
```

### 4. Curriculum Progression (4 Stages)
- **Stage 0**: Baseline (0 NPCs) - establish reward baselines
- **Stage 1**: Forced Overtaking (1 slow NPC) - learn overtaking with clear incentive
- **Stage 2**: Selective Decisions (2 mixed NPCs) - learn WHEN to overtake
- **Stage 3**: Complex Scenarios (4 variable NPCs) - robust navigation

---

## Expected Outcomes

**Timeline**: ~25 minutes (3.0M steps)
- Stage 0 (750K steps): +600-800 reward
- Stage 1 (750K steps): +1000-1200 reward
- Stage 2 (750K steps): +1200-1500 reward
- Stage 3 (750K steps): +1500-1800 reward

**Success Criteria**:
- Mean reward >= +1500 (minimum) or +1600-1800 (expected)
- Overtaking events > 150 (minimum) or > 300 (expected)
- Collision rate < 5% (max)
- Goal completion > 90%

**Failure Triggers Redesign**:
- Mean reward < +1200 at 2.5M steps
- Overtaking events = 0 after Stage 1 validation
- Collision rate > 8% (safety critical)

---

## Risk Management

**Overall Risk Level**: MEDIUM-LOW (most contingencies are quick fixes)

### Contingency Plans

**Contingency A**: Overtaking still not detected
- Trigger: Detection rate < 50%
- Action: Implement explicit lane-change logging
- Recovery time: 500K steps

**Contingency B**: Speed performance drops >15%
- Trigger: Speed < 80% of Phase A
- Action: Revert weight 0.3 → 0.4 or 0.5
- Recovery time: 250K steps

**Contingency C**: Safety compromised (collision >5%)
- Trigger: Collision rate > 5%
- Action: Increase penalties (-10→-15, -2→-5)
- Recovery time: Immediate

**Contingency D**: Training stalls
- Trigger: No improvement for 500K steps
- Action: Increase learning rate 3e-4 → 5e-4
- Recovery time: 250K steps

**Fallback**: If multiple contingencies needed AND not resolved → Rollback to Phase A checkpoint and redesign

---

## File Locations

All deliverables in: `/c/Users/user/Desktop/dev/physical-unity/experiments/phase-B-decision/`

```
phase-B-decision/
├── README.md                 (3.6 KB)
├── HYPOTHESIS.md             (5.2 KB)
├── DESIGN-SUMMARY.md         (6.8 KB)
├── TRAINING-GUIDE.md         (6.7 KB)
├── COMPARISON.md             (7.5 KB)
├── logs/                     (monitoring)
└── checkpoints/              (results)
```

Configuration: `/python/configs/planning/vehicle_ppo_phase-B.yaml` (7.1 KB)

---

## Approval Checklist

Before starting Phase B training:

- [ ] Review all 5 design documents
- [ ] Approve root cause hypothesis (speed dominance)
- [ ] Approve initialization strategy (Phase 0)
- [ ] Approve reward rebalancing
- [ ] Approve curriculum design
- [ ] Verify Phase 0 checkpoint available
- [ ] Run 10K steps sanity check
- [ ] Enable overtaking detection logging
- [ ] Set up TensorBoard monitoring
- [ ] Confirm timeline and resources (30-40 minutes)

---

## Next Steps

### Day 1: Approval & Preparation
1. Review all Phase B design documents
2. Run sanity check: 10K steps test
3. Verify Phase 0 checkpoint loads correctly
4. Validate YAML config
5. Enable overtaking detection logging in Unity
6. Set up TensorBoard

### Day 2-3: Training
1. Execute full training: 3.0M steps (~25 minutes)
2. Monitor Stage 0-3 progression
3. Watch for critical milestones (750K, 1.5M, 2.25M, 3.0M)
4. Implement contingencies if needed

### Day 3: Evaluation
1. Verify success criteria
2. Analyze overtaking events and success rate
3. Compare with Phase A
4. Document learnings
5. Approve for Phase C or redesign Phase B

---

## Key Metrics to Track

**Per-Stage Validation**:
- 750K steps: Reward +600-800, Speed >80% Phase A
- 1.5M steps: Reward +1000-1200, Overtaking events >50
- 2.25M steps: Reward +1200-1500, Success rate >70%
- 3.0M steps: Reward +1500-1800, Collision <5%

**Decision Point at 1.5M steps** (~12 minutes):
- If >50 overtaking events AND >1000 reward → Continue confidently
- If 0 events OR <1000 reward → Assess contingency options

---

## Why This Design Matters

**The Problem**: Phase A learned to go fast (+2113.75) but didn't learn to make overtaking decisions (0 events detected). This leaves us uncertain about the agent's actual capabilities.

**The Solution**: Phase B isolates the root cause (speed dominance) and tests it systematically with rebalanced rewards and progressive curriculum.

**The Impact**: If successful, Phase B validates overtaking capability and provides foundation for Phase C (multi-vehicle scenarios) and beyond.

**The Risk**: If Phase B fails, we have clear contingency plans and data to redesign. Worst case: rollback to Phase A and redesign.

---

## Summary

Phase B design is complete and data-driven:
- **Root cause**: Speed reward = 93.9% of Phase A
- **Fix**: Rebalance to 30%, add decision incentives
- **Testing**: 4-stage curriculum with explicit milestones
- **Risk**: MEDIUM-LOW with 4 contingency plans
- **Timeline**: ~25 minutes training + 10 minutes monitoring

**All documents delivered and ready for approval.**

---

Generated: 2026-01-28
ML Experiment Planning Orchestrator
Status: DESIGN COMPLETE - READY FOR TRAINING APPROVAL
