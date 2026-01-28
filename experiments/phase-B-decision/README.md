# Phase B: Decision Learning & Overtaking Validation

**Status**: Design Complete - Ready for Training  
**Version**: v12_phaseB  
**Created**: 2026-01-28  
**Expected Duration**: ~25 minutes (3.0M steps)  
**Expected Reward**: +1500-1800  

---

## Phase Overview

Phase A achieved exceptional reward (+2113.75) with perfect safety, but revealed a critical gap: **zero detected overtaking events** despite +3.0 bonus. Phase B addresses this by:

1. **Validating overtaking detection** - Does the agent actually switch lanes?
2. **Rebalancing reward structure** - Reduce speed dominance, incentivize decisions
3. **Progressive curriculum** - Simple → complex overtaking scenarios
4. **Conservative initialization** - Phase 0 checkpoint (clean slate)

---

## Key Design Decisions

### Initialization: Phase 0 (Not Phase A)
- **Why not Phase A?** Risk of inheriting speed-only bias
- **Phase 0 advantage**: Clean slate, unbiased for decision-learning
- **Trade-off**: Slower convergence (acceptable for correctness)

### Reward Rebalancing
```
Speed reward:           0.5 → 0.3 (reduce dominance)
Lane center:            — → 0.2 (new, encourage positioning)
Following penalty:      — → -0.5 (new, create urgency)
Overtaking bonus:       3.0 → 5.0 (increase incentive)
Collision/safety:       Unchanged
```

### Curriculum Progression (4 Stages)
1. **Stage 0** (0-750K): Baseline, no NPCs
2. **Stage 1** (750K-1.5M): 1 slow NPC, forced overtaking
3. **Stage 2** (1.5M-2.25M): 2 mixed-speed NPCs, selective decisions
4. **Stage 3** (2.25M-3.0M): 4 variable NPCs, complex scenarios

---

## Success Metrics

**Minimum Success** (Proceed to Phase C):
- Mean reward >= +1500
- Overtaking events > 150
- Collision rate < 5%
- Goal completion > 90%

**Expected Success** (Ideal):
- Mean reward +1600-1800
- Overtaking events > 300
- Overtaking success rate > 70%
- Collision rate 0-2%

---

## Files in This Directory

1. **HYPOTHESIS.md** - Root cause analysis, testable predictions, contingency plans
2. **DESIGN-SUMMARY.md** - Executive overview, reward structure, curriculum detail
3. **TRAINING-GUIDE.md** - Commands, monitoring, troubleshooting procedures
4. **COMPARISON.md** - Phase A vs B detailed analysis, risk assessment, rollback strategy

---

## Quick Start

### Training Command
```bash
cd /c/Users/user/Desktop/dev/physical-unity
source .venv/bin/activate

mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B-decision \
  --initialize-from=results/phase-0-foundation/E2EDrivingAgent/E2EDrivingAgent-8000047.pt \
  --force
```

### Monitor Training
```bash
tensorboard --logdir=results/phase-B-decision
```

---

## Critical Validation Points

| Milestone | Steps | Expected Reward | Key Metric |
|-----------|-------|-----------------|-----------|
| Stage 0 | 750K | +600-800 | Speed not degraded |
| Stage 1 | 1.5M | +1000-1200 | >50 overtaking events |
| Stage 2 | 2.25M | +1200-1500 | >70% correct decisions |
| Stage 3 | 3.0M | +1500-1800 | Stable convergence |

---

## Rollback Plan

If ANY of these occur, revert to Phase A checkpoint and redesign:
1. Mean reward < +1200 at 2.5M steps
2. Collision rate > 8% (safety critical)
3. Overtaking events = 0 after Stage 1 validation
4. Training stalls >500K consecutive steps

---

## Next Steps

1. Approval: Review all documents (HYPOTHESIS, DESIGN-SUMMARY, TRAINING-GUIDE, COMPARISON)
2. Sanity check: Run 10K steps test to validate setup
3. Training: Execute full 3.0M steps (expected ~25 minutes)
4. Evaluation: Validate success criteria
5. Decision: Approve for Phase C or redesign Phase B

---

*Phase B Design Complete - 2026-01-28*
