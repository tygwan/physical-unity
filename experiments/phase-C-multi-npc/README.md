# Phase C: Multi-NPC Interaction & Advanced Decision-Making

**Status**: Design Complete - Ready for Training
**Created**: 2026-01-29
**Expected Duration**: 3.6M steps (45-50 minutes)
**Target Reward**: +1500 (71% improvement over Phase B v2: +877)

---

## Quick Start

### Training Command
```bash
cd /c/Users/user/Desktop/dev/physical-unity
source .venv/bin/activate

mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-C.yaml \
  --run-id=phase-C-multi-npc \
  --initialize-from=phase-B-decision-v2 \
  --force \
  --no-graphics \
  --time-scale=20
```

### Success Checklist
- [ ] Stage 4 reward >= +1200
- [ ] Collision rate < 5%
- [ ] Overtaking events > 5 per 1000 steps
- [ ] Goal completion > 85%
- [ ] All 5 stages completed

---

## What's Phase C About?

**Problem**: Phase B v2 agent learned to safely drive 3 NPCs but never overtook (OvertakePhase_Active=0).

**Challenge**: Can agent scale to 8 NPCs while learning selective overtaking?

**Solution**: 
1. Start with Phase B v2 checkpoint (has decision-making)
2. Gradually scale: 3→4→5→6→8 NPCs
3. Dynamic overtaking bonus (easy:+1, medium:+2, hard:+3)
4. Blocked detection suspension (don't penalize for being stuck)

---

## 5-Stage Curriculum

| Stage | NPCs | Speed Range | Steps | Target |
|-------|------|-------------|-------|--------|
| 0 | 3 | 0.85 mixed | 0-500K | +850 |
| 1 | 4 | 0.4-0.55 | 500K-1.2M | +950 |
| 2 | 5 | 0.55-0.7 | 1.2M-2.0M | +1100 |
| 3 | 6 | 0.7-0.85 | 2.0M-2.8M | +1250 |
| 4 | 8 | 0.5-1.0 | 2.8M-3.6M | +1500 |

**Key Insight**: Each stage adds one NPC + increases speed variation, introducing incremental complexity.

---

## Critical Technical Changes

### 1. Initialization (Key Decision)
```
Why Phase B v2 over Phase A?
- Phase B v2 has 3-NPC decision-making experience
- Phase A has raw overtaking speed but no multi-NPC judgment
- Phase C is fundamentally about DECISIONS
- Transfer should be more efficient from B v2
```

### 2. Dynamic Overtaking Bonus (New)
```
OLD (Phase B v2): Fixed +2.0 bonus
NEW (Phase C):
  blocked_duration < 10s:  +1.0 (easy)
  blocked_duration 10-30s: +2.0 (medium)
  blocked_duration > 30s:  +3.0 (hard)

Multiplied by: overtaking_aggressiveness (0.5 → 1.0 curriculum)
```

### 3. Blocked Detection Suspension (Critical Fix)
```
Problem (Phase B v1): Speed penalty -0.2/step caused agent to STOP
Math: -0.2 × 501 steps = -100.2 accumulated penalty

Solution (Phase C):
When blocked (lead vehicle close & slow):
  - Increment blocked_timer
  - SKIP speed_under_penalty for this period
  - Encourage overtaking attempt instead

Effect: Agent can wait strategically or attempt overtaking (not penalized for stuck)
```

---

## Validation Checkpoints (MANDATORY)

### Checkpoint 1: 100K steps (15 min)
```
- Speed averaging > 5 m/s (basic driving works)
- Reward trend positive or stable (not negative)
- No CUDA errors
```

### Checkpoint 2: 500K steps (Stage 0 complete)
```
- Reward > +800 (match Phase B v2 baseline)
- Episode length ~2500 steps (stable)
- Ready to progress to Stage 1
```

### Checkpoint 3: 1.5M steps (Stage 2 transition)
```
- Reward > +1000 (30% improvement)
- Overtaking events visible in logs
- Collision rate stable < 5%
```

### Checkpoint 4: 3.6M steps (Final)
```
- Reward approaching +1500 or plateau
- All 5 stages progressed successfully
- Safety metrics maintained
```

---

## Emergency Stop Conditions

STOP training immediately if:
1. Mean reward < -50 for 3 consecutive checkpoints (Phase B v1 repeat)
2. Collision rate > 10% sustained (safety issue)
3. Episode length < 1000 steps (agent giving up)

---

## Files in This Directory

```
README.md                          (this file)
PLAN.md                            (training plan + timeline)
HYPOTHESIS.md                      (validation criteria)
PHASE_C_DESIGN_REQUEST.md          (detailed requirements)
SUMMARY.md                         (executive summary)
config/vehicle_ppo_phase-C.yaml    (hyperparameters + curriculum)
```

---

## Expected Timeline

| Milestone | Time | Status |
|-----------|------|--------|
| Setup | 10 min | Start |
| Validation | 5 min | Sanity checks |
| Stage 0 (3 NPCs) | 15 min | Baseline |
| Stage 1-2 (4-5 NPCs) | 25 min | Overtaking starts |
| Stage 3-4 (6-8 NPCs) | 30 min | Complex scenarios |
| Analysis | 10 min | Results |

**Total**: ~95 minutes (~1.5 hours including setup/analysis)

---

## Success = All of These

1. Stage 4 final reward >= +1200 (71% improvement)
2. Collision rate < 5% (safety)
3. Overtaking events > 5 per 1000 steps (behavior change)
4. Goal completion > 85% (route finishing)
5. All 5 stages progressed smoothly

## Failure = Any of These

1. Stage 4 reward < +1000 (regression)
2. Collision rate > 10% (unsafe)
3. Zero overtaking after 2M steps (no progress)
4. Stage stuck for >1M steps (curriculum failure)

---

## Key Hypotheses Being Tested

### H1: Phase B v2 Checkpoint Transfers Better Than Phase A
- B v2: 3-NPC decision-making + safety
- A: Raw 1-NPC overtaking speed
- Prediction: B v2 converges faster at Stage 0

### H2: Dynamic Overtaking Bonus Drives Behavior
- Fixed bonus insufficient for varying complexity
- Dynamic (+1/+2/+3) better
- Prediction: Overtaking events increase 5→10→15→20 across stages

### H3: Blocked Detection Suspension Critical
- Prevents "learned to stop" failure
- Allows strategic waiting
- Prediction: Suspension prevents reward collapse

### H4: Smooth Curriculum Progression
- Each stage one challenge
- Expected: +850→+950→+1100→+1250→+1500
- Tolerance: Within ±5% of predictions

---

## Next Phase (Phase D)

If Phase C succeeds (+1200+):
- Phase D: 6-8 NPCs + Curved Roads
- Phase E: Intersections
- Phase F: Multi-lane switching

If Phase C plateaus (1000-1200):
- Extend training or adjust curriculum
- Can still proceed to Phase D

If Phase C fails (<1000):
- Root cause analysis (like Phase B v1)
- May need redesign or fallback

---

## Post-Training Analysis

After training completes, analyze:
1. Reward progression: Match expected (+850→+1500)?
2. Overtaking events: Increase across stages?
3. Collision rate: Stay < 5%?
4. Curriculum progression: All 5 stages reached?
5. Stage transition times: Typical (500-700K steps)?

Generate:
- `RESULTS.md` - Actual results vs predictions
- `ANALYSIS.md` - Key findings and lessons
- Checkpoint comparison charts

---

## References

- Phase B v1 Failure Analysis: `experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md`
- Phase B v2 Design: `experiments/phase-B-decision-v2/DESIGN.md`
- Phase A Results: `experiments/phase-A-overtaking/ANALYSIS.md`

---

**Created**: 2026-01-29
**Status**: Design Complete - Ready for Approval & Training
**Next**: Technical review → Approval → Training execution
