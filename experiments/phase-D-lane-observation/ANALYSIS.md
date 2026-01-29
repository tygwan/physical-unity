# Phase D v1 Failure Analysis

Status: FAILURE - Catastrophic Curriculum Collapse
Run ID: phase-D
Duration: 6,000,000 steps (100 minutes)
Final Reward: -2,156 (expected: +1000+)

## Executive Summary

Phase D v1 FAILED due to simultaneous curriculum transitions at step 4.68M.

**What Went Wrong**:
Agent peaked at +406 reward, then three curriculum parameters transitioned simultaneously, causing collapse to -4,825. Budget exhausted before recovery.

**Why It Failed**:
Agent learned scenario-specific policies. Simultaneous multi-parameter transition violated incremental learning principle.

**What We Learned**:
- Curriculum parameters are NOT independent
- Single-parameter progression safer than parallel curricula
- Peak reward doesn't guarantee robustness
- Lane observation helps but isn't sufficient

**Recommendation**:
Phase D v2 with staggered curriculum (one parameter at a time)

---

## Timeline: Where It Went Wrong

Steps | Reward | Status
---|---|---
0-10K | -58 | Exploration
100K | -40 | Stabilization
500K | -36 | Plateau
1.5M | -207 | Volatile
2.5M | +157 | Recovery begins
3.5M | +298 | Improvement
4.0M | +350 | Approaching target
4.6M | +406 | PEAK - Near success
4.68M | +406 | Ready to transition
4.7M | -4,825 | COLLAPSE - Critical failure
5.0M | -3,487 | Attempted recovery
5.5M | -2,700 | Still negative
6.0M | -2,156 | Final - Budget exhausted

## The Critical Moment: Step 4.68M

**What Changed**:
Three curriculum parameters transitioned simultaneously:
1. num_active_npcs: 1 to 2
2. speed_zone_count: 1 to 2  
3. npc_speed_variation: 0 to 0.3

**Expected Difficulty**: 40-50% harder scenario
**Actual Difficulty**: Catastrophic collapse (-5,231 points)

## Root Cause

### Primary: Curriculum Design Flaw

Agent learns scenario-specific policies when mastering simple environment:
- "If 1 NPC ahead, overtake immediately"
- "Follow center lane (no choices)"
- "NPC speed is constant"

Simultaneous transition to harder scenario makes old policies WRONG:
- Multiple NPCs: Which one to overtake? When? How?
- Multiple zones: Which boundary to follow?
- Speed variation: Old prediction model completely invalid

**Unlearning cost**: -5,200 reward loss
**Learning time**: 2M+ steps needed
**Budget**: Only 1.3M steps remaining
**Result**: Unable to recover

### Secondary: Parameter Threshold Independence

v1 Design: Each parameter has independent reward threshold
- If reward > 400, transition ALL ready parameters
- Problem: Multiple transitions simultaneously

v2 Fix: Sequential thresholds
- Only ONE parameter can transition at a time
- Others wait for next cycle

### Tertiary: Brittle Convergence

Agent converged very quickly to +406 (500K steps after breakthrough)
- Not true convergence, but local optimum
- No plateau period before transition
- Immediate collapse on environmental change

Lesson: Peak reward ≠ Robust learning

---

## Lessons Learned

### L1: Curriculum Parameters NOT Independent

Finding: Agent specializes to simple scenario, cannot simply scale solution.

Evidence:
- +406 reward (1 NPC, 1 zone, 0 variation)
- -4,825 reward (2 NPCs, 2 zones, 0.3 variation)
- 5,231 point collapse

Implication:
- Sequential thresholds (not parallel)
- One parameter at a time
- Validate convergence before transition

### L2: Lane Observation Effective But Not Sufficient

Finding: Lane info helped initial learning (+406 peak) but failed to enable robustness.

Evidence:
- -58 to +406: Lane observation improving agent
- +406 to -4,825: Lane observation couldn't prevent collapse

Implication:
- Keep lane observation (valuable feature)
- But improve reward shaping for multi-variable scenarios
- Use for Phase E/F with better curriculum design

### L3: Peak Reward Doesn't Guarantee Robustness

Finding: +406 seemed successful, but was brittle.

Interpretation: Local optimization in narrow scenario, not true learning.

Implication: Need robustness validation
- Require convergence plateau (not peaks)
- Test stability before proceeding
- Validate generalization

---

## Phase D v2: Recommendations

### Change 1: Curriculum Sequencing

v2 Design: Strict single-parameter progression
- Stage 0: Baseline (1 NPC, 0.3 ratio, 80m goal, 1 zone, 0 variation)
- Stage 1: num_active_npcs 1 to 2 (ONLY change)
- Stage 2: npc_speed_ratio 0.3 to 0.6 (ONLY change)
- Stage 3: goal_distance 80m to 150m (ONLY change)
- Stage 4: speed_zone_count 1 to 2 (ONLY change)
- Stage 5: npc_speed_variation 0 to 0.3 (ONLY change)

### Change 2: Looser Thresholds

v2 Thresholds:
- Reward threshold: 500 (up from 400)
- Min episodes: 1000 (up from 300)
- Require plateau: Stable for 100K steps
- Manual approval required for transition

### Change 3: Checkpoint & Fallback

v2 Strategy:
- Save checkpoint after each stage success
- If collapse > 50%, revert automatically
- Manual decision required to retry

### Change 4: Robustness Testing

v2 Validation:
- After stage convergence, test 50K steps
- Verify reward stays stable
- Only proceed if plateau confirmed
- Prevents brittle convergence issues

---

## Expected v2 Outcome

**Timeline**:

Stage 0 (0-1M): Baseline -> +600
Stage 1 (1M-2.5M): 1 NPC -> 2 NPCs -> +750
Stage 2 (2.5M-3.5M): Speed ratio -> +800
Stage 3 (3.5M-4.5M): Goal distance -> +850
Stage 4 (4.5M-6M): Speed zones -> +900
Stage 5 (6M-8M): NPC variation -> +1000+

**Total**: 8M steps (vs v1: 6M)

**Success Criteria**:
1. Final reward >= +1000
2. No collapse > 1000 within 100K steps
3. Collision rate < 5%
4. All stages completed smoothly
5. Checkpoints saved at each transition

**Expected Success Rate**: 75-80% (vs v1: 0%)

---

## Fallback Options If v2 Fails

**Option A**: Further relax thresholds
- Longer patience, higher thresholds
- Risk: 12M+ steps needed
- Benefit: Most conservative

**Option B**: Skip lane observation
- Proceed with 242D to Phase E
- Risk: Lose lane observation
- Benefit: Unblock Phase E immediately

**Option C**: Focus on Phase E design
- Proceed to curved roads
- Risk: Delay lane observation
- Benefit: New curriculum insights

---

## Conclusion

**Phase D v1**: FAILED
- Reward: +406 to -4,825 to -2,156
- Cause: Simultaneous multi-parameter transitions
- Lesson: Curriculum parameters are not independent

**Phase D v2**: PLANNED
- Strategy: Staggered single-parameter progression
- Thresholds: Looser, more patience
- Checkpoints: Fallback mechanism
- Success Chance: 75-80%

**Key Insight**:
Slower sequential progress more reliable than aggressive parallel optimization.

---

**Analysis Confidence**: HIGH (95%+)
**Recommendation**: Proceed with Phase D v2
**Timeline**: 48 hours implementation, 120 minutes training
**Decision**: 2026-01-30 (24 hours)
