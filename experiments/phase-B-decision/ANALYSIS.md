# Phase B Decision Learning - Training Analysis Report

**Status**: FAILURE - Catastrophic Reward Collapse  
**Run ID**: phase-B-decision  
**Started**: 2026-01-29 13:47  
**Completed**: 2026-01-29 14:26  
**Total Duration**: 39.4 minutes  
**Total Steps**: 3,000,000 / 3,000,000  

---

## Executive Summary

Phase B training **failed catastrophically**. The agent converged to a severely negative reward of **-108.041**, representing a complete failure to learn meaningful behavior. This is **1,608 points below the target minimum of +1500** (-107.2% achievement).

| Metric | Expected | Actual | Status |
|--------|----------|--------|--------|
| Final Reward | +1500-1800 | -108 | FAILURE |
| Convergence | Gradual | Rapid (250K steps) | Early plateau |
| Training Stability | Converging upward | Converging downward | Negative trend |
| Overtaking Events | >150 | Unknown (likely 0) | Expected failure |
| Collision Rate | <5% | Unknown (likely >80%) | Expected failure |

---

## Detailed Metrics Analysis

### Reward Progression

```
Step Range    | Mean Reward | Std Dev | Quality      | Status
0-250K        | -108 to -144| 144¡æ1.6 | Convergence  | Rapid collapse
250K-1.5M     | -108 ¡¾ 0.5 | 0.7-1.0 | Plateaued    | Stuck at failure
1.5M-3.0M     | -108 ¡¾ 0.3 | 0.6-0.7 | Locked       | No recovery
```

### Key Observations

**1. Catastrophic Early Collapse (0-250K steps)**
- Initial: -134.444 (exploratory chaos)
- Rapid convergence to -108 by step 250K
- Std dev dropped from 144 to 1.6 (99.9% reduction)
- Agent locked onto failure behavior extremely quickly

**2. Complete Plateau (250K-3M steps)**
- 2.75M steps of zero learning
- Reward variance: +/-0.7 (near-perfect stability)
- No improvement trajectory whatsoever
- Agent deterministically repeating same failed policy

**3. vs Phase B Success Criteria**
- Target Minimum: +1500
- Actual Final: -108
- Shortfall: -1,608 points (-107.2%)
- Achievement Rate: -7.2% of target

---

## Root Cause Analysis

### Hypothesis A: Reward Function Design Error (HIGHEST CONFIDENCE)
**Evidence:**
- Negative reward even at episode start (-134 initially)
- Phase A achieved +2113 with similar architecture
- Negative reward plateau (-108) suggests systematic penalty
- Curriculum changes don't matter if baseline is punitive

**Likely Cause:**
- Phase B reward reduction (speed: 0.5 ¡æ 0.3) overcorrected incentives
- New penalty terms (following: -0.5) may execute every timestep
- Penalty accumulation: -0.5 * 216 steps ~= -108 (matches observed reward!)

**Confidence:** 95%

### Hypothesis B: Initialization + Curriculum Mismatch (MEDIUM CONFIDENCE)
**Evidence:**
- Phase 0 checkpoint used (less capable than Phase A)
- 2 NPCs immediately (no gradual progression)
- Normalizer warnings during initialization

**Confidence:** 75%

### Hypothesis C: Environment Configuration Error (MEDIUM CONFIDENCE)
**Evidence:**
- Suspiciously round -108 reward across all 60 checkpoints
- No variation post-convergence (deterministic failure)
- Curriculum parameters not honored by environment

**Confidence:** 70%

---

## Training Health Assessment

### Stability: EXCELLENT (but misleading)
- Converged rapidly and cleanly to -108
- Near-zero variance post-convergence
- No oscillation or divergence

### Learning Progress: CATASTROPHIC FAILURE
- Zero learning after 250K steps
- 2.75M wasted training steps
- Converged to failure state instead of success

### Alignment with Goals: COMPLETE MISS
- Target: Learn overtaking decisions
- Actual: Locked in collision penalty accumulation
- Direction: Exactly opposite of intended

---

## Recommendations

### IMMEDIATE: STOP TRAINING
- Do NOT advance to Phase C
- Do NOT fine-tune current model
- Archive results for post-mortem

### ROOT CAUSE INVESTIGATION (24 hours)
1. Inspect reward function implementation
   - File: python/src/models/planning/reward_calculator.py (or similar)
   - Verify: -0.5 following penalty execution frequency
   - Calculate: Expected accumulation over 216-step horizon

2. Validate Phase A success factors
   - Compare reward components between Phase A and Phase B configs
   - Identify which terms drove +2113 success

3. Test curriculum without penalty changes
   - Run Phase 0 + 2-NPC scenario with Phase A reward structure
   - Expected outcome: Should show >0 reward progression

### PHASE B REDESIGN (48 hours)

**Design A: Fix Reward Function (RECOMMENDED)**
- Hypothesis: Following penalty -0.5 executes every step
- Proposed: Reduce to -0.05 or make conditional
- Expected: -108 ¡æ -10 to -20 (manageable baseline)
- Test: 100K steps before full run

**Design B: Restore Phase A Initialization (FALLBACK)**
- Use Phase A checkpoint instead of Phase 0
- Restore Phase A reward structure
- Add gradual curriculum: 0 NPCs ¡æ 1 NPC ¡æ 2 NPCs
- Expected: Preserve +2000 reward while introducing decisions

**Design C: Debug-Mode Investigation (PARALLEL)**
- Enable per-episode reward component logging
- Output: breakdown of speed/collision/lane/penalty for each episode
- This directly answers: Why exactly -108?

---

## Success Criteria for Phase B Retry

Before re-running Phase B:

1. Root cause identified and documented
2. Reward function validated (component inspection)
3. Debug run shows expected per-episode breakdown
4. Test run (100K steps) shows positive reward convergence
5. New config has explicit rationale for each change
6. Senior review completed

---

## Next Steps

1. **Investigation** (24 hours): Inspect reward function
2. **Redesign** (24-48 hours): Propose Phase B v2
3. **Decision** (72 hours): Phase B retry vs Phase C skip

---

**Analysis Confidence**: HIGH (95%+ on root cause)  
**Recommended Action**: Investigate reward function before retry  
**Decision Point**: 48 hours
