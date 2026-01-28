# Phase B: Decision Learning - Hypothesis & Analysis

**Date**: 2026-01-28
**Goal**: Validate and enhance overtaking decision-making
**Initialization**: Phase 0 checkpoint (conservative, unbiased)
**Expected Outcome**: +1500-1800 reward with >70% overtaking success rate

---

## Root Cause Analysis: Why 0 Overtaking Events in Phase A?

### Critical Finding

Phase A achieved +2113.75 reward (107% above Phase 0) with perfect safety, suggesting enhanced capabilities. However, zero detected overtaking events despite +3.0/overtake bonus is problematic.

**Three Competing Hypotheses**:

### Hypothesis 1: Speed Reward Dominates Decision Logic (PRIMARY - 70% probability)

**Evidence**:
- Speed reward component: 93.9% of Phase A reward composition
- Overtaking bonus (+3.0) is negligible relative to speed accumulation
- Agent learned higher speed → higher cumulative reward
- Lane changes become secondary optimization

**Mechanism**:
- Speed reward: ~+0.5 per step × ~1000 steps per episode = +500 per episode
- Overtaking bonus: +3.0 per event × ~0.5 events average = +1.5 per episode
- Rational choice: Optimize speed, not explicit overtaking

**Implication**: Agent moves faster but not via explicit lane-changing strategy

**Fix**: Rebalance reward weights (0.5 → 0.3), add competing objectives (following penalty -0.5)

---

### Hypothesis 2: Overtaking Detection System Malfunction (TECHNICAL - 20% probability)

**Evidence**:
- Zero events despite +3.0 bonus (statistically improbable if behavior exists)
- Detection relies on Unity-side lane-change detection
- No explicit log validation of detection code firing

**Mechanism**: Event counter not incremented OR event condition never triggered

**Implication**: Agent may switch lanes but detection fails

**Fix**: Add explicit logging, verify detection logic in ROS/Unity bridge

---

### Hypothesis 3: Environmental Constraint - NPC Positioning (10% probability)

**Evidence**:
- Phase A used single slow NPC
- If NPC positioned in center lane consistently, overtaking opportunity minimal
- Agent learns to navigate around without explicit lane-switching

**Fix**: Validate NPC positioning, introduce varied positions/speeds

---

## Phase B Testing Strategy

### Stage 1: Overtaking Detection Validation (Weeks 1-2)

**Objective**: Determine if agent behavior includes lane-switching despite zero detected events

**Testing Approach**:
1. Add explicit logging for every lane change event
2. Save video clips of 10 random episodes
3. Manual review for actual lane-switching behavior
4. Calculate:
   - Lane Change Frequency: Times agent lateral position > 1m from initial lane
   - Detection Rate: Detected events / Observed lane changes
   - False Negative Rate: (Observed - Detected) / Observed

**Decision Point**: 
- If Detection Rate > 90%: Proceed to Hypothesis 1 (reward rebalancing)
- If Detection Rate < 50%: Implement Hypothesis 2 fix (explicit logging)

---

## Success/Failure Criteria

### Success Criteria (ALL MUST BE MET)

| Metric | Target | Threshold |
|--------|--------|-----------|
| **Mean Reward** | +1600 | >= +1500 |
| **Overtaking Events** | >300 | >150 |
| **Overtaking Success Rate** | >75% | >70% |
| **Collision Rate** | 0-2% | <5% |
| **Goal Completion** | >95% | >90% |

**Overall Success**: 4/5 criteria met (can tolerate 1 minor miss)

### Failure Criteria (TRIGGERS ROLLBACK)

1. **Mean Reward < +1200** after 2M steps
2. **Collision Rate > 8%** (safety violation)
3. **Overtaking Events = 0** after Stage 1 complete
4. **Unstable training** (reward oscillates >500 for 500K steps)
5. **Curriculum not progressing**

**Rollback Plan**: If ANY failure criterion triggered, revert to Phase A checkpoint and redesign.

---

## Contingency Plans

### Contingency A: Overtaking Still Not Detected
**Trigger**: Detection rate remains <50% after Stage 1 validation

**Action**:
1. Implement explicit Lane-Change Detection
2. Add visual validation pipeline with video overlays
3. Use lateral displacement as proxy metric

### Contingency B: Speed Performance Degrades >15%
**Trigger**: Final speed tracking drops >15% vs Phase A

**Action**:
1. Revert speed weight from 0.3 back to 0.5
2. Increase following penalty instead (-0.5 to -1.0)
3. Test in mini-curriculum (500K steps)

### Contingency C: Safety Compromised (Collision Rate > 5%)
**Trigger**: Any episode with >3 collisions

**Action**:
1. Increase collision penalty: -10.0 to -15.0
2. Add near-collision penalty: -2.0 to -5.0
3. Reduce overtaking bonus: +5.0 to +3.0

### Contingency D: Training Stalls (No Progress >500K Steps)
**Trigger**: Reward doesn't improve for 500K consecutive steps

**Action**:
1. Check curriculum progression
2. Increase learning rate: 3e-4 to 5e-4
3. Reduce curriculum threshold by 10-20%

---

## Summary

**Primary Hypothesis**: Speed reward dominates Phase A, suppressing explicit overtaking decision-making

**Testing Strategy**:
1. Stage 1: Validate detection system works
2. Stage 2: Rebalance rewards to incentivize decisions
3. Stage 3-4: Progressive curriculum for complex scenarios

**Success Definition**: >70% overtaking success rate, +1500+ reward, <5% collision rate

**Timeline**: 4-6 weeks for complete validation and potential iteration

---

*Phase B Hypothesis Document - Created: 2026-01-28*
