# Phase B: Decision Learning - Design Summary

**Phase**: B (Decision Learning & Overtaking Validation)
**Version**: v12_phaseB
**Created**: 2026-01-28
**Status**: Ready for Approval and Training Start

---

## Executive Overview

Phase A successfully achieved +2113.75 reward with perfect safety, BUT with 0 detected overtaking events despite +3.0 bonus. Phase B addresses this critical gap by:

1. **Validating overtaking detection system** (does agent actually switch lanes?)
2. **Rebalancing reward structure** (reduce speed dominance, incentivize decisions)
3. **Progressive curriculum** (simple → complex overtaking scenarios)
4. **Conservative initialization** (Phase 0 checkpoint for clean slate)

**Expected Outcome**: +1500-1800 reward, >70% overtaking success, <5% collision rate

---

## Key Changes from Phase A

### Reward Structure Rebalancing

| Component | Phase A | Phase B | Change | Rationale |
|-----------|---------|---------|--------|-----------|
| **Speed reward** | 0.5 | 0.3 | -40% | Reduce dominance (93.9% of reward) |
| **Lane center** | None | 0.2 | NEW | Encourage positioning before overtake |
| **Following penalty** | None | -0.5 | NEW | Create urgency to overtake |
| **Overtaking bonus** | 3.0 | 5.0 | +67% | Increase relative incentive |
| **Collision penalty** | -10.0 | -10.0 | — | Maintain safety |

**Effect**: Reduces speed-per-step from ~+0.5 to ~+0.3 (40%), adds following penalty (-0.5 when TTC<5s), increases overtaking bonus 5x, adds lane positioning reward (0.2).

---

## Initialization Strategy: Phase 0 (RECOMMENDED)

### Why NOT Phase A?

| Factor | Phase A Init | Phase 0 Init | Decision |
|--------|-------------|-------------|----------|
| **Convergence Speed** | 2x faster | Slower | Phase 0 safer |
| **Bias Risk** | High (speed-only) | None | Phase 0 unbiased |
| **Decision Learning** | Inherits suboptimal | Fresh start | Phase 0 better |
| **Overtaking Bias** | Learned non-overtaking | No preconceptions | Phase 0 clearer |

**Recommendation**: Phase 0 checkpoint (E2EDrivingAgent-8000047.pt)
- Provides foundation (lane keeping, NPC coexistence)
- No speed-only bias to unlearn
- Clean slate for decision-learning curriculum
- Slower convergence acceptable for correctness

**Fallback**: If Phase B underperforms Phase 0 baseline at 1.5M steps, switch to Phase A init.

---

## Curriculum Design (4 Stages)

### Progressive Overtaking Difficulty

**Stage 0: Baseline (0-750K steps)**
- NPCs: 0
- Objective: Establish reward baselines with rebalanced weights
- Expected reward: +600-800
- Success metric: Stable speed tracking with new reward structure

**Stage 1: Forced Overtaking (750K-1500K steps)**
- NPCs: 1 slow (8 m/s)
- Requirement: MUST overtake to progress
- Objective: Learn overtaking behavior with clear incentive
- Expected reward: +1000-1200
- Success metric: >50% overtaking attempts per episode

**Stage 2: Selective Decisions (1500K-2250K steps)**
- NPCs: 2 mixed-speed (8 m/s + 15 m/s)
- Requirement: Selective overtaking
- Objective: Learn WHEN to overtake (decision-making)
- Expected reward: +1200-1500
- Success metric: >70% correct overtaking decisions

**Stage 3: Complex Scenarios (2250K-3000K steps)**
- NPCs: 4 variable-speed (6-16 m/s)
- Requirement: Multi-vehicle navigation
- Objective: Robust overtaking in dense environment
- Expected reward: +1500-1800 (target)
- Success metric: Stable +1500+ with <5% collision

---

## Reward Structure (Phase B Detail)

```
Progress:              +1.0 per step (towards goal)
Goal Reached:          +10.0 (completion bonus)
Speed Reward:          +0.3 per step * (speed / target_speed)
                       Target: 18 m/s, Max: 20 m/s
Lane Center:           +0.2 when in center lane (NEW)
Following Penalty:     -0.5 per step when TTC < 5s (NEW)
Overtaking Bonus:      +5.0 per successful overtake
Collision Penalty:     -10.0 per collision
Near-Collision:        -2.0 per near-collision (TTC < 2s)
```

**Expected Reward Trajectory**:
- Stage 0: ~+0.65/step → +490K per 750K steps
- Stage 1: ~+0.75/step → +560K per 750K steps
- Stage 2: ~+0.80/step → +600K per 750K steps
- Stage 3: ~+0.85/step → +637K per 750K steps
- **Total**: ~2.3M across 3M steps = **+1700 expected mean**

---

## Expected Training Timeline

| Phase | Steps | Duration | Cumulative |
|-------|-------|----------|-----------|
| Stage 0 (Baseline) | 750K | ~6 min | 6 min |
| Stage 1 (Forced) | 750K | ~6 min | 12 min |
| Stage 2 (Selective) | 750K | ~6 min | 18 min |
| Stage 3 (Complex) | 750K | ~7 min | 25 min |
| **Total** | **3.0M** | **~25 min** | **~25 min** |

**Assumption**: 5M steps/hour throughput (based on Phase A: 5.05M/hour)

---

## Critical Validation Milestones

1. **750K steps**: Stage 0 complete
   - Expected: +600-800 reward with rebalanced weights
   - Validation: Speed performance not degraded >15%

2. **1.5M steps**: Stage 1 complete
   - Expected: +1000-1200 reward
   - Validation: >50 overtaking events total

3. **2.25M steps**: Stage 2 complete
   - Expected: +1200-1500 reward
   - Validation: >70% correct overtaking decisions

4. **3.0M steps**: Stage 3 complete
   - Expected: +1500-1800 reward
   - Validation: Stable convergence, <5% collision rate

---

## Pre-Training Checklist

- [ ] Verify Phase 0 checkpoint exists
- [ ] Update Unity environment for overtaking detection logging
- [ ] Implement explicit lane-change event detection
- [ ] Prepare video recording for episode validation
- [ ] Create TensorBoard dashboard for monitoring
- [ ] Verify YAML config valid: vehicle_ppo_phase-B.yaml
- [ ] Test training command with 1K steps (sanity check)
- [ ] Document baseline metrics from Phase A

---

## Risk Analysis

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Speed degrades >15% | Low | High | Contingency B (revert weight) |
| Overtaking still not detected | Low | High | Contingency A (explicit logging) |
| Safety compromised (>5% collision) | Very Low | Critical | Contingency C (increase penalties) |
| Training stalls | Low | Medium | Contingency D (learning rate boost) |
| Curriculum progression fails | Low | Medium | Monitor lesson transitions |

---

## Success Metrics Summary

**Minimum Success**:
- Mean reward >= +1500
- Overtaking events > 150
- Overtaking success rate > 70%
- Collision rate < 5%
- Goal completion > 90%

**Expected Success**:
- Mean reward +1600-1800
- Overtaking events > 300
- Overtaking success rate > 75%
- Collision rate 0-2%
- Goal completion > 95%

---

## Files & Artifacts

- **Config**: /python/configs/planning/vehicle_ppo_phase-B.yaml
- **Experiment Dir**: /experiments/phase-B-decision/
- **Hypothesis**: HYPOTHESIS.md (root cause analysis)
- **Training Guide**: TRAINING-GUIDE.md (commands & monitoring)
- **Comparison**: COMPARISON.md (Phase A vs B detailed analysis)

---

*Phase B Design Summary - Created: 2026-01-28*
*Status: Ready for Approval*
