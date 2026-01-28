# Phase A vs Phase B: Detailed Comparison & Transition Plan

**Date**: 2026-01-28
**Context**: Phase A achieved +2113.75 reward but with 0 overtaking events
**Goal**: Phase B addresses overtaking decision-making gap

---

## Side-by-Side Comparison

### Initialization & Architecture

| Aspect | Phase A | Phase B | Change |
|--------|---------|---------|--------|
| **Base Checkpoint** | Phase 0 (Phase 0) | Phase 0 (Phase 0) | Same |
| **Init Speed** | 68% faster | Baseline | Slower (conservative) |
| **Bias Risk** | Speed-only | None | Reduced |
| **Network** | 512x3 | 512x3 | Unchanged |

### Reward Structure

| Component | Phase A | Phase B | Change | Rationale |
|-----------|---------|---------|--------|-----------|
| **Progress** | 1.0 | 1.0 | — | Unchanged |
| **Speed** | 0.5 | 0.3 | -40% | Reduce dominance |
| **Lane Center** | None | 0.2 | NEW | Encourage positioning |
| **Following Penalty** | None | -0.5 | NEW | Create urgency |
| **Overtaking Bonus** | 3.0 | 5.0 | +67% | Increase incentive |
| **Collision** | -10.0 | -10.0 | — | Safety maintained |

**Phase A Composition**:
```
Total: 2113.75
Speed Tracking:  ~1980 (93.9%)
Progress:        ~289 (13.7%)
Overtaking:      0 events (0.0%)
```

**Phase B Expected**:
```
Total: 1600
Speed:    ~480 (30%)
Progress: ~200 (12.5%)
Lane:     ~50 (3.1%)
Overtaking: ~300 (18.7%)
```

### Curriculum

| Aspect | Phase A | Phase B |
|--------|---------|---------|
| **Stage 1** | 1 NPC (density) | 0 NPCs (baseline) |
| **Stage 2** | 2-3 NPCs | 1 slow NPC (forced overtaking) |
| **Stage 3** | 4-5 NPCs | 2 mixed NPCs (selective) |
| **Stage 4** | — | 4 variable NPCs (complex) |
| **Focus** | NPC density | Overtaking decisions |

### Performance Expectations

| Metric | Phase A | Phase B | Change |
|--------|---------|---------|--------|
| **Mean Reward** | +2113.75 | +1500-1800 | -29% (by design) |
| **Collision Rate** | 0.0% | 0-2% | Maintained |
| **Training Steps** | 2.5M | 3.0M | +20% |
| **Training Time** | 29.6 min | ~25 min | Faster |
| **Overtaking Events** | 0 | >150 | NEW FOCUS |

---

## Risk Analysis: Phase A → Phase B

### Risk 1: Speed Performance Regression

**Probability**: Medium (40%)
**Impact**: High
**Description**: Reducing speed weight 0.5 → 0.3 may reduce speed tracking

**Mitigation**:
- Add lane-center reward (0.2) to compensate
- Contingency B: Can revert to 0.4-0.5 if needed
- Monitor speed component in TensorBoard

**Acceptance**: Final speed >= 80% of Phase A

---

### Risk 2: Overtaking Detection Still Fails

**Probability**: Low (20%)
**Impact**: Critical
**Description**: Phase B still zero overtaking events despite fixes

**Mitigation**:
- Stage 1 validation milestone (explicit logging)
- Contingency A: Implement lateral displacement tracking
- Video review protocol for manual validation

**Acceptance**: >150 overtaking events by 2.25M steps

---

### Risk 3: Safety Compromised

**Probability**: Very Low (5%)
**Impact**: Critical
**Description**: Collision rate spikes above 5%

**Mitigation**:
- Conservative penalty structure (-10 collision, -2 near-collision)
- Contingency C: Can increase penalties by 50% immediately
- Curriculum staged (single NPC in Stage 1)

**Acceptance**: Collision rate stays 0-5%

---

### Risk 4: Training Instability

**Probability**: Low (15%)
**Impact**: Medium
**Description**: Reward oscillates or stalls

**Mitigation**:
- Learning rate proven at 3e-4 from Phase A
- Contingency D: Can increase to 5e-4 if stalling
- Conservative curriculum thresholds

**Acceptance**: Smooth convergence, oscillation <300

---

## Rollback & Recovery Strategy

### Scenario 1: Overtaking Events Still = 0 at 1.5M Steps

**Trigger**: Zero detected events after Stage 1

**Immediate Action**:
1. PAUSE at 1.5M steps
2. Implement explicit lane-change detection logging
3. Save checkpoint

**Investigation**:
1. Review 10 episode videos (manual lane-change count)
2. Compare detected vs observed lane changes
3. Identify detection system gap

**Recovery Path A** (If agent IS switching lanes):
1. Implement lateral displacement metric
2. Resume from 1.5M checkpoint
3. Use new metric for overtaking validation

**Recovery Path B** (If agent NOT switching lanes):
1. Switch to Phase A initialization
2. Add explicit "required overtakes" threshold
3. Retrain Stage 1
4. Resume training

---

### Scenario 2: Speed Performance Drops >15%

**Trigger**: Speed reward <60% of Phase A baseline at 750K

**Recovery**:
1. Edit YAML: speed_reward_weight: 0.3 → 0.4
2. Resume from checkpoint
3. Retrain Stage 0 (100K test)
4. Validate speed improvement

**Alternative**:
1. Reduce following penalty: -0.5 → -0.2
2. Increase lane-center: 0.2 → 0.3
3. Keep speed at 0.3

---

### Scenario 3: Collision Rate >5%

**IMMEDIATE ACTION**:
1. PAUSE training
2. Increase collision penalty: -10.0 → -15.0
3. Increase near-collision: -2.0 → -5.0
4. Resume from latest checkpoint

**Secondary**:
1. Reduce overtaking bonus: +5.0 → +3.0
2. Go back one stage
3. Retrain conservatively

**Hard Stop**: If >8% persists after adjustments

---

## Decision Tree: Phase B Progression

```
START Phase B
├─ 750K: Stage 0 complete?
│  ├─ YES: Reward +600-800? Speed >80% Phase A?
│  │  ├─ YES: CONTINUE
│  │  └─ NO: Contingency B (adjust weights)
│  └─ NO: Extend Stage 0
│
├─ 1.5M: Stage 1 complete?
│  ├─ YES: >1 overtaking event/episode?
│  │  ├─ YES: CONTINUE to Stage 2
│  │  └─ NO: Contingency A (detection fix)
│  └─ NO: Reward >1000? Extend Stage 1
│
├─ 2.25M: Stage 2 complete?
│  ├─ YES: >70% correct decisions? >150 events?
│  │  ├─ YES: CONTINUE to Stage 3
│  │  └─ NO: Adjust curriculum/weights
│  └─ NO: Reward >1400? Extend Stage 2
│
├─ 3.0M: Stage 3 complete?
│  ├─ YES: Reward +1500+? Collision <5%?
│  │  ├─ YES: PHASE B SUCCESS ✓
│  │  └─ NO: Document and iterate
│  └─ NO: Evaluate current checkpoint
│
└─ Throughout: Collision >8%?
   └─ YES: EMERGENCY STOP, Contingency C
```

---

## Contingency Quick Reference

| Contingency | Trigger | Action | Recovery |
|-------------|---------|--------|----------|
| **A** | Overtaking = 0 | Add logging | 500K steps |
| **B** | Speed -15% | Adjust weights | 250K steps |
| **C** | Collision >5% | Increase penalties | Immediate |
| **D** | Stall >500K | Increase LR | 250K steps |
| **Rollback** | Multiple failures | Switch Phase A | Restart |

---

## Success Definition

### Minimum Success (Proceed to Phase C)
- Mean reward >= +1500
- Overtaking events > 150
- Collision rate < 5%
- Goal completion > 90%
- Smooth convergence

### Expected Success (Ideal)
- Mean reward +1600-1800
- Overtaking events > 300
- Overtaking success rate > 70%
- Collision rate 0-2%
- Goal completion > 95%

### Failure Criteria (Triggers Redesign)
- Mean reward < +1200 at 2.5M steps
- Overtaking events = 0 after Stage 1 validation
- Collision rate > 8% at any point
- Multiple contingencies required AND not resolved
- Curriculum fails to progress

---

## Approval Checklist

Before training starts:

- [ ] Phase A analysis complete (0 overtaking events documented)
- [ ] Root cause hypothesis approved (speed dominance likely)
- [ ] Phase 0 checkpoint verified available
- [ ] YAML config validated and tested (10K steps sanity check)
- [ ] Overtaking detection logging enabled
- [ ] TensorBoard dashboard prepared
- [ ] Emergency stop procedures documented
- [ ] Contingency plans reviewed
- [ ] Timeline and resources confirmed

---

*Phase A vs Phase B Comparison - Created: 2026-01-28*
