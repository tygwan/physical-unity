# Phase C Design Request - Multi-NPC Interaction & Advanced Decision-Making

**Created**: 2026-01-29
**Based on**: Phase B v2 completion (+877 reward, 3 NPCs)
**Status**: Ready for design phase

## Context Summary

### Previous Phase Results
- **Phase A**: +2113 reward (single NPC overtaking)
- **Phase B v1**: -108 reward (FAILED - reward design error)
- **Phase B v2**: +877 reward (decision learning, 3 NPCs, no overtaking observed)

### Key Challenge
Phase B v2 succeeded in stable driving with 3 NPCs but OvertakePhase_Active=0.
Agent learned defensive behavior (no overtaking) but driving is safe.

### Phase C Objectives
1. Scale to 5-8 NPCs (multi-agent complexity)
2. Encourage selective overtaking in constrained environment
3. Learn when to overtake vs wait vs switch lanes
4. Maintain safety (collision rate < 5%)
5. Achieve +1200+ reward (2-3% improvement over B v2)

## Design Hypotheses

### H1: Overtaking Risk Increases with NPC Count
**Assumption**: In 3-NPC environment, overtaking risk > benefit
**Prediction**: More NPCs = harder to find safe overtaking window
**Test**: Phase C Stage 0 (3 NPCs identical) should match Phase B v2 rewards

### H2: Overtaking Bonus Needs Scaling
**Assumption**: Fixed +2.0 overtaking bonus insufficient for 8-NPC scenario
**Prediction**: Overtaking should scale: easy_overtake(+1.0), medium(+2.0), hard(+3.0)
**Test**: Implement dynamic overtaking bonus based on blocked_duration

### H3: Curriculum Design is Critical
**Assumption**: Gradual NPC scaling (3→4→5→6→8) better than sudden jump
**Prediction**: Each stage teaches different interaction pattern
**Test**: Measure stage completion times and reward progression

### H4: Phase A vs B v2 Initialization
**Assumption**: Phase B v2 (+877) has decision-making not in Phase A (+2113 = pure overtaking)
**Prediction**: Phase B v2 checkpoint better for Phase C (already knows limitations)
**Test**: Compare convergence speed from each checkpoint

## Design Specifications

### Initialization Strategy
- **Candidate A**: Phase A checkpoint (E2EDrivingAgent-2500155.pt)
  - Pros: Proven overtaking capability, clean baseline
  - Cons: No decision-making experience, might regress
- **Candidate B**: Phase B v2 checkpoint (latest from phase-B-decision)
  - Pros: Already knows 3-NPC management, decision-making
  - Cons: Hasn't explicitly practiced overtaking, risk aversion encoded

**Recommendation**: Phase B v2 (builds on learned safety + decision-making)

### Curriculum Design (5 Stages)

**Stage 0: Consolidation** (0-500K steps)
- num_active_npcs: 3 (same as Phase B v2)
- Objective: Verify learned behavior transfers
- Target reward: +850 (match Phase B v2)
- Purpose: Baseline, no new learning

**Stage 1: Gradual Scaling** (500K-1.2M steps)
- num_active_npcs: 4
- npc_speed_ratio: 0.4→0.6 (slow NPCs encourage overtaking)
- Objective: Learn 4-vehicle coordination
- Target reward: +1000

**Stage 2: Medium Complexity** (1.2M-2.0M steps)
- num_active_npcs: 5
- npc_speed_ratio: 0.5→0.8 (varied speeds)
- Objective: Selective overtaking decisions
- Target reward: +1200

**Stage 3: High Complexity** (2.0M-2.8M steps)
- num_active_npcs: 6
- npc_speed_ratio: 0.6→0.9 (mostly fast NPCs)
- Objective: Multi-lane coordination
- Target reward: +1400

**Stage 4: Full Challenge** (2.8M-3.6M steps)
- num_active_npcs: 8
- npc_speed_ratio: 0.5→1.0 (full range)
- Objective: Complex real-world scenario
- Target reward: +1500

### Reward Structure Changes

**Keep from Phase B v2** (proven):
- Speed reward: +0.3/step
- Lane center bonus: +0.2/step
- Collision penalty: -10.0
- Off-road penalty: -5.0
- Time penalty: -0.001/step

**NEW for Phase C** (to encourage overtaking):
1. **Dynamic Overtaking Bonus**:
   - Easy overtake (blocked_duration < 10s): +1.0
   - Medium overtake (10-30s blocked): +2.0
   - Hard overtake (>30s blocked): +3.0
   
2. **Blocked Detection Suspension** (critical):
   - When NPC ahead AND blocked_duration > 5s
   - Suspend speed penalty (don't penalize for stopping when stuck)
   - Incentivize overtaking decision instead

3. **Lane Change Encouragement**:
   - Lateral movement penalty reduced: -0.02/rad (from -0.1)
   - Lane change completion bonus: +0.5 (from 0)

4. **Selective Following Penalty** (refined):
   - Only when actually following (moving at <80% speed of leader)
   - NOT when moving faster or when completely blocked
   - Value: -0.1/step (softer than B v2 experiment)

### Environmental Parameters

**NPC Behavior**:
- Speed range: 30-60 km/h (8-16 m/s) - realistic highway
- Density: 5-8 vehicles (16 training areas = 80-128 NPCs total)
- Spacing: Varied, creating natural blocking scenarios
- Predictability: Mix of predictable + random behavior

**Goal Distance**:
- Stage 0-1: 80-120m
- Stage 2-3: 120-200m
- Stage 4: 200-300m

**Speed Zones**:
- Always 2-3 zones for variety
- Speed limits: 40, 50, 60 km/h (realistic mixture)

## Success Metrics

| Metric | Target | Validation |
|--------|--------|-----------|
| Stage 4 Final Reward | +1500 | TensorBoard mean reward |
| Overtaking Events | >20 per 1000 steps | Custom logging |
| Collision Rate | <5% | Episode statistics |
| Safety Maintained | Yes | No catastrophic failures |
| Convergence Speed | 3.6M steps | Compared to Phase B v2 |

## Monitoring Plan

### Real-Time Checks (TensorBoard)
- Mean Reward trajectory
- Episode length (should stay ~2500 steps)
- Collision rate (should stay <5%)
- Speed compliance (>85%)

### Validation Checkpoints
- 100K steps: Speed > 5 m/s (basic driving)
- 500K steps: Reward > +500 (making progress)
- 1.2M steps: Transition to 5 NPCs smooth
- 3.6M steps: Stage 4 started converging

### Root Cause Prevention
- Auto-stop if mean reward < -50 for 3 checkpoints (prevent Phase B v1 repeat)
- Alert if collision rate > 10% (safety issue)
- Track overtaking events separately (debug if zero)

## Expected Timeline

| Item | Duration |
|------|----------|
| Total Training | 3.6M steps ≈ 45-50 minutes |
| Setup & validation | 15 minutes |
| Stage 0-2 | 1.2M steps ≈ 15 minutes |
| Stage 2-4 | 2.4M steps ≈ 30 minutes |

**Total Wall Time**: ~1 hour on RTX 4090

