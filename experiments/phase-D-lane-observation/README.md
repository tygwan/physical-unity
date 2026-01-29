# Phase D: Lane Observation & Perception Integration

**Status**: Training Complete - Post-Mortem Analysis
**Created**: 2026-01-29
**Training Duration**: 6,000,000 steps (~100 minutes)
**Outcome**: FAILURE - Critical Curriculum Collapse at Step 4.68M

## Quick Reference

Run ID: phase-D
Config: python/configs/planning/vehicle_ppo_phase-D.yaml
Scene: PhaseB_DecisionLearning (16 training areas, single-lane straight road)
Agent: E2EDrivingAgentBv2 with enableLaneObservation=true

## Key Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Initial Reward | -58 | Exploratory phase |
| Peak Reward | +406 @ 4.6M steps | Near success |
| Final Reward | -2,156 @ 6M steps | Catastrophic collapse |
| Total Duration | 100 minutes | Budget exhausted |
| Curriculum Completion | 3/5 parameters reached max | Partial |

## Phase D: What & Why?

### Background: Phase C Success
- Phase C Result: +1,372 reward with 8 NPCs across 3.6M steps
- Observation Space: 242D (ego state + route + surrounding vehicles)
- Limitation: No explicit lane boundary information
- Insight: Agent may not understand road constraints at physics level

### Phase D Innovation: Lane Observation (12D)

Adding explicit lane boundary information to observation space:

Old Space: 242D
- ego_state (8D): position, velocity, heading, acceleration
- route_info (30D): waypoints, distances
- surrounding (40D): 8 vehicles x 5 features

New Space: 254D (+12D)
- ego_state (8D)
- route_info (30D)
- surrounding (40D)
- LANE OBSERVATION (12D) <- NEW
  - left_lane_distance (4 values at relative positions)
  - right_lane_distance (4 values at relative positions)

### Why Lane Observation?

1. Implicit vs Explicit Knowledge
   - Phase C: Agent infers lane from collision penalties
   - Phase D: Agent observes lane directly (faster learning)

2. Safety & Precision
   - Lane boundaries = hard constraints
   - Removes ambiguity about road boundaries

3. Overtaking Decisions
   - Clear lane geometry helps judge safe lane changes
   - Relevant for Phase E (curved roads) and Phase F (multi-lane)

## Scene Setup: Lane Marking System

### LaneMarking GameObjects
Each of 16 TrainingAreas includes:
- LeftEdge: Solid white lane marking (left boundary)
- RightEdge: Solid white lane marking (right boundary)
- Layer: LaneMarking (index 8)
- Shape: Linear road edges across training area

### Detection Implementation

In E2EDrivingAgentBv2.cs:

```csharp
// Physics raycast from ego position
foreach (float relativePos in new[] { -20, 0, 20, 40 })
{
    Vector3 queryPos = egoPos + forward * relativePos;
    
    Physics.Raycast(queryPos, Vector3.right, out hit, rayDistance, 
                    1 << LayerMask.NameToLayer("LaneMarking"));
    float rightDistance = hit.distance ?? maxDistance;
    
    Physics.Raycast(queryPos, -Vector3.right, out hit, rayDistance,
                    1 << LayerMask.NameToLayer("LaneMarking"));
    float leftDistance = hit.distance ?? maxDistance;
    
    observation[242 + offset] = rightDistance / maxDistance;
    observation[243 + offset] = leftDistance / maxDistance;
}
```

## Training Configuration

### Curriculum: 5 Parameters, 3 Stages Each

| Parameter | Stage 0 | Stage 1 | Stage 2 |
|-----------|---------|---------|---------|
| num_active_npcs | 1 | 2 | 4 |
| npc_speed_ratio | 0.3 | 0.6 | 0.9 |
| goal_distance | 80m | 150m | 230m |
| speed_zone_count | 1 | 2 | 4 |
| npc_speed_variation | 0.0 | - | 0.3 |

## Training Timeline

### Success Window (3.5M-4.68M steps)
- Reward: +298 to +406 (steady climb)
- Peak: +406 at 4.6M steps
- Status: On track for success

### CRITICAL: Simultaneous Curriculum Transition (4.68M-4.7M steps)

What Happened:
Three curriculum parameters transitioned simultaneously:
1. num_active_npcs: 1 to 2
2. speed_zone_count: 1 to 2
3. npc_speed_variation: 0 to 0.3

Actual Result:
- Step 4.68M: +406 reward
- Step 4.7M: -4,825 reward
- Collapse: 5,231 points in less than 20K steps

### Attempted Recovery (4.7M-6M steps)
- Reward: -4,825 to -2,156 (slow recovery)
- Status: Failed before budget exhausted

## Root Cause: Curriculum Collapse

Why Simultaneous Transitions Failed:
- Agent learned scenario-specific policies
- Old policies (1 NPC) became liabilities
- New policies not yet formed
- Unlearning cost > Learning benefit

## Lessons Learned

### L1: Curriculum Design - Independence Assumption
Finding: Not all curriculum parameters are truly independent
Implication: Stagger transitions, one parameter at a time

### L2: Lane Observation - Effective but Not Sufficient
Finding: Lane info helps (+406 peak) but doesn't solve decision-making
Implication: Keep for Phase E/F but improve reward shaping

### L3: Peak Performance Doesn't Guarantee Robustness
Finding: +406 reward was brittle, not truly converged
Implication: Need better validation criteria

## Success Criteria: Why This Was Failure

| Criterion | Target | Achieved |
|-----------|--------|----------|
| Stage 2 Completion | All 5 params | Only 3 |
| Final Reward | +1000+ | -2,156 |
| Stable Convergence | Monotonic improve | Volatile, collapse |

Verdict: PHASE D v1 = FAILURE

## Recommendations for Phase D v2

### Approach: Conservative Curriculum Redesign
1. Single-Parameter Progression (one at a time)
2. Looser Thresholds (time-based + reward-based)
3. Checkpoint Strategy (save at each stage)
4. Expected Timeline: 8M-10M steps (vs v1: 6M)

## Files

README.md - experiment overview
DESIGN.md - detailed design and lane observation specs
ANALYSIS.md - v1 failure analysis and v2 recommendations

## References

- Phase C Success: experiments/phase-C-multi-npc/ANALYSIS.md
- Phase B v2 Success: experiments/phase-B-decision-v2/DESIGN.md
- Phase B v1 Failure: experiments/phase-B-decision/ANALYSIS.md

Status: Phase D v1 Complete (FAILURE)
Decision Point: 2026-01-30

