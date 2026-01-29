# Phase D Design: Lane Observation & Perception Integration

Created: 2026-01-29
Status: Complete (v1) - Analysis & v2 Design
Expected Duration: 6M steps (v1) to 8-10M steps (v2)
Expected Reward: +1000 (v1 failed) to +1200+ (v2)

## Executive Summary

Phase D extends Phase C success (+1,372 reward, 8 NPCs) by adding explicit lane boundary perception (12D).

Problem: Phase C agent infers lane from collision penalties implicitly
Solution: Add explicit lane observation to help agent understand road geometry
Status: v1 Failed due to simultaneous curriculum transitions
Next: v2 with staggered curriculum

## 1. Background: Why Lane Observation?

Phase C Achievements:
- Reward: +1,372 (72% improvement over Phase B v2)
- NPCs: 3 to 8 (scaled successfully)
- Overtaking: Learned selective overtaking decisions
- Safety: Collision rate maintained less than 5%

Phase C Limitations:
- Observation Space: 242D (no explicit lane info)
- Lane Learning: Implicit (from collision penalties)
- Road Understanding: Limited to vehicle-relative features
- Future Phases: Cannot handle multi-lane scenarios

Phase D Hypothesis:

H1: Explicit lane observation enables faster learning
- Phase C learned implicitly (slow)
- Phase D learns explicitly (fast)
- Expected: Converge 2-3x faster

H2: Lane info helps multi-zone navigation
- Phase C: Single speed zone per area
- Phase D: Multiple speed zones in same area
- Clear lane boundaries help transition decisions

H3: Preparation for Phase E (Curved Roads)
- Straight roads: Lane is obvious (physics)
- Curved roads: Need explicit road curvature
- Phase D: Bridge between straight and curved

## 2. Technical Design: Lane Observation (12D)

Observation Space Expansion:

Old (242D):
- ego_state (8D): position, velocity, heading, acceleration
- route_info (30D): 6 waypoints x 5 values
- surrounding (40D): 8 vehicles x 5 features
- padding (164D): Reserved

New (254D):
- ego_state (8D): unchanged
- route_info (30D): unchanged
- surrounding (40D): unchanged
- LANE OBSERVATION (12D): NEW
- padding (164D): same

Lane Observation Details:

Query Geometry:
- 4 query points along ego vehicle's forward direction
- Point 0: -20m (behind ego)
- Point 1: 0m (at ego position)
- Point 2: +20m (ahead)
- Point 3: +40m (far ahead)

Per-Point Features (3D):
- right_distance: Distance to right lane marking
- left_distance: Distance to left lane marking
- marking_type: Enum (0=none, 1=solid_white, 2=dashed)

Total: 4 points x 3 features = 12D

## 3. Scene Setup: PhaseB_DecisionLearning

TrainingArea Structure:

Each of 16 TrainingAreas:
- Training ground: 200m long straight road
- Width: 6 meters (3m lane width + 1.5m per side)
- Markings: LeftEdge (at -3m), RightEdge (at +3m)
- Layer: 8 (LaneMarking)

GameObject Names:
- Parent: TrainingArea_0 through TrainingArea_15
  - LeftEdge: Line renderer or collider
  - RightEdge: Line renderer or collider
  - Road: Visual representation
  - NPCs: Dynamic vehicles

Verification Checklist:
1. All 16 TrainingAreas have LeftEdge and RightEdge
2. Both objects on Layer 8 (LaneMarking)
3. Both extend full length (200m)
4. No collision (triggers, not rigid bodies)
5. Test raycasts in editor

## 4. Curriculum Design (v1 - Failed)

5-Parameter Curriculum:

Parameter 1: num_active_npcs
- Stage 0: 1 NPC
- Stage 1: 2 NPCs
- Stage 2: 4 NPCs
- Progression: Reward-based

Parameter 2: npc_speed_ratio
- Stage 0: 0.3
- Stage 1: 0.6
- Stage 2: 0.9
- Progression: Reward-based

Parameter 3: goal_distance
- Stage 0: 80m
- Stage 1: 150m
- Stage 2: 230m
- Progression: Reward-based

Parameter 4: speed_zone_count
- Stage 0: 1 zone
- Stage 1: 2 zones
- Stage 2: 4 zones
- Progression: Reward-based

Parameter 5: npc_speed_variation
- Stage 0: 0.0
- Stage 1: skip
- Stage 2: 0.3
- Progression: Reward-based

Progression Strategy:

v1 (Failed) Strategy: Independent curricula
- Each parameter progresses independently
- When reward reached, transition immediately
- Problem: All 5 can transition simultaneously
- Result: Catastrophic at 4.68M steps

v2 (Proposed) Strategy: Staggered curricula
- Progression strictly ordered
- One parameter at a time
- Manual validation
- Fallback mechanism

## 5. Failure Analysis: v1 Curriculum Collapse

Timeline of Failure:

0-100K steps: Exploration
- Reward: -58 to -40
- Status: Normal

100K-1.5M: Stabilization
- Reward: -40 to -36
- Status: Slow but expected

1.5M-3.5M: Recovery
- Reward: -36 to +298
- Status: Recovery in progress

3.5M-4.68M: Success phase
- Reward: +298 to +406
- Peak: +406 at 4.6M
- Status: ON TRACK

4.68M-4.7M: CATASTROPHIC COLLAPSE
Three parameters transitioned simultaneously:
1. num_active_npcs: 1 to 2
2. speed_zone_count: 1 to 2
3. npc_speed_variation: 0.0 to 0.3

Result: Reward collapsed from +406 to -4,825
Magnitude: 5,231 point drop in less than 20K steps

4.7M-6M: Attempted recovery
- Reward: -4,825 to -2,156
- Status: UNRECOVERED

Root Cause:

Agent Learning Paradigm:
When agent masters simple scenario, it learns scenario-specific policies:
1. Overtaking: When to overtake 1 NPC
2. Navigation: How to follow 1 zone
3. Prediction: Deterministic NPC behavior

Transition Effect:
Suddenly: 2 NPCs + 2 zones + 0.3 variation
- Old overtaking policy now wrong
- Navigation policy no longer applies
- Prediction model broken

Unlearning Cost:
- Cost to forget old policies: 10,000+ reward loss
- Time to learn new policies: over 1M steps
- Net effect: Collapse and slow recovery

## 6. Design for Phase D v2

Conservative Curriculum Strategy:

Principle: Staggered Single-Parameter Progression

Stage Sequence:

Stage 0 (Baseline):
- num_active_npcs: 1
- npc_speed_ratio: 0.3
- goal_distance: 80m
- speed_zone_count: 1
- npc_speed_variation: 0.0
- Target: +600
- Duration: 1M steps or convergence
- Completion: Reward greater than 700 or 500K steps

Stage 1 (NPC Scaling):
- num_active_npcs: 2 (increase by 1)
- Others: Unchanged
- Target: +750
- Duration: 1.5M steps

Stage 2 (Speed Ratio):
- npc_speed_ratio: 0.6
- Others: Unchanged
- Target: +800
- Duration: 1M steps

Stage 3 (Goal Distance):
- goal_distance: 150m
- Others: Unchanged
- Target: +850
- Duration: 1M steps

Stage 4 (Speed Zones):
- speed_zone_count: 2
- Others: Unchanged
- Target: +900
- Duration: 1.5M steps

Stage 5 (NPC Variation):
- npc_speed_variation: 0.3
- Others: Unchanged
- Target: +1000+
- Duration: 2M steps

Total Expected: 8M-9M steps (vs v1: 6M)

Checkpoint Strategy:

Save after each stage:
- ckpt_stage_0_baseline.pt
- ckpt_stage_1_1npc.pt
- ckpt_stage_2_2npc.pt
- etc.

Fallback mechanism:
- If reward drops greater than 50% for 200K steps, revert
- Manual decision required to proceed

## 7. Expected Results

v2 Reward Progression:

Stage 0: -50 to 600 (converge to baseline)
Stage 1: 600 to 750 (NPC scaling)
Stage 2: 750 to 800 (speed ratio)
Stage 3: 800 to 850 (goal distance)
Stage 4: 850 to 900 (speed zones)
Stage 5: 900 to 1000+ (NPC variation)

Final: 1000-1200 (success)

Success Criteria:
1. Stage 5 reward >= 1000
2. No collapse greater than 1000 within 100K steps
3. Collision rate less than 5%
4. All stages completed
5. Checkpoint saved at each transition

Failure Criteria (any triggers STOP):
1. Final reward less than 800 (regression)
2. Stage stuck greater than 2M steps
3. Collision rate greater than 10%
4. Hardware errors

## 8. Files & References

Config Files:
- python/configs/planning/vehicle_ppo_phase-D.yaml (v1)
- python/configs/planning/vehicle_ppo_phase-D-v2.yaml (v2)

Unity Assets:
- Assets/Scripts/Agents/E2EDrivingAgentBv2.cs
- Assets/Scenes/PhaseB_DecisionLearning.unity

Results:
- results/phase-D/ (logs + checkpoints)
- experiments/phase-D-lane-observation/ (docs)

---

Design Status: Ready for v2 Implementation
Next Step: Update vehicle_ppo_phase-D-v2.yaml
Timeline: 24 hours approval, 48 hours training
