# Development Phases

Autonomous Driving ML Platform의 개발 단계별 기술 설계서입니다.

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS DRIVING ML PLATFORM - PHASES                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│  │ Phase 1  │   │ Phase 2  │   │ Phase 3  │   │ Phase 4  │                  │
│  │Foundation│   │  Data    │   │Perception│   │Prediction│                  │
│  │    DONE  │   │   DONE   │   │ SUSPENDED│   │ SUSPENDED│                  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘                  │
│                                      │              │                        │
│                                      │ (Simplified) │                        │
│                                      └──────┬───────┘                        │
│                                             │                                │
│                                             v                                │
│                                      ┌──────────┐                            │
│                                      │ Phase 5  │ PRIMARY FOCUS              │
│                                      │ Planning │ IN PROGRESS (~47%)         │
│                                      │    <<    │                            │
│                                      └────┬─────┘                            │
│                                           │                                  │
│                                           v                                  │
│                   ┌──────────┐     ┌──────────┐                              │
│                   │ Phase 6  │  ->  │ Phase 7  │                              │
│                   │Integration│     │ Advanced │                              │
│                   │   PLANNED  │     │ PLANNED  │                              │
│                   └──────────┘     └──────────┘                              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Phase Status Summary (1-7)

| Phase | Name | Status | Completion | Description |
|-------|------|--------|------------|-------------|
| **Phase 1** | Foundation & Architecture | COMPLETED | 2026-01-22 | Unity 6, ML-Agents 4.0, Windows Native |
| **Phase 2** | Data Infrastructure | COMPLETED | 2026-01-22 | nuPlan/Waymo pipeline, preprocessing |
| **Phase 3** | Perception Models | SUSPENDED | - | Ground Truth method for simplification |
| **Phase 4** | Prediction Models | SUSPENDED | - | Constant Velocity baseline for simplification |
| **Phase 5** | Planning Models | IN PROGRESS | ~47% | RL/IL Motion Planner (PRIMARY FOCUS) |
| **Phase 6** | Integration & Evaluation | NOT STARTED | - | E2E system integration, nuPlan benchmark |
| **Phase 7** | Advanced Topics | NOT STARTED | - | World Model, LLM Planning, Sim-to-Real |

## Phase 5: Planning - Detailed Progress

Phase 5 is the **primary focus** of this project. It is divided into macro stages (0-L) representing progressive complexity levels.

### Planning Sub-Phases: Macro Stages (0-L)

| Sub-Phase | Focus | Status | Peak Reward | Steps | Key Metric |
|-----------|-------|--------|-------------|-------|-----------|
| **Stage 0** | Foundation: Lane keeping | COMPLETED | +1018 | 8.0M | Perfect safety |
| **Stage A** | Dense Overtaking (1 slow NPC) | COMPLETED | +2114 | 2.5M | Excellent generalization |
| **Stage B v1** | Decision Making (0→3 NPCs) | FAILED | -108 | 1.5M | Learned to STOP (harsh penalty) |
| **Stage B v2** | Decision Making (recovery) | COMPLETED | +877 | 3.5M | Recovery from v1 failure |
| **Stage C** | Multi-NPC (4-8 NPCs) | SKIPPED | - | - | D v3 242D 런으로 간접 검증 (+835) |
| **Stage D v1** | Lane Observation (254D) | FAILED | +406 | 6.0M | Curriculum collapse @ 4.68M |
| **Stage D v2** | Lane Obs (staggered curriculum) | FAILED | +447 | 10M | Collapse @ 7.87M (4 NPC transition) |
| **Stage D v3** | Lane Obs (fixed env, 254D) | COMPLETED | +895 | 5M | P-009 적용, lane obs +7.2% 향상 |
| **Stage E** | Curved Roads + Curves | COMPLETED | +938 | 6.0M | Sharp curves mastered, curriculum collapse recovery |
| **Stage F v1** | Multi-Lane (Wrong Scene) | FAILED | -8 | 5.82M | Scene mismatch (P-011) |
| **Stage F v2** | Multi-Lane Switching | IN PROGRESS | - | 6M | Correct scene loaded |
| **Stage G** | Intersections | PLANNED | - | 6-8M | Crossing logic |
| **Stage H-L** | Advanced Scenarios | PLANNED | - | TBD | Pending SOTIF redesign |

### Completed Stages - Detailed Summary

#### Stage 0: Foundation (Lane Keeping)
- **Completion Date**: 2026-01-27
- **Result**: SUCCESSFUL
- **Peak Reward**: +1018.43
- **Safety**: 0% collision rate (perfect)
- **Duration**: 1.17 hours
- **Key Achievement**: Basic lane keeping + speed control established

#### Stage A: Dense Overtaking
- **Completion Date**: 2026-01-28
- **Result**: SUCCESSFUL
- **Peak Reward**: +3161 at 1.999M steps | Final: +2113
- **Safety**: 0% collision rate (perfect)
- **Duration**: 29.6 minutes
- **Episodes**: 238/238 successful (100%)
- **Key Achievement**: Agent reliably overtakes slow NPC (1.5 m/s)

#### Stage B v1: Decision Making (FAILED)
- **Attempt Date**: 2026-01-28
- **Result**: FAILED - Root cause: Harsh reward penalty
- **Final Reward**: -108 (completely degenerate)
- **Root Cause**: speedUnderPenalty=-0.1 per step caused agent to learn STOP as optimal policy
- **Curriculum Issue**: Started at 2 NPCs immediately (no warmup)
- **Duration**: 1.5 hours before termination
- **Recovery**: Redesigned as v2 with penalty reduction and gradual curriculum

#### Stage B v2: Decision Making (SUCCESS - Recovery from v1)
- **Completion Date**: 2026-01-29
- **Result**: SUCCESSFUL (Recovery from v1)
- **Peak Reward**: +897 at 3.49M steps | Final: +877
- **Safety**: ~0% collision rate
- **Duration**: 11.8 minutes (1M new steps from Phase A checkpoint)
- **Curriculum**: All 4 stages completed (0->1->2->3 NPCs)
- **Key Fixes vs v1**:
  - Used Phase A checkpoint (not Phase 0)
  - Reduced speedUnderPenalty from -0.1 to -0.02 (80% reduction)
  - Added gradual warmup (0 NPC baseline before 1->2->3)
  - Reverted 7 hyperparameters to Phase A baseline
- **Key Achievement**: Agent makes real decisions: overtake vs follow based on NPC speed

#### Stage C: Multi-NPC Generalization
- **Status**: Design Complete - Ready for Training
- **Created**: 2026-01-29
- **Expected Duration**: 3.6M steps (45-50 minutes)
- **Target Reward**: +1500 (71% improvement over B v2)
- **Scope**: 3->4->5->6->8 NPCs with increasing speed variation
- **Critical Change**: Dynamic overtaking bonus (+1/+2/+3 based on difficulty)
- **Init Checkpoint**: Phase B v2 (+877 reward)
- **Success Criteria**:
  - Stage 4 reward >= +1200
  - Collision rate < 5%
  - Overtaking events > 5 per 1000 steps
  - All 5 stages completed
- **Status**: Awaiting approval for training execution

#### Stage D v1: Lane Observation (FAILED - Curriculum Collapse)
- **Completion Date**: 2026-01-29 (premature termination)
- **Result**: FAILED - Curriculum collapse at step 4.68M
- **Peak Reward**: +406 @ 4.6M steps | Final: -2,156 @ 6M steps
- **Observation Space**: 254D (added 12D lane markings)
- **Curriculum**: 5 parameters with 3 stages each (15 total transitions)
- **Root Cause**: Three curriculum parameters transitioned simultaneously:
  - num_active_npcs: 1->2
  - speed_zone_count: 1->2
  - npc_speed_variation: 0->0.3
- **Collapse Magnitude**: 5,231 points drop in <20K steps
- **Lessons**:
  - Curriculum parameters are NOT independent
  - Peak reward doesn't guarantee robustness
  - Lane observation is helpful but not sufficient alone
- **Duration**: 100 minutes
- **Recovery**: Redesigned as v2 with staggered thresholds

#### Stage D v2: Lane Observation - Staggered Curriculum (FAILED)
- **Completion Date**: 2026-01-30
- **Result**: FAILED - Collapse at 7.87M steps during 4 NPC transition
- **Peak Reward**: +447 @ 5.8M steps | Final: -756
- **Key Change**: Staggered transition thresholds (P-002 적용)
- **Root Cause**: Observation space change (254D) + environment change (NPC increase) 동시 적용
- **Recovery**: P-009 원칙 도출 → Phase D v3 설계

#### Stage D v3: Lane Observation - Fixed Environment (COMPLETED)
- **Completion Date**: 2026-01-30
- **Result**: SUCCESSFUL
- **Peak Reward**: +895.5 @ 5M steps
- **Observation Space**: 254D (242D + 12D lane observation)
- **Strategy**: 커리큘럼 완전 제거, 고정 환경 (3 NPC, 0.6 speed ratio)
- **Key Principles**: P-009 (관측-환경 결합 금지), P-010 (Scene-Config-Code 일관성)
- **Duration**: ~22분 (5M steps)
- **242D vs 254D**: +835 → +895 (+7.2% lane observation 효과 검증)
- **Note**: 최초 실행 시 VectorObservationSize=242 오류 발견 → 254로 수정 후 재학습
- **Checkpoint**: `results/phase-D-v3-254d/E2EDrivingAgent/E2EDrivingAgent-4999947.pt`

### Not Started Stages (F-L)

#### Stage E: Curved Roads (COMPLETED)
- **Completion Date**: 2026-01-30
- **Result**: SUCCESSFUL
- **Peak Reward**: +938.2 at 4.5M steps | Final: +892.6 at 6M steps
- **Observation Space**: 254D (same as Phase D v3)
- **Init Checkpoint**: Phase D v3 (--initialize-from=phase-D-v3-254d)
- **Duration**: 6M steps
- **Curriculum**: 7 parameters, all completed to final lesson
  - road_curvature: Straight → GentleCurves(0.3) → ModerateCurves(0.6) → SharpCurves(1.0)
  - curve_direction_variation: SingleDirection → MixedDirections
  - num_active_npcs: 0 → 1 → 2 NPCs
  - npc_speed_ratio: 0.4 → 0.7
  - goal_distance: 100m → 150m → 200m
  - speed_zone_count: 1 → 2
  - npc_speed_variation: 0.0 → 0.2
- **Critical Event**: Curriculum collapse at 1.68M steps
  - 4 parameters (npc_speed_ratio, goal_distance, speed_zone_count, npc_speed_variation) transitioned simultaneously
  - Reward crashed from +362 to -3,863 (P-002 violation)
  - **Recovery**: Agent recovered to positive rewards by 2.44M steps (~800K recovery period)
  - Unlike Phase D v1/v2, the Phase E agent successfully recovered from curriculum shock
- **Curvature Progression**: After recovery, curvature curriculum advanced smoothly
  - 3.47M: GentleCurves (0.3)
  - 3.81M: ModerateCurves (0.6)
  - 4.15M: SharpCurves (1.0)
  - Peak +956 at 3.58M, stable +920-940 through completion
- **Key Achievement**: Agent drives on sharp curved roads with 2 NPCs, mixed directions, 200m goal

#### Stage F v1: Multi-Lane (FAILED - Wrong Scene)
- **Attempt Date**: 2026-01-31
- **Result**: FAILED - Wrong Unity scene loaded (P-011 violation)
- **Peak Reward**: +303 at 1.52M steps | Final: -8.15 (stuck for 4.27M steps)
- **Root Cause**: `PhaseE_CurvedRoads` scene (4.5m road) active instead of `PhaseF_MultiLane` (11.5m road)
- **Failure Mechanism**: `num_lanes` 1->2 transition at step 1520K → road too narrow → instant off-road → -8.19
- **Duration**: 5.82M steps before manual termination
- **Policy Discovered**: P-011 (Scene-Phase Matching)

#### Stage F v2: Multi-Lane (FAILED - Waypoint Destruction)
- **Attempt Date**: 2026-01-31
- **Result**: FAILED - Catastrophic reward collapse at num_lanes 1->2 transition
- **Peak Reward**: +317 at 2.78M steps | Final: -14.2 (stuck for 1.1M steps)
- **Scene**: `PhaseF_MultiLane.unity` (11.5m road, correct scene)
- **Root Cause**: `WaypointManager.SetLaneCount()` called `GenerateWaypoints()` which destroyed all waypoint GameObjects
  - Observation continuity broken: agent's routeWaypoints references invalidated
  - Agent policy locked to [0,0] action (entropy collapse, Std=0.08)
  - Speed=0 -> -0.2/step penalty -> -14.2 per episode (mathematically verified)
- **Duration**: 4.1M steps before manual termination
- **Key Events**:
  - 0-2.78M: Successful single-lane training (+317)
  - 2.78M: center_line_enabled -> 1.0 (smooth transition)
  - 2.99M: num_lanes 1->2 (CATASTROPHIC: +317 -> -14.2)
  - 2.99-4.1M: Stuck at -14.2 (entropy collapsed, no recovery)
- **Fix for v3**: Pre-generate waypoints, shift positions without destruction

#### Stage F v3: Multi-Lane (READY)
- **Status**: Code fix applied, ready for training
- **Scene**: `PhaseF_MultiLane.unity`
- **Key Fix**: WaypointManager.SetLaneCount() now shifts existing waypoint X positions instead of destroying/regenerating
- **Secondary Fix**: GenerateWaypoints() reuses existing GameObjects (position update, no Destroy)
- **Init Checkpoint**: Phase E (--initialize-from=phase-E)
- **Config**: `python/configs/planning/vehicle_ppo_phase-F.yaml` (9 curriculum params, P-002 staggered)
- **Scene Validation**: DrivingSceneManager now validates active scene at startup (P-011)

#### Stage G: Intersections
- **Status**: Planned
- **Purpose**: Crossing logic, traffic rules at intersections
- **Expected Duration**: 6-8M steps
- **Key Features**: T-junctions, 4-way intersections, priority rules

#### Stages H-L: Advanced Scenarios
- **Status**: Planned (pending SOTIF naming redesign)
- **Components**:
  - Stage H: Traffic signals + stop lines
  - Stage I: U-turns and special maneuvers
  - Stage J: Pedestrians + crosswalks
  - Stage K: Obstacles + emergency scenarios
  - Stage L: Complex scenario integration (10-15M steps)
- **Note**: Naming may change per SOTIF (Safety of the Intended Functionality) strategy

## Directory Structure

```
docs/phases/
├── README.md              # This file - Complete overview
├── phase-1/
│   └── SPEC.md           # Foundation & Architecture - COMPLETED
├── phase-2/
│   └── SPEC.md           # Data Infrastructure - COMPLETED
├── phase-3/
│   └── SPEC.md           # Perception Models (Simplified) - SUSPENDED
├── phase-4/
│   └── SPEC.md           # Prediction Models (Simplified) - SUSPENDED
├── phase-5/
│   └── SPEC.md           # Planning Models (PRIMARY FOCUS) - IN PROGRESS
├── phase-6/
│   └── SPEC.md           # Integration & Evaluation - NOT STARTED
└── phase-7/
    └── SPEC.md           # Advanced Topics - NOT STARTED
```

## Document Relationship

```
PRD.md                    # Overall product requirements
    │
    ├── docs/phases/      # Infrastructure + design (Phases 1-7)
    │   ├── phase-1/      # Foundation (COMPLETED)
    │   ├── phase-2/      # Data (COMPLETED)
    │   ├── phase-3/      # Perception (SUSPENDED)
    │   ├── phase-4/      # Prediction (SUSPENDED)
    │   ├── phase-5/      # Planning (PRIMARY - IN PROGRESS)
    │   ├── phase-6/      # Integration (NOT STARTED)
    │   └── phase-7/      # Advanced (NOT STARTED)
    │
    └── PROGRESS.md       # Overall completion tracking

experiments/             # Training experiments by stage
├── phase-0-foundation/   # Stage 0 - COMPLETED
├── phase-A-overtaking/   # Stage A - COMPLETED
├── phase-B-decision/     # Stage B v1 - FAILED
├── phase-B-decision-v2/  # Stage B v2 - COMPLETED
├── phase-C-multi-npc/    # Stage C - DESIGN READY
├── phase-D-lane-observation/  # Stage D v1 - FAILED
├── phase-D-lane-observation-v2/ # Stage D v2 - IN PROGRESS
└── phase-E,F,G/         # Future stages
```

## Key Milestones

| Milestone | Target | Status | Date |
|-----------|--------|--------|------|
| M1 | Unity-ML-Agents integration | COMPLETED | 2026-01-22 |
| M2 | Data infrastructure pipeline | COMPLETED | 2026-01-22 |
| M3 | Perception MVP (simplified) | SUSPENDED | - |
| M4 | Prediction MVP (simplified) | SUSPENDED | - |
| M5 | Planning MVP - Stage A complete | COMPLETED | 2026-01-28 |
| M6 | Planning - Stage B v2 recovery | COMPLETED | 2026-01-29 |
| M7 | Planning - Stage D v3 lane observation | COMPLETED | 2026-01-30 |
| M8 | Planning - Stage E curved roads | COMPLETED | 2026-01-30 |
| M9 | E2E integration system | PLANNED | - |

## Success Criteria (Overall)

| Category | Metric | Target | Current | Status |
|----------|--------|--------|---------|--------|
| Safety | Collision Rate | <5% | ~0% (Stages 0-B v2) | ON TRACK |
| Comfort | Jerk | <2 m/s³ | ~1.5 m/s³ | ACHIEVED |
| Progress | Route Completion | >85% | >95% | EXCEEDED |
| Latency | End-to-end | <200ms | ~150ms | ACHIEVED |
| Behavior | Overtaking skill | Demonstrated | Stage A (+3161 peak) | ACHIEVED |
| Decision Making | Multi-agent handling | Stage D capable | Stage D v3 (+895) | ACHIEVED |
| Curve Handling | Sharp curve navigation | Stage E capable | Stage E (+938 peak) | ACHIEVED |

## Training Summary: Completed Stages

| Stage | Status | Peak Reward | Final Reward | Steps | Duration | Date |
|-------|--------|-------------|--------------|-------|----------|------|
| Stage 0 | COMPLETED | 1018 | 1018 | 8.0M | 1.17h | 2026-01-27 |
| Stage A | COMPLETED | 3161 | 2114 | 2.5M | 29.6m | 2026-01-28 |
| Stage B v1 | FAILED | -108 | -108 | 1.5M | ~1.5h | 2026-01-28 |
| Stage B v2 | COMPLETED | 897 | 877 | 3.5M* | 11.8m | 2026-01-29 |
| Stage D v1 | FAILED | 406 | -2156 | 6.0M | 100m | 2026-01-29 |
| Stage D v2 | FAILED | 447 | -756 | 10M | ~135m | 2026-01-30 |
| Stage D v3 | COMPLETED | 895 | 895 | 5M | ~22m | 2026-01-30 |
| Stage E | COMPLETED | 938 | 893 | 6.0M | ~35m | 2026-01-30 |
| Stage F v1 | FAILED | 303 | -8 | 5.82M | ~40m | 2026-01-31 |

* B v2 resumed from A at 2.5M; 1M new steps

## Related Documents

- [PRD.md](../PRD.md) - Product requirements
- [PROGRESS.md](../PROGRESS.md) - Overall progress tracking
- [LEARNING-ROADMAP.md](../LEARNING-ROADMAP.md) - RL/IL training roadmap (stages 0-L)
- [TRAINING-LOG.md](../TRAINING-LOG.md) - Detailed training logs

## Experiment References

### Successful Completions
- [Stage 0 Foundation](../../experiments/phase-0-foundation/README.md)
- [Stage A Overtaking](../../experiments/phase-A-overtaking/README.md)
- [Stage B v2 Recovery](../../experiments/phase-B-decision-v2/README.md)

### Failure Analysis
- [Stage B v1 Root Cause](../../experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md)
- [Stage D v1 Analysis](../../experiments/phase-D-lane-observation/ANALYSIS.md)

### Design Complete
- [Stage C Design](../../experiments/phase-C-multi-npc/README.md)

### Successful Completions (continued)
- [Stage E Curved Roads](../../experiments/phase-E-curved-roads/) - COMPLETED

### In Progress
- Stage F Multi-Lane (NEXT)

## Current Environment

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Windows 11 | Native (no WSL) |
| Unity | 6000.x (Unity 6) | LTS version |
| ML-Agents | 4.0.1 | Unity Package |
| Sentis | 2.4.1 | ONNX inference |
| Python | 3.10.11 | Windows native |
| PyTorch | 2.1+ | CUDA 12.x |
| GPU | RTX 4090 | 24GB VRAM |

## Quick Start: View Training Status

```powershell
# View latest PROGRESS.md
cat C:\Users\user\Desktop\dev\physical-unity\docs\PROGRESS.md

# Check current training (if running)
Get-Process mlagents-learn -ErrorAction SilentlyContinue

# View TensorBoard (if logs exist)
tensorboard --logdir=experiments
```

## Phase 5 Progress Calculation

```
Completed Stages: 5 (Stage 0, A, B v2, D v3, E)
Total Planned Stages: 12 (Stages 0-L)
Progress = 5/12 = 42% + ~5% (design/analysis) = 47%

Key Milestones:
- Foundation (Stage 0): 8% ✓
- Overtaking (Stage A): 17% ✓
- Decision Making (Stage B v2): 25% ✓
- Lane Observation (Stage D v3): 37% ✓ (C 건너뛰고 D 완료)
- Curves (Stage E): 47% ✓ (Sharp curves + 2 NPCs)
- Lanes (Stage F): 57% (NEXT)
- Intersections (Stage G): 67% (planned)
- Advanced (H-L): 100% (planned)
```

---

**Last Updated**: 2026-01-30
**Current Status**: Phase 5 planning models at ~47% complete with Stages 0, A, B v2, D v3, E successful
**Next Action**: Phase F (Multi-Lane) 학습 시작
