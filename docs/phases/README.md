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
│                                      │ Planning │ IN PROGRESS (~27%)         │
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
| **Phase 5** | Planning Models | IN PROGRESS | ~27% | RL/IL Motion Planner (PRIMARY FOCUS) |
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
| **Stage C** | Multi-NPC (4-8 NPCs) | DESIGN COMPLETE | - | 3.6M planned | Ready for training |
| **Stage D v1** | Lane Observation (254D) | FAILED | +406 | 6.0M | Curriculum collapse @ 4.68M |
| **Stage D v2** | Lane Obs (staggered curriculum) | IN TRAINING | - | 10M planned | Running with conservative approach |
| **Stage E** | Curved Roads | PLANNED | - | 6-8M | Curve handling |
| **Stage F** | Multi-Lane Switching | PLANNED | - | 4-6M | Lane changing |
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

#### Stage D v2: Lane Observation - Staggered Curriculum (IN PROGRESS)
- **Status**: Currently training or awaiting training
- **Expected Duration**: 8-10M steps (longer than v1 due to conservative approach)
- **Key Change**: Staggered transition thresholds per parameter
  - num_active_npcs (1->2): threshold=200
  - npc_speed_variation (0->0.3): threshold=300
  - npc_speed_ratio/goal/zones: threshold=350
  - num_active_npcs (2->3): threshold=300
  - Final transitions: threshold=350
- **Target Reward**: >+300 (conservative target)
- **Expected Progression**: One parameter at a time, steady improvement
- **Observation Space**: 254D (same as v1)
- **Init Checkpoint**: Fresh start (learns from scratch)
- **Safety**: Monitor collision rate <5%

### Not Started Stages (E-L)

#### Stage E: Curved Roads
- **Status**: Planned
- **Purpose**: Add curve handling to agent capabilities
- **Expected Duration**: 6-8M steps
- **Key Features**: Curved lane geometry, varying road angles
- **Init Checkpoint**: Will use Stage D v2 checkpoint (if successful)

#### Stage F: Multi-Lane Switching
- **Status**: Planned
- **Purpose**: Explicit lane changing with 2-3 lane roads
- **Expected Duration**: 4-6M steps
- **Key Features**: Multiple parallel lanes, lane change safety rules
- **Dependencies**: Stage D completion required

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
| M7 | Planning - Stage C design & approval | IN PROGRESS | 2026-01-29 |
| M8 | E2E integration system | PLANNED | - |

## Success Criteria (Overall)

| Category | Metric | Target | Current | Status |
|----------|--------|--------|---------|--------|
| Safety | Collision Rate | <5% | ~0% (Stages 0-B v2) | ON TRACK |
| Comfort | Jerk | <2 m/s³ | ~1.5 m/s³ | ACHIEVED |
| Progress | Route Completion | >85% | >95% | EXCEEDED |
| Latency | End-to-end | <200ms | ~150ms | ACHIEVED |
| Behavior | Overtaking skill | Demonstrated | Stage A (+3161 peak) | ACHIEVED |
| Decision Making | Multi-agent handling | Stage C capable | Stage B v2 (+877) | IN PROGRESS |

## Training Summary: Completed Stages

| Stage | Status | Peak Reward | Final Reward | Steps | Duration | Date |
|-------|--------|-------------|--------------|-------|----------|------|
| Stage 0 | COMPLETED | 1018 | 1018 | 8.0M | 1.17h | 2026-01-27 |
| Stage A | COMPLETED | 3161 | 2114 | 2.5M | 29.6m | 2026-01-28 |
| Stage B v1 | FAILED | -108 | -108 | 1.5M | ~1.5h | 2026-01-28 |
| Stage B v2 | COMPLETED | 897 | 877 | 3.5M* | 11.8m | 2026-01-29 |
| Stage D v1 | FAILED | 406 | -2156 | 6.0M | 100m | 2026-01-29 |
| Stage D v2 | IN PROGRESS | - | - | 10M planned | TBD | 2026-01-30+ |

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

### In Progress
- [Stage D v2 Setup](../../experiments/phase-D-lane-observation-v2/README.md)

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
Completed Stages: 3 (Stage 0, A, B v2)
Total Planned Stages: 12 (Stages 0-L)
Progress = 3/12 = 25% + ~2% (design work) = 27%

Key Milestones:
- Foundation (Stage 0): 8% ✓
- Overtaking (Stage A): 17% ✓
- Decision Making (Stage B v2): 27% ✓
- Generalization (Stage C): 37% (design ready)
- Perception (Stage D v2): 47% (in progress)
- Curves (Stage E): 57% (planned)
- Lanes (Stage F): 67% (planned)
- Intersections (Stage G): 77% (planned)
- Advanced (H-L): 100% (planned)
```

---

**Last Updated**: 2026-01-30
**Current Status**: Phase 5 planning models at ~27% complete with Stages 0, A, and B v2 successful
**Next Action**: Monitor Stage D v2 training progress; prepare Stage C training start
