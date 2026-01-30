# Autonomous Driving ML Platform

Unity ML-Agents based Motion Planning AI training platform with SOTIF compliance and safety standards integration.

> **Development Infrastructure**: This project uses [cc-initializer](https://github.com/tygwan/cc-initializer) for Claude Code workflow automation, including custom agents, skills, hooks, and development lifecycle management.

---

## Project Overview

This platform combines Unity simulation, ML-Agents reinforcement learning, and PyTorch to develop and validate autonomous driving motion planning algorithms. The project focuses on systematic progression through increasingly complex driving scenarios while maintaining alignment with international safety standards (ISO 21448 SOTIF, UN R171/R157).

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Simulation | Unity 6 (6000.x) | Physics-accurate driving environment |
| ML Framework | ML-Agents 4.0.1, PyTorch 2.3.1 | Reinforcement learning training |
| Safety Standards | ISO 21448, UN R171/R157 | SOTIF compliance and validation |
| Hardware | RTX 4090 (24GB VRAM), 128GB RAM, 4TB SSD | Large-scale model training |

### Key Differentiators

1. **Trial-Error-Policy Discovery**: Systematic learning from failures to derive safety principles that converge with international standards
2. **Phase-based Curriculum**: Progressive complexity from lane keeping to multi-agent interactions
3. **SOTIF Integration**: Functional Insufficiency (FI) and Triggering Condition (TC) analysis for each training phase
4. **Realistic Constraints**: RTX 4090-optimized architecture, achievable with academic research resources

---

## Current Status

| Phase | Focus | Status | Completion |
|-------|-------|--------|-----------|
| Phase 1-2 | Foundation & Data Infrastructure | COMPLETED | 100% |
| Phase 3-4 | Perception & Prediction | SUSPENDED | - |
| **Phase 5** | **Planning Models (RL/IL)** | **IN PROGRESS** | **~27%** |
| Phase 6-7 | Integration & Advanced Topics | PLANNED | - |

### Phase 5 Progress: Planning Sub-Phases

**Current Stage**: Phase D v2 (Lane Observation - 254D vector space with staggered curriculum)

**Completed Stages** (3/12 = 25% + design work = 27%):

| Stage | Scenario | Peak Reward | Steps | Status | Notes |
|-------|----------|-------------|-------|--------|-------|
| **Stage 0** | Foundation: Lane Keeping | +1,018 | 8.0M | COMPLETED | Perfect safety, 0% collision |
| **Stage A** | Dense Overtaking (1 slow NPC) | +2,114 | 2.5M | COMPLETED | Excellent generalization |
| **Stage B v1** | Decision Making (0→3 NPCs) | -108 | 1.5M | FAILED | Learned to STOP (harsh penalty bug) |
| **Stage B v2** | Decision Making (recovery) | +877 | 3.5M | COMPLETED | Recovery from v1 failure |
| **Stage C** | Multi-NPC (4-8 NPCs) | +1,372 | 3.6M | COMPLETED | Perfect safety, 8 concurrent NPCs |
| **Stage D v1** | Lane Observation (254D) | +406 → -2,156 | 6.0M | FAILED | Curriculum collapse at 4.68M steps |
| **Stage D v2** | Lane Obs (staggered curriculum) | TBD | 10M planned | IN PROGRESS | Conservative single-param progression |

**Planned Stages** (E-L): Curved Roads, Multi-Lane, Intersections, Curvature Nodes (SOTIF), Cut-in/Cut-out (UN R171), Sensor Degradation, Boundary Violations, Integrated Complex, SOTIF Validation

---

## Architecture

### Current Implementation

```
┌─────────────────────────────────────────────────────────────────────┐
│                   CURRENT ARCHITECTURE (Phase D)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │  Vector Obs      │ → │   MLP Policy     │ → │  Action Space  │  │
│  │  254D Ground     │   │   (PPO/SAC)      │   │  Steering +    │  │
│  │  Truth Features  │   │   3-layer NN     │   │  Acceleration  │  │
│  └──────────────────┘   └──────────────────┘   └────────────────┘  │
│                                                                      │
│  Observation Components (254D):                                     │
│    - Ego State (8D): position, velocity, heading, acceleration      │
│    - Route Info (30D): waypoints, distances to goal                 │
│    - Surrounding (40D): 8 vehicles x 5 features                     │
│    - NPC Observations (152D): detailed NPC states                   │
│    - Lane Observations (12D): left/right lane markings (4 points)   │
│    - Goal Info (12D): goal distance, direction, completion status   │
│                                                                      │
│  Action Space (Continuous):                                         │
│    - Steering: [-0.5, +0.5] rad                                     │
│    - Acceleration: [-4.0, +4.0] m/s²                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Target Architecture (Phase 6+)

```
┌─────────────────────────────────────────────────────────────────────┐
│                   TARGET E2E ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │  Vision      │ → │  Planning    │ → │  Control     │            │
│  │  BEV Encoder │   │  Transformer │   │  MPC/Neural  │            │
│  │  (Camera)    │   │  + MCTS      │   │  Tracker     │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│                                                                      │
│  Future Components:                                                 │
│    - Multi-camera BEV fusion (simplified vs Tesla 8-camera)         │
│    - Occupancy network for spatial reasoning                        │
│    - Trajectory optimization with kinematic constraints             │
│    - SOTIF-validated cost functions (comfort, safety, intervention) │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Results Summary

### Main Training Progression

| Phase | Base Checkpoint | Steps | Peak Reward | Final Reward | Key Achievement |
|-------|-----------------|-------|-------------|--------------|-----------------|
| **Phase 0** | From scratch | 8M | +1,018 | +1,018 | Lane keeping, NPC coexistence |
| **Phase A** | Phase 0 | 2.5M | +2,114 | +2,114 | Overtaking mastery (235% of target) |
| **Phase B v1** | Phase 0 (wrong) | 3M | -108 | -108 | FAILED - Wrong checkpoint + harsh penalty |
| **Phase B v2** | Phase A (correct) | 1M | +877 | +877 | Decision learning (recovery success, 146% of target) |
| **Phase C** | Phase B v2 | 3.6M | +1,372 | +1,372 | Multi-NPC generalization (228% of target, 8 NPCs) |
| **Phase D v1** | Phase C | 6M | +406 | -2,156 | FAILED - Curriculum collapse (3 params simultaneous) |
| **Phase D v2** | Phase C | 10M planned | TBD | TBD | IN PROGRESS - Staggered curriculum approach |

### Success Metrics

| Category | Metric | Target | Current (Stages 0-C) | Status |
|----------|--------|--------|---------------------|--------|
| Safety | Collision Rate | <5% | ~0% | EXCEEDED |
| Comfort | Jerk | <2 m/s³ | ~1.5 m/s³ | ACHIEVED |
| Progress | Route Completion | >85% | >95% | EXCEEDED |
| Behavior | Overtaking Skill | Demonstrated | +2,114 peak (Stage A) | ACHIEVED |
| Decision Making | Multi-agent Handling | Stage C capable | 8 concurrent NPCs | ACHIEVED |

### Key Lessons from Failures

**Phase B v1 Failure** (Reward -108):
- Root Cause: Wrong checkpoint selection (Phase 0 lacks overtaking capability) + harsh speed penalty (-0.1/step)
- Recovery: Phase A checkpoint + reduced penalty (-0.02/step) + gradual curriculum (0→1→2→3 NPCs)
- Discovered Policy: **P-001 (Single Variable Isolation)**, **P-003 (Capability-based Checkpoint)**, **P-004 (Conservative Penalty Design)**

**Phase D v1 Failure** (Reward +406 → -2,156):
- Root Cause: 3 curriculum parameters transitioned simultaneously at step 4.68M (num_active_npcs, speed_zone_count, npc_speed_variation)
- Collapse: 5,231 points drop in <20K steps
- Recovery: Staggered thresholds (200K/300K/350K) for single-parameter progression
- Discovered Policy: **P-002 (Staggered Curriculum Principle)**

---

## Policy Discovery: From Trial to Standards

This project employs a **trial-error-policy discovery** approach where empirical findings from training failures naturally converge with international safety standards.

### Discovery Process

```
Experimental Failure → Root Cause Analysis → Design Principle → Standard Alignment
       ↓                        ↓                     ↓                   ↓
   Phase B v1            Harsh Penalty          P-004 Conservative   Reward Shaping
   Phase D v1         Curriculum Collapse       P-002 Staggered      SOTIF Gradual
```

### Discovered Policies (Registry)

| ID | Principle | Discovered From | Aligned Standard | Status |
|----|-----------|----------------|------------------|--------|
| P-001 | Single Variable Isolation | Phase B v1→v2 | Controlled Experiment | Validated |
| P-002 | Staggered Curriculum | Phase D v1→v2 | SOTIF Gradual Complexity | In Validation |
| P-003 | Capability-based Checkpoint | Phase B v1→v2 | Transfer Learning | Validated |
| P-004 | Conservative Penalty Design | Phase B v1→v2 | Reward Shaping Theory | Validated |
| P-005 | Lateral Accel Limit (planned) | Phase E | UN R157 (lat_accel <0.3g) | Not Tested |
| P-006 | TTC-based Reaction (planned) | Phase I | UN R171 (TTC 1.5-5.0s) | Not Tested |
| P-007 | Curvature Rate Response (planned) | Phase H | SOTIF FI + UN R157 dk/ds | Not Tested |
| P-008 | Sensor Degradation (planned) | Phase K | SOTIF TC (Triggering Condition) | Not Tested |

**Full Documentation**: [docs/POLICY-DISCOVERY-LOG.md](docs/POLICY-DISCOVERY-LOG.md)

---

## SOTIF & Safety Standards Integration

This project integrates **ISO 21448 (SOTIF)**, **UN R171 (DCAS)**, and **UN R157 (ALKS)** to ensure systematic safety validation.

### SOTIF Framework

**Safety of the Intended Functionality** (ISO 21448) addresses risks from functional limitations rather than system faults - critical for ML/AI systems.

**4-Quadrant Model**:

```
              KNOWN                    UNKNOWN
         +--------------+         +--------------+
  SAFE   | Quadrant 1   |         | Quadrant 4   |
         | Known Safe   |         | Unknown Safe |
         | Phases A-C   |         | Field Data   |
         +--------------+         +--------------+
         +--------------+         +--------------+
 UNSAFE  | Quadrant 2   |         | Quadrant 3   |
         | Known Unsafe |         | Unknown Unsafe|
         | Phases D-G   |         | Phases H-L   |
         +--------------+         +--------------+
```

**Goal**: Reduce Quadrant 2/3 area, expand Quadrant 1 through systematic FI/TC analysis.

### UN Regulations Integration

**UN R171 (DCAS - Level 2 Driving Assistance)**:
- Max deceleration: -7.0 m/s²
- Jerk limit: ≤ 3.0 m/s³
- TTC maintenance: 1.5-5.0s
- **Application**: Phase I (Cut-in/Cut-out scenarios)

**UN R157 (ALKS - Level 3 Lane Keeping)**:
- Lateral acceleration: ≤ 0.3g (2.94 m/s²)
- Curvature rate: dk/ds ≤ 0.1 /m²
- Crosstrack error: ≤ 0.3m (curves)
- **Application**: Phase H (Curvature Transition Nodes)

### Phase-Specific SOTIF Mapping

| Phase | SOTIF Quadrant | Key FI/TC | Regulatory Reference |
|-------|---------------|-----------|---------------------|
| A-C | 1 (Known Safe) | Baseline curriculum | - |
| D-G | 2 (Known Unsafe) | Lane observation, curvature control | UN R157 |
| H | 3 (Unknown Unsafe) | Curvature prediction error, speed-steering mismatch | SOTIF FI, UN R157 dk/ds |
| I | 3 (Unknown Unsafe) | Reaction latency, decel limit | UN R171 |
| J-K | 3 (Unknown Unsafe) | Sensor degradation, ODD boundary | SOTIF TC |
| L-M | 1 (Target) | Integrated validation | ISO 21448 compliance |

**Full Strategy**: [docs/SOTIF-EDGE-CASE-STRATEGY.md](docs/SOTIF-EDGE-CASE-STRATEGY.md)

---

## Tesla FSD Gap Analysis

This project acknowledges the fundamental gap between academic research platforms and commercial autonomous driving systems like Tesla FSD 12/13.

### Key Differences

| Aspect | Tesla FSD | This Project |
|--------|-----------|--------------|
| Data Scale | 400M+ miles, 4M+ fleet vehicles | nuPlan 1,282 hours, Unity simulation |
| Compute | Dojo Supercomputer (1.1 ExaFLOPS) | RTX 4090 (82.6 TFLOPS FP32) |
| Architecture | 8-camera multi-view → Occupancy Network → MCTS Planner | Ground Truth Vector → MLP Policy (current) |
| Model Size | ~400M parameters, ~120GB VRAM | ~10M parameters, <24GB VRAM |
| Deployment | Level 2-3 production (no geo-fence) | Simulation validation platform |

### Realistic Goals

**What RTX 4090 Can Achieve**:
- Validate planning algorithms in controlled simulation
- Systematic curriculum learning with phase-based progression
- SOTIF/UN regulation compliance verification
- Academic research-level E2E pipeline demonstration

**What It Cannot**:
- Match Tesla's multi-camera Occupancy Network (VRAM constraint)
- Process 4M vehicle fleet data (data scale)
- Achieve production-level robustness (deployment constraint)

**Project Focus**: Algorithm validation and safety standard integration, not commercial deployment.

**Full Analysis**: [docs/TESLA-FSD-GAP-ANALYSIS.md](docs/TESLA-FSD-GAP-ANALYSIS.md)

---

## Phase Structure

### Overview

This project follows a 7-phase structure with Phase 5 (Planning) as the primary focus:

```
Phase 1: Foundation & Architecture (COMPLETED)
Phase 2: Data Infrastructure (COMPLETED)
Phase 3: Perception Models (SUSPENDED - Ground Truth used)
Phase 4: Prediction Models (SUSPENDED - Constant Velocity baseline)
Phase 5: Planning Models (IN PROGRESS - 12 sub-stages 0-L)
Phase 6: Integration & Evaluation (PLANNED)
Phase 7: Advanced Topics (PLANNED)
```

### Phase 5 Sub-Stages (0-L)

**Foundation (0-C)**: COMPLETED
- Stage 0: Lane keeping (+1,018 reward)
- Stage A: Dense overtaking (+2,114 reward)
- Stage B v2: Decision making (+877 reward)
- Stage C: Multi-NPC generalization (+1,372 reward, 8 NPCs)

**Observation & Environment (D-G)**: IN PROGRESS
- Stage D v2: Lane observation (254D vector space) - IN PROGRESS
- Stage E: Curved roads - PLANNED
- Stage F: Multi-lane switching - PLANNED
- Stage G: Intersections - PLANNED

**SOTIF & Advanced (H-L)**: PLANNED
- Stage H: Curvature transition nodes (SOTIF FI analysis)
- Stage I: Cut-in/Cut-out (UN R171 compliance)
- Stage J: Sensor degradation (SOTIF TC)
- Stage K: Boundary violations (ODD edge detection)
- Stage L: Integrated complex scenarios

**Validation (M)**: PLANNED
- Stage M: SOTIF validation (ISO 21448 compliance)

**Full Phase Documentation**: [docs/phases/README.md](docs/phases/README.md)

---

## Tech Stack

| Component | Version | Purpose |
|-----------|---------|---------|
| OS | Windows 11 | Native (no WSL) |
| Unity | 6000.x (Unity 6) | Physics simulation, rendering |
| ML-Agents | 4.0.1 (Unity Package) | RL training framework |
| Sentis | 2.4.1 | ONNX model inference in Unity |
| Python | 3.10.11 | ML training scripts |
| PyTorch | 2.3.1 | Deep learning framework |
| mlagents | 1.2.0 (Python Package) | Training CLI tool |
| CUDA | 12.x | GPU acceleration |

---

## Hardware Specifications

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | NVIDIA RTX 4090 | 24GB VRAM, 82.6 TFLOPS FP32 |
| RAM | 128GB DDR4/DDR5 | Large-scale data processing |
| Storage | 4TB SSD | Full dataset + experiment logs |
| CPU | Modern multi-core | Unity simulation parallelization |

**GPU Utilization**:
- Training: 70-90% GPU usage during PPO/SAC learning
- Inference: 10-20% GPU usage during policy evaluation
- VRAM: Current models use <10GB, allows future vision model expansion

---

## Project Structure

```
physical-unity/
├── .claude/                    # cc-initializer config (38 agents, 22 skills)
├── Assets/                     # Unity project
│   ├── Scripts/
│   │   ├── Agents/            # E2EDrivingAgent.cs, E2EDrivingAgentBv2.cs
│   │   ├── Environment/       # DrivingSceneManager.cs, curriculum logic
│   │   └── Sensors/           # Future: CameraSensor, LiDARSensor
│   └── Resources/Models/      # Trained ONNX models
├── docs/
│   ├── PRD.md                 # Product requirements
│   ├── TRAINING-LOG.md        # Detailed training logs
│   ├── POLICY-DISCOVERY-LOG.md    # Trial-error-policy discovery
│   ├── SOTIF-EDGE-CASE-STRATEGY.md # SOTIF integration
│   ├── TESLA-FSD-GAP-ANALYSIS.md   # Tesla FSD comparison
│   └── phases/                # Phase-specific documentation
├── python/
│   ├── configs/planning/      # Training YAML configs
│   │   ├── vehicle_ppo_phase-A.yaml
│   │   ├── vehicle_ppo_phase-B-v2.yaml
│   │   ├── vehicle_ppo_phase-C.yaml
│   │   └── vehicle_ppo_phase-D-v2.yaml
│   └── src/
│       ├── models/            # PyTorch model definitions
│       └── training/          # Training scripts
├── experiments/               # Phase-specific experiment results
│   ├── phase-0-foundation/
│   ├── phase-A-overtaking/
│   ├── phase-B-decision/
│   ├── phase-B-decision-v2/
│   ├── phase-C-multi-npc/
│   ├── phase-D-lane-observation/
│   └── phase-D-lane-observation-v2/
├── results/                   # TensorBoard logs
└── models/planning/           # Final ONNX models for inference
```

---

## Quick Start

### Training a New Phase

```powershell
# Windows PowerShell
cd C:\Users\user\Desktop\dev\physical-unity

# Start TensorBoard monitoring
tensorboard --logdir=results

# Launch training (example: Phase D v2)
mlagents-learn python/configs/planning/vehicle_ppo_phase-D-v2.yaml --run-id=phase-D-v2

# In Unity Editor: Click Play button to start training
```

### Monitoring Training Progress

```powershell
# TensorBoard (real-time metrics)
tensorboard --logdir=results
# Open browser: http://localhost:6006

# Check training status
cat docs/TRAINING-LOG.md

# View experiment results
cat experiments/phase-D-lane-observation-v2/README.md
```

### Running Inference

1. Copy trained model: `results/<run-id>/E2EDrivingAgent.onnx` → `Assets/Resources/Models/`
2. In Unity Inspector:
   - Set BehaviorParameters > Model to your ONNX file
   - Change BehaviorType to "Inference Only"
3. Click Play to run autonomous driving inference

---

## Development Workflow

### cc-initializer Integration

This project uses [cc-initializer](https://github.com/tygwan/cc-initializer) for automated workflows:

**Key Features**:
- 38 AI agents (26 core + 12 ML-specific)
- 22 skills (18 core + 4 ML-specific)
- 6 hooks (pre-tool, post-tool, error recovery)
- 6 commands (/feature, /experiment, /phase, etc.)

**ML-Specific Agents**:
- `ad-experiment-manager`: Experiment creation, execution, comparison
- `training-analyst`: Training result analysis, success/failure detection
- `forensic-analyst`: Root cause analysis for failures
- `training-planner`: Experiment design and config generation
- `training-monitor`: Real-time training status monitoring
- `training-orchestrator`: Workflow coordination
- `training-doc-manager`: Documentation synchronization
- `training-site-publisher`: GitHub Pages publishing

**ML-Specific Skills**:
- `/experiment`: Create/run/compare experiments
- `/train`: Start/monitor RL/IL training
- `/evaluate`: Benchmark evaluation
- `/dataset`: Data curation and preprocessing

---

## Documentation

| Document | Purpose |
|----------|---------|
| [PRD.md](docs/PRD.md) | Product requirements and success criteria |
| [TRAINING-LOG.md](docs/TRAINING-LOG.md) | Detailed training experiment logs |
| [POLICY-DISCOVERY-LOG.md](docs/POLICY-DISCOVERY-LOG.md) | Trial-error-policy convergence tracking |
| [SOTIF-EDGE-CASE-STRATEGY.md](docs/SOTIF-EDGE-CASE-STRATEGY.md) | ISO 21448 integration strategy |
| [TESLA-FSD-GAP-ANALYSIS.md](docs/TESLA-FSD-GAP-ANALYSIS.md) | Tesla FSD architecture comparison |
| [phases/README.md](docs/phases/README.md) | Phase structure and progress |
| [PROGRESS.md](docs/PROGRESS.md) | Overall project progress tracking |

---

## References

### Standards
- [ISO 21448:2022 - SOTIF](https://www.iso.org/standard/77490.html)
- [UN Regulation No. 171 - DCAS](https://unece.org/sites/default/files/2025-03/R171e.pdf)
- [UN Regulation No. 157 - ALKS](https://unece.org/transport/documents/2021/03/standards/un-regulation-no-157-automated-lane-keeping-systems-alks)

### Technical Frameworks
- [Unity ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [PPO Algorithm (Schulman et al. 2017)](https://arxiv.org/abs/1707.06347)

### Academic Research
- [SOTIF Analysis for MPC Planner (arXiv)](https://arxiv.org/html/2407.21569v1)
- [LKA Performance Evaluation (arXiv)](https://arxiv.org/html/2505.11534v1)
- [Curriculum DRL for Autonomous Driving (CuRLA)](https://arxiv.org/html/2501.04982v1)

---

## License

[Specify License Here]

---

## Contributors

[Specify Contributors/Acknowledgments Here]

---

**Last Updated**: 2026-01-30 | **Phase 5 Status**: ~27% Complete (Stages 0, A, B v2, C completed; Stage D v2 in progress)

**Current Focus**: Phase D v2 training with staggered curriculum (254D observation space, single-parameter progression)

**Next Milestone**: Complete Stage D v2 → Begin Stage E (Curved Roads)
