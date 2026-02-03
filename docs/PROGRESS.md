# Progress Tracker

## Project: Autonomous Driving ML Platform

**Focus**: Planning (RL/IL Motion Planning)

---

## Current Status

| Metric | Value |
|--------|-------|
| **Latest Completion** | Phase L v5 Crosswalks + Pedestrians (3/3 peds, reward 504) - 2026-02-03 |
| **Current Phase** | Phase M: Multi-Agent Test Field (12 agents, inference) |
| **Overall Progress** | Phase 0~L (12 phases, 32 runs) + Phase M test field |
| **Latest Model** | E2EDrivingAgent-5000046.onnx (Phase L v5, peak 504.3) |
| **Last Updated** | 2026-02-03 |

---

## Training Dashboard

### All Phases

| Phase | Steps | Peak Reward | Final Reward | Status | Grade | Date |
|-------|-------|-------------|--------------|--------|-------|------|
| Phase 0 | 8.0M | 1018 | 1018 | COMPLETE | A+ | 2026-01-27 |
| Phase A | 2.5M | 3161 | 2114 | COMPLETE | A | 2026-01-28 |
| Phase B v2 | 3.5M | 897 | 877 | COMPLETE | A | 2026-01-29 |
| Phase C | 3.6M | 1390 | 1372 | COMPLETE | A | 2026-01-29 |
| Phase D v3-254d | 5.0M | 912 | 904 | COMPLETE | A- | 2026-01-30 |
| Phase E | 6.0M | 956 | 924 | COMPLETE | A- | 2026-01-30 |
| Phase F v5 | 10.0M | 913 | 643 | COMPLETE | B+ | 2026-01-31 |
| Phase G v1 | 10.0M | 516 | 494 | PARTIAL | C+ | 2026-02-01 |
| Phase G v2 | 5.0M | 633 | 628 | COMPLETE | A- | 2026-02-01 |
| Phase H v1 | ~5.0M | 700 | ~550 | CRASHED | D | 2026-02-01 |
| Phase H v2 | 5.0M | 706 | 681 | PARTIAL (9/11) | B | 2026-02-01 |
| Phase H v3 | 5.0M | 708 | 701 | COMPLETE | A- | 2026-02-01 |
| Phase I v1 | 5.0M | 724 | 623 | PARTIAL (crash) | B- | 2026-02-01 |
| **Phase I v2** | **5.0M** | **775** | **770** | **COMPLETE** | **A+** | **2026-02-01** |
| Phase J v1 | ~40K | N/A | N/A | FAILED | F | 2026-02-02 |
| Phase J v2 | 10.0M | 660.6 | 632 | PARTIAL (9/13) | B- | 2026-02-02 |
| Phase J v3 | 5.0M | 658 | 477 | PARTIAL (12/13) | C+ | 2026-02-02 |
| Phase J v4 | 5.0M | 616 | 497 | PARTIAL (3/4) | B- | 2026-02-02 |
| **Phase J v5** | **5.0M** | **605.7** | **537** | **COMPLETE (5/5)** | **B+** | **2026-02-02** |
| **Phase K v1** | **5.0M** | **703** | **590** | **COMPLETE (3/3)** | **A-** | **2026-02-02** |
| Phase L v1 | 15.0M | 787 | 5799 | EXPLOIT (P-026) | C | 2026-02-03 |
| Phase L v2 | 1.8M | 6759 | 3965 | EXPLOIT (still) | D | 2026-02-03 |
| Phase L v3 | 5.0M | 28675 | 6923 | EXPLOIT (still) | D | 2026-02-03 |
| Phase L v4 | 5.0M | 505 | 474 | COMPLETE (2/3 peds) | B+ | 2026-02-03 |
| **Phase L v5** | **5.0M** | **504** | **494** | **COMPLETE (3/3 peds)** | **A-** | **2026-02-03** |

### Failed/Superseded Runs

| Run | Steps | Peak | Issue | Superseded By |
|-----|-------|------|-------|---------------|
| Phase D v1 | 6.0M | 407 | Reward collapse -2156 | D v3 |
| Phase D v2 | 8.0M | 448 | Stuck at -690 | D v3 |
| Phase F v2 | 4.4M | 318 | Collapse to -14 | F v5 |
| Phase F v3 | 7.1M | 407 | Collapse to 0 | F v5 |
| Phase F v4 | 10.0M | 488 | Degraded to 106 | F v5 |
| Phase H v1 | ~5.0M | 700 | Crash at variation=0.15 | H v3 |
| Phase H v2 | 5.0M | 706 | Stuck at variation=0.05 | H v3 |
| Phase I v1 | 5.0M | 724 | Triple-param crash (700/702/705) | I v2 |
| Phase J v1 | ~40K | N/A | Tensor mismatch (260D vs 268D) | J v2 |
| Phase J v3 | 5.0M | 658 | Signal ordering: green_ratio changed before signals ON | J v4 |
| Phase J v4 | 5.0M | 616 | Plateau at green_ratio=0.5, threshold 540 unreachable | J v5 |
| Phase L v1 | 15.0M | 787 | Yield reward exploit (P-026): agent farms crosswalk indefinitely | L v4 |
| Phase L v2 | 1.8M | 6759 | Yield cap insufficient, still exploiting | L v4 |
| Phase L v3 | 5.0M | 28675 | From-scratch still exploiting (massive reward) | L v4 |

---

## Phase 0: Foundation - Lane Keeping

**COMPLETED 2026-01-27** | Grade: A+

- Final Reward: +1018.43 (101.8% of target)
- Steps: 8.0M | Collision: 0% | Goal: 100%
- Skills: Lane keeping, NPC coexistence (0->4 NPCs)

---

## Phase A: Dense Overtaking

**COMPLETED 2026-01-28** | Grade: A

- Final Reward: +2113.75 (235% of target: +900)
- Steps: 2.5M | Collision: 0% | Goal: 100%
- Skills: Single NPC overtaking at high speed

---

## Phase B v2: Decision Learning

**COMPLETED 2026-01-29** | Grade: A (Recovery from v1 failure)

- Final Reward: +877 (146% of target: +600)
- Steps: 3.5M (resumed from Phase A) | Collision: ~0%
- Skills: Multi-agent decision making (0->3 NPCs)
- Key fix: Reduced speedUnderPenalty -0.1 -> -0.02

---

## Phase C: Multi-NPC Generalization

**COMPLETED 2026-01-29** | Grade: A

- Final Reward: +1372 | Peak: +1390
- Steps: 3.6M | Collision: Low
- Skills: Complex multi-vehicle interaction (4-5+ NPCs)

---

## Phase D v3-254d: Speed Zones & Observation Expansion

**COMPLETED 2026-01-30** | Grade: A-

- Final Reward: +904 | Peak: +912
- Steps: 5.0M
- Skills: Speed zone compliance, expanded 254D observation space
- Note: v1/v2 failed (reward collapse). v3 solved via observation redesign.

---

## Phase E: Curved Roads

**COMPLETED 2026-01-30** | Grade: A-

- Final Reward: +924 | Peak: +956
- Steps: 6.0M
- Skills: Curved road navigation, direction variation

---

## Phase F v5: Multi-Lane Highway

**COMPLETED 2026-01-31** | Grade: B+

- Final Reward: +643 | Peak: +913
- Steps: 10.0M
- Skills: 3-lane highway, lane changes, center line enforcement
- Curriculum: num_lanes 3/3, road_curvature 2/2, goal_distance 2/2
- Note: v2-v4 failed (reward collapse). v5 solved via P-002/P-012 threshold separation.

---

## Phase G v1: Intersection Navigation

**COMPLETED 2026-02-01** | Grade: C+ (Partial)

- Final Reward: +494 | Peak: +516 @ 9.1M
- Steps: 10.0M (budget exhausted)
- Collision: 0% | Goal: 67.9% | WrongWay: 31.9%
- Skills: T-junction + Cross intersection, all turn directions
- Curriculum achieved: intersection_type 2/3, turn_direction 2/2, num_lanes 1/1
- Curriculum missed: Y-junction (threshold 550), NPC interaction (threshold 500+)

### Bottleneck Analysis
1. **WrongWay termination (32%)** -- agent overshoots turns, fails heading check
2. **4M-step plateau** at reward ~400-440 (4M-8M steps)
3. **Fresh start penalty** -- 254D->260D prevented checkpoint transfer
4. **Overcrowded curriculum** -- 9 parameters competing for reward bandwidth

### Visual Enhancement (2026-02-01)
- Added intersection road geometry: curbs, lane markings, stop lines
- Added intersection arms (Left/Right/Angled) with runtime toggling
- DrivingSceneManager auto-switches visuals per curriculum `intersection_type`
- Scene regenerated via `Tools > Create Phase Scenes > Phase G`

**Next: Phase G v2 with WrongWay fix and simplified curriculum**

---

## Phase G v2: Intersection Navigation (WrongWay Fix)

**COMPLETED 2026-02-01** | Grade: A-

- Final Reward: +628 | Peak: +633 @ 4.72M
- Steps: 5.0M (warm start from v1 checkpoint)
- Collision: 0% | All intersection types mastered
- Curriculum: 7/7 lessons completed (all transitions successful)

### Artifacts
- **Final Model**: `results/phase-G-v2/E2EDrivingAgent-5000074.onnx`
- **Config**: `python/configs/planning/vehicle_ppo_phase-G-v2.yaml`

---

## Phase H v3: NPC Interaction in Intersections

**COMPLETED 2026-02-01** | Grade: A-

- Final Reward: +701 | Peak: +708 @ 1.55M
- Steps: 5.0M (warm start from v2 3.5M checkpoint)
- Training: Build-based, 3 parallel envs, ~26 min
- Curriculum: **11/11 complete** (3 NPCs + speed_variation 0.15)

### v1 → v2 → v3 Evolution

| Metric | v1 | v2 | v3 |
|--------|----|----|-----|
| Final Reward | ~550 (crashed) | 681 | **701** |
| Curriculum | 7/11 | 9/11 | **11/11** |
| speed_variation | crash at 0.15 | stuck at 0.05 | **0.15 complete** |
| Duration | ~90 min | ~26 min | ~26 min |

### Key Innovations
1. **Build Training (P-017)**: 헤드리스 빌드 + num_envs=3로 ~3x 속도 향상
2. **Gradual Variation**: 0->0.05->0.10->0.15 (v1의 abrupt jump 수정)
3. **Achievable Thresholds (P-016)**: 685/690/693 (v2의 710/720 도달 불가 수정)
4. **CUDA Fix (P-018)**: threaded=false로 device mismatch 해결

### Artifacts
- **Final Model**: `results/phase-H-v3/E2EDrivingAgent-5000501.onnx`
- **Config**: `python/configs/planning/vehicle_ppo_phase-H-v3.yaml`
- **Build**: `Builds/PhaseH/PhaseH.exe` (118MB)

---

## Phase I v2: Curved Roads + NPC (Project Record)

**COMPLETED 2026-02-01** | Grade: A+

- Final Reward: +770 | Peak: +775 @ 4.83M
- Steps: 5.0M (pure recovery from v1)
- Skills: Full curvature + 3 NPCs + speed variation + 2 speed zones
- v1 crashed (triple-param threshold 700/702/705), v2 recovered with fixed params
- **Project-wide reward record**: +770

---

## Phase J v1/v2: Traffic Signal Compliance

### Phase J v1: FAILED (2026-02-02) | Grade: F

- Steps: ~40K (immediate crash)
- Error: Adam optimizer tensor mismatch (260D checkpoint vs 268D observation)
- Lesson P-020: Observation dimension change requires fresh start

### Phase J v2: PARTIAL (2026-02-02) | Grade: B-

- Final Reward: +632 | Peak: +660.6 @ 7.5M
- Steps: 10.0M (from scratch, 268D)
- Curriculum: 9/13 complete (missed Y-Junction, traffic signals, green ratio)
- Resume at 3.7M: LR 3e-4 -> 1.5e-4, thresholds lowered by 30

### Phase J v3: PARTIAL (2026-02-02) | Grade: C+

- Peak Reward (pre-signal): +658 @ 900K | Final: +477
- Steps: 5.0M (warm start from v2 9.5M)
- Curriculum: 12/13 (Y-Junction, signal ON, green 0.5 -- missed green 0.4)
- Issue: green_ratio threshold < signal threshold, causing ordering conflict (P-022)
- Signal activation caused 177-point crash (647 -> 470), never recovered

### v1 -> v2 -> v3 -> v4 -> v5 Evolution

| Metric | v1 | v2 | v3 | v4 | v5 |
|--------|----|----|----|----|-----|
| Init from | Phase I v2 (260D) | None (scratch) | v2 9.5M | v2 9.5M | v2 9.5M |
| Steps | ~40K | 10M | 5M | 5M | 5M |
| Curriculum | 0/13 | 9/13 | 12/13 | 3/4 green_ratio | **5/5 green_ratio** |
| Peak Reward | N/A | +660 | +658 | +616 | **+605.7** |
| Final Reward | N/A | +632 | +477 | +497 | +537 |
| Issue | Tensor mismatch | Peak plateau | Signal ordering (P-022) | Plateau at 0.5 (P-023) | **COMPLETE** |

### Phase J v5: COMPLETE (2026-02-02) | Grade: B+

- Peak Reward: +605.7 @ 1.44M | Final: +537
- Steps: 5.0M (warm start from v2 9.5M, same 268D)
- Curriculum: **5/5 green_ratio COMPLETE** (0.8->0.7->0.6->0.5->0.4)
- Code fixes: false violation bug (wasPastStopLineAtRedStart), deceleration reward
- P-023 fix: lowered thresholds (450/470/475/475 vs v4's 450/480/510/540)
- P-024: BehaviorType=InferenceOnly left in build caused silent training failure
- Goal rate: 56%, Collision: 0%, Stuck: 4%

### Phase J v4: PARTIAL (2026-02-02) | Grade: B-

- Peak Reward: +616 @ 680K | Final: +497
- Steps: 5.0M (warm start from v2 9.5M, same 268D)
- Curriculum: 3/4 green_ratio (0.8->0.7->0.6->0.5 -- missed 0.5->0.4)
- Signal-first approach: signals ON + Y-Junction locked from step 0
- P-022 fix validated: no signal crash (v3 had -177 point drop)
- P-023: Reward compression at green_ratio=0.5, plateau ~490-500 vs threshold 540

### Artifacts
- **Best Checkpoint (v5)**: `results/phase-J-v5/E2EDrivingAgent/E2EDrivingAgent-5000148.pt`
- **Best Checkpoint (v2)**: `results/phase-J-v2/E2EDrivingAgent/E2EDrivingAgent-9499888.pt` (v2 9.5M, ~652)
- **Config (v5)**: `python/configs/planning/vehicle_ppo_phase-J-v5.yaml`
- **Config (v4)**: `python/configs/planning/vehicle_ppo_phase-J-v4.yaml`
- **Results (v5)**: `results/phase-J-v5/E2EDrivingAgent/`

---

## Phase K v1: Dense Urban Integration

**COMPLETED 2026-02-02** | Grade: A-

- Final Reward: +590 | Peak: +703 @ 4.67M
- Steps: 5.0M (warm start from Phase J v5)
- Training: Build-based, 3 parallel envs, ~25 min
- Observation: 268D (same as Phase J)
- Skills: Combined ALL driving skills -- curved roads + intersections + signals + NPCs
- Curriculum: **3/3 complete** (road_curvature 0 -> 0.3 -> 0.5)

### Artifacts
- **Final Model**: `results/phase-K-v1/E2EDrivingAgent/E2EDrivingAgent-4999930.onnx`
- **Config**: `python/configs/planning/vehicle_ppo_phase-K.yaml`

---

## Phase L: Crosswalks + Pedestrians

### Phase L v1: EXPLOIT (2026-02-03) | Grade: C

- Peak Reward (stable): +787 @ 12.09M | Peak (exploit): +20,148 @ 14.66M
- Steps: 15.0M (fresh start, 280D = 268D + 12D pedestrian/crosswalk)
- Driving skills solid (+730 at 12.5M), but reward exploit found at ~13.3M
- Agent learned to stop at crosswalk indefinitely, farming unbounded yield reward
- Best usable checkpoint: `E2EDrivingAgent-12499788.pt` (12.5M, +730)
- **Lesson P-026**: Yield reward cap needed (see Lessons Learned)

### Phase L v2: EXPLOIT (2026-02-03) | Grade: D

- Steps: 1.8M / 5.0M (stopped early - exploit persisted)
- Peak: +6759 (exploit inflated) | Init from v1 12.5M checkpoint
- P-026 yield cap insufficient - agent still reward hacking

### Phase L v3: EXPLOIT (2026-02-03) | Grade: D

- Steps: 5.0M (from scratch, 280D)
- Peak: +28,675 (massive exploit) | Ped curriculum: 1/3
- Training from scratch did not fix reward exploit

### Phase L v4: COMPLETE (2026-02-03) | Grade: B+

- Final Reward: +474 | Peak: +505 @ ~2.2M
- Steps: 5.0M | Pedestrian curriculum: 2/3 (reached 2 pedestrians)
- Complete reward function redesign eliminated exploit vectors
- First clean training with pedestrians

### Phase L v5: COMPLETE (2026-02-03) | Grade: A-

- Final Reward: +494 | Peak: +504 @ 2.19M
- Steps: 5.0M (warm start from v4 best checkpoint at 4M)
- Pedestrian curriculum: **3/3 complete** (1 -> 2 -> 3 pedestrians)
- Collision: 0% | Off-Road: 0% | Pedestrian collision: 0%
- Last 10 avg: 476.0 (stable range 462.9 - 495.2)

### v1 -> v2 -> v3 -> v4 -> v5 Evolution

| Metric | v1 | v2 | v3 | v4 | v5 |
|--------|----|----|----|----|-----|
| Init from | Scratch | v1 12.5M | Scratch | Scratch | v4 4M |
| Steps | 15M | 1.8M | 5M | 5M | 5M |
| Peak Reward | +787 | +6759 | +28675 | +505 | **+504** |
| Exploit | YES | YES | YES | NO | NO |
| Ped Curriculum | 2/3 | 1/3 | 1/3 | 2/3 | **3/3** |
| Issue | P-026 reward hack | Cap insufficient | Still hacking | 2 peds only | **COMPLETE** |

### Code Changes (Phase L)
1. **PedestrianController.cs** (NEW): Kinematic capsule crossing road at crosswalk
2. **E2EDrivingAgent.cs**: +12D observations (2 nearest pedestrians + crosswalk info), SetStartPose(), inference MaxStep guard
3. **DrivingSceneManager.cs**: num_pedestrians curriculum param, SpawnPedestrians(), TestFieldManager guard
4. **PhaseSceneCreator.cs**: PhaseL_Crosswalks scene + PhaseM_TestField scene

### Artifacts
- **Best Model (v5)**: `results/phase-L-v5/E2EDrivingAgent-5000046.onnx`
- **Deployed Model**: `Assets/ML-Agents/Models/E2EDrivingAgent_PhaseL.onnx`
- **Config (v5)**: `python/configs/planning/vehicle_ppo_phase-L-v5.yaml`
- **Build**: `Builds/PhaseL/PhaseL.exe`

---

## Phase M: Multi-Agent Test Field

**ACTIVE (2026-02-03)** | Inference Only (No Training)

12 trained RL agents driving simultaneously on a shared 2000m road, interacting with each other, 25 NPCs, 8 pedestrians, and traffic signals.

### Configuration
| Item | Value |
|------|-------|
| **Agents** | 12 E2EDrivingAgent (InferenceOnly, Phase L v5 ONNX) |
| **NPCs** | 25 (all active, scattered along road) |
| **Pedestrians** | 8 (at crosswalk Z=75) |
| **Road** | 2000m, 3 lanes, curved, with intersection + crosswalk |
| **Traffic Signals** | Active at intersection |
| **Model** | E2EDrivingAgent_PhaseL.onnx (L v5, peak 504) |
| **Camera** | Tab to cycle agents, F for free-fly |

### New Code
1. **TestFieldManager.cs** (NEW): Central orchestrator for multi-agent inference
2. **FreeFlyCamera.cs** (NEW): WASD + mouse free-fly camera (F toggle)
3. **FollowCamera.cs** (MOD): Multi-target cycling (Tab, 1-9 keys)
4. **PhaseSceneCreator.cs** (MOD): CreatePhaseM_TestField menu item
5. **BuildHelper.cs** (MOD): Build Phase M menu item

### Agent Interaction
- All 12 agents + 25 NPCs share "Vehicle" tag
- Each agent's OverlapSphere (50m) detects other agents as vehicles
- Agent-agent collision triggers respawn via TestFieldManager
- Intersection and crosswalk negotiation per individual agent

### Artifacts
- **Scene**: `Assets/Scenes/PhaseM_TestField.unity`
- **Build**: `Builds/PhaseM/PhaseM.exe` (pending)

---

## Key Achievements

### Safety Record
- **0% collision across all phases** (0 through L v5)
- Perfect safety maintained with 3 NPCs + 3 pedestrians in intersections

### Curriculum Progression
- Phase 0: Basic driving -> Phase L v5: Crosswalk + 3 pedestrians -> Phase M: Multi-agent test field
- 12 training phases completed (32 total runs including failures)
- Agent handles: overtaking, multi-NPC, speed zones, curves, multi-lane, T/Cross/Y-junction intersections, NPC speed variation, traffic signals, crosswalks, pedestrian yielding (280D)
- Phase M: 12 agents + 25 NPCs + 8 pedestrians on shared 2000m road (inference)

### Lessons Learned

| Lesson ID | Description | First Seen |
|-----------|-------------|------------|
| P-002 | Unique thresholds across ALL curriculum parameters | Phase F v4 |
| P-009 | Observation-environment coupling requires fresh start | Phase D v3 |
| P-011 | Scene validation prevents silent training failures | Phase F |
| P-012 | No two params share threshold (simultaneous transition) | Phase F v3 |
| P-013 | WrongWay rate dominates when turns are imprecise | Phase G v1 |
| P-014 | Intersection zone requires WrongWay exemption | Phase G v1→v2 |
| P-015 | DecisionRequester required after scene regeneration | Phase G v2 |
| P-016 | Curriculum thresholds must be achievable under target conditions | Phase H v2 |
| P-017 | Build training (num_envs=3, no_graphics) enables rapid iteration | Phase H v2 |
| P-018 | threaded=false required with CUDA warm start (device mismatch) | Phase H v2 |
| P-019 | min_lesson_length should scale with reward noise | Phase H v3 |
| P-020 | Observation dimension change = fresh start required (no warm start) | Phase J v1 |
| P-021 | From-scratch training needs lower thresholds than warm start | Phase J v2 |
| P-022 | Feature activation must precede param tuning; use single-param curriculum | Phase J v3 |
| P-023 | Reward compression under signal constraints; lower green = lower reward ceiling | Phase J v4 |
| P-024 | BehaviorType=InferenceOnly in build silently prevents training (no brain registration) | Phase J v5 |
| P-025 | BehaviorType enum: value 1 = HeuristicOnly, not InferenceOnly (use value 2) | Phase L v1 |
| P-026 | Unbounded per-step positive reward enables reward hacking; cap yield rewards per episode | Phase L v1 |
| P-027 | MaxStep=3000 for training, MaxStep=0 for inference (unlimited steps) | Phase M |

---

*Document updated: 2026-02-03*
*Phase L v5 Complete (peak 504, 3/3 pedestrians) -- Phase M Multi-Agent Test Field active*
