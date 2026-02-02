---
layout: default
title: Phase J - Traffic Signals
---

# Phase J: Traffic Signal Compliance

Traffic signal recognition, stop line behavior, and green ratio adaptation

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-J-v5 (FINAL) |
| **Status** | COMPLETE (5/5 green_ratio curriculum) |
| **Date** | 2026-02-02 |
| **Total Steps** | 10M (v2) + 5M (v3) + 5M (v4) + 5M (v5) = 25M |
| **Training Time** | ~28 min per 5M (Build + 3 parallel envs) |
| **Final Reward** | **+537** (v5, green_ratio=0.4) |
| **Peak Reward** | **+605.7** (v5, at 1.44M) |
| **Observation** | 268D (260D + 8D traffic signal) |
| **Initialize From** | v2 9.5M checkpoint (v3/v4/v5) |
| **Training Mode** | Build + 3 parallel envs (no_graphics) |

---

## Objective

Add traffic signal compliance to the driving agent. Agent must recognize traffic light states (red/yellow/green), stop at red lights behind stop lines, and proceed when green. This requires a new 8D observation extension for signal state and timing information.

### New Capabilities
- Traffic signal state recognition (red/yellow/green)
- Stop line compliance (deceleration and stopping)
- Green ratio adaptation (varying signal timing)
- Y-junction navigation (from Phase G, not yet mastered)

---

## Observation Space

**268D (260D + 8D traffic signal)**

| Component | Dimensions | Source |
|-----------|-----------|--------|
| ego_state | 8D | Phase 0 |
| route_info | 30D | Phase 0 |
| surrounding vehicles | 160D (40D x 4 history) | Phase D |
| speed_zones | 4D | Phase D |
| lane_observation | 12D | Phase D |
| intersection_info | 6D | Phase G |
| curvature_info | 40D | Phase E |
| **traffic_signal** | **8D** | **Phase J (NEW)** |

### Traffic Signal Observation (+8D)
| Feature | Count | Description |
|---------|-------|-------------|
| signal_state | 3D | One-hot: [red, yellow, green] |
| signal_distance | 1D | Distance to next signal (normalized) |
| stop_line_distance | 1D | Distance to stop line (normalized) |
| time_remaining | 1D | Estimated time to next change |
| signal_green_ratio | 1D | Current green phase ratio |
| is_signal_active | 1D | Whether signals are enabled |

---

## Version History: v1 -> v2 -> v3 -> v4

| Aspect | v1 (FAILED) | v2 (PARTIAL) | v3 (PARTIAL) | v4 (PLANNED) |
|--------|-------------|--------------|--------------|--------------|
| Init from | Phase I v2 (260D) | None (scratch) | v2 9.5M (268D) | v2 9.5M (268D) |
| Steps | ~40K (crash) | 10M | 5M | 5M |
| Observation | 268D | 268D | 268D | 268D |
| Batch size | 4096 | 2048 | 4096 | 4096 |
| Num epoch | 5 | 3 | 5 | 5 |
| Curriculum | 13 params | 13 params | 3 remaining | 1 (green_ratio) |
| Completed | 0/13 | 9/13 | 12/13 | 3/4 green_ratio |
| Final reward | N/A | +632 | +477 | +497 |
| Issue | Tensor mismatch | Peak plateau | Signal ordering (P-022) | Plateau at 0.5 (P-023) |

### v1 Failure: Observation Dimension Mismatch (P-020)
- init_path pointed to Phase I v2 checkpoint (260D observation)
- Phase J scene uses 268D observation (+8D traffic signal)
- Adam optimizer state tensor size mismatch: 260 vs 268
- Training crashed within ~40K steps
- Lesson: Observation dimension changes ALWAYS require fresh start

### v2: From Scratch Success (Partial)
- Trained from scratch with 268D observation
- Rebuilt all skills from Phase 0 through Phase G curriculum
- Reached 9/13 curriculum (all pre-traffic-signal params completed)
- Missed Y-Junction (threshold 650) and all traffic signal params (670+)
- Mid-training resume at 3.7M: LR 3e-4 -> 1.5e-4, thresholds lowered

### v3: Warm Start (12/13 -- Signal Ordering Problem)
- Warm start from v2 best checkpoint (9.5M, ~652 reward)
- Same 268D observation = no tensor mismatch
- 12/13 curriculum completed (Y-Junction + signals ON + green 0.5)
- **Missed**: green_ratio 0.5 -> 0.4 (threshold 640 unreachable after signal crash)
- Signal activation caused 177-point crash (647 -> 470), never recovered
- Root cause: curriculum ordering conflict (P-022)

### v4: Signal-First Single-Param Curriculum (3/4 -- Plateau)
- Warm start from v2 9.5M (pre-signal baseline, same 268D)
- Locked traffic_signal_enabled=ON and intersection_type=Y-Junction from step 0
- Only curriculum: signal_green_ratio 0.8 -> 0.7 -> 0.6 -> 0.5 -> 0.4
- Single-param curriculum eliminated ordering conflicts (P-022 fix validated)
- **No signal crash** (v3 had -177 point crash; v4 smooth transition)
- 3/4 green_ratio transitions completed (0.8 -> 0.7 -> 0.6 -> 0.5)
- **Missed**: green_ratio 0.5 -> 0.4 (threshold 540, agent plateau ~490-500)
- Root cause: reward range narrows as green ratio decreases (P-023)

---

## v2 Training Progress

### Reward Curve

```
Reward
+660  |                           * Peak (7.5M)
      |                        __/ \__
+640  |                     __/       \___
      |                  __/              \______ 632
+620  |               __/
      |            __/
+600  |         __/ <-- Resume (3.7M): LR + thresholds adjusted
      |      __/
+550  |   __/
      |  /
+400  | /
      |/
  +0  |*
      +--------------------------------------------------
       0   1.0   2.0   3.0   4.0   5.0   6.0   7.0   8.0   9.0   10.0 M steps
```

### Curriculum Transition Timeline (v2)

| Step | Parameter | Transition | Threshold |
|------|-----------|-----------|-----------|
| ~100K | goal_distance | 50m -> 100m | 30 |
| ~200K | goal_distance | 100m -> 150m | 100 |
| ~300K | num_lanes | 1 -> 2 | 150 |
| ~400K | center_line_enabled | Off -> On | 200 |
| ~500K | goal_distance | 150m -> 200m | 200 |
| ~700K | goal_distance | 200m -> 230m | 400 |
| ~1.5M | num_active_npcs | 0 -> 1 | 250 |
| ~2.0M | num_active_npcs | 1 -> 2 | 350 |
| ~2.5M | num_active_npcs | 2 -> 3 | 450 |
| ~2.8M | npc_speed_ratio | 0.5 -> 0.7 | 300 |
| ~3.0M | npc_speed_ratio | 0.7 -> 0.85 | 400 |
| ~3.5M | npc_speed_variation | 0 -> 0.05 | 500 |
| ~4.0M | npc_speed_variation | 0.05 -> 0.10 | 550 |
| ~4.5M | npc_speed_variation | 0.10 -> 0.15 | 600 |
| ~5.0M | intersection_type | None -> T-junction | 590 |
| ~6.0M | intersection_type | T -> Cross | 620 |
| ~6.0M | turn_direction | Straight -> Left | 605 |
| ~7.0M | turn_direction | Left -> Right | 635 |
| ~10.0M | intersection_type | Cross -> Y-junction | **MISSED (650)** |
| ~10.0M | traffic_signal_enabled | Off -> On | **MISSED (670)** |

---

## v3 Training Progress

### Reward Curve

```
Reward
+660  |  *658 (pre-signal peak, ~900K)
      | / \
+640  |/   \____
      |         \__ 647
+600  |             | <-- signal_green_ratio -> 0.5 (no effect, signals OFF)
      |             |
+550  |             |
      |             * traffic_signal_enabled -> ON (1.84M)
+500  |             |\
      |             | \_____ 500
+480  |             |        \_____ 490
      |             |               \_____ 477
+450  |             |
      +--------------------------------------------------
       0    0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0 M steps
```

### Curriculum Transition Timeline (v3)

| Step | Parameter | Transition | Threshold | Status |
|------|-----------|-----------|-----------|--------|
| ~520K | intersection_type | Cross -> Y-Junction | 580 | DONE |
| ~1.04M | signal_green_ratio | 0.7 -> 0.5 | 620 | DONE (no effect, signals OFF) |
| ~1.84M | traffic_signal_enabled | Off -> On | 600 | DONE |
| - | signal_green_ratio | 0.5 -> 0.4 | 640 | MISSED |

### Signal Ordering Problem (P-022)

ML-Agents curriculum parameters transition independently based on reward thresholds. When `signal_green_ratio` threshold (620) was lower than the reward level (~650), it changed from 0.7 to 0.5 at ~1.04M while signals were still OFF -- having no effect. When signals finally turned ON at ~1.84M, the green ratio was already at 0.5 (harder), not the intended 0.7 (easy start).

```
INTENDED:  signals ON (easy green 0.7) -> reduce green 0.7->0.5->0.4
ACTUAL:    green 0.7->0.5 (no effect) -> signals ON (already hard green 0.5)
```

The agent faced Y-Junction + signals ON + 0.5 green ratio simultaneously, causing a 177-point crash (647 -> 470) with no recovery over 3M+ steps.

---

## v4 Training Progress

### Reward Curve

```
Reward
+620  |  *616 (peak, green=0.8, ~680K)
      | / \
+600  |/   \____
      |         \__ 580
+550  |             \___
      |                 \__ 530
+500  |                     \_____ 510
      |                            \__ 500
+480  |                                \_____ 497
      |                                       \______ 490
+450  |
      +--------------------------------------------------
       0    0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0 M steps
                ^         ^              ^
               0.7       0.6            0.5
```

### Curriculum Transition Timeline (v4)

| Step | Parameter | Transition | Threshold | Status |
|------|-----------|-----------|-----------|--------|
| ~700K | signal_green_ratio | 0.8 -> 0.7 | 450 | DONE |
| ~1.4M | signal_green_ratio | 0.7 -> 0.6 | 480 | DONE |
| ~2.06M | signal_green_ratio | 0.6 -> 0.5 | 510 | DONE |
| - | signal_green_ratio | 0.5 -> 0.4 | 540 | MISSED |

### Green Ratio Plateau (P-023)

With green_ratio=0.5, the agent's reward stabilized at ~490-500, approximately 40 points below the threshold of 540 for the final transition. As the green ratio decreases, the agent spends more time waiting at red lights, reducing the per-episode progress and speed rewards. This narrows the achievable reward range, making higher thresholds unreachable.

The signal-first approach successfully eliminated the v3 ordering problem (no crash at signal activation), validating P-022. But the reward compression at lower green ratios creates a new ceiling.

---

## Key Findings

### 1. From-Scratch 268D Training Works
Despite starting from zero, the agent rebuilt all 9 phases of driving skills within 10M steps. This validates that the 268D observation space is learnable, though it requires more steps than warm-start training.

### 2. Threshold Lowering at Resume (P-021)
From-scratch training reaches lower reward ceilings than warm-start training. Original thresholds (designed for warm-start levels) were too high. Lowering intersection thresholds by ~30 points at 3.7M helped agent progress past the 600 plateau.

### 3. Reward Plateau at ~630
Agent plateaued at 630-660 after mastering Cross intersections. The Y-Junction threshold (650) was borderline reachable, and traffic signal thresholds (670+) were out of range given the complexity.

### 4. 10M Steps Insufficient for Full Curriculum
From-scratch 268D with 13 curriculum parameters needs more than 10M steps. Warm-start v3 should be more efficient with only 3 remaining params.

### 5. Independent Curriculum Ordering Conflict (P-022)
ML-Agents curriculum parameters transition independently. When parameter B only has meaning when parameter A is active (green_ratio only matters when signals ON), independent reward thresholds cannot enforce ordering. The solution is to lock dependent features and use single-parameter curriculum.

### 6. Signal-First Approach Validated (v4)
The v4 signal-first strategy (lock signals ON from step 0, single-param green_ratio curriculum) eliminated the v3 ordering crash. No signal activation shock occurred. However, reward compression at lower green ratios (P-023) limits how far the green ratio can be reduced within a single training run.

### 7. Reward Compression Under Signal Constraints (P-023)
As green_ratio decreases from 0.8 to 0.5, the agent's reward ceiling drops proportionally (616 -> ~490-500). Red light waiting reduces progress/speed rewards per episode. Threshold design must account for this compression -- or reward shaping must compensate for lost progress during red phases.

---

## Lessons Learned

### P-020: Observation Dimension Change = Fresh Start Required
When observation space changes (260D -> 268D), the Adam optimizer's internal state tensors have fixed sizes matching the old observation dimension. Warm start from a checkpoint with different observation size causes an immediate tensor mismatch crash. Always train from scratch when obs dimensions change.

### P-021: From-Scratch Training Needs Lower Thresholds
Warm-start training begins near the previous phase's reward level (e.g., 600+), so curriculum thresholds can be set at 650, 670, etc. From-scratch training starts at 0 and must rebuild all skills, reaching lower steady-state rewards. Thresholds must account for this lower ceiling.

### P-022: Feature Activation Must Precede Parameter Tuning
When parameter B only has meaning when parameter A is active (e.g., green_ratio only matters when signals are ON), they cannot be reliably sequenced through independent reward thresholds. Solution: lock the dependent feature from start and use single-parameter curriculum to avoid ordering conflicts.

### P-023: Reward Compression Under Signal Constraints
As traffic signal green_ratio decreases, agents spend more time waiting at red lights, reducing per-episode progress and speed rewards. This compresses the achievable reward range, making higher thresholds unreachable. Threshold spacing must account for reward ceiling reduction at each green_ratio level, or reward shaping must compensate for lost progress during red phases.

---

## Artifacts

- **Config (v2)**: `python/configs/planning/vehicle_ppo_phase-J-v2.yaml`
- **Config (v3)**: `python/configs/planning/vehicle_ppo_phase-J-v3.yaml`
- **Config (v4)**: `python/configs/planning/vehicle_ppo_phase-J-v4.yaml`
- **Results (v2)**: `results/phase-J-v2/E2EDrivingAgent/`
- **Results (v3)**: `results/phase-J-v3/E2EDrivingAgent/`
- **Results (v4)**: `results/phase-J-v4/E2EDrivingAgent/`
- **Best Checkpoint**: `E2EDrivingAgent-9499888.pt` (v2 9.5M, ~652)
- **Build**: `Builds/PhaseJ/PhaseJ.exe`
- **Scene**: `Assets/Scenes/PhaseJ_TrafficSignals.unity`
- **Experiment Archive (v2)**: `experiments/phase-J-traffic-signals-v2/`
- **Experiment Archive (v3)**: `experiments/phase-J-traffic-signals-v3/`
- **Experiment Archive (v4)**: `experiments/phase-J-traffic-signals-v4/`

---

[Phase I](./phase-i) | [Phase K](./phase-k) | [Home](../)

*Last Updated: 2026-02-02 (Phase J v4 Partial 3/4 green_ratio)*
