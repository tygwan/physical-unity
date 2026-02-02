# Phase J v2: Traffic Signals (From Scratch, 268D)

## Experiment Overview

| Item | Value |
|------|-------|
| **Run ID** | phase-J-v2 |
| **Date** | 2026-02-02 |
| **Steps** | 10,000,000 |
| **Final Reward** | +632.2 |
| **Peak Reward** | +660.6 (at ~7.5M) |
| **Observation** | 268D (260D + 8D traffic signal) |
| **Initialize From** | None (from scratch) |
| **Training Mode** | Build + 3 parallel envs (no_graphics) |
| **Scene** | PhaseJ_TrafficSignals |

---

## Strategy: From Scratch (268D)

Phase J v1 attempted warm start from Phase I v2 checkpoint (260D), but the 260D -> 268D observation mismatch caused an Adam optimizer tensor crash. v2 trains from scratch with the full 268D observation space.

### v1 Failure (P-020)
- **init_path**: Phase I v2 checkpoint (260D)
- **Error**: `RuntimeError: Adam optimizer state tensor size mismatch (260 vs 268)`
- **Steps completed**: ~40K (crashed immediately)
- **Lesson**: Observation dimension changes require fresh start (no warm start possible)

### v2 Design Choices
- **No init_path**: Training from scratch
- **goal_distance starts at 50m**: Like Phase 0, build fundamentals first
- **Lower curriculum thresholds**: From-scratch needs achievable early targets
- **Smaller batch_size (2048)**: Faster initial gradient updates
- **10M step budget**: Double the typical warm-start budget

---

## Resume Details (at 3.7M steps)

At 3.7M steps, reward plateaued around 600. Applied mid-training adjustments:

| Parameter | Before Resume | After Resume |
|-----------|---------------|--------------|
| learning_rate | 3e-4 | 1.5e-4 |
| intersection thresholds | 620/650/680 | 590/620/650 |
| Downstream thresholds | Adjusted | Maintained 15+ spacing |

Rationale: Agent was oscillating around 600, unable to push through. Learning rate reduction + threshold lowering gave it room to progress.

---

## Results

### Final Metrics

| Metric | Value |
|--------|-------|
| Total Steps | 10,000,000 |
| Final Reward | 632.2 |
| Peak Reward | 660.6 (at ~7.5M) |
| Curriculum Completed | 9/13 |
| Observation | 268D |
| Collision Rate | ~0% |

### Curriculum Status

| # | Parameter | Status | Final Lesson | Final Value |
|---|-----------|--------|--------------|-------------|
| 1 | goal_distance | DONE | FullGoal | 230m |
| 2 | num_lanes | DONE | TwoLanes | 2 |
| 3 | center_line_enabled | DONE | CenterLineEnforced | 1 |
| 4 | num_active_npcs | DONE | ThreeNPCs | 3 |
| 5 | npc_speed_ratio | DONE | NormalNPCs | 0.85 |
| 6 | npc_speed_variation | DONE | FullVariation | 0.15 |
| 7 | intersection_type (T) | DONE | TJunction | 1 |
| 8 | intersection_type (Cross) | DONE | CrossIntersection | 2 |
| 9 | turn_direction | DONE | RightTurn | 2 |
| 10 | **intersection_type (Y)** | **MISS** | CrossIntersection | 2 (threshold 650 not reached) |
| 11 | **traffic_signal_enabled** | **MISS** | NoSignal | 0 (threshold 670 not reached) |
| 12 | **signal_green_ratio (0.5)** | **MISS** | EasyGreen | 0.7 (blocked by signal) |
| 13 | **signal_green_ratio (0.4)** | **MISS** | EasyGreen | 0.7 (blocked by signal) |

### Training Progression

| Step | Reward | Event |
|------|--------|-------|
| 500K | ~30 | Basic driving from scratch |
| 1.0M | ~100 | goal_distance progression |
| 1.5M | ~200 | Multi-lane, center line |
| 2.0M | ~350 | NPCs introduced |
| 2.5M | ~450 | 2-3 NPCs |
| 3.0M | ~550 | Speed variation starting |
| 3.5M | ~580 | Approaching plateau |
| 3.7M | ~600 | **Resume: LR + thresholds adjusted** |
| 5.0M | ~610 | T-junction unlocked |
| 6.0M | ~630 | Cross intersection, turns |
| 7.5M | **660** | **PEAK** |
| 8.0M | ~645 | Slight decline from peak |
| 10.0M | ~632 | Budget exhausted |

---

## Best Checkpoint for v3

| File | Step | Reward |
|------|------|--------|
| **E2EDrivingAgent-9499888.pt** | **9.5M** | **~652** |

Selected as v3 init_path because:
- Near peak performance window
- Same 268D observation (no tensor mismatch)
- Post-curriculum completion (9/13 stable)
- Before late-stage oscillation

---

## v3 Handoff

Phase J v3 will warm start from the 9.5M checkpoint:
- **Config**: `python/configs/planning/vehicle_ppo_phase-J-v3.yaml`
- **init_path**: `results/phase-J-v2/E2EDrivingAgent/E2EDrivingAgent-9499888.pt`
- **Remaining curriculum**: Y-Junction (580), traffic signals (600), green ratio (620/640)
- **Budget**: 5M steps
- **Changes**: batch_size 4096, buffer_size 40960, num_epoch 5

---

## Artifacts

- **Config**: `config/vehicle_ppo_phase-J-v2.yaml` (copy from python/configs)
- **Results**: `results/phase-J-v2/E2EDrivingAgent/`
- **Build**: `Builds/PhaseJ/PhaseJ.exe`
- **Scene**: `Assets/Scenes/PhaseJ_TrafficSignals.unity`
