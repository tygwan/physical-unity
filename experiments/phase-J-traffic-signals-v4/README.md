# Phase J v4: Traffic Signals (Signal-First, Green Ratio Curriculum)

## Experiment Overview

| Item | Value |
|------|-------|
| **Run ID** | phase-J-v4 |
| **Date** | 2026-02-02 |
| **Steps** | 5,000,000 |
| **Final Reward** | +497 |
| **Peak Reward** | +616 (at ~680K, green_ratio=0.8) |
| **Observation** | 268D |
| **Initialize From** | v2 9.5M checkpoint (E2EDrivingAgent-9499888.pt) |
| **Training Mode** | Build + 3 parallel envs (no_graphics) |
| **Duration** | ~28 min (1,689 seconds) |

---

## Strategy: Signal-First Single-Param Curriculum

Fix for P-022 (v3 signal ordering conflict). Lock signals ON and Y-Junction from step 0. Only curriculum parameter: signal_green_ratio.

### Key Design Decisions
1. **Warm start from v2 9.5M** (not v3): v2 checkpoint is pre-signal baseline (~652 reward)
2. **Signals ON from step 0**: Agent learns signal compliance immediately, no ordering issues
3. **Single parameter**: Only green_ratio changes (0.8 -> 0.7 -> 0.6 -> 0.5 -> 0.4)
4. **30-point threshold spacing**: 450/480/510/540 (post-signal reward range)
5. **keep_checkpoints=10**: Preserve progression for analysis

### Locked Parameters (from step 0)
- traffic_signal_enabled = 1 (ON)
- intersection_type = 3 (Y-Junction)
- All other params at v2 final values

---

## Results

### Curriculum Transitions (3/4 completed)

| Step | Parameter | Transition | Threshold | Status |
|------|-----------|-----------|-----------|--------|
| ~700K | signal_green_ratio | 0.8 -> 0.7 | 450 | DONE |
| ~1.4M | signal_green_ratio | 0.7 -> 0.6 | 480 | DONE |
| ~2.06M | signal_green_ratio | 0.6 -> 0.5 | 510 | DONE |
| - | signal_green_ratio | 0.5 -> 0.4 | 540 | MISS |

### Training Progression

| Step | Reward | Event |
|------|--------|-------|
| 30K | ~570 | Warm start stabilized (signals ON from start) |
| 680K | **616** | **Peak** (green_ratio=0.8) |
| 700K | ~570 | signal_green_ratio -> 0.7 (threshold 450) |
| 1.0M | ~540 | Recovering at 0.7 |
| 1.4M | ~530 | signal_green_ratio -> 0.6 (threshold 480) |
| 2.06M | ~510 | signal_green_ratio -> 0.5 (threshold 510) |
| 3.0M | ~500 | Plateau at green_ratio=0.5 |
| 4.0M | ~495 | Still plateau |
| 5.0M | 497 | Budget exhausted |

### Reward by Green Ratio Level

| Green Ratio | Peak at Level | Plateau | Threshold | Reached? |
|-------------|---------------|---------|-----------|----------|
| 0.8 | 616 | ~570 | 450 | Yes |
| 0.7 | ~540 | ~530 | 480 | Yes |
| 0.6 | ~520 | ~510 | 510 | Yes |
| 0.5 | ~500 | ~490-500 | 540 | No (-40 gap) |

---

## Key Findings

### 1. P-022 Fix Validated
Signal-first approach eliminated the v3 ordering crash. No reward drop at signal activation. The agent started with signals ON and adapted from step 0. Smooth, predictable curriculum transitions.

### 2. Reward Compression (P-023)
As green_ratio decreased, the reward ceiling dropped proportionally. Each 0.1 decrease in green_ratio reduced the reward plateau by ~30-40 points. This is expected: more red light time = less progress/speed reward per episode.

### 3. Threshold 540 Unreachable
With green_ratio=0.5, the agent plateaus at ~490-500. The 40-point gap to threshold 540 is too large. The agent cannot compensate for the lost progress during red phases.

### 4. No Catastrophic Collapse
Unlike v3 (-177 point crash), v4 showed smooth, gradual reward decrease with each green_ratio step. The single-param approach prevents shock transitions.

---

## Lesson P-023

**Reward compression under signal constraints.** As green ratio decreases, agents spend more time stopped at red lights, reducing per-episode progress and speed rewards. This compresses the achievable reward range. Threshold spacing must account for this ceiling reduction at each difficulty level, or reward shaping must compensate for lost progress during red phases.

---

## Next Steps (Options)

1. **v5 with lower threshold**: Set green_ratio 0.5->0.4 threshold to ~470-480 instead of 540
2. **v5 with reward shaping**: Add positive reward for correct stop-line behavior during red
3. **Accept green_ratio=0.5**: Move to Phase K with current signal compliance level
4. **Longer training**: Extend v4-style training beyond 5M steps at green_ratio=0.5

---

## Artifacts

- **Config**: `config/vehicle_ppo_phase-J-v4.yaml`
- **Results**: `results/phase-J-v4/E2EDrivingAgent/`
- **Checkpoints**: 10 checkpoints (999850 through 5000029)
- **Build**: `Builds/PhaseJ/PhaseJ.exe`
