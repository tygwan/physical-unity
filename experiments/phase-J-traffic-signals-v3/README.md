# Phase J v3: Traffic Signals (Warm Start from v2)

## Experiment Overview

| Item | Value |
|------|-------|
| **Run ID** | phase-J-v3 |
| **Date** | 2026-02-02 |
| **Steps** | 5,000,000 |
| **Final Reward** | +477 |
| **Peak Reward (pre-signal)** | +658 (at ~900K) |
| **Peak Reward (with signal)** | +538 (at ~1.94M) |
| **Observation** | 268D |
| **Initialize From** | v2 9.5M checkpoint (E2EDrivingAgent-9499888.pt) |
| **Training Mode** | Build + 3 parallel envs (no_graphics) |
| **Duration** | ~25.5 min (1,530 seconds) |

---

## Strategy: Warm Start Remaining Curriculum

v3 warm-started from v2's best checkpoint (9.5M, ~652 reward) with only 3 remaining curriculum parameters:
- intersection_type: Cross(2) -> Y-Junction(3)
- traffic_signal_enabled: Off(0) -> On(1)
- signal_green_ratio: 0.7 -> 0.5 -> 0.4

All other 10 parameters locked at final values.

---

## Results

### Curriculum Transitions (3/4 completed)

| Step | Parameter | Transition | Threshold | Status |
|------|-----------|-----------|-----------|--------|
| ~520K | intersection_type | Cross -> Y-Junction | 580 | DONE |
| ~1.04M | signal_green_ratio | 0.7 -> 0.5 | 620 | DONE |
| ~1.84M | traffic_signal_enabled | Off -> On | 600 | DONE |
| - | signal_green_ratio | 0.5 -> 0.4 | 640 | MISS |

### Training Progression

| Step | Reward | Std | Event |
|------|--------|-----|-------|
| 10K | 282 | 6 | Initial buffer noise |
| 30K | 643 | 14 | Warm start stabilized |
| 300K | 650 | 18 | Stable at v2 level |
| 520K | 651 | 11 | **Y-Junction unlocked** |
| 600K | 643 | 9 | Y-Junction adaptation |
| 900K | 658 | 8 | **Pre-signal peak** |
| 1.04M | 645 | 35 | **signal_green_ratio -> 0.5** (no effect, signals OFF) |
| 1.84M | 647 | 12 | **traffic_signal_enabled -> ON** |
| 1.86M | 470 | 218 | **SIGNAL CRASH (-177 points)** |
| 2.0M | 489 | 119 | Post-signal plateau |
| 3.0M | 500 | 120 | Oscillating |
| 4.0M | 490 | 115 | No recovery |
| 5.0M | 477 | 126 | Budget exhausted |

---

## Root Cause: Curriculum Ordering Problem (P-022)

ML-Agents curriculum parameters are **independent** -- each parameter transitions based solely on smoothed reward crossing its threshold, regardless of other parameters' states.

### What Happened

1. **signal_green_ratio threshold (620) < reward level (~650)**: Green ratio changed from 0.7 to 0.5 at ~1.04M steps, while `traffic_signal_enabled=0` (signals OFF). This had **no actual effect** since signals weren't active.

2. **traffic_signal_enabled threshold (600) crossed later (~1.84M)**: When signals finally turned ON, the green ratio was already at 0.5 (harder setting), not the intended 0.7 (easy start).

3. **Double shock**: Agent faced Y-Junction + signals ON + 0.5 green ratio simultaneously, causing a 177-point crash that it never recovered from.

### The Ordering Issue

```
INTENDED:  signals ON (easy green 0.7) -> reduce green 0.7->0.5->0.4
ACTUAL:    green 0.7->0.5 (no effect) -> signals ON (already hard green 0.5)
```

Independent curriculum params cannot enforce ordering when thresholds overlap in the reward range.

---

## Lesson P-022

**Feature activation must precede parameter tuning -- use single-param curriculum to avoid independent ordering conflicts.**

When parameter B only has meaning when parameter A is active (e.g., green_ratio only matters when signals are ON), they cannot be reliably sequenced through independent reward thresholds. Solution: lock the dependent feature from start and use single-parameter curriculum.

---

## v4 Design (fix)

- Lock traffic_signal_enabled=ON and intersection_type=Y-Junction from step 0
- Only curriculum: signal_green_ratio 0.8 -> 0.7 -> 0.6 -> 0.5 -> 0.4
- Single-parameter curriculum eliminates ordering conflicts
- Thresholds: 450/480/510/540 (post-signal reward range, 30-point spacing)
- keep_checkpoints=10 (preserve early checkpoints for analysis)
- Config: `python/configs/planning/vehicle_ppo_phase-J-v4.yaml`

---

## Artifacts

- **Config**: `config/vehicle_ppo_phase-J-v3.yaml`
- **Results**: `results/phase-J-v3/E2EDrivingAgent/`
- **Checkpoints**: E2EDrivingAgent-3499745.pt through E2EDrivingAgent-5000109.pt
- **Note**: Early checkpoints (500K, 1M) rotated out by keep_checkpoints=5
