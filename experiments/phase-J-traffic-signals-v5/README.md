# Phase J v5: Traffic Signals (Deceleration Reward + Lower Thresholds)

## Status: COMPLETE (2026-02-02)

## Results

| Metric | Value |
|--------|-------|
| Steps | 5,000,000 |
| Peak Reward | 605.7 (at 1.44M) |
| Final Reward | 537.7 |
| Curriculum | 5/5 green_ratio COMPLETE |
| Goal Rate | 56% |
| Collision Rate | 0% |
| Stuck Rate | 4% |

## Strategy

Warm start from v2 best checkpoint (9.5M, 268D observation space). Two key improvements over v4:

### 1. Code Fixes (E2EDrivingAgent.cs)

- **False violation bug**: `hasPassedStopLine` carried from Green to Red phase, causing agents that passed on Green to be flagged when signal changed to Red. Fixed with `wasPastStopLineAtRedStart` tracking on Red phase transitions.
- **Deceleration reward**: Distance-proportional target speed within 50m of red light stop line. Gives agent gradient to learn braking behavior.
- **Yellow approach penalty**: Speed-based penalty when >15m from stop line during Yellow phase.

### 2. Lower Thresholds (P-023 Fix)

v4 plateau analysis showed reward ceiling drops with each green_ratio decrease:

| green_ratio | v4 Plateau | v4 Threshold | v5 Threshold |
|-------------|-----------|-------------|-------------|
| 0.8 | ~570 | 450 | 450 |
| 0.7 | ~530 | 480 | 470 |
| 0.6 | ~510 | 510 | 475 |
| 0.5 | ~490 | 540 (MISSED) | 475 |

## Curriculum Progression

| Transition | Threshold | Status |
|-----------|-----------|--------|
| 0.8 -> 0.7 | 450 | DONE |
| 0.7 -> 0.6 | 470 | DONE |
| 0.6 -> 0.5 | 475 | DONE |
| 0.5 -> 0.4 | 475 | DONE |

## Reward Progression (100K intervals)

```
  100K: 489  ########################
  200K: 498  ########################
  400K: 568  ############################
  600K: 522  ##########################
  800K: 517  #########################
 1000K: 580  #############################
 1200K: 573  ############################
 1440K: 606  ############################## <-- PEAK
 2000K: 545  ###########################
 3000K: 546  ###########################
 4000K: 541  ###########################
 5000K: 538  ##########################
```

## Bug Found: P-024 (BehaviorType Build Issue)

BehaviorType=InferenceOnly left in scene from inference test caused build to produce non-trainable agents. Environments connected, episodes ran, but no brain registration occurred. Training silently produced zero output for ~30 minutes before diagnosis.

## Config

See `config/vehicle_ppo_phase-J-v5.yaml`

## Artifacts

- Config: `python/configs/planning/vehicle_ppo_phase-J-v5.yaml`
- Results: `results/phase-J-v5/E2EDrivingAgent/`
- Final Model: `E2EDrivingAgent-5000148.pt` / `.onnx`
- Peak Model: `E2EDrivingAgent-1499929.pt` (near peak at 1.44M)
