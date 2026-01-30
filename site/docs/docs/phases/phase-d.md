# Phase D: Lane Observation (254D)

| Metric | Value |
|--------|-------|
| **Reward** | IN PROGRESS |
| **Status** | v2 Training |
| **Steps** | 10M target |
| **Observations** | 254D |
| **Tags** | PPO, Lane Detection, 254D, Raycast |

## Overview

Adding 12D lane marking observations (left/right edge detection via raycasts) expanding observation space from 242D to 254D. Fresh training required due to dimension change.

## v1 Failure

Three curriculum parameters shared the same threshold (~400), causing simultaneous transitions:

```
Step 4.68M: reward hits 400
  → num_active_npcs: 1→2
  → speed_zone_count: 1→2
  → npc_speed_variation: 0→0.3
  Result: reward crashed +406 → -4,825
```

## v2 Fix: Staggered Curriculum

```
Step ~2M:   reward 200 → num_active_npcs 1→2 (FIRST)
Step ~4M:   reward 300 → npc_speed_variation 0→0.3 (SECOND)
Step ~5.5M: reward 350 → zones/ratio/goal advance (THIRD)
Step ~7M:   reward 300 → num_active_npcs 2→3
Step ~9M:   reward 300 → num_active_npcs 3→4 (FINAL)
```

## Current Status

v2 training in progress with staggered thresholds ensuring one parameter advances at a time.
