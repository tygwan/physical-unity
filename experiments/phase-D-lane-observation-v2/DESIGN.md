# Phase D v2 Design: Staggered Curriculum

## Problem Statement

Phase D v1 demonstrated that 254D observation (with lane info) works - the agent reached +406 reward with 1 NPC. However, ML-Agents evaluates all curriculum parameters against reward independently. When reward reached ~400, three parameters with similar thresholds (400/400/400) all crossed simultaneously, causing a 5,231-point reward crash.

## Root Cause (from v1)

```
Step 4.68M: reward ~406
  → num_active_npcs threshold 400 → ADVANCE (1→2)
  → speed_zone_count threshold 400 → ADVANCE (1→2)
  → npc_speed_variation threshold 400 → ADVANCE (0→0.3)
  = Triple difficulty spike = catastrophic policy failure
```

## Solution: Staggered Thresholds

Each parameter uses a **different threshold** so they cannot advance at the same time:

```
Reward 200: num_active_npcs 1→2   (FIRST - primary difficulty)
Reward 300: npc_speed_variation 0→0.3   (SECOND - after NPC adaptation)
Reward 350: npc_speed_ratio 0.3→0.6   (THIRD - moderate group)
            goal_distance 80→150
            speed_zone_count 1→2
Reward 300: num_active_npcs 2→3   (FOURTH - recovered after moderate group)
Reward 350: npc_speed_ratio 0.6→0.9   (FIFTH - final group)
            goal_distance 150→230
            speed_zone_count 2→4
Reward 300: num_active_npcs 3→4   (SIXTH - final NPC count)
```

### Why This Works

1. When reward hits 200, ONLY num_active_npcs advances (nothing else has threshold 200)
2. Reward drops after NPC increase, then recovers
3. When reward hits 300, npc_speed_variation advances (num_active_npcs already past lesson 1)
4. The 350-threshold group advances together, but these are moderate changes
5. num_active_npcs at lesson 2→3 needs 300 again, which comes after the 350 group causes a dip

### Key Insight

The **order** of advancement is determined by threshold value:
- Lower threshold = advances earlier
- Parameters at the SAME threshold still advance together, but we group only moderate-impact changes at 350

## Training Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| max_steps | 10,000,000 | Extended from 6M for more curriculum stages |
| keep_checkpoints | 10 | More checkpoints for analysis |
| checkpoint_interval | 500,000 | Standard |
| min_lesson_length | 500 | All parameters |
| signal_smoothing | true | All parameters |

## Hyperparameters (unchanged from v1)

| Parameter | Value |
|-----------|-------|
| batch_size | 4096 |
| buffer_size | 40960 |
| learning_rate | 3.0e-4 |
| hidden_units | 512 |
| num_layers | 3 |
| gamma | 0.99 |
| lambd | 0.95 |
| num_epoch | 5 |

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| 350-group still advances together | Medium | ratio/goal/zones are moderate changes, not catastrophic |
| 10M steps insufficient | Low | v1 reached +406 by 4.6M, more budget available |
| Agent never reaches 200 reward | Low | v1 passed 200 by ~2.5M steps |
| Staggered transitions slow progress | Medium | Acceptable tradeoff for stability |
