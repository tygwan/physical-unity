# Curriculum Learning

## Overview

Curriculum learning progressively increases environment difficulty as the agent's performance improves. This prevents early catastrophic failures while enabling the agent to handle complex scenarios.

## Parameters

| Parameter | Stages | Description |
|-----------|--------|-------------|
| `num_active_npcs` | 1 → 2 → 3 → 4+ | Number of NPC vehicles |
| `npc_speed_variation` | 0 → 0.3 | Speed randomness range |
| `npc_speed_ratio` | 0.3 → 0.6 → 0.9 | NPC speed relative to ego |
| `goal_distance` | 80 → 150 → 230 | Target distance in meters |
| `speed_zone_count` | 1 → 2 → 4 | Number of speed limit zones |

## Staggered Threshold Design

After Phase D v1's catastrophic failure, thresholds are deliberately staggered:

```
Reward 200 → num_active_npcs: 1→2 (easiest impact)
Reward 300 → npc_speed_variation: 0→0.3
Reward 300 → num_active_npcs: 2→3
Reward 350 → speed_zone_count: 1→2
Reward 350 → npc_speed_ratio: 0.3→0.6
Reward 350 → goal_distance: 80→150
```

## Lesson Transition Rules

Each curriculum parameter uses:

- **measure**: `reward` (smoothed cumulative reward)
- **signal_smoothing**: `true` (prevents noise-triggered transitions)
- **min_lesson_length**: `500` steps (minimum before advancing)
- **threshold**: varies per parameter (staggered)

## Phase D v1 Failure Analysis

Three parameters shared the same threshold (~400):

```
Step 4.68M: reward hits 400
  → num_active_npcs: 1→2 (difficulty ++)
  → speed_zone_count: 1→2 (difficulty ++)
  → npc_speed_variation: 0→0.3 (difficulty ++)
  Result: reward crashed +406 → -4,825
```

This "curriculum shock" overwhelmed the agent with simultaneous difficulty increases.

## Best Practices

1. **Stagger thresholds**: Each parameter should advance at a different reward level
2. **Start with easiest impact**: NPC count increases before speed or distance
3. **Use signal smoothing**: Prevents noise-triggered premature transitions
4. **Monitor lesson numbers**: Track which parameters have advanced via TensorBoard
5. **Budget for recovery**: Each transition temporarily decreases reward; allow steps for recovery
