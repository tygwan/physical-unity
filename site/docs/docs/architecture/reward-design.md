# Reward Design

## Overview

The agent uses a **dense reward function** with 7 components providing clear learning signals at every timestep.

## Reward Components

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| Progress | +1.0 | [0, ~15] | Forward distance toward goal |
| Speed | +0.5 | [0, ~10] | Maintaining target speed |
| Lane Keeping | +0.3 | [0, ~5] | Centering within lane |
| Overtaking | +2.0 | [0, +10] | Successful NPC overtake |
| Violation | -5.0 | [-10, 0] | Lane/road boundary violation |
| Jerk | -0.1 | [-2, 0] | Comfort penalty for sudden accel |
| Time | -0.01 | [-1, 0] | Small penalty for elapsed time |

## Design Principles

### Dense vs Sparse Rewards
Dense reward shaping was critical for Phase A success. Early experiments with sparse rewards (goal-only) failed to provide learning signal, as episodes terminated before the agent could reach goals.

### Curriculum Interaction
Reward thresholds drive curriculum transitions:

- **200**: First NPC count increase
- **300**: Speed variation and additional NPCs
- **350**: Speed zones, goal distance, speed ratio

### Staggered Thresholds
Phase D v1 failure taught that parameters sharing the same threshold advance simultaneously, causing "curriculum shock". Phase D v2 fixed this with staggered thresholds.

```yaml
# Bad: Same threshold (Phase D v1)
num_active_npcs: threshold: 400
speed_zone_count: threshold: 400
npc_speed_variation: threshold: 400

# Good: Staggered thresholds (Phase D v2)
num_active_npcs: threshold: 200      # Advances FIRST
npc_speed_variation: threshold: 300  # Advances SECOND
speed_zone_count: threshold: 350     # Advances THIRD
```
