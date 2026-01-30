# Phase A: Dense Overtaking

| Metric | Value |
|--------|-------|
| **Reward** | +937 |
| **Status** | SUCCESS |
| **Steps** | 2.5M |
| **Observations** | 242D |
| **Tags** | PPO, Dense Reward, Curriculum |

## Overview

Foundation phase establishing core driving behavior with dense reward shaping. Agent learns lane keeping, speed control, and basic overtaking on a straight single-lane road.

## Environment

- Single straight lane (300m)
- 16 parallel training areas
- Curriculum: 1 → 4 NPCs
- Goal distance: 80 → 230m

## Key Results

- Reward stabilized at +937 after 2.5M steps
- Agent learned smooth lane keeping and speed control
- Basic overtaking behavior emerged through curriculum
- Dense reward with 7 components provided clear learning signal

## Key Insight

Dense reward shaping with 7 reward components enabled stable learning from scratch. Curriculum learning (1→4 NPCs) prevented early catastrophic failures.

## Config

```yaml
trainer_type: ppo
batch_size: 4096
buffer_size: 40960
learning_rate: 3.0e-4
hidden_units: 512
num_layers: 3
max_steps: 2500000
```
