# Phase B: Decision Learning

| Metric | Value |
|--------|-------|
| **Reward** | +994 |
| **Status** | SUCCESS (v2) |
| **Steps** | 3.0M |
| **Observations** | 242D |
| **Tags** | PPO, NPC Interaction, Overtaking Decision |

## Overview

Extended Phase A with intelligent overtaking decisions. Agent learns when to overtake vs follow, using NPC interaction observations.

## v1 Failure

- Fresh training collapsed at 1.8M steps
- Reward crashed due to NPC interaction complexity
- Agent couldn't learn overtaking decisions without prior driving knowledge

## v2 Success

- Warm-started from Phase A checkpoint
- Checkpoint transfer provided critical bootstrap
- Agent built on existing driving policy to learn decisions
- Final reward: +994 (exceeded Phase A)

## Key Insight

Checkpoint transfer from Phase A provided critical bootstrap. Fresh training (v1) collapsed due to NPC interaction complexity; warm-started training (v2) succeeded.
