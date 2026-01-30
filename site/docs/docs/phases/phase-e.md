# Phase E: Curved Roads

| Metric | Value |
|--------|-------|
| **Reward** | +931 |
| **Status** | SUCCESS |
| **Steps** | 3.0M |
| **Observations** | 242D |
| **Tags** | PPO, Curvature, Banked Roads |

## Overview

Introduced curved road geometry with variable radius turns and banked surfaces. Agent learns speed adaptation for curves and maintains lane discipline through turns.

## Results

- Agent learned to decelerate before curves
- Accelerates on curve exits (matching human patterns)
- Maintains lane discipline through banked turns
- Curvature observations critical for safe cornering

## Key Insight

Curvature observations critical for safe cornering. Agent learned to decelerate before curves and accelerate on exits, matching human driving patterns.
