# Phase D v3: Lane Observation - Technical Design

**Created**: 2026-01-30
**Status**: COMPLETED
**Original Design**: `../phase-D-lane-observation-v2/DESIGN_V3.md`

---

## Overview

Phase D v3 adds **12D lane observation** to the 242D observation space (total 254D).
Key constraint: 242D->254D dimension change prevents checkpoint transfer, requiring fresh training.

## Design Principles

### P-009: Observation-Curriculum Coupling Ban
When observation space changes, **freeze all curriculum parameters**.
- Phase D v1/v2 failed by changing observations AND curriculum simultaneously
- v3 fixes this by using a fixed environment (no curriculum progression)

### P-010: Scene-Config-Code Consistency
VectorObservationSize in Unity must match the model's expected input dimension.
- v3 Run 1 (242D) was inadvertently trained without lane obs due to VectorObservationSize mismatch
- v3 Run 2 (254D) fixed the Unity config to match 254D

## Observation Space (254D)

| Component | Dimensions | Source |
|-----------|-----------|--------|
| Ego state | 8D | Position, velocity, heading, acceleration |
| Route info | 30D | Waypoints, distances |
| NPC vehicles | 192D | 8 vehicles x 24 features |
| Lane markings | 12D | Left/right edge detection (6D each) |
| **Total** | **254D** | |

### Lane Observation (12D) Detail
- 6 raycasts per side (left + right = 12D)
- Each raycast returns normalized distance to lane marking
- `LaneMarking` component on BoxCollider objects with "LaneMarking" layer
- Raycasts: -60, -30, 0, +30, +60, +90 degrees from vehicle heading

## Environment Configuration

**Fixed environment** (no curriculum):
- 3 NPCs active (fixed)
- NPC speed ratio: 0.6 (fixed)
- Goal distance: 230m (fixed)
- Single speed zone
- Straight road, single lane

## Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| batch_size | 4096 | Standard |
| buffer_size | 40960 | Standard |
| learning_rate | 3.0e-4 | Standard |
| hidden_units | 512 | Standard |
| num_layers | 3 | Standard |
| max_steps | 5,000,000 | Fresh start budget |
| checkpoint_interval | 500,000 | Standard |
| init_path | None | 242D->254D incompatible |

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| Final Reward | > +800 | Match 242D baseline (+835) |
| Lane Obs Benefit | > 0% | Prove lane info adds value |
| Stability | No collapse | Smooth convergence |
| Convergence | < 4M steps | Reasonable for fresh start |

## Two-Run Strategy

### Run 1: 242D Baseline (Accidental)
- VectorObservationSize=242 (lane obs disabled in Unity)
- Result: +835 @ 5M steps
- Served as 242D baseline for comparison

### Run 2: 254D with Lane Observation
- VectorObservationSize=254 (lane obs enabled)
- Result: +895.5 @ 5M steps
- +7.2% improvement over 242D baseline

---

**Related**: `ANALYSIS.md`, `../phase-D-lane-observation-v2/DESIGN_V3.md`
