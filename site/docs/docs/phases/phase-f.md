# Phase F: Multi-Lane

| Metric | Value |
|--------|-------|
| **Reward** | +988 |
| **Status** | SUCCESS |
| **Steps** | 3.5M |
| **Observations** | 242D |
| **Tags** | PPO, 2-Lane, Center Line Rules |

## Overview

Expanded road to 2 lanes with center line rules. Agent learns proper lane selection, lane change timing, and center line discipline while navigating traffic.

## Results

- Proper lane selection and lane change timing
- Center line violation penalty was essential
- Without penalty, agent treated both lanes as single wide road
- Clean lane change trajectories emerged

## Key Insight

Center line violation penalty was essential for teaching lane discipline. Without it, agent treated both lanes as a single wide road.
