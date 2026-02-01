# Phase G v1 Analysis: Intersection Navigation

**Date**: 2026-02-01
**Run ID**: phase-G
**Config**: `python/configs/planning/vehicle_ppo_phase-G.yaml`
**Steps**: 10,000,153 / 10,000,000 (budget exhausted)

---

## Executive Summary

Phase G v1 achieved **partial success**. The agent learned straight-through and turning behavior at T-junctions and Cross intersections, but reward plateaued at ~494 (peak 516) -- failing to reach the Y-junction curriculum threshold (550). Key bottleneck: **WrongWay terminations at 32%** indicate the agent struggles with turn execution accuracy after intersection maneuvers.

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Final Reward | 550+ (Y-junction) | 494.09 | MISS |
| Peak Reward | 600+ | 516.24 | MISS |
| Collision Rate | <5% | 0.0% | PASS |
| Goal Completion | >80% | 67.9% | MISS |
| WrongWay Rate | <5% | 31.9% | FAIL |
| Curriculum | 3/3 intersection | 2/3 (Cross) | PARTIAL |

**Grade: C+ (Partial -- Plateau Before Target)**

---

## Training Progression

### Reward Curve

| Step | Reward | Phase | Notes |
|------|--------|-------|-------|
| 0.5M | -107 | Struggling | Random policy, negative rewards |
| 1.0M | -106 | Struggling | Still exploring |
| 1.5M | -97 | Emerging | First signs of learning |
| 2.0M | +143 | Breakthrough | Lane keeping acquired |
| 2.5M | +236 | Climbing | Intersection curriculum begins |
| 3.0M | +321 | Strong growth | T-junction -> Cross transition |
| 3.5M | +373 | Growth | Turn directions expanding |
| 4.0M | +403 | Slowing | Approaching 400 plateau |
| 5.0M | +406 | **Plateau** | Growth stalls at ~400 |
| 7.0M | +439 | Slow crawl | +33 over 2M steps |
| 9.0M | +498 | Late spike | Goal distance curriculum kicks in |
| 10.0M | +494 | Final | Slight regression from 9.5M peak |

**Key observation**: 4M-8M was a 4M-step plateau at ~400-440. The late push to ~500 coincided with goal_distance (8.7M) and num_active_npcs (8.9M) curriculum transitions, not intersection mastery improvement.

### Curriculum Transitions

| Parameter | Lesson 0->1 | Lesson 1->2 | Lesson 2->3 | Final |
|-----------|------------|------------|------------|-------|
| intersection_type | 2.1M | 3.4M | **never** | 2 (Cross) |
| turn_direction | 2.4M | 3.9M | -- | 2 (Right) |
| num_lanes | 2.6M | -- | -- | 1 (2-lane) |
| center_line_enabled | 2.9M | -- | -- | 1 (On) |
| goal_distance | **8.7M** | -- | -- | 1 (Medium) |
| num_active_npcs | **8.9M** | -- | -- | 1 (1 NPC) |
| npc_speed_ratio | never | -- | -- | 0 (Slow) |
| npc_speed_variation | never | -- | -- | 0 (Uniform) |

**Analysis**: Early curriculum (intersection, turn, lanes) progressed well by 4M steps. But reward then stagnated, blocking later curriculum stages. Goal distance and NPC transitions happened very late (8.7-8.9M) as reward briefly crossed their lower thresholds.

---

## Episode Analysis

### End Reasons (final 50-point average)

| Reason | Rate | Assessment |
|--------|------|------------|
| Goal Reached | 67.9% | Below 80% target |
| WrongWay | 31.9% | **Primary failure mode** |
| Collision | 0.0% | Excellent safety |
| Stuck | 0.2% | Negligible |
| OffRoad | 0.0% | Good |
| LaneViolation | 0.0% | Good |

**Root cause**: 1/3 of episodes end in WrongWay termination. This occurs when the agent completes a turn but ends up facing backward on the exit road, or overshoots the turn arc and leaves the valid heading range.

### Reward Component Breakdown (10M)

| Component | Value | % of Total | Trend |
|-----------|-------|-----------|-------|
| Progress | +145.30 | 60.3% | Steady growth |
| Speed | +90.37 | 37.5% | Good |
| LaneKeeping | -7.49 | -3.1% | Improving (was -25 at 3M) |
| Jerk | -0.01 | ~0% | Excellent smoothness |
| Time | -0.10 | ~0% | Negligible |
| **Total** | **~228** | -- | Per-step average |

**Observation**: Progress and Speed dominate positively. LaneKeeping penalty improved from -25 to -7 over training, suggesting the agent learned to stay closer to lane center over time.

### Agent Stats (final averages)

| Stat | Value | Assessment |
|------|-------|------------|
| Speed | 15.6 m/s | 93.8% of limit (good) |
| Steering | 0.173 rad | Moderate (turns require more) |
| Acceleration | 1.30 m/s^2 | Smooth |
| Distance/Episode | 317 m | Goal at 150m, some overshoot |
| Episode Length | 1065 steps | ~21s at 50fps |
| Collisions | 0.0 | Perfect safety |

---

## Root Cause Analysis: Why Plateau at ~500?

### 1. WrongWay Termination (31.9%)

The dominant failure mode. After completing a turn (left/right), the agent's heading may exceed the WrongWay detection angle threshold. This is a **geometry problem** -- the exit waypoints after a turn have specific expected headings, and the agent's turn arc doesn't always align precisely.

**Evidence**: WrongWay rate is consistent throughout training, not decreasing. The agent cannot "learn away" a geometry misalignment issue.

### 2. Fresh Start Penalty

Phase G started from scratch (254D->260D dimension change). Unlike Phase F v5 (which inherited Phase E knowledge), Phase G had no pre-trained driving skill. The first 2M steps were essentially relearning basic lane keeping.

### 3. High Curriculum Density

9 curriculum parameters with thresholds packed into reward range 150-800. Cross intersection threshold (550) requires the agent to master turns while simultaneously handling 2 lanes, center line enforcement, and beginning NPC interaction. This is a lot of simultaneous complexity.

### 4. Goal Distance Too Short Initially

`goal_distance` starts at 120m, but with intersection at Z=100, the goal is only 20m past the intersection. Episodes where the agent successfully navigates the intersection but slightly overshoots get WrongWay terminated before reaching the goal, losing the +10 goal bonus.

---

## V2 Strategy Recommendations

### Priority 1: Fix WrongWay Issue
- Increase WrongWay angle threshold from current value to be more tolerant post-intersection
- OR add a WrongWay grace period after intersection traversal
- OR tune exit waypoint headings to better match actual turn trajectories

### Priority 2: Simplify Curriculum
- Reduce to 2 macro phases instead of 3
- Move NPC-related parameters to Phase H (separate training)
- Focus Phase G purely on intersection geometry mastery

### Priority 3: Adjust Thresholds
- Lower Y-junction threshold from 550 to 400-450 (agent plateaus at ~500, Y-junction should be reachable)
- OR increase max_steps to 15M with current thresholds

### Priority 4: Warm Start Option
- Use Phase G v1 checkpoint (10M steps, 494 reward) as init_path for v2
- This preserves the already-learned intersection navigation while fixing issues

---

## Comparison with Previous Phases

| Phase | Steps | Peak | Final | Goal% | Collision% | Grade |
|-------|-------|------|-------|-------|------------|-------|
| Phase 0 | 8M | 1018 | 1018 | 100% | 0% | A+ |
| Phase A | 2.5M | 3161 | 2114 | 100% | 0% | A |
| Phase B v2 | 3.5M | 897 | 877 | ~100% | ~0% | A |
| Phase C | 3.6M | 1390 | 1372 | -- | -- | A |
| Phase D v3-254d | 5M | 912 | 904 | -- | -- | A- |
| Phase E | 6M | 956 | 924 | -- | -- | A- |
| Phase F v5 | 10M | 913 | 643 | -- | -- | B+ |
| **Phase G** | **10M** | **516** | **494** | **68%** | **0%** | **C+** |

Phase G is the hardest phase attempted, with a fresh start and complex intersection geometry. The C+ grade reflects partial success -- the agent learned basic intersection navigation but couldn't fully master all turn types.

---

## Artifacts

- Model: `results/phase-G/E2EDrivingAgent/E2EDrivingAgent-10000153.onnx`
- Best checkpoint: `results/phase-G/E2EDrivingAgent/E2EDrivingAgent-9499790.pt` (~506 reward)
- Config: `python/configs/planning/vehicle_ppo_phase-G.yaml`
- TensorBoard: `results/phase-G/E2EDrivingAgent/events.out.tfevents.*`
