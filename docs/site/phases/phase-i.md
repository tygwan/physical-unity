---
layout: default
title: Phase I - Curved Roads + NPC
---

# Phase I: Curved Roads + NPC Traffic

Curved road driving with 3 NPC vehicles and speed variation

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-I-v2 (final) |
| **Status** | Completed |
| **Date** | 2026-02-01 |
| **Total Steps** | 5,000,000 (v2) + 5,000,000 (v1) |
| **Training Time** | ~25 min (v2), ~25 min (v1) |
| **Final Reward** | **+770** |
| **Peak Reward** | **+775** (at 4.83M steps, v2) |
| **Observation** | 260D (unchanged from Phase H) |
| **Initialize From** | Phase I v1 5M checkpoint (via Phase H v3) |
| **Training Mode** | Build + 3 parallel envs (no_graphics) |

---

## Objective

Combine Phase E's curved road driving with Phase H's NPC traffic. Agent drives curved roads (curvature 0-1.0, S-curves) with 3 NPC vehicles at varying speeds. Intersections disabled (mutually exclusive with curves in WaypointManager).

### New Capabilities
- Full curvature (1.0) with NPC traffic
- Random S-curves (curve_direction_variation = 1.0)
- 2 speed zones on curved roads
- 3 NPCs with speed variation on curves

---

## Observation Space

**260D (unchanged from Phase H)**

Curvature observation (60D) was already in the space since Phase E but locked at 0. Phase I activates it alongside the existing NPC observation.

---

## Key Design: Curves vs Intersections

`WaypointManager.GenerateWaypoints()` is mutually exclusive:

```
if (intersectionType > 0)      -> intersection paths (curvature ignored)
else if (roadCurvature <= 0.01) -> straight paths
else                            -> curved paths
```

Phase I locks `intersection_type=0` to enable curved path generation.

---

## Curriculum Design (v1)

### Three-Tier Architecture

```
Tier 1: Quick-unlock (70-130)     -> ~300K steps
Tier 2: NPC re-unlock (550-693)   -> ~3.5M steps
Tier 3: Curves NEW (695-705)      -> ~3.76M steps
```

### All 14 Curriculum Parameters (17 Transitions)

| # | Parameter | Lessons | Final Value | Source |
|---|-----------|---------|-------------|--------|
| 1 | num_lanes | 1->2 | 2 | Phase F |
| 2 | center_line_enabled | off->on | 1 | Phase F |
| 3 | goal_distance | 150->200->230 | 230m | Phase F |
| 4 | num_active_npcs | 0->1->2->3 | 3 | Phase H |
| 5 | npc_speed_ratio | 0.5->0.7->0.85 | 0.85 | Phase H |
| 6 | npc_speed_variation | 0->0.05->0.10->0.15 | 0.15 | Phase H v3 |
| 7 | **road_curvature** | **0->0.3->0.6->1.0** | **1.0** | **NEW** |
| 8 | **curve_direction_variation** | **0->1.0** | **1.0** | **NEW** |
| 9 | **speed_zone_count** | **1->2** | **2** | **NEW** |
| 10 | intersection_type | (locked) | 0 | - |
| 11 | turn_direction | (locked) | 0 | - |

---

## Version History: v1 -> v2

| Aspect | v1 | v2 |
|--------|----|----|
| Init from | Phase H v3 5M | Phase I v1 5M |
| Steps | 5M | 5M |
| Curriculum | 17/17 transitions | None (all final) |
| Curve thresholds | 695/698/700/702/705 | N/A (fixed) |
| Final reward | +623 | **+770** |
| Peak reward | +724 (pre-crash) | **+775** |
| Issue | Triple-param crash | None |

### v1 Issue: Simultaneous Curve Transition (P-018)

Thresholds 700/702/705 were too tight. At 3.76M steps, `road_curvature=1.0`, `curve_direction_variation=1.0`, and `speed_zone_count=2` all unlocked within ~20K steps:

```
Step 3770K: reward 724 (normal)
Step 3790K: reward  96 (crash)
Step 3810K: reward -40 (bottom)
...1.2M steps of recovery...
Step 5000K: reward 623 (still climbing)
```

### v2 Fix: Pure Recovery Training

All parameters fixed at final values. No curriculum transitions needed since v1 already completed all 17/17. Agent just needed more time to stabilize curves + NPCs:

```
v2    0K: reward 623 (from v1 end)
v2  200K: reward 700 (rapid recovery)
v2 1000K: reward 730 (exceeded v1 peak)
v2 5000K: reward 770 (converged, project record)
```

---

## Training Progress (v2 - Final)

### Reward Curve

```
Reward
+775  |                                              * Peak (4.83M)
      |                                   __________/
+750  |                            ______/
      |                     ______/
+730  |               _____/
      |         _____/
+700  |   _____/
      |  /
+623  | * (v1 warm start)
      +----------------------------------------------------
       0    0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0 M steps
```

### Step-by-Step Progress (v2)

| Step | Reward | Std | Note |
|------|--------|-----|------|
| 10K | ~-200 | - | Initial buffer noise |
| 100K | ~650 | 30 | Warm start kicking in |
| 500K | ~715 | 12 | Rapid recovery |
| 1000K | ~730 | 13 | Exceeded v1 peak (724) |
| 2000K | ~755 | 10 | Steady climb |
| 3000K | ~765 | 9 | Approaching plateau |
| 4000K | ~768 | 8 | Near convergence |
| 4830K | **775** | 10 | **PEAK** |
| 5000K | **770** | 8 | **FINAL** |

---

## Key Achievements

### 1. Project-Wide Reward Record
```
Phase H v3: +701 (intersections + NPCs)
Phase I v2: +770 (curves + NPCs)  <- NEW RECORD (+69)
```

### 2. Full Curvature + NPC Mastery
Agent simultaneously handles:
- road_curvature = 1.0 (maximum)
- curve_direction_variation = 1.0 (random S-curves)
- 3 NPC vehicles at 0.85x speed
- NPC speed variation +/- 15%
- 2 speed zones
- 2 lanes with center line
- 230m goal distance

### 3. Recovery Training Validated
v1 crashed from 724 to -40 but recovered to 623 by 5M. v2 continued from 623 and reached 770 — proving that crashed policies can fully recover with fixed-param continuation training.

### 4. Efficient Pipeline
Two 25-min runs (50 min total) with build-based multi-env training. Rapid iteration from crash analysis to recovery.

---

## Checkpoints (v2)

| File | Step | Reward |
|------|------|--------|
| E2EDrivingAgent-3499xxx.pt | 3.5M | ~765 |
| E2EDrivingAgent-3999xxx.pt | 4.0M | ~768 |
| E2EDrivingAgent-4499xxx.pt | 4.5M | ~770 |
| E2EDrivingAgent-4999xxx.pt | 5.0M | ~770 |
| **E2EDrivingAgent-5000080.onnx** | **5.0M** | **~770 (FINAL)** |

---

## Lessons Learned

1. **P-018: Minimum threshold spacing >= 15 points**: Tighter spacing causes simultaneous transitions and catastrophic reward collapse
2. **Recovery training is viable**: A crashed policy (623) can reach new highs (770) with fixed-param continuation
3. **Curves + NPCs are compatible**: Despite being trained separately (Phase E curves, Phase H NPCs), skills combine successfully
4. **Build reuse works**: PhaseH build supports curvature with `intersection_type=0` — no new build needed
5. **Tight thresholds near ceiling are dangerous**: At reward ~700, thresholds 695-705 leave no margin for noise

---

## Next Phase

**Phase J**: TBD — potential candidates:
- Intersections + Curves combined (both active)
- Traffic signals + stop lines
- U-turns + special maneuvers

---

[Phase H](./phase-h) | [Home](../)
