---
layout: default
title: Phase H - NPC Intersection
---

# Phase H: NPC Interaction in Intersections

NPC waypoint-following through T/Cross/Y-junction intersections

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | phase-H-v3 (final) |
| **Status** | Completed |
| **Date** | 2026-02-01 |
| **Total Steps** | 5,000,501 (v3) + 5,000,000 (v2) |
| **Training Time** | ~26 min (v3), ~26 min (v2) |
| **Final Reward** | **+701** |
| **Peak Reward** | **+708** (at 1.55M steps, v3) |
| **Observation** | 260D (unchanged from Phase G) |
| **Initialize From** | Phase H v2 3.5M checkpoint (via Phase G v2) |
| **Training Mode** | Build + 3 parallel envs (no_graphics) |

---

## Objective

Add NPC vehicle interaction to intersection scenarios. Agent maintains all intersection geometry mastery from Phase G v2 while learning to navigate with 1-3 NPC vehicles following waypoints through intersections.

### New Capabilities
- NPC avoidance in intersection turns
- 1/2/3 NPC density handling
- Variable NPC speed ratios (0.5 - 0.85x)
- NPC speed variation tolerance (+/- 15%)
- Build-based multi-environment training (3x speedup)

---

## Observation Space

**260D (unchanged from Phase G)**

NPC interaction uses the existing 160D surrounding vehicle observation (OverlapSphere-based), which was already part of the observation space since Phase D.

---

## Key Innovation: Waypoint-Following NPCs

Phase G NPCs moved in straight lines (broke at intersection turns). Phase H NPCs follow the same waypoint path as the ego agent, enabling realistic intersection interaction.

```
Phase G: NPC drives straight -> off road at turns
Phase H: NPC follows waypoints -> correct turns at intersections
```

### Build-Based Training

First phase using headless Unity builds for training:
- `env_path: Builds/PhaseH/PhaseH.exe` (118MB)
- `num_envs: 3` (3 parallel Unity processes)
- `no_graphics: true` (headless rendering)
- ~183K steps/min (~11M steps/hour) vs ~5M/hour in editor

---

## Curriculum Design

### Full Curriculum Parameters (11/11 Completed)

| Parameter | Final Lesson | Final Value | Status |
|-----------|--------------|-------------|--------|
| intersection_type | Y-Junction | 3 | Completed |
| turn_direction | RightTurn | 2 | Completed |
| num_lanes | TwoLanes | 2 | Completed |
| center_line_enabled | CenterLineEnforced | 1 | Completed |
| goal_distance | FullGoal | 230m | Completed |
| num_active_npcs | ThreeNPCs | 3 | Completed |
| npc_speed_ratio | NormalNPCs | 0.85 | Completed |
| npc_speed_variation | MildVariation | 0.15 | Completed |
| road_curvature | Straight | 0 | Locked |
| curve_direction_variation | SingleDirection | 0 | Locked |
| speed_zone_count | SingleZone | 1 | Locked |

---

## Version History: v1 -> v2 -> v3

| Aspect | v1 | v2 | v3 |
|--------|----|----|-----|
| Init from | Phase G v2 5M | Phase G v2 5M | Phase H v2 3.5M |
| Steps | 8M (crashed at 5M) | 5M | 5M |
| speed_variation | 0 -> 0.15 (abrupt) | 0 -> 0.05 -> 0.10 -> 0.15 | same |
| Variation thresholds | 700 (single) | 700 / 710 / 720 | **685 / 690 / 693** |
| min_lesson_length | 500 | 1000 | **1500** (variation) |
| Curriculum complete | 7/11 (crash) | 9/11 (stuck) | **11/11** |
| Final reward | ~550 (collapsed) | ~681 | **+701** |
| threaded | true | **false** | false |
| Training mode | Editor | Build x3 | Build x3 |

### v1 Failure: Abrupt Speed Variation
- Single jump from variation=0 to variation=0.15 at threshold 700
- Reward crashed 700 -> 550, Std spiked to 300+
- Agent couldn't adapt to sudden NPC speed unpredictability

### v2 Issue: Thresholds Too High
- Gradual variation (0 -> 0.05 -> 0.10 -> 0.15) fixed the crash
- But thresholds 710/720 unreachable: agent averages ~690-700 with variation active
- speed_variation stuck at lesson 2/4 (0.05)

### v3 Fix: Lowered Thresholds
- Thresholds: 685/690/693 (achievable even with variation noise)
- min_lesson_length: 1500 for variation stability
- Warm start from v2 3.5M (peak ~700, Std ~5, pre-variation)
- Result: all 4 variation lessons completed

---

## Training Progress (v3 - Final)

### Reward Curve

```
Reward
+708  |      * Peak (3 NPCs, pre-variation)
      |     /
+700  |____/                           ___________*__ 701
      |                          _____/
+680  |                    _____/
      |                   / (variation dips)
+660  |                  /
      |           ______/
+640  |     _____/  * 641 (variation=0.15 dip)
      |____/
+620  |
      +------------------------------------------------
       0.5   1.0   1.5   2.0   2.5   3.0   3.5   4.0   4.5   5.0 M steps
```

### Step-by-Step Progress

| Step | Reward | Event |
|------|--------|-------|
| 10K | ~630 | Warm start from v2 3.5M checkpoint |
| 150K | ~680 | Phase G params unlocking (50-130 thresholds) |
| 340K | ~695 | All Phase G params complete (Y-junction) |
| 560K | ~700 | npc_speed_ratio -> 0.85 |
| 800K | ~680 | 1 NPC introduced (threshold 550) |
| 1030K | ~690 | 2 NPCs (threshold 620) |
| 1260K | ~700 | 3 NPCs (threshold 680) |
| 1550K | **708** | **PEAK** - speed_variation -> 0.05 |
| 1980K | ~690 | speed_variation -> 0.10 |
| 3780K | ~670 | speed_variation -> 0.15 (dip to 641) |
| 4200K | ~690 | Recovery from variation dip |
| 5000K | **701** | Training complete, all 11/11 curriculum |

---

## Curriculum Transitions (v3)

| Step | Parameter | From -> To | Threshold |
|------|-----------|-----------|-----------|
| ~150K | intersection_type | None -> T -> Cross -> Y | 50/75/100 |
| ~150K | turn_direction | Straight -> Left -> Right | 60/85 |
| ~150K | num_lanes | Single -> Two | 70 |
| ~150K | center_line_enabled | Off -> On | 90 |
| ~340K | goal_distance | 150m -> 200m -> 230m | 110/130 |
| ~560K | npc_speed_ratio | 0.5 -> 0.7 -> 0.85 | 600/660 |
| ~800K | num_active_npcs | 0 -> 1 | 550 |
| ~1030K | num_active_npcs | 1 -> 2 | 620 |
| ~1260K | num_active_npcs | 2 -> 3 | 680 |
| ~1550K | npc_speed_variation | 0 -> 0.05 | 685 |
| ~1980K | npc_speed_variation | 0.05 -> 0.10 | 690 |
| ~3780K | npc_speed_variation | 0.10 -> 0.15 | 693 |

---

## Key Achievements

### 1. Full NPC Interaction in Intersections
```
Phase G: 0 NPCs, intersections only
Phase H: 3 NPCs + speed variation, through intersections
```

### 2. Speed Variation Mastered (3 attempts)
- v1: Abrupt jump crashed training
- v2: Gradual but thresholds too high (stuck at 0.05)
- v3: Lowered thresholds (685/690/693) -> full completion to 0.15

### 3. Build Training Pipeline
- First phase using headless builds (3x parallel envs)
- ~26 min for 5M steps (vs ~56 min in editor for Phase G v2)
- Enabled rapid v2->v3 iteration

### 4. CUDA Threading Fix
- `threaded: true` caused `RuntimeError: Expected all tensors on same device`
- Fix: `threaded: false` (multi-env provides the main speedup)

### 5. Reward Comparison
```
Phase G v2: +628 (0 NPC, intersections)
Phase H v3: +701 (3 NPC, speed variation 0.15, intersections)
```

Phase H reward is HIGHER than Phase G despite 3x more NPCs and speed variation. This is because longer goal_distance (230m vs 200m) provides more reward opportunity.

---

## Checkpoints (v3)

| File | Step | Reward |
|------|------|--------|
| E2EDrivingAgent-3499821.pt | 3.5M | ~695 |
| E2EDrivingAgent-3999810.pt | 4.0M | ~690 |
| E2EDrivingAgent-4499997.pt | 4.5M | ~697 |
| E2EDrivingAgent-4999989.pt | 5.0M | ~701 |
| **E2EDrivingAgent-5000501.onnx** | **5.0M** | **~701 (FINAL)** |

---

## Bugs Fixed

1. **CUDA Device Mismatch (v2)**: `threaded: true` creates separate training thread that loses CUDA context with warm start models. Fixed by setting `threaded: false`.
2. **TensorBoard Stale Tracking**: `--force` re-run deleted old tfevents file, TensorBoard cached stale reference. Fixed by restarting TensorBoard.
3. **Deleted Checkpoint (v2 first attempt)**: Phase H v1 checkpoint rotated out by `keep_checkpoints=5`. Fixed by using Phase G v2 5M checkpoint instead.

---

## Lessons Learned

1. **Curriculum thresholds must be achievable under target conditions**: v2 thresholds (710/720) were set based on pre-variation performance, but variation itself lowers average reward by ~10-20 points
2. **Warm start checkpoint selection matters**: v3 used v2's 3.5M (peak stable, pre-variation) over v2's 5M (noisy with stuck variation)
3. **Build training enables rapid iteration**: 26 min per 5M run allowed v2->v3 turnaround within same session
4. **threaded=false with multi-env is sufficient**: num_envs=3 provides the main speedup; threaded adds marginal benefit but risks CUDA errors
5. **min_lesson_length should scale with noise**: Higher variation = more reward noise = need longer lesson for stable threshold crossing

---

## Next Phase

**[Phase I: Curved Roads + NPC](./phase-i)** - COMPLETED (+770, project record)
- Combined Phase E curve skills + Phase H NPC traffic skills
- Progressive road curvature with 3 NPCs at varying speeds
- v1: Triple-param crash (724->-40), recovered to 623
- v2: Pure recovery training, reached 770 (new project record)

---

[Phase G](./phase-g) | [Phase I](./phase-i) | [Home](../)
