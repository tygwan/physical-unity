# Phase I v1: Curved Roads + NPC Traffic

## Experiment ID
- **Run ID**: `phase-I`
- **Config**: `python/configs/planning/vehicle_ppo_phase-I.yaml`
- **Build**: `Builds/PhaseH/PhaseH.exe` (reused)
- **Date**: 2026-02-01

## Motivation

Phase H v3 completed (+701 reward) mastering 3 NPCs with speed variation at intersections. Phase I combines Phase E's curve driving with Phase H's NPC traffic. The agent learns to drive curved roads with 3 NPC vehicles at varying speeds.

Key insight: In `WaypointManager.GenerateWaypoints()`, intersections and curves are mutually exclusive:
- `intersection_type > 0` -> intersection paths (curvature ignored)
- `intersection_type == 0 && road_curvature > 0.01` -> curved paths

Phase I locks `intersection_type=0` and unlocks `road_curvature`.

## Strategy

**WARM START** from Phase H v3 final checkpoint (reward ~701, 260D obs, 5M steps).

### Three-Tier Curriculum
1. **Tier 1** Quick-unlock (thresholds 70-130): num_lanes, center_line, goal_distance
2. **Tier 2** NPC re-unlock (thresholds 550-693): npcs, speed_ratio, speed_variation
3. **Tier 3** NEW curves (thresholds 695-705): curvature, direction, speed_zones

## Results

### Outcome: Partial Success -> v2 Needed

- **Steps completed**: 5,000,000 / 5,000,000
- **Final reward**: **623** (Std 13.7)
- **Peak reward**: **724.7** (at 3,770K, pre-curve-crash)
- **Curriculum**: **17/17 complete** (all transitions done)
- **Training time**: ~25 min (3 envs, build mode)

### Critical Event: Triple-Param Crash at 3.76M

```
Step 3760K: road_curvature=1.0 + curve_direction=1.0 + speed_zones=2 (simultaneous)
Step 3770K: reward 724 (normal)
Step 3790K: reward  96  (collapse begins)
Step 3800K: reward -13  (bottom)
Step 3810K: reward -40  (absolute minimum)
...recovery...
Step 4060K: reward 248
Step 5000K: reward 623  (still climbing)
```

Thresholds 700/702/705 were too tight -> 3 params unlocked simultaneously -> **760-point crash**.

### Curriculum Completion

| Tier | Parameters | Status | Steps |
|------|-----------|--------|-------|
| Tier 1 (70-130) | lanes, center_line, goal_distance | 4/4 Complete | ~300K |
| Tier 2 (550-693) | npcs, speed_ratio, speed_variation | 8/8 Complete | ~3.5M |
| Tier 3 (695-705) | curvature, direction, speed_zones | 5/5 Complete | ~3.76M |

### Key Lesson (P-018)
**Threshold spacing must be >= 15 points for independent transitions.** 700/702/705 spacing caused simultaneous unlock of 3 difficult params. Future configs should use wider gaps (e.g., 680/695/715).

## Handoff to v2
- v2 warm starts from v1 final checkpoint (reward 623, all curriculum complete)
- v2 uses fixed final values (no curriculum transitions)
- v2 is pure recovery training
