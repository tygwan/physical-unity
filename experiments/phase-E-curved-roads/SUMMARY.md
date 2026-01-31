# Phase E Training Summary

**Date**: 2026-01-30
**Status**: COMPLETED
**Confidence**: High (all curriculum stages completed, stable final reward)

---

## Results

| Metric | Value |
|--------|-------|
| Peak Reward | +938.2 @ 4.5M steps |
| Final Reward | +892.6 @ 6.0M steps |
| Curriculum Completion | 7/7 parameters at final lesson |
| Collapse Events | 1 (at 1.68M, recovered by 2.44M) |
| Best Checkpoint | E2EDrivingAgent-4499975.pt (+938.2) |

## Training Timeline

```
Step 0        : Training start (init from Phase D v3, +895)
Step 0-1.68M  : Straight road adaptation, NPC curriculum advancing
                Reward: 0 -> +362 (rising)

Step 1.68M    : CURRICULUM COLLAPSE (P-002 violation)
                4 params transitioned simultaneously
                Reward: +362 -> -3,863

Step 1.68-2.44M: Recovery period (~800K steps)
                Agent relearns NPC interaction on straight roads
                Reward: -3,863 -> 0 (crossing zero)

Step 2.44-3.47M: Consolidation
                Straight road mastery restored
                Reward: 0 -> +800

Step 3.47M    : road_curvature -> GentleCurves (0.3)
Step 3.58M    : Peak reward +956
Step 3.81M    : road_curvature -> ModerateCurves (0.6)
Step 4.15M    : road_curvature -> SharpCurves (1.0)
                FULL CURVATURE MASTERY

Step 4.5M     : Best checkpoint (+938.2)
Step 4.5-6.0M : Stable performance (+880-938)
Step 6.0M     : Training complete (+892.6)
```

## Curriculum Final State (training_status.json)

| Parameter | Final Lesson | Meaning |
|-----------|-------------|---------|
| road_curvature | 3 | SharpCurves (1.0) |
| curve_direction_variation | 1 | MixedDirections |
| num_active_npcs | 2 | 2 NPCs |
| npc_speed_ratio | 1 | Medium (0.7) |
| goal_distance | 2 | 200m |
| speed_zone_count | 1 | 2 zones |
| npc_speed_variation | 1 | Varied (0.2) |

## Key Findings

### 1. Curriculum Collapse is Recoverable with Strong Base Policy

Phase D v1/v2 both suffered curriculum collapse and NEVER recovered (fresh starts).
Phase E also collapsed at 1.68M but recovered within 800K steps because:
- Phase D v3 checkpoint provided robust 254D observation processing
- Straight road driving was already deeply learned
- Only NPC parameters changed (not observation dimensions)

### 2. Curvature Transitions are Smooth

Unlike NPC-related transitions (risky), road curvature transitions were smooth:
- Straight -> GentleCurves: no reward drop
- GentleCurves -> ModerateCurves: minimal dip, fast recovery
- ModerateCurves -> SharpCurves: stable progression

This suggests geometric complexity is easier to adapt to than agent interaction complexity.

### 3. Sharp Curves at maxCurveAngle=45 deg

The agent mastered curves up to 45 degrees per segment with:
- Mixed left/right direction variation
- 2 NPCs at 70% speed limit
- 200m goal distance
- Variable speed zones

## Checkpoint Artifacts

```
results/phase-E/
  E2EDrivingAgent/
    E2EDrivingAgent-4499975.pt    # Best (peak +938.2)
    E2EDrivingAgent-4499975.onnx
    E2EDrivingAgent-5999820.pt    # Final
    E2EDrivingAgent-5999820.onnx
    E2EDrivingAgent-6000076.pt    # Final checkpoint
    E2EDrivingAgent-6000076.onnx
    checkpoint.pt                 # Latest state
    events.out.tfevents.*         # TensorBoard data
  run_logs/
    training_status.json          # Curriculum state
    timers.json                   # Performance data
  configuration.yaml              # Runtime config snapshot
  E2EDrivingAgent.onnx            # Exported final model
```

## Comparison with Previous Phases

| Phase | Peak Reward | Steps to Peak | Collapse? | Recovery? |
|-------|-------------|---------------|-----------|-----------|
| D v1 | +406 | 4.6M | YES (4.68M) | NO |
| D v2 | +447 | 5.8M | YES (7.87M) | NO |
| D v3 | +895 | 5.0M | No | N/A |
| **E** | **+938** | **4.5M** | **YES (1.68M)** | **YES (2.44M)** |

## Next Phase

Phase F: Multi-Lane Roads
- Init from: Phase E checkpoint (--initialize-from=phase-E)
- New features: num_lanes (1-4), center_line_enabled
- Observation: 254D (same as Phase E)
