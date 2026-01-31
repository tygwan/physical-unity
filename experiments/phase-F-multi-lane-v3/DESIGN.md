# Phase F v3: Multi-Lane Roads - WaypointManager Fix

## Experiment ID
- **Run ID**: `phase-F-v3`
- **Config**: `python/configs/planning/vehicle_ppo_phase-F.yaml`
- **Scene**: `PhaseF_MultiLane`
- **Date**: 2026-01-30

## Motivation

Phase F v2 failed due to `WaypointManager.GenerateWaypoints()` destroying all waypoints during `num_lanes` curriculum transition. v3 applies a code fix to maintain waypoint position continuity.

### Code Fix: WaypointManager.GenerateWaypoints()

**Before (v2):**
```csharp
foreach (Transform child in transform)
{
    Destroy(child.gameObject);  // ALL WAYPOINTS DESTROYED
}
waypoints.Clear();
```

**After (v3):**
```csharp
// Reuse existing waypoint positions instead of destroying
// Only add/remove waypoints as needed for new lane count
// Existing lane 0 waypoints maintain their Transform references
```

The fix ensures that when `SetLaneCount()` is called during curriculum transitions, existing waypoint Transforms are preserved and new lanes are added without disrupting the agent's route observations.

## Training Config

Same base config as v2 (`vehicle_ppo_phase-F.yaml`):

### Curriculum: 9 Parameters
| Parameter | Lessons | Thresholds |
|-----------|---------|------------|
| num_lanes | 1->2->3->4 | 300, 250, 200 |
| center_line_enabled | off->on | 250 |
| road_curvature | 0->0.3->0.6 | 400, 350 |
| curve_direction_variation | 0->1 | 450 |
| num_active_npcs | 0->1->2->3 | 500, 400, 350 |
| npc_speed_ratio | 0.5->0.7 | 500 |
| goal_distance | 150->200->250m | 350, 300 |
| speed_zone_count | 1->2 | 350 |
| npc_speed_variation | 0->0.2 | 450 |

### Shared Threshold Analysis (P-002 Violation)
| Threshold | Parameters |
|-----------|-----------|
| **350** | **goal_distance L0, speed_zone_count L0, road_curvature L1, num_active_npcs L2** (4 params!) |
| 250 | center_line L0, num_lanes L1 |
| 300 | num_lanes L0, goal_distance L1 |
| 400 | road_curvature L0, num_active_npcs L1 |
| 450 | curve_direction L0, npc_speed_variation L0 |
| 500 | num_active_npcs L0, npc_speed_ratio L0 |

Threshold 350 was shared by **4 parameters** - the most dangerous collision point.

### Hyperparameters
| Parameter | Value |
|-----------|-------|
| batch_size | 4096 |
| buffer_size | 40960 |
| learning_rate | 1.5e-4 |
| learning_rate_schedule | **linear** (decays to 0 at max_steps) |
| hidden_units | 512 |
| num_layers | 3 |
| max_steps | 6,000,000 |

### Initialization
- **Source**: Phase E checkpoint via `--initialize-from=phase-E`
- **Phase E final reward**: +892.6

## Expected Behavior

With the WaypointManager fix:
- `num_lanes` transitions should NOT cause entropy collapse
- Lane transitions should show mild reward dip with recovery
- Primary risk: shared thresholds causing simultaneous transitions

## Known Issues (Post-Mortem)
1. **Threshold 350 shared by 4 params** - caused simultaneous triple transition
2. **linear LR schedule** - lr decayed to ~0 by step 6M, extension produced no learning
3. **GenerateSpeedZones() ordering** - first zone is Residential (30 km/h), not discovered until v4
