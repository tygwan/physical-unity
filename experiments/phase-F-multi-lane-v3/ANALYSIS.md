# Phase F v3: Training Analysis

## Run Summary
- **Run ID**: `phase-F-v3`
- **Steps**: 6,000,021 / 6,000,000 (100%)
- **Final Reward**: +4.715
- **Peak Reward**: ~+317 (estimated, pre-crash)
- **Initialized From**: Phase E checkpoint (+892.6)
- **Verdict**: **FAILURE** - Simultaneous curriculum transition crash, inadequate recovery

## Curriculum Final State

| Parameter | Final Lesson | Completed? |
|-----------|-------------|------------|
| num_lanes | FourLanes (4) | 3/3 transitions |
| center_line_enabled | CenterLineEnforced (1) | 1/1 |
| road_curvature | Straight (0) | 0/2 (never started) |
| curve_direction_variation | SingleDirection (0) | 0/1 (never started) |
| num_active_npcs | NoNPCs (0) | 0/3 (never started) |
| npc_speed_ratio | SlowNPCs (0.5) | 0/1 (never started) |
| goal_distance | MediumGoal (200m) | 1/2 |
| speed_zone_count | TwoZones (2) | 1/1 |
| npc_speed_variation | Uniform (0) | 0/1 (never started) |

**6 of 16 total lesson transitions completed.** All secondary/tertiary parameters (curvature, NPCs) never advanced.

## Training Trajectory

### Phase 1: Learning (0-3.5M)
- Agent learns multi-lane driving from Phase E checkpoint
- Lane transitions (1->2->3->4) proceed smoothly thanks to WaypointManager fix
- Validates that the v2 root cause (waypoint destruction) was correctly identified and fixed
- Reward climbs steadily toward +317

### Phase 2: Simultaneous Transition Crash (~4.38M)
When smoothed reward reached threshold 350, **three parameters transitioned simultaneously**:
- `goal_distance`: 150m -> 200m
- `speed_zone_count`: 1 -> 2 zones
- `road_curvature`: attempted 0 -> 0.3 (but its threshold was 400, not 350)

Actually, the 4 parameters sharing threshold 350:
- goal_distance L0 (150->200m)
- speed_zone_count L0 (1->2 zones)
- num_active_npcs L2 (their L2 threshold was 350)
- road_curvature L1 (their L1 threshold was 350)

Multiple parameters transitioning at once overwhelmed the agent's ability to adapt.

**Reward crash**: ~+317 -> -2480 (estimated -2797 drop)

### Phase 3: Partial Recovery (4.38M-6M)
| Checkpoint | Steps | Reward |
|------------|-------|--------|
| 4.5M | 4,499,823 | -2,043.7 |
| 5.0M | 4,999,816 | -1,735.2 |
| 5.5M | 5,499,992 | -305.3 |
| 6.0M | 5,999,995 | +4.7 |
| Final | 6,000,021 | +4.7 |

Recovery took 1.62M steps (4.38M -> 6.0M) to reach merely +4.7 reward.

## Why Recovery Was Incomplete

### 1. Linear LR Decay
```
learning_rate_schedule: linear
learning_rate: 1.5e-4
max_steps: 6,000,000
```

At step 4.38M (crash point): lr = 1.5e-4 * (1 - 4.38M/6M) = 1.5e-4 * 0.27 = **4.05e-5**
At step 5.5M: lr = 1.5e-4 * (1 - 5.5M/6M) = 1.5e-4 * 0.083 = **1.25e-5**
At step 6.0M: lr = **~0** (effectively frozen)

The agent was learning at <5% of initial learning rate during the critical recovery period. By 5.5M steps, lr was too low for meaningful policy updates.

### 2. Speed Zone Bug (Not Discovered Until v4)
The speed_zone 1->2 transition placed Residential (30 km/h) as the first zone while the agent was driving at 60 km/h. This alone caused a -3.0/step penalty (max overspeed cap). This was masked by the simultaneous transitions and not identified until v4 isolated the speed_zone transition.

### 3. Too Many Changes at Once
Three curriculum parameters changing simultaneously meant:
- Goal distance increased (longer episodes)
- Speed zones introduced (different speed limits per section)
- Environment fundamentally different from what agent learned

## Comparison: v2 vs v3

| Aspect | v2 | v3 |
|--------|-----|-----|
| Scene | PhaseF_MultiLane | PhaseF_MultiLane |
| WaypointManager fix | No | Yes |
| Lane transitions | FATAL (entropy collapse) | SMOOTH |
| Multi-param crash | N/A (failed at first transition) | Yes (threshold 350) |
| Recovery possible? | No (StdDev=0.08) | Partial (StdDev maintained) |
| Final reward | -14.2 (locked) | +4.7 (recovering) |
| Root cause | Waypoint destruction | Shared thresholds + speed zone bug |

Key insight: v3 validated the WaypointManager fix. Lane transitions no longer cause entropy collapse. The remaining failure was due to curriculum design (shared thresholds) and a hidden speed zone implementation bug.

## Policy Discoveries

### P-002 Reinforced: Staggered Curriculum (Strict)
The threshold 350 collision (4 params sharing it) proved that "spread" thresholds (v2's 250-500 range) are insufficient. **Every threshold must be unique across ALL parameters**. Applied strictly in v4.

### P-012: No Shared Thresholds
All curriculum parameters must use distinct threshold values. When N parameters share threshold T, they all transition when smoothed reward crosses T - regardless of whether the agent can handle N simultaneous changes.

## Checkpoint Inventory

| Steps | Reward | File |
|-------|--------|------|
| 4,499,823 | -2,043.7 | `E2EDrivingAgent-4499823.pt` |
| 4,999,816 | -1,735.2 | `E2EDrivingAgent-4999816.pt` |
| 5,499,992 | -305.3 | `E2EDrivingAgent-5499992.pt` |
| 5,999,995 | +4.7 | `E2EDrivingAgent-5999995.pt` |
| 6,000,021 | +4.7 | `E2EDrivingAgent-6000021.pt` (final) |

Note: Earlier checkpoints (before 4.5M) were overwritten due to `keep_checkpoints: 5`.

## Files Referenced

| File | Relevance |
|------|-----------|
| `python/configs/planning/vehicle_ppo_phase-F.yaml` | v3 base config |
| `results/phase-F-v3/configuration.yaml` | Actual runtime config (captured by ML-Agents) |
| `results/phase-F-v3/run_logs/training_status.json` | Checkpoint and curriculum state |
| `Assets/Scripts/Agents/WaypointManager.cs` | Fixed waypoint destruction |
