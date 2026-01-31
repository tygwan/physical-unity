# Phase F v2: Training Analysis

## Run Summary
- **Run ID**: `phase-F-v2`
- **Steps**: ~4,100,000 / 6,000,000 (68%, manually stopped)
- **Final Reward**: -14.2 (locked)
- **Peak Reward**: +317 (step 2.98M)
- **Initialized From**: Phase E checkpoint (+892.6)
- **Verdict**: **FAILURE** - Unrecoverable policy entropy collapse

## Training Trajectory

| Phase | Step Range | Reward | Description |
|-------|-----------|--------|-------------|
| Learning | 0-2.98M | -700 -> +317 | Steady improvement on single-lane |
| Transition | 2.99M | +317 -> -14.2 | num_lanes 1->2 curriculum transition |
| Stuck | 2.99M-4.1M | -14.2 (locked) | Policy entropy collapse, zero exploration |

## Root Cause: Waypoint Destruction at Curriculum Transition

### Call Chain
```
E2EDrivingAgent.OnEpisodeBegin()
  -> sceneManager.ResetEpisode()
    -> DrivingSceneManager.ApplyCurriculumParameters()
      -> waypointManager.SetLaneCount(numLanes)
        -> WaypointManager.GenerateWaypoints()
          -> Destroy(child.gameObject) for ALL waypoints
```

When `num_lanes` transitions from 1 to 2 at step 2.99M:
1. All existing waypoints are destroyed via `Destroy(child.gameObject)`
2. New 2-lane waypoints are regenerated from scratch
3. Agent's route observations (30D) reference entirely new Transform objects
4. Previously learned single-lane policy becomes misaligned with new geometry
5. Agent tries alternatives, all perform worse than [0,0] (stationary)
6. PPO converges to [0,0] action with 99%+ probability
7. Policy entropy collapses: StdDev = 0.08 (near minimum)
8. Zero exploration -> cannot discover better actions -> permanently stuck

### Mathematical Verification

When agent speed = 0 m/s:
```
speedRatio = 0 / speedLimit = 0.0
progressivePenalty = speedUnderPenalty * (2.0 - 0.0 * 2.0)
                   = -0.1 * 2.0 = -0.2 per step

For 90-step episode: -0.2 * 90 = -18.0
Plus time penalty:   -0.001 * 90 = -0.09
Plus bonuses:        +3 to +4
Expected total:      -14 to -15
Observed total:      -14.2  [MATCHES within 0.1% tolerance]
```

### TensorBoard Evidence

| Metric | Before (2.98M) | After (2.99M+) |
|--------|----------------|-----------------|
| Reward | +317 | -14.2 |
| Speed | Normal | 0.0 m/s |
| Episode Length | ~500 steps | 70-140 steps |
| StdDev | Normal | 0.08 (entropy collapse) |

## Why This Failure Was Hard to Detect

Unlike v1 (off-road collision, obvious), v2's failure was **silent**:
- No collision markers in logs
- Agent stays on road (11.5m wide, plenty of room)
- Reward is negative but not catastrophically low (-14.2 vs v1's immediate -5 collisions)
- Required 1.1M steps to confirm the stuck state was permanent
- Root cause required code inspection of WaypointManager internals

## Recommendations (Applied in v3)

1. **Pre-generate all waypoints** - Switch lanes without destruction (implemented as position-reuse in v3)
2. **Verify Phase E checkpoint** compatibility before training
3. **Test curriculum transitions** with short preflight runs
4. **Monitor entropy/StdDev** as early warning of policy collapse

## Policy Discovery

### P-009: Waypoint Persistence
**Curriculum transitions that modify scene geometry (waypoints, road structure) must NOT destroy existing navigation references.** Pre-generate or reuse positions to maintain observation continuity.

## Files Referenced

| File | Relevance |
|------|-----------|
| `Assets/Scripts/Agents/WaypointManager.cs:196-202` | GenerateWaypoints() - destroys all children |
| `Assets/Scripts/Agents/WaypointManager.cs:339-344` | SetLaneCount() - triggers destruction |
| `Assets/Scripts/Environment/DrivingSceneManager.cs:176-182` | ApplyCurriculumParameters() - calls SetLaneCount |
| `Assets/Scripts/Agents/E2EDrivingAgent.cs:1054-1062` | CalculateSpeedPolicyReward() - zero-speed penalty math |
| `experiments/phase-F-multi-lane/ROOT-CAUSE-ANALYSIS.txt` | Full 213-line investigation |
