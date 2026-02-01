# Phase G v2 Strategy

**Date**: 2026-02-01
**Based on**: Phase G v1 Analysis (ANALYSIS-v1.md)

---

## Problem Statement

Phase G v1 plateaued at reward ~494 (target: 550 for Y-junction). Root cause: **32% WrongWay termination rate** caused by `IsWrongWayDriving()` triggering during/after intersection turns.

The WrongWay check (`xPos < -0.5`) was designed for straight roads (Phase F). On left turns, the agent exits at negative X positions (X=-8.25 and decreasing), which always triggers WrongWay termination. The agent physically cannot complete left turns without being terminated.

---

## Changes Required

### Change 1: Fix WrongWay Check for Intersections (CODE)

**File**: `Assets/Scripts/Agents/WaypointManager.cs`, line 335

**Current**: `IsWrongWayDriving(float xPos)` returns true if `xPos < -0.5`

**Fix**: Add intersection zone awareness. Disable WrongWay check when:
- Agent Z position is within or beyond intersection zone (Z >= intersectionZoneStart)
- OR intersection is active (`intersectionType > 0`)

**Implementation**:
```csharp
public bool IsWrongWayDriving(float xPos, float zPos)
{
    if (!centerLineEnabled || numLanes <= 1)
        return false;

    // Disable WrongWay check in/after intersection zone
    if (intersectionType > 0 && zPos >= intersectionDistance - intersectionWidth)
        return false;

    float tolerance = 0.5f;
    return xPos < -tolerance;
}
```

Must also update callers in `E2EDrivingAgent.cs` and `E2EDrivingAgentBv2.cs` to pass zPos.

### Change 2: Simplify Curriculum (CONFIG)

Remove NPC-related parameters from Phase G. Intersection mastery should be learned without NPC interaction complexity. Defer NPCs to Phase H.

**v1 curriculum** (9 params):
- intersection_type, turn_direction, num_lanes, center_line_enabled
- goal_distance, num_active_npcs, npc_speed_ratio, npc_speed_variation, road_curvature

**v2 curriculum** (5 params):
- intersection_type, turn_direction, num_lanes, center_line_enabled, goal_distance

Remove: num_active_npcs (set to 0), npc_speed_ratio, npc_speed_variation

### Change 3: Lower Curriculum Thresholds (CONFIG)

v1 thresholds were too high (550 for Y-junction). With WrongWay fix, reward should be higher, but lower thresholds provide safety margin.

| Parameter | v1 Thresholds | v2 Thresholds |
|-----------|--------------|--------------|
| intersection_type | 150, 350, 550 | 150, 300, 450 |
| turn_direction | 200, 400 | 200, 350 |
| num_lanes | 250 | 250 |
| center_line_enabled | 300 | 300 |
| goal_distance | 450, 600 | 400, 500 |

### Change 4: Warm Start from v1 Checkpoint (CONFIG)

Use Phase G v1 final checkpoint as init_path. This is safe because:
- Same 260D observation space (no dimension change)
- v1 already learned basic intersection navigation
- Only WrongWay fix changes runtime behavior, not observation/action space

`init_path: results/phase-G/E2EDrivingAgent/E2EDrivingAgent-10000153.pt`

### Change 5: Increase Goal Distance Initial Value (CONFIG)

v1 started at 120m (only 20m past intersection center at Z=100). This was barely enough for turns. v2 starts at 150m.

### Change 6: Reduce max_steps (CONFIG)

With warm start and WrongWay fix, 5M steps should suffice. If plateau persists, can extend.

---

## Expected Outcome

With WrongWay fix eliminating the 32% termination rate:
- Goal completion should jump from 68% to 85%+
- Reward should increase by ~100-150 (from lost WrongWay episodes becoming goal completions)
- Y-junction curriculum (threshold 450) should be reachable
- Full curriculum completion within 5M steps

---

## Config Summary

| Setting | v1 | v2 | Rationale |
|---------|----|----|-----------|
| init_path | none (fresh) | v1 checkpoint | Preserve learned intersection skills |
| max_steps | 10M | 5M | Warm start needs less budget |
| intersection_type thresholds | 150/350/550 | 150/300/450 | Lower bar for Y-junction |
| num_active_npcs | curriculum 0->2 | fixed 0 | Remove NPC complexity |
| goal_distance initial | 120m | 150m | More room for turns |
| WrongWay check | X-only | X + Z zone | Fix termination bug |

---

## Risk Assessment

| Risk | Probability | Mitigation |
|------|------------|-----------|
| v1 checkpoint incompatible | Low | Same obs/action space |
| WrongWay fix too permissive | Medium | Only disable in intersection zone |
| Reward still plateaus | Medium | Lower thresholds provide buffer |
| Y-junction geometry issues | Medium | Monitor EndReason_WrongWay specifically |
