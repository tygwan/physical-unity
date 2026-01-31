# Phase F v4: Training Analysis

## Run Summary
- **Run ID**: `phase-F-v4`
- **Steps**: 10,000,000 / 10,000,000 (100%)
- **Duration**: 7,626 seconds (2h 7m)
- **Final Reward**: +105.5
- **Peak Reward**: +483 (step 3.65M, pre-crash)
- **Initialized From**: Phase E checkpoint (+892.6)

## Curriculum Transitions (6 of 15 completed)

| Step | Transition | Reward Before | Reward After | Drop | Recovery |
|------|-----------|---------------|-------------|------|----------|
| 2.68M | num_lanes 1→2 | +163 | +166 | none | instant |
| 2.94M | num_lanes 2→3 | +271 | +283 | none | instant |
| 3.16M | center_line off→on | +365 | +188 | -177 | ~200K steps |
| 3.38M | num_lanes 3→4 | +429 | +255 | -174 | ~270K steps |
| 3.65M | goal_distance 150→200m | +483 | +483 | none | instant |
| 3.78M | **speed_zone 1→2** | **+481** | **-2309** | **-2790** | **6.22M (incomplete)** |

## Staggered Curriculum: SUCCESS

v4's strict P-002 compliance (every threshold unique) worked perfectly:
- All 6 transitions occurred individually (no simultaneous transitions)
- Lane transitions (1→2, 2→3, 3→4) were handled smoothly
- goal_distance transition had zero impact

This confirms the v3 diagnosis was partially correct: simultaneous transitions are bad. But v4 reveals a deeper issue.

## Root Cause: speed_zone Implementation Bug

### The Crash Mechanism

When `speed_zone_count` transitions from 1 to 2, `GenerateSpeedZones()` assigns:
- Zone 0 (road first half): **Residential = 30 km/h** (8.33 m/s)
- Zone 1 (road second half): **UrbanNarrow = 50 km/h** (13.89 m/s)

The agent was trained at **60 km/h** (single zone default). Suddenly the speed limit halves.

### Reward Impact Calculation

Agent driving at 60 km/h in a 30 km/h zone:
```
speedRatio = 60 / 30 = 2.0 (100% over limit)
overRatio = 2.0 - 1.0 = 1.0
penalty = -0.5 * min(1.0 * 10, 6) = -3.0 per step (max penalty cap)
```

Every decision step incurs **-3.0 penalty** until the agent learns to slow to 30 km/h. This overwhelms all positive rewards.

### Why v3 Had The Same Crash

In v3, the speed_zone transition happened simultaneously with num_lanes and goal_distance. We attributed the crash to simultaneous transitions. In reality, speed_zone alone was sufficient to cause -2790 drop.

## Recovery Analysis

| Phase | Step Range | Reward | Duration |
|-------|-----------|--------|----------|
| Crash | 3.78M | +481 → -2309 | instant |
| Deep negative | 3.78M-4.25M | -2309 → -1930 | 470K steps |
| Fast recovery | 4.25M-4.69M | -1930 → +0.5 | 440K steps |
| Plateau | 4.69M-5.5M | +0.5 → +10 | 810K steps |
| Slow climb | 5.5M-10.0M | +10 → +105 | 4.5M steps |

Total recovery: 6.22M steps, only reached +105 (22% of pre-crash peak +483).

The slow recovery suggests the 30 km/h zone fundamentally limits the agent's achievable reward - driving slowly yields less progress reward and extends episode time.

## Curriculum Final State

| Parameter | Final Lesson | Next Threshold | Gap |
|-----------|-------------|----------------|-----|
| num_lanes | FourLanes (4) | - (complete) | - |
| center_line | Enforced | - (complete) | - |
| goal_distance | MediumGoal (200m) | 500 | -395 |
| speed_zone_count | TwoZones (2) | - (complete) | - |
| road_curvature | Straight (0) | 550 | -445 |
| curve_direction | SingleDir (0) | 650 | -545 |
| num_active_npcs | NoNPCs (0) | 700 | -595 |
| npc_speed_ratio | SlowNPCs (0.5) | 850 | -745 |
| npc_speed_variation | Uniform (0) | 900 | -795 |

9 of 15 transitions remain blocked. Agent reward (+105) far below next threshold (500).

## Policy Discovery

### P-013: Speed Zone Curriculum Ordering
**When introducing multi-zone speed limits via curriculum, the first zone must match the previous single-zone speed limit.** Placing the slowest zone first (Residential 30 km/h) when the agent learned at 60 km/h causes catastrophic overspeed penalties. Zone ordering should descend from familiar to unfamiliar speeds.

## Verdict

**PARTIAL SUCCESS** - Curriculum staggering validated, but speed_zone implementation blocks further progress. Requires code fix to `GenerateSpeedZones()` before v5.
