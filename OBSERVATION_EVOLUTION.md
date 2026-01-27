# E2EDrivingAgent Observation Size Evolution

## Summary Table

| Phase | Size | Key Features | Status | Commit |
|-------|------|--------------|--------|--------|
| A | 242D | Ego(8) + History(40) + Agents(160) + Route(30) + Speed(4) | ✅ | 04428a3 |
| B | 242D | Same as A, different curriculum | ✅ | 04428a3 |
| C | 242D | Multi-NPC generalization (1→4) | ✅ | 04428a3 |
| D | 254D | +12D Lane info | ⏸️ | 04428a3 |
| E | 254D | Lane + curved roads | ✅ | 04428a3 |
| F | 254D | Lane + multi-lane roads | ✅ | 04428a3 |
| G | 260D | +6D Intersection info | 🔄 | 70a536d |

## Phase A-C: Base Observation (242D)

### Composition (8 + 40 + 160 + 30 + 4 = 242D)

1. **Ego State (8D)**: x, y, vx, vy, cos_h, sin_h, ax, ay
2. **Ego History (40D)**: 5 past timesteps × 8D
3. **Surrounding Agents (160D)**: 20 agents × 8 features
4. **Route Information (30D)**: 10 waypoints × 3D
5. **Speed Information (4D)**: current_norm, limit_norm, ratio, next_limit

### Phase A: Dense Overtaking
- Steps: 2,000,209
- Final: +714 (Peak: +937)
- Env: 1 NPC @ 30%
- Status: ✅ Success

### Phase B: Decision Making
- Steps: 2,000,150
- Final: +903 (Peak: +994)
- Env: 1 NPC @ 30-90%
- Status: ✅ Success

### Phase C: Multi-NPC
- Steps: 4,000,000
- Final: +961 (Peak: +1086)
- Env: 1-4 NPCs
- Status: ✅ Success

## Phase D-F: Lane Observation (254D)

### Expansion: +12D Lane Info

254D = 242D + 12D

Lane components:
- left_lane_type (4D): one-hot
- right_lane_type (4D): one-hot
- left_lane_distance (1D): normalized
- right_lane_distance (1D): normalized
- left_lane_crossing (1D): binary
- right_lane_crossing (1D): binary

### Phase E: Curved Roads
- Steps: 6,000,090
- Final: +931 (Peak: +931)
- Env: Curves 0→1.0
- Init: Phase D
- Status: ✅ Success (+180% improvement)

### Phase F: Multi-Lane Roads
- Steps: 6,000,000
- Final: +988 (Peak: +988)
- Env: 2 lanes, center line, curves
- Init: Phase E
- Status: ✅ Success (BEST in v12\!)

## Phase G: Intersection Navigation (260D)

### Expansion: +6D Intersection Info

260D = 254D + 6D

Intersection components:
- intersection_type_none (1D)
- intersection_type_t (1D)
- intersection_type_cross (1D)
- intersection_type_y (1D)
- distance_to_intersection (1D)
- turn_direction (1D)

### Phase G Progress
- Current: 3,560,000 / 8,000,000 (44.5%)
- Reward: +792 (Peak: +882)
- Status: 🔄 In Progress
- Curriculum: NoIntersection ✓, T-Junction ✓, Cross ✓, Y-Junction (TBD)

## Performance Progression

Phase A:  +937 (1 NPC @ 30%, 242D)
Phase B:  +903 (1 NPC @ 30-90%, 242D)
Phase C:  +961 (4 NPC, 242D)
Phase E:  +931 (2 NPC, curves, 254D)
Phase F:  +988 (3 NPC, curves+lane, 254D) ← BEST
Phase G:  +792 (intersections, 260D, in-progress)

## Expansion Rationale

### 242D → 254D (+12D Lane)
- Needed: Multi-lane selection, lane-keeping, lateral awareness
- Benefits: Enables curve navigation, center line compliance
- Result: Phase E (+931), Phase F (+988)

### 254D → 260D (+6D Intersection)
- Needed: Intersection type, turn planning, distance awareness
- Benefits: Enables intersection navigation
- Result: Phase G learning in progress

## Key Findings

### Git History
- Commit 04428a3: Phase A-F (242D/254D)
- Commit 70a536d: Phase G (260D)

### Code References
- E2EDrivingAgent.cs: Main agent with 254D comments
- ObservationSizeUpdater.cs: Batch update tool (242D/254D configs)

### Lessons Learned
1. Incremental expansion effective (+5%, +2.3%)
2. Backwards compatible: append new features
3. Phase checkpoints preserve 60-80% reward
4. Don't transfer encoder across expansions (HybridPolicy failed)

## Summary

Evolution: 242D (A-C) → 254D (D-F) → 260D (G)
Expansion: 7.4% total increase over time
Status: Phase G at 44.5% completion

