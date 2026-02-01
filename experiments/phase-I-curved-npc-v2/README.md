# Phase I v2: Curved Roads + NPC Recovery

## Experiment ID
- **Run ID**: `phase-I-v2`
- **Config**: `python/configs/planning/vehicle_ppo_phase-I-v2.yaml`
- **Build**: `Builds/PhaseH/PhaseH.exe` (reused)
- **Date**: 2026-02-01

## Motivation

Phase I v1 completed all 17/17 curriculum transitions but suffered a ~760-point reward crash when road_curvature, curve_direction_variation, and speed_zone_count unlocked simultaneously at 3.76M steps (thresholds 700/702/705 too tight). v1 recovered from -40 to 623 by 5M and was still climbing. v2 continues with all params at final values for pure recovery training.

## Strategy

**WARM START** from Phase I v1 final checkpoint (reward ~623, all 17/17 curriculum complete).

### Key Changes from v1
1. **NO CURRICULUM**: All parameters fixed at final values (no transitions)
2. **PURE RECOVERY**: Agent already knows the task, just needs stability time
3. **WARM START**: v1 5M checkpoint (reward ~623, climbing at +40/100K steps)

## Results

### Outcome: SUCCESS - Project-Wide Record

- **Steps completed**: 5,000,000 / 5,000,000
- **Final reward**: **770** (Std 8.1)
- **Peak reward**: **774.8** (at 4,830K)
- **Training time**: ~25 min (3 envs, build mode)
- **Reward range**: 762-775 (last 1M steps)

### Recovery Trajectory

```
v1 end (5M):     623  (recovering from curve crash)
v2 200K:         ~700  (rapid recovery)
v2 1M:           ~730  (exceeded v1 pre-crash peak of 724)
v2 2M:           ~750  (steady climb)
v2 3M:           ~765  (approaching plateau)
v2 5M:           ~770  (converged, project record)
```

### Final Agent Capabilities
- 260D observation space
- 3 NPC vehicles, speed_ratio 0.85, speed_variation 0.15
- Full curvature (1.0) with random S-curves
- 2 speed zones
- 2 lanes with center line
- Goal distance 230m
- No intersections (locked at 0)

## Phase I Summary (v1 + v2)

| Version | Steps | Peak | Final | Issue |
|---------|-------|------|-------|-------|
| v1 | 5M | 724 | 623 | Triple-param crash (thresholds too tight) |
| v2 | 5M | **775** | **770** | None (pure recovery, project record) |

Total Phase I effort: 10M steps across 2 versions (~50 min).

## Key Lessons

1. **P-018**: Threshold spacing < 5 causes simultaneous transitions -> catastrophic crash
2. **Recovery training works**: Crashed policy (623) recovered to 770 with fixed-param continuation
3. **No curriculum needed for recovery**: All params at final values, just let the agent train
4. **Curves + NPCs achievable**: Agent handles full curvature + 3 NPCs + speed variation simultaneously
