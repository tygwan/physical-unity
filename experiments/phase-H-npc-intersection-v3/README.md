# Phase H v3: NPC Speed Variation Completion

## Experiment ID
- **Run ID**: `phase-H-v3`
- **Config**: `python/configs/planning/vehicle_ppo_phase-H-v3.yaml`
- **Scene**: `PhaseH_NPCIntersection` (Builds/PhaseH/PhaseH.exe)
- **Date**: 2026-01-31

## Motivation

Phase H v2 completed 3 NPCs + speed_ratio 0.85, but speed_variation was stuck at 0.05/4. Thresholds 710/720 were unreachable because the agent averages ~690-700 under active variation. v3 lowers thresholds to achievable levels.

## Strategy

**WARM START** from Phase H v2 3.5M checkpoint (peak ~700, Std ~5, 3 NPCs + speed_ratio 0.85, variation=0.05).

### Key Changes from v2
1. **LOWER THRESHOLDS**: speed_variation 685/690/693 (was 700/710/720)
2. **LONGER LESSONS**: min_lesson_length 1500 for stability before advancing
3. **WARM START**: v2 3.5M checkpoint (best performance point)

## Threshold Fix Analysis

| v2 Threshold | Agent Avg w/ Variation | Reachable? | v3 Threshold |
|-------------|----------------------|-----------|-------------|
| 700 | ~695 | Barely | 685 |
| 710 | ~690-700 | No | 690 |
| 720 | ~690-700 | No | 693 |

v3 thresholds are set 5-10 points BELOW observed averages to ensure achievability.

## Curriculum (Speed Variation Only)

All other params unlock immediately from ~700 warm start.

| Lesson | Threshold | Value | min_lesson_length |
|--------|-----------|-------|-------------------|
| NoVariation | 685 | 0.0 | 1000 |
| TinyVariation | 690 | 0.05 | 1500 |
| SmallVariation | 693 | 0.10 | 1500 |
| MildVariation | (final) | 0.15 | - |

## Training Command

```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-H-v3.yaml \
  --run-id=phase-H-v3 --force
```

## Results

### Outcome: SUCCESS

- **Steps completed**: 5M / 5M
- **Final reward**: ~701 (stable at full variation)
- **All curriculum complete**: 11/11 params fully transitioned
- **Speed variation**: 4/4 lessons complete (0 -> 0.05 -> 0.10 -> 0.15)

### Final Agent Capabilities
- 260D observation space
- 3 NPC vehicles with waypoint-following through intersections
- npc_speed_ratio: 0.85
- npc_speed_variation: 0.15
- All intersection types: T-junction, Cross, Y-junction
- All turn directions: Straight, Left, Right
- 2 lanes with center line
- Goal distance: 230m

### Curriculum Completion Timeline

| Parameter | Status | Approximate Steps |
|-----------|--------|-------------------|
| Phase G params (9 transitions) | Complete | ~200K |
| NPC count (0->1->2->3) | Complete | ~1M |
| NPC speed_ratio (0.5->0.7->0.85) | Complete | ~1.5M |
| NPC speed_variation (0->0.05->0.10->0.15) | Complete | ~3M |

### Key Lessons
1. **P-016**: Threshold must be below expected reward under active conditions
2. **P-017**: Warm start at peak checkpoint (not final) gives best results
3. **Build-based multi-env**: 3x throughput, essential for iteration speed

## Phase H Summary (v1 -> v2 -> v3)

| Version | Steps | Peak Reward | Variation Complete | Issue |
|---------|-------|-------------|-------------------|-------|
| v1 | 3.5M/8M | ~700 | 0/2 | Single-step 0->0.15 crash |
| v2 | 5M/5M | ~701 | 1/4 | Thresholds 710/720 unreachable |
| v3 | 5M/5M | ~701 | 4/4 | Success |

Total Phase H effort: ~13.5M steps across 3 versions.
