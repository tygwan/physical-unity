# Phase H v2: NPC Speed Variation (Multi-Env Build)

## Experiment ID
- **Run ID**: `phase-H-v2`
- **Config**: `python/configs/planning/vehicle_ppo_phase-H-v2.yaml`
- **Scene**: `PhaseH_NPCIntersection` (built as Builds/PhaseH/PhaseH.exe)
- **Date**: 2026-01-31

## Motivation

Phase H v1 crashed at npc_speed_variation=0.15 (reward 700->550, Std 300+). The single-step jump from 0 to 0.15 variation was too aggressive. v2 fixes this with gradual speed variation introduction and switches to build-based multi-env training.

## Strategy

**WARM START** from Phase G v2 final checkpoint (reward ~628).

Phase H v1 3.5M checkpoint was deleted by keep_checkpoints rotation, so we restart from Phase G v2.

### Key Changes from v1
1. **BUILD TRAINING**: env_path points to built executable (3-4x speedup)
2. **MULTI-ENV**: num_envs=3 (3 parallel Unity processes)
3. **GRADUAL VARIATION**: npc_speed_variation 0->0.05->0.10->0.15 (was 0->0.15)
4. **NO GRAPHICS**: Headless builds for maximum throughput
5. **REDUCED BUDGET**: 5M steps (down from 8M, multi-env compensates)

## Curriculum Changes (vs v1)

### Speed Variation Fix
| Version | Thresholds | Steps | Outcome |
|---------|-----------|-------|---------|
| v1 | 700 (single jump) | 0->0.15 | CRASHED |
| v2 | 700/710/720 | 0->0.05->0.10->0.15 | Stuck at 0.05 |

The v2 thresholds (710, 720) turned out to be too high: the agent averages ~690-700 with variation=0.05 active, making 710 unreachable.

## Training Command

```bash
# Build first in Unity: Build > Build Phase H (NPC Intersection)
mlagents-learn python/configs/planning/vehicle_ppo_phase-H-v2.yaml \
  --run-id=phase-H-v2 --force
```

## Results

### Outcome: Partial Success -> v3 Needed

- **Steps completed**: 5M / 5M
- **Peak reward**: ~701 (at 3.5M steps)
- **Final reward**: ~700 (stable)
- **Speed variation**: Stuck at 1/4 (0.05 only, thresholds 710/720 unreachable)

### Curriculum Completion

| Parameter | Status | Steps |
|-----------|--------|-------|
| Phase G params (9 transitions) | Complete | ~200K |
| num_active_npcs (0->1->2->3) | Complete | ~1.5M |
| npc_speed_ratio (0.5->0.7->0.85) | Complete | ~2M |
| npc_speed_variation 0->0.05 | Complete | ~3M |
| npc_speed_variation 0.05->0.10 | STUCK | Threshold 710 unreachable |
| npc_speed_variation 0.10->0.15 | STUCK | Threshold 720 unreachable |

### Analysis
- Agent reward with variation=0.05 averages ~690-700
- Threshold 710 requires sustained reward above 710, but variation itself caps reward at ~700
- **P-016 updated**: Thresholds for variation params must be BELOW the expected reward under active variation

### Build Training Performance
- ~15-20M steps/hour with 3 envs (vs ~5M in editor)
- 3x speedup validated

## Handoff to v3
- v3 lowers speed_variation thresholds: 700/710/720 -> 685/690/693
- v3 warm starts from v2 3.5M checkpoint (peak performance)
- v3 increases min_lesson_length to 1500 for stability
