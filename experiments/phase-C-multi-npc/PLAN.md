# Phase C Training Plan: Multi-NPC Interaction & Advanced Decision-Making

**Status**: Ready for Training
**Created**: 2026-01-29
**Duration**: 3.6M steps (~50 minutes on RTX 4090)
**Target**: +1500 reward (+71% vs Phase B v2)

## Quick Summary

**Phase B v2 Results**: +877 reward, 3 NPCs, safe driving, no overtaking observed
**Phase C Goal**: Scale to 8 NPCs with selective overtaking strategy

**Key Changes**:
1. Initialize from Phase B v2 checkpoint (has decision-making)
2. 5-stage curriculum: 3→4→5→6→8 NPCs
3. Dynamic overtaking bonus based on blocked duration
4. Blocked detection suspension (don't penalize speed when stuck)

## Why Phase C Matters

Phase B v2 agent learned "don't overtake in tight spaces" (smart defensive strategy).
Phase C needs to learn "overtake WHEN SAFE, follow WHEN NOT."

This requires:
- Progressive complexity (3→8 NPCs)
- Clear overtaking incentives (dynamic +1.0/+2.0/+3.0 bonus)
- Intelligent penalty suspension (don't punish being blocked)

## 5-Stage Curriculum

| Stage | NPCs | Speed | Steps | Target Reward |
|-------|------|-------|-------|---|
| 0 | 3 | 0.85 mixed | 500K | +850 (baseline) |
| 1 | 4 | 0.4-0.55 slow | 700K | +950 (+15%) |
| 2 | 5 | 0.55-0.7 med | 800K | +1100 (+30%) |
| 3 | 6 | 0.7-0.85 fast | 800K | +1250 (+45%) |
| 4 | 8 | 0.5-1.0 mixed | 800K | +1500 (+75%) |

## Critical Configuration

### Initialization
```yaml
init_path: results/phase-B-decision-v2/E2EDrivingAgent/E2EDrivingAgent-latest.pt
```

### Hyperparameters (identical to Phase B v2)
```yaml
batch_size: 4096
learning_rate: 3e-4
beta: 5e-3
lambd: 0.95
```

### YAML Config
```
File: python/configs/planning/vehicle_ppo_phase-C.yaml
Status: ✅ Ready
```

## Validation Checkpoints

MANDATORY pre-flight checks:

1. **100K steps**: Speed > 5 m/s (basic driving)
2. **500K steps**: Reward > +800 (Stage 0 baseline)
3. **1.5M steps**: Reward > +1000 (Stage 2 progression)
4. **3.6M steps**: Final convergence

## Emergency Stops

Auto-abort if:
- Mean reward < -50 for 3 consecutive checkpoints
- Collision rate > 10% sustained
- Episode length < 1000 steps (agent giving up)

## Expected Outcomes

### Stage 0-1 (First 1.2M steps, ~15 min)
- Agent adapts to 4-NPC environment
- Establishes baseline with slow-moving vehicles
- Should see first overtaking attempts

### Stage 2-3 (Next 1.6M steps, ~20 min)
- Agent learns selective overtaking decisions
- Handles 5-6 NPCs with mixed speeds
- Reward should reach +1200+

### Stage 4 (Final 0.8M steps, ~10 min)
- Peak challenge: 8 NPCs, full speed range
- Tests generalization to complex scenarios
- Final reward target: +1500

## What Could Go Wrong (Lessons from Phase B)

**Phase B v1 failed (-108)** because:
- Speed penalty (-0.2/step when stopped) → Agent learned to STOP
- Math: -0.2 × 501 steps = -100.2 (matches observed)

**Phase C Prevention**:
- Blocked detection suspension: Don't penalize speed when stuck
- If reward drops suddenly: Check if suspension is working
- If overtaking never happens: Verify dynamic bonus is being applied

## Training Command

```bash
cd /c/Users/user/Desktop/dev/physical-unity
source .venv/bin/activate

mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-C.yaml \
  --run-id=phase-C-multi-npc \
  --initialize-from=phase-B-decision-v2 \
  --force \
  --no-graphics \
  --time-scale=20
```

## Success Criteria (PASS/FAIL)

**PASS (All must be true)**:
- [ ] Stage 4 final reward ≥ +1200 (meaningful improvement)
- [ ] Collision rate < 5% (safety maintained)
- [ ] Overtaking events > 5 per 1000 steps (learned behavior)
- [ ] Goal completion > 85% (route finishing)

**FAIL (Any one fails)**:
- [ ] Stage 4 reward < +1000 (regression from Phase B)
- [ ] Collision rate > 10% (safety issue)
- [ ] Zero overtaking events (no progress)

## Timeline

| Milestone | Time | Status |
|-----------|------|--------|
| Setup | 10 min | Start |
| Stage 0 | 15 min | Baseline |
| Stage 1-2 | 25 min | Overtaking |
| Stage 3-4 | 30 min | Complex |
| Analysis | 10 min | Done |

**Total**: ~90 minutes (training + setup + analysis)

---

**Created**: 2026-01-29
**Next Step**: Execute training, monitor checkpoints, analyze results
