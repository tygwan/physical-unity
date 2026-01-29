# Phase C Hypothesis & Validation

Created: 2026-01-29

## Primary Hypothesis

Agent learns selective overtaking strategy with gradual curriculum, achieving +1200+ reward by Stage 4.

## Key Sub-Hypotheses

### H1: Phase B v2 Checkpoint Better for Multi-NPC Transfer
- B v2 has decision-making experience (3 NPCs)
- Phase A has pure overtaking speed (1 NPC)
- Prediction: B v2 converges faster at Stage 0

### H2: Dynamic Overtaking Bonus Drives Behavior
- Easy (+1), Medium (+2), Hard (+3) bonuses based on blocked_duration
- Prediction: Overtaking events increase 5→10→15→20 across stages

### H3: Blocked Detection Suspension Critical
- Don't penalize speed when agent is legitimately stuck
- Prevents "learned to stop" failure like Phase B v1
- Prediction: Suspension prevents reward collapse

### H4: Smooth Curriculum Progression
- Each stage introduces one challenge
- Prediction: +850 → +950 → +1100 → +1250 → +1500 progression

## Validation Checkpoints

| Stage | Expected Reward | TensorBoard Check |
|-------|-----------------|------------------|
| 0 (500K) | +850 | Mean reward crosses 800 |
| 1 (1.2M) | +950 | Smooth upward trend |
| 2 (2.0M) | +1100 | Overtaking events >10/1000 |
| 3 (2.8M) | +1250 | Collision rate <5% |
| 4 (3.6M) | +1500 | Final convergence |

## Success Criteria

PASS if all true:
- [ ] Stage 4 reward >= +1200
- [ ] Collision rate < 5%
- [ ] Overtaking events > 5 per 1000 steps
- [ ] Goal completion > 85%

FAIL if any true:
- [ ] Stage 4 reward < +1000
- [ ] Collision rate > 10%
- [ ] Zero overtaking after 2M steps

