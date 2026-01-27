# Phase A: Dense Overtaking - Testable Hypothesis

## Primary Hypothesis

"An RL agent initialized from v10g foundation (lane-keeping) can learn overtaking maneuvers when trained with progressive NPC density curriculum and explicit overtaking rewards (+3.0/overtake), achieving >70% overtaking success while maintaining <5% collision rate within 2.5M steps."

---

## Hypothesis Components

### 1. Skill Transfer from v10g Foundation
- **Claim**: Lane-keeping behaviors transfer to Phase A, maintaining 0% collision in Stage 1
- **Metric**: Lane-keeping bonus remains >50% of v10g levels
- **Expected**: First collisions are learning explorations, not systematic failures

### 2. Overtaking Learning Signal
- **Claim**: Explicit overtaking rewards (+3.0/overtake) create strong learning signal
- **Metric**: Overtaking success increases from 0% → >70% by step 2.5M
- **Expected**: First successful overtakes by 300K-400K steps

### 3. Progressive Curriculum Enables Safety
- **Claim**: Gradual NPC density increase (1→2→4→6) maintains <5% collision throughout
- **Metric**: Collision rate stable across all stages
- **Expected**: Stage 1 <2%, Stage 3 <5%

### 4. Safety Preservation
- **Claim**: Collision penalty (-10) prevents catastrophic behavior degradation
- **Metric**: No unplanned collision spikes >10% in any 100K window
- **Expected**: Smooth collision curve, no sharp increases

### 5. Convergence by 2.5M Steps
- **Claim**: Mean reward plateaus by step 2.2M, indicating learning completion
- **Metric**: Reward std <50 for final 200K steps
- **Expected**: Similar convergence pattern to v10g (smooth S-curve)

---

## Key Assumptions & Risks

| Assumption | Risk | Validation | Contingency |
|-----------|------|-----------|-------------|
| Transfer learning works | Weights don't transfer | Stage 1 collision <2% | Lower learning rate to 2e-4 |
| Curriculum thresholds correct | Too fast/slow progression | Check stage completion times | Adjust thresholds ±10% |
| Overtaking reward magnitude | Too small/large incentive | Overtaking >20% of rewards | Adjust ±2.0 points |
| Penalties sufficient | Agent willing to collide | Collision/overtake ratio <0.1 | Increase penalty to -15 |
| NPC distribution realistic | Doesn't match real traffic | Monitor overtaking patterns | Adjust NPC spawn logic |

---

## Success Criteria

### Must-Pass Metrics
✅ **ACCEPT if ALL met at step 2.5M:**
1. Overtaking success ≥ 70%
2. Collision rate ≤ 5%
3. Mean reward ≥ 900
4. Goal completion ≥ 90%

❌ **REJECT if ANY occur:**
1. Overtaking success <50%
2. Collision rate >10%
3. Mean reward <650 by 1.5M steps
4. Unrecoverable failure mode

### Stretch Goals
- Overtaking success >80%
- Collision rate <2%
- Convergence by 2M steps

---

## Expected Training Timeline

| Milestone | Steps | Expected | Validation |
|-----------|-------|----------|-----------|
| First overtakes | 300-400K | Success rate jumps 0%→5% | Check logs for overtake events |
| Stage 1 plateau | 700K | Stable at +400 reward | Std deviation <30 |
| Stage 2 mid-point | 1.2M | Handling 4 NPCs, +550 | Collision rate stable <5% |
| Convergence start | 1.8M | Stage 3, reward rising to +700 | Smooth progression |
| Final plateau | 2.2M-2.5M | Stable at +950, std<50 | Learning complete |

---

## Failure Mode Responses

### Mode 1: Agent Won't Overtake
**Symptom**: Overtaking <30% at 2.5M  
**Fix**: Increase overtaking reward +3.0→+5.0, reduce lane-keeping bonus, resume training

### Mode 2: Reckless Overtaking
**Symptom**: Collision rate >8%  
**Fix**: Increase collision penalty -10→-15, reduce overtaking reward +3.0→+1.5, reload v10g

### Mode 3: Reward Collapse at 1.2M
**Symptom**: Reward drops from +400→+150  
**Fix**: Load 1M checkpoint, increase curriculum thresholds 50%→60%, resume

### Mode 4: Catastrophic Forgetting
**Symptom**: Collision rate 2%→8% from Stage 1 to Stage 3  
**Fix**: Load 1.5M checkpoint, reduce Stage 3 NPCs, add intermediate stage

### Mode 5: Training Instability
**Symptom**: Reward variance >150, erratic oscillations  
**Fix**: Load stable checkpoint, reduce learning rate 3e-4→2e-4, increase batch size

---

## Validation Protocol

### Checkpoints (Every 500K)
- Save models at 0K, 500K, 1M, 1.5M, 2M, 2.5M
- Log curriculum stage, reward, collision rate, overtaking metrics

### Metrics to Monitor
```
Per Episode:
  - episode_reward
  - num_collisions
  - goal_reached (boolean)
  - num_overtakes
  - max_speed_achieved

Per 10K Steps:
  - cumulative_reward (moving avg, window=100 episodes)
  - collision_rate (%)
  - goal_completion_rate (%)

Per 100K Steps:
  - overtaking_success_rate (%)
  - stage_transition_event
  - policy_loss
  - value_loss
```

### Statistical Tests
- **Convergence**: Reward variance [1.8M-2M] vs [2.3M-2.5M], expect >30% reduction
- **Safety**: Binomial test collision rate <5% per stage, p<0.05
- **Learning**: Exponential fit to overtaking curve, R² > 0.85

---

## Success Story (Best Case)

```
Step 200K: Agent discovers lateral movement (lane changes possible)
Step 300K: First successful overtakes, reward +50 jump, success rate jumps to 5%
Step 500K: Stage 1 plateau at +400 reward, overtaking success 65%, collision <2%
Step 700K: Smooth transition to Stage 2 (2 NPCs), minimal disruption
Step 1.0M: Learning 2-NPC coordination, reward +450
Step 1.2M: Full Stage 2 capability (4 NPCs), reward +550, collision <4%
Step 1.8M: Smooth transition to Stage 3 (6 NPCs)
Step 2.1M: Complex overtaking chains learned, reward rising
Step 2.5M: Final convergence at +950 (70% overtake success, 2% collision)

Overall: Smooth progression, all metrics on target, production ready
```

---

## Acceptance Criteria

**ACCEPT Phase A if:**
- ✅ Overtaking success ≥ 70% (primary goal achieved)
- ✅ Collision rate ≤ 5% throughout (safety maintained)
- ✅ Mean reward ≥ 900 (target reached)
- ✅ Convergence by 2.5M steps (efficient learning)
- ✅ No unrecoverable failure modes

**REJECT Phase A if:**
- ❌ Overtaking success < 50% (insufficient learning)
- ❌ Collision rate > 10% at any point (safety failure)
- ❌ Mean reward < 650 by 1.5M (convergence failure)
- ❌ Multiple rollbacks required (design issue)

---

**Version**: 1.0 | **Created**: 2026-01-27 | **Status**: Ready for Validation
