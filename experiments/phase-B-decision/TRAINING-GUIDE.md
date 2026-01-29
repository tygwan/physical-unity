# Phase B: Decision Learning - Training Guide

**Version**: phase-B
**Created**: 2026-01-28
**Duration**: ~25 minutes (3.0M steps)
**GPU**: RTX 4090 (24GB VRAM) - Sufficient

---

## Pre-Training Checklist

### Environment Validation
- [ ] Unity environment built and ready
- [ ] ROS2 Humble running (Windows native or WSL)
- [ ] Python ML-Agents environment set up (v3.0+)
- [ ] PyTorch 2.0+ installed
- [ ] CUDA 12.1+ available

### Artifact Validation
- [ ] Phase 0 checkpoint exists: results/phase-0-foundation/E2EDrivingAgent/E2EDrivingAgent-8000047.pt
- [ ] YAML config valid: python/configs/planning/vehicle_ppo_phase-B.yaml
- [ ] TensorBoard log directory accessible: experiments/phase-B-decision/logs/

### Data & Logging
- [ ] Experiment directory created: experiments/phase-B-decision/logs and checkpoints
- [ ] Video recording enabled in Unity (for validation)
- [ ] Overtaking detection logging enabled

---

## Training Commands

### Option 1: Full Training (Recommended)

```bash
cd /c/Users/user/Desktop/dev/physical-unity
source .venv/bin/activate

mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B-decision \
  --initialize-from=results/phase-0-foundation/E2EDrivingAgent/E2EDrivingAgent-8000047.pt \
  --force \
  --no-graphics \
  --time-scale=1.0
```

### Option 2: With TensorBoard Monitoring

```bash
# Terminal 1: Training
mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B-decision \
  --initialize-from=results/phase-0-foundation/E2EDrivingAgent/E2EDrivingAgent-8000047.pt \
  --force

# Terminal 2: TensorBoard
tensorboard --logdir=results/phase-B-decision/Agent
```

### Option 3: Quick Sanity Check (10K steps)

```bash
mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B-sanity-check \
  --initialize-from=results/phase-0-foundation/E2EDrivingAgent/E2EDrivingAgent-8000047.pt \
  --force \
  --time-scale=4.0 \
  --max-steps=10000
```

**Expected output**: Training starts without errors, reward improves, no CUDA errors

---

## Monitoring Strategy

### TensorBoard Key Metrics

**1. Mean Reward (Cumulative)**
- Expected trajectory: 0→600→1100→1400→1600+
- Red flag: Stalled <500 for >500K steps
- Green flag: Smooth monotonic increase

**2. Episode Length**
- Expected: 2400-2600 steps per episode
- Should remain stable across all stages

**3. Collision Rate**
- Expected: 0-2%
- Red flag: >5% at any point (STOP training immediately)
- Action: Increase collision penalties, resume

**4. Policy Loss**
- Expected: Decreasing trend
- Red flag: Diverging or oscillating >1.0
- Action: Check learning rate, reduce if needed

### Stage-Specific Monitoring

**Stage 0 (0-750K): Baseline**
- Monitor: Speed reward component
- Expected: +600-800 by 750K
- Action if <500: Investigate speed degradation

**Stage 1 (750K-1500K): Forced Overtaking**
- Monitor: Overtaking event frequency
- Expected: >1 event per episode by 1.5M
- Action if 0 events: Implement Contingency A (logging)

**Stage 2 (1500K-2250K): Selective Decisions**
- Monitor: Correct decision ratio
- Expected: >70% correct decisions by 2.25M
- Action if <50%: Adjust NPC speed differential

**Stage 3 (2250K-3000K): Complex Scenarios**
- Monitor: Final convergence
- Expected: +1500+ by 3.0M
- Action if <1200: Investigate curriculum progression

---

## Troubleshooting Guide

### Issue 1: Overtaking Events = 0

**Symptoms**: After Stage 1 (1.5M steps), zero overtaking events

**Solutions**:
1. Check Unity logs for detection messages
2. Enable explicit lane-change logging
3. Review video clips (manual validation)
4. Implement lateral displacement metric as proxy
5. If still 0: Implement Contingency A

**Recovery**:
1. Pause at 1.5M steps
2. Add explicit logging to code
3. Resume from latest checkpoint
4. Monitor new logs

### Issue 2: Speed Performance Drops >15%

**Symptoms**: Speed reward <60% of Phase A baseline

**Solutions**:
1. Increase speed weight: 0.3 → 0.4
2. Decrease following penalty: -0.5 → -0.2
3. Retrain mini-curriculum (500K test)
4. If restored, continue full training

**Recovery**:
1. Edit YAML: speed_reward_weight: 0.3 → 0.4
2. Resume from latest checkpoint
3. Monitor speed component

### Issue 3: Collision Rate >5%

**Symptoms**: >5% collision rate at any stage

**IMMEDIATE ACTION**:
1. PAUSE training immediately
2. Increase collision penalty: -10.0 → -15.0
3. Increase near-collision penalty: -2.0 → -5.0
4. Resume from latest checkpoint

**Alternative**:
1. Reduce overtaking bonus: +5.0 → +3.0
2. Go back one curriculum stage
3. Retrain conservatively

### Issue 4: Training Stalls >500K Steps

**Symptoms**: Reward plateau or oscillation for 500K steps

**Solutions**:
1. Increase learning rate: 3e-4 → 5e-4
2. Lower curriculum thresholds by 10-20%
3. Check curriculum progression logs
4. If none work: Rollback to Phase A and redesign

**Recovery**:
1. Edit YAML: learning_rate: 3e-4 → 5e-4
2. Resume from checkpoint before plateau
3. Monitor for recovery signs
4. If no improvement in 250K steps, rollback

### Issue 5: Memory Error (Out of VRAM)

**Symptoms**: CUDA out of memory error

**Solutions**:
1. Reduce batch size: 4096 → 2048
2. Reduce buffer size: 40960 → 20480
3. Clear GPU memory: Restart training
4. Reduce time horizon: 2048 → 1024

---

## Checkpoint Management

### Automatic Checkpoints
- Saved every 500K steps
- Keep last 5 checkpoints
- Located: results/phase-B-decision/

### Manual Backup
```bash
cp results/phase-B-decision/E2EDrivingAgent-XXX.pt \
   results/phase-B-decision/E2EDrivingAgent-backup-YYYYMMDD.pt
```

### Resume from Checkpoint
```bash
mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B-decision-resume \
  --initialize-from=results/phase-B-decision/E2EDrivingAgent-1500000.pt \
  --resume
```

---

## Expected Performance Timeline

| Milestone | Steps | Duration | Reward |
|-----------|-------|----------|--------|
| Sanity check | 10K | <5 sec | ~100-200 |
| Stage 0 | 750K | ~6 min | +600-800 |
| Stage 1 | 1.5M | ~12 min | +1000-1200 |
| Stage 2 | 2.25M | ~18 min | +1200-1500 |
| Stage 3 | 3.0M | ~25 min | +1500-1800 |

---

## Post-Training Steps

### 1. Evaluation
```bash
mlagents-learn \
  python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B-eval \
  --initialize-from=results/phase-B-decision/E2EDrivingAgent-3000000.pt \
  --inference \
  --num-episodes=20
```

### 2. Success Validation

Verify:
1. Mean Reward: >= +1500
2. Overtaking Events: > 150
3. Collision Rate: < 5%
4. Goal Completion: > 90%

**If ALL met**: Phase B SUCCESS → Proceed to Phase C
**If ANY failed**: Review contingencies and redesign

---

*Phase B Training Guide - Created: 2026-01-28*
