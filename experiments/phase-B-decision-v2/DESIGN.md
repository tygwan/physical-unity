# Phase B v2: Decision Learning - Design Document

**Created**: 2026-01-29
**Parent**: Phase A Overtaking (checkpoint init)
**v1 Failure Ref**: `experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md`
**Approach**: Option C (Hybrid Approach)

---

## 1. Background

### v1 Failure Summary (Confidence: 100%)

Phase B v1 failed catastrophically with -108 reward. Root cause:

```
Agent learned to STOP (speed=0) as optimal policy.
Speed penalty: -0.2/step × 501 steps = -100.2 (92.8% of total)
```

**Contributing Factors**:
1. Speed under penalty too harsh (-0.2/step when stopped)
2. 7 hyperparameters changed simultaneously from Phase A
3. No warm-up period (immediate 2 NPCs)
4. Phase 0 checkpoint used (no overtaking experience)

**Full Analysis**: `experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md`

---

## 2. v2 Design Philosophy

### Option C: Hybrid Approach

**Core Principle**: Change ONE variable at a time from proven Phase A baseline.

```
Phase A (proven +2113) + Fixed Rewards + Gradual Curriculum = Phase B v2
```

### Three Pillars

| Pillar | v1 Approach (FAILED) | v2 Approach |
|--------|---------------------|-------------|
| **Init** | Phase 0 (no overtake skill) | Phase A (overtake capable) |
| **Hyperparams** | 7 changes from Phase A | **ZERO** changes from Phase A |
| **Curriculum** | Immediate 2 NPCs | Gradual 0→1→2→3 |

---

## 3. Config Design

### Hyperparameters (IDENTICAL to Phase A)

```yaml
# Every parameter matches Phase A exactly
batch_size: 4096
buffer_size: 40960
learning_rate: 3.0e-4
beta: 5.0e-3
epsilon: 0.2
lambd: 0.95        # v1 ERROR: changed to 0.99
num_epoch: 5       # v1 ERROR: changed to 10
lr_schedule: linear # v1 ERROR: changed to constant
normalize: false    # v1 ERROR: changed to true
gamma: 0.99        # v1 ERROR: changed to 0.995
time_horizon: 256  # v1 ERROR: changed to 2048
```

**Rationale**: These 6 parameters were all changed in v1. Any one of them could have contributed to the failure. By reverting ALL to Phase A values, we isolate the reward function as the only variable.

### Initialization

```yaml
init_path: results/phase-A-overtaking/E2EDrivingAgent/E2EDrivingAgent-2500155.pt
```

**Rationale**: Phase A achieved +2113 with overtaking capability. The agent already knows:
- Basic lane keeping
- Speed compliance
- Overtaking maneuvers (single slow NPC)

Phase B v2 only needs to learn **when** to overtake vs follow.

### 4-Stage Curriculum

```
Stage 0: Solo Warmup (0 NPCs)
├── Purpose: Verify agent still drives after loading checkpoint
├── Target: reward > 200
├── Expected Steps: 200K-500K
└── Validates: Basic driving intact

Stage 1: Single Slow NPC (1 NPC, 30% speed)
├── Purpose: Re-learn overtaking decision with fixed rewards
├── Target: reward > 400
├── Expected Steps: 300K-800K
└── Validates: Overtaking + speed compliance

Stage 2: Two Mixed NPCs (2 NPCs, 50-70% speed)
├── Purpose: Selective overtaking (slow → overtake, fast → follow)
├── Target: reward > 600
├── Expected Steps: 500K-1M
└── Validates: Decision-making capability

Stage 3: Three Mixed NPCs (3 NPCs, 85% speed)
├── Purpose: Complex multi-vehicle decisions
├── No completion criteria (final stage)
├── Expected Steps: 500K-1M
└── Validates: Generalization
```

---

## 4. Unity Code: Separate Script (Copy Approach)

### Approach: `E2EDrivingAgentBv2.cs`

Instead of modifying the original `E2EDrivingAgent.cs`, a **separate copy** was created.
This allows manual swapping between original and v2 in the Unity Inspector.

**File**: `Assets/Scripts/Agents/E2EDrivingAgentBv2.cs`
**Class**: `E2EDrivingAgentBv2 : Agent`

### 4.1. Enum Renamed to Current Naming System

```csharp
// ORIGINAL (old naming):
public enum TrainingVersion { v10g, v11, v12 }

// V2 COPY (current naming):
public enum TrainingVersion { Phase0, Phase0Sparse, PhaseA }
// Phase0 = v10g (Lane keeping + following)
// Phase0Sparse = v11 (Sparse overtake, deprecated)
// PhaseA = v12 (Dense overtake)
```

### 4.2. Phase B v2 Reward Values (hardcoded in Initialize)

```csharp
// Phase B v2: Boosted overtaking + reduced penalties
overtakeInitiateBonus = 2.0f;       // 0.5 -> 2.0 (4x increase)
overtakeBesideBonus = 0.5f;          // 0.2 -> 0.5 (2.5x increase)
overtakeAheadBonus = 2.0f;           // 1.0 -> 2.0 (2x increase)
overtakeCompleteBonus = 3.0f;        // 2.0 -> 3.0 (1.5x increase)
stuckBehindPenalty = -0.05f;         // -0.1 -> -0.05 (50% reduction)
stuckBehindTimeout = 5.0f;           // 3.0 -> 5.0 (more patience)
speedUnderPenalty = -0.02f;          // -0.1 -> -0.02 (80% reduction)
```

### 4.3. Blocked Detection in Speed Penalty

In `CalculateSpeedPolicyReward()`, speed under penalty is suspended when blocked by NPC:

```csharp
else if (speedRatio < 0.5f)
{
    // Phase B v2: Check if agent is blocked by NPC ahead
    bool isBlockedByNPC = isBlocked &&
        leadDist < safeFollowingDistance * 1.5f;

    if (!isBlockedByNPC)
    {
        float progressivePenalty = speedUnderPenalty * (2f - speedRatio * 2f);
        reward += progressivePenalty;
    }
    // When blocked by NPC: no speed penalty
    // Agent shouldn't be punished for physics impossibility
}
```

**Note**: `isBlocked`, `leadDist` are already available from `GetLeadVehicleInfo()` called earlier in the same method.

### 4.4. Unity Scene Setup

Phase B v2 Scene에서 Agent GameObject의 컴포넌트를 교체:
1. `E2EDrivingAgent` 컴포넌트 제거
2. `E2EDrivingAgentBv2` 컴포넌트 추가
3. Inspector에서 `Training Version: PhaseA` 확인 (기본값)
4. Scene/Environment 참조 재연결 (goalTarget, routeWaypoints, sceneManager 등)

---

## 5. Expected Reward Calculation

### Per-Episode (256 steps, Stage 1: 1 slow NPC)

**With fixed reward function**:

```
Component Calculation:
  Progress:          +0.5/step × 200 steps moving ≈ +100
  Speed Compliance:  +0.3/step × 150 steps in range ≈ +45
  Overtake Complete: +7.5 per overtake × 2 overtakes ≈ +15
  Jerk Penalty:      -0.1 × 0.05 avg × 256 ≈ -1.3
  Time Penalty:      -0.001 × 256 ≈ -0.3
  Speed Under (blocked): 0 (suspended when NPC ahead)

  EXPECTED TOTAL: +158 per episode (vs v1: -108)
```

### Per-Episode (Stage 3: 3 mixed NPCs)

```
  Progress:          +0.5/step × 220 steps ≈ +110
  Speed Compliance:  +0.3/step × 180 steps ≈ +54
  Overtakes:         +7.5 × 3 overtakes ≈ +22.5
  Following (fast):  0 (PhaseA has no following bonus)
  Penalties:         -5 (minor jerk, time)

  EXPECTED TOTAL: +181 per episode
```

---

## 6. Validation Protocol

### Pre-Flight (MANDATORY before full training)

**Step 1: 10K Step Sanity Check**
```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-B-v2.yaml \
  --run-id=phase-B-v2-preflight-10k \
  --initialize-from=phase-A-overtaking \
  --max-steps=10000
```

**Pass Criteria**:
- [ ] Stats/Speed > 5 m/s for >50% of steps
- [ ] No compilation errors in console
- [ ] Agent visibly moving in Unity editor

**Step 2: 100K Step Validation**
```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-B-v2.yaml \
  --run-id=phase-B-v2-preflight-100k \
  --initialize-from=phase-A-overtaking \
  --max-steps=100000
```

**Pass Criteria**:
- [ ] Positive reward trend (> +50 improvement over 50K)
- [ ] Episode/Length > 200 steps (not stuck)
- [ ] Stats/Speed > 8 m/s average
- [ ] Stage 0 curriculum progressing

### During Training (Monitoring Checkpoints)

**Every 500K steps**:
- Check reward trend (should be increasing)
- Check curriculum stage (should advance)
- Check speed ratio (should be > 0.5)

**Auto-Abort Conditions**:
- Speed < 1 m/s for 50K consecutive steps
- Reward < -50 for 3 consecutive checkpoints (150K steps)
- No curriculum advancement after 1M steps

### Post-Training

**Success Criteria (Stage 3 Final)**:
| Metric | Target | v1 Actual | v2 Expected |
|--------|--------|-----------|-------------|
| Final Reward | > +600 | -108 | +800-1200 |
| Speed | > 12 m/s | 0.0 | 14-18 |
| Overtake Count | > 1/episode | 0 | 2-3 |
| Episode Length | > 200 | 501 (stuck) | 256 (full) |
| Collision Rate | < 10% | 0% (stopped) | < 5% |

---

## 7. Rollback Plan

### If v2 Also Fails

**Option A: Further reward reduction**
- speedUnderPenalty: -0.02 → -0.005
- Remove all negative penalties except collision/off-road

**Option B: Positive-only reward**
- Remove ALL penalties
- Only positive rewards (progress, speed compliance, overtaking)
- Let agent discover penalties through collision/off-road episode termination

**Option C: Fresh start with simplified environment**
- No checkpoint initialization
- Very simple environment (straight road, 1 slow NPC)
- Train from scratch with corrected reward function

---

## 8. File Structure

```
Assets/Scripts/Agents/
├── E2EDrivingAgent.cs             ← Original (Phase A, unchanged)
├── E2EDrivingAgentBv2.cs          ← Phase B v2 copy (this experiment)

experiments/phase-B-decision-v2/
├── DESIGN.md                      ← THIS FILE

python/configs/planning/
├── vehicle_ppo_phase-B-v2.yaml    ← Training config

results/phase-B-decision-v2/       ← Will be created by mlagents-learn
├── E2EDrivingAgent/
│   ├── events.out.tfevents.*
│   ├── *.pt (checkpoints)
│   └── *.onnx (final model)
```

---

## 9. Cross-References

- **v1 Failure Analysis**: `experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md`
- **v1 Config (reference)**: `python/configs/planning/vehicle_ppo_phase-B-minimal.yaml`
- **Phase A Config (base)**: `python/configs/planning/vehicle_ppo_phase-A.yaml`
- **Phase A Checkpoint**: `results/phase-A-overtaking/E2EDrivingAgent/E2EDrivingAgent-2500155.pt`
- **Unity Agent Code (original)**: `Assets/Scripts/Agents/E2EDrivingAgent.cs`
- **Unity Agent Code (v2 copy)**: `Assets/Scripts/Agents/E2EDrivingAgentBv2.cs`
- **TensorBoard Guide**: `docs/TENSORBOARD-METRICS-GUIDE.md`

---

**Design Approved**: Pending
**Implementation Status**: Config + Unity script (E2EDrivingAgentBv2.cs) created
**Remaining**: Scene setup (swap component on Agent GameObject), pre-flight validation
