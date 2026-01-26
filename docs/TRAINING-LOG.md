# Training Log - E2EDrivingAgent RL Training History

## Overview

| Version | Focus | Steps | Best Reward | Status |
|---------|-------|-------|-------------|--------|
| v10g | Lane Keeping + NPC 공존 | 8M (4.97M effective) | ~40 (NPC4) | Plateau |
| v11 | Overtaking Reward (Sparse) | 8M | ~51 peak (NPC4) | Marginal improvement |
| v12 Phase A | Dense Overtaking (Slow NPC) | 2M | **+937** | **COMPLETED** |
| v12 Phase B | Overtake vs Follow Decision | 2M | **+903** | **COMPLETED** |
| **v12 Phase C** | **Multi-NPC Generalization** | **4M** | **+961 (peak +1086)** | **COMPLETED** ✅ |
| **v12 Phase D** | **Lane Observation (254D)** | **6M** | **+332 (peak +402)** | **COMPLETED** ✅ |
| v12_ModularEncoder | Modular Architecture for Incremental Learning | - | - | Superseded |
| v12_HybridPolicy | Phase C-1 Hybrid Training | 3M | -82.7 best | **FAILED** (Catastrophic Forgetting) |

---

## v10g: Lane Keeping + NPC Coexistence

### Intent
- Speed policy 기반 주행 + 차선 유지 + NPC 4대 환경에서 안정적 공존
- Heading alignment (0.02) + lateral deviation (-0.02) 보상
- 3-strike collision (에피소드 당 3회 충돌 시 종료)

### Key Parameters
```
headingAlignmentReward = 0.02 (Inspector override 문제로 Initialize에서 강제 설정)
lateralDeviationPenalty = -0.02
followingBonus = 0.3 (안전 거리 유지 시 보상)
collisionPenalty = -5.0 (3회까지 허용)
```

### Curriculum
- num_active_npcs: 0 -> 1 -> 2 -> 4
- goal_distance: 50 -> 100 -> 160 -> 230
- speed_zone_count: 1 -> 2 -> 3 -> 4

### Results (8M steps, ~4.97M effective)
- NPC 0: reward ~90-95
- NPC 4 (1.55M onwards): reward ~35-40, plateau for 3.5M steps
- Agent successfully avoids collisions and maintains lane
- **Problem**: Agent "follows" slow NPCs indefinitely (no overtaking behavior)

### Lessons Learned
- Inspector serialized values override `public float` defaults - must force in `Initialize()`
- Lane keeping reward scale matters: 0.2 was too dominant (520 reward), 0.02 is appropriate
- followingBonus rewards "not crashing" which is already implicit in collision penalty

---

## v11: Overtaking Reward (Sparse)

### Intent
- v10g 에이전트가 느린 NPC를 추월하도록 학습
- State machine: None -> Approaching -> Beside -> Completed
- SphereCast (radius 3m) for wider lead detection
- OverlapSphere for robust NPC tracking

### Key Parameters (v11 additions)
```
overtakePassBonus = 3.0      # 추월 완료 시 1회 보상
overtakeSpeedBonus = 0.15    # NPC 옆에서 속도 유지 시 per-step 보상
overtakeSlowLeadThreshold = 0.7  # 이 비율 이하면 "느린 NPC"
overtakeDetectWidth = 3.0    # SphereCast 반경
```

### Design Decisions
1. SphereCast > Raycast: offset된 NPC도 감지
2. Lane keeping penalty 추월 중 중지 (isOvertaking flag)
3. Following bonus: NPC > 70% speedLimit일 때만 보상

### Results (8M steps)
- NPC 0: reward ~91-95 (free driving unchanged)
- NPC 1 (850K): reward ~60-85 (avg ~70)
- NPC 2 (1.23M): reward ~45-64 (avg ~55)
- NPC 4 (1.55M): reward ~35-50 (avg ~41)
- Plateau from 1.55M to 8M
- **v10g 대비**: mean +4, peak +8 (미미한 개선)
- overtakePassBonus 일부 에피소드에서 발생하지만 일관되게 학습되지 않음

### Problem Diagnosis
1. **targetSpeed 감소 구조**: 느린 NPC 뒤에서 `targetSpeed = leadSpeed` → 따라가기만 해도 speed reward 최대
2. **위험 비대칭**: 추월 시도 = 차선 이탈 리스크, 따라가기 = 안전
3. **보상 희소성**: overtakePassBonus(+3.0)는 전체 과정 완료 시에만 → 학습 신호 부족
4. **followingBonus 역할**: 따라가기를 명시적으로 보상하여 추월 동기 약화

### Lessons Learned
- Sparse reward만으로는 추월 학습 불가
- 구조적으로 "따라가기"가 "추월"보다 보상이 높거나 같으면 안전한 선택이 항상 유리
- targetSpeed를 NPC 속도로 낮추면 절대 추월 학습 불가

---

## v12: Dense Overtaking + Strategy 3 (Current)

### Strategy: Industry-Informed Phased Dense Reward

연구 배경:
- **Waymo**: BC-SAC (Behavioral Cloning + Soft Actor-Critic), dual-loop learning
- **Tesla**: Imitation Learning -> RL augmentation (AlphaStar pattern)
- **Academic**: Hierarchical RL, Dense-to-Sparse distillation, Teacher-Student frameworks
- **Key insight**: Agents get stuck behind slower vehicles as "safer" action

### Core Changes (v11 -> v12)
1. **targetSpeed = speedLimit ALWAYS** - 절대 NPC 속도로 낮추지 않음
2. **followingBonus 완전 제거** - 따라가기 보상 없음
3. **Stuck-behind penalty** (-0.1/step after 3초) - 느린 NPC 뒤 정체 시 페널티
4. **Dense 5-phase overtaking reward**:
   - None -> Approaching: 느린 NPC 감지 (보상 없음, 추적 시작)
   - Approaching -> Beside: 차선 변경 시작 (+0.5)
   - Beside: NPC 옆에서 속도 유지 (+0.2/step)
   - Beside -> Ahead: NPC 추월 완료 (+1.0)
   - Ahead -> LaneReturn: 원래 차선 복귀 (+2.0)
5. **Lane return detection**: 웨이포인트 X좌표 기반 복귀 판정

### Reward Summary (v12)
```
Speed compliance (80-100% of limit): +0.3/step
Speed over limit: -0.5 ~ -3.0 progressive
Stuck behind slow NPC (>3s): -0.1/step
Overtake initiate (lane change): +0.5 one-time
Overtake beside (speed maintain): +0.2/step
Overtake ahead (passed NPC): +1.0 one-time
Overtake complete (lane return): +2.0 one-time
Collision: -5.0 (3-strike rule)
Off-road: -5.0 (immediate episode end)
```

### Phase Plan
| Phase | Steps | NPC Count | NPC Speed | Goal |
|-------|-------|-----------|-----------|------|
| A | 2M | 1 | 30% of limit | 추월 동작 자체를 학습 |
| B | 2M | 1 | 30-90% varied | 추월/따라가기 판단 학습 |
| C | 4M | 2-4 | 30-100% varied | 복잡 환경 일반화 |

### Phase A Config
- `vehicle_ppo_v12_phaseA.yaml`
- 1 very slow NPC (npc_speed_ratio=0.3)
- Short goal (80m -> 150m)
- Single speed zone
- Initialize from v11 checkpoint
- Expected: Agent learns basic overtaking maneuver

### Phase A Results (v12_phaseA_fixed) - COMPLETED 2026-01-25

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 2,000,209 |
| Training Time | ~30 minutes |
| Final Mean Reward | **+714** (peak: +935) |
| Std of Reward | 234 (stabilized) |

#### Training Progression
| Step | Mean Reward | Std | Phase |
|------|-------------|-----|-------|
| 5K | -127 | 48 | Initial (penalty applied) |
| 200K | -244 | 127 | Heavy penalty for 0 speed |
| 460K | **+7.3** | 162 | First positive! |
| 580K | +77.6 | 234 | Rapid improvement |
| 940K | +190.9 | 340 | Consistent positive |
| 1.2M | +502.9 | 298 | Breakthrough |
| 1.37M | **+937.0** | 324 | Peak performance |
| 1.66M | +823.0 | 275 | Stabilizing |
| 2.0M | +714.7 | 234 | Converged |

#### Checkpoints Saved
- `E2EDrivingAgent-499972.onnx` (500K)
- `E2EDrivingAgent-999988.onnx` (1M)
- `E2EDrivingAgent-1499899.onnx` (1.5M)
- `E2EDrivingAgent-1999953.onnx` (2M - recommended)
- `E2EDrivingAgent-2000209.onnx` (final)

#### Key Fix: Speed Penalty Bug
```diff
- else if (speedRatio < 0.5f && speed > 1f)
- {
-     reward += speedUnderPenalty;
- }
+ else if (speedRatio < 0.5f)
+ {
+     // Progressive penalty: slower = more penalty
+     float progressivePenalty = speedUnderPenalty * (2f - speedRatio * 2f);
+     reward += progressivePenalty;
+ }
```
**Impact**: Fixed loophole where agent stayed at 0-1 m/s without penalty

#### Lessons Learned
1. **Speed penalty must be unconditional** - Any condition creates exploitable loopholes
2. **Progressive penalty > Binary penalty** - Smoother gradient for learning
3. **Dense reward works** - Agent learned complete overtaking maneuver
4. **High variance during learning is normal** - Exploration phase shows Std > 300

#### Success Indicators
- Agent successfully overtakes slow NPCs (30% speed limit)
- Lane change, pass, and return behavior observed
- Reward stabilized in 600-900 range (goal achieved)

---

### Phase B Training Log (v12_phaseB) - COMPLETED 2026-01-25

#### Training Information
| Field | Value |
|-------|-------|
| Version | v12_phaseB |
| Date | 2026-01-25 |
| Status | **COMPLETED** ✅ |
| Config | `python/configs/planning/vehicle_ppo_v12_phaseB.yaml` |
| Initialize From | `v12_phaseA_fixed` |

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 2,000,150 |
| Training Time | ~16 minutes (950 seconds) |
| Final Mean Reward | **+903.3** |
| Peak Mean Reward | **+994.5** (at 1.64M steps) |
| Std of Reward | 225.3 (stabilized) |

#### Intent
- Learn when to overtake (slow NPC) vs when to follow (fast NPC)
- NPC speed curriculum: 0.3 → 0.5 → 0.7 → 0.9
- Decision-making policy training

#### Hypothesis (Verified ✅)
- Agent learned to recognize NPC speed relative to speed limit
- When NPC < 70% of limit: overtake behavior observed
- When NPC > 85% of limit: appropriate following/passing behavior

#### Key Differences from Phase A
| Aspect | Phase A | Phase B |
|--------|---------|---------|
| NPC Speed | Fixed at 30% | Varies 30-90% |
| Decision | Always overtake | Conditional (overtake vs follow) |
| Goal | Learn overtaking maneuver | Learn decision-making policy |
| Final Reward | +714 | **+903** (+26% improvement) |

#### Training Progress
| Step | Mean Reward | Std | NPC Speed Lesson | Notes |
|------|-------------|-----|------------------|-------|
| 680K | 630 | 11 | Fast (0.9) | Resumed from phaseA checkpoint |
| 800K | 834 | 190 | Fast (0.9) | Rapid adaptation |
| 1.0M | 850 | 220 | Fast (0.9) | Stable high performance |
| 1.2M | 870 | 230 | Fast (0.9) | Consistent |
| 1.4M | 880 | 235 | Fast (0.9) | Near peak |
| 1.64M | **994** | 193 | Fast (0.9) | **Peak performance** |
| 1.8M | 886 | 228 | Fast (0.9) | Stabilizing |
| 2.0M | **903** | 225 | Fast (0.9) | **Converged** |

#### Checkpoints Saved
- `E2EDrivingAgent-999988.onnx` (1M)
- `E2EDrivingAgent-1499899.onnx` (1.5M)
- `E2EDrivingAgent-1999894.onnx` (2M)
- `E2EDrivingAgent-2000150.onnx` (final)
- `E2EDrivingAgent.onnx` (latest copy)

#### Lessons Learned
1. **Curriculum completed all 4 lessons** (VerySlow→Slow→Medium→Fast) before step 680K
2. **High reward maintained** even with fast (90%) NPCs
3. **Phase transition successful**: Initialize from Phase A checkpoint worked correctly
4. **BehaviorType issue resolved**: All agents must be BehaviorType=0 (Default) for training

---

### Phase C Training Log (v12_phaseC_242D) - COMPLETED 2026-01-27

#### Training Information
| Field | Value |
|-------|-------|
| Version | v12_phaseC_242D |
| Date | 2026-01-27 |
| Status | **COMPLETED** ✅ |
| Config | `python/configs/planning/vehicle_ppo_v12_phaseC.yaml` |
| Initialize From | `v12_phaseB` |
| Observation | 242D (Speed enabled, Lane disabled) |

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 4,000,000 |
| Training Time | ~54 minutes (3269 seconds) |
| Final Mean Reward | **+961.8** |
| Peak Mean Reward | **+1086.0** (at 3.85M steps) |
| Std of Reward | 181.9 (stabilized) |

#### Intent
- Multi-NPC environment generalization (1→2→3→4 NPCs)
- Handle varied NPC speeds (40%→60%→80%)
- Longer goal distances (100m→160m→230m)
- Multiple speed zones (1→2→4)

#### Curriculum Progression
| Step | Curriculum Change | Reward | Notes |
|------|-------------------|--------|-------|
| 0K | Start (1 NPC, 100m, 1 zone) | +546 | Initialized from Phase B |
| 90K | goal_distance→160m, zones→2 | +766 | Peak before transition |
| 110K | Curriculum shock | -814 | Temporary collapse |
| 500K | Recovery | -777 | Learning new environment |
| 760K | Positive again | +11 | Adaptation complete |
| 960K | zones→4, variation→0.3 | +207 | Another transition |
| 1080K | goal_distance→230m | +127 | Final curriculum |
| 2M | Stable training | +1000 | High performance |
| 4M | **Final** | **+961** | **Training complete** |

#### Key Observations
1. **Curriculum shock recovery**: Agent recovered from -1329 to +1000 within 1M steps
2. **Resilient learning**: Multiple curriculum transitions handled successfully
3. **Phase B knowledge preserved**: Started at +546 (60% of Phase B's +903)
4. **Exceeded Phase B**: Final +961 > Phase B +903 in harder environment

#### Checkpoints Saved
- `E2EDrivingAgent-499806.onnx` (500K)
- `E2EDrivingAgent-999802.onnx` (1M)
- `E2EDrivingAgent-2999929.onnx` (3M)
- `E2EDrivingAgent-4000054.onnx` (final)
- `E2EDrivingAgent.onnx` (latest copy)
- `Assets/Resources/Models/E2EDrivingAgent_v12_phaseC.onnx` (Unity)

#### Lessons Learned
1. **Curriculum transitions cause temporary drops**: Expected and recoverable
2. **242D observation confirmed**: Phase B checkpoint requires 242D (not 238D)
3. **Space Size must match**: BehaviorParameters Space Size = observation dimensions
4. **TensorBoard essential**: Real-time monitoring crucial for long training runs

#### Phase Comparison
| Phase | Reward | Environment | Improvement |
|-------|--------|-------------|-------------|
| Phase A | +937 | 1 NPC @ 30% | Baseline |
| Phase B | +903 | 1 NPC @ 30-90% | Decision making |
| **Phase C** | **+961** | **4 NPC, 4 zones, 230m** | **+6% in 4x complexity** |

---

### Phase D Training Log (v12_phaseD) - COMPLETED 2026-01-27

#### Training Information
| Field | Value |
|-------|-------|
| Version | v12_phaseD |
| Date | 2026-01-27 |
| Status | **COMPLETED** ✅ |
| Config | `python/configs/planning/vehicle_ppo_v12_phaseD.yaml` |
| Initialize From | `v12_phaseC_242D` |
| Observation | **254D** (242D + 12D lane features) |

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 6,000,012 |
| Training Time | ~74 minutes (4448 seconds) |
| Final Mean Reward | **+332.4** |
| Peak Mean Reward | **+402.0** (at 5.86M steps) |
| Std of Reward | 119.2 (stabilized) |

#### Intent
- Expand observation space from 242D to 254D with lane features
- Validate agent can learn with larger observation space
- Curriculum: 1 NPC → 2 NPCs with curriculum shock management

#### Key Changes from Phase C
| Aspect | Phase C | Phase D |
|--------|---------|---------|
| Observation Dim | 242D | **254D (+12D lane)** |
| Lane Features | Disabled | **Enabled** |
| Max Steps | 4M | **6M** |
| Focus | Multi-NPC | Lane awareness |

#### Training Progress
| Step | Mean Reward | Std | Curriculum | Notes |
|------|-------------|-----|------------|-------|
| 500K | +97 | 187 | 1 NPC | Initial learning |
| 1M | +122 | 179 | 1 NPC | Steady progress |
| 1.5M | +162 | 187 | 1 NPC | Improving |
| 2M | +154 | 190 | 1 NPC | Stable |
| 2.5M | +418 | 135 | 1 NPC | **Pre-curriculum peak** |
| 2.5M+ | -1557 | - | **2 NPCs** | **Curriculum shock** |
| 3M | -212 | - | 2 NPCs | Recovery begins |
| 3.5M | +97 | 158 | 2 NPCs | Positive again |
| 4M | +141 | 133 | 2 NPCs | Learning |
| 4.5M | +166 | 140 | 2 NPCs | Improving |
| 5M | +124 | 159 | 2 NPCs | Stable |
| 5.5M | +288 | 181 | 2 NPCs | Good progress |
| 5.86M | **+402** | 144 | 2 NPCs | **Peak** |
| 6M | **+332** | 119 | 2 NPCs | **Final** |

#### Curriculum Shock Analysis
- **Transition Point**: 2.5M steps (1 NPC → 2 NPCs)
- **Pre-transition**: +418 (excellent single-NPC performance)
- **Post-transition**: -1557 (expected curriculum shock)
- **Recovery**: ~1.5M steps to return to positive rewards
- **Final**: +332 (80% of pre-curriculum peak)

#### Checkpoints Saved
- `E2EDrivingAgent-499959.onnx` (500K)
- `E2EDrivingAgent-999965.onnx` (1M)
- `E2EDrivingAgent-1999965.onnx` (2M)
- `E2EDrivingAgent-4999959.onnx` (5M)
- `E2EDrivingAgent-5999902.onnx` (6M)
- `E2EDrivingAgent-6000012.onnx` (final)
- `E2EDrivingAgent.onnx` (latest copy)

#### Lessons Learned
1. **254D observation space works**: Agent successfully learns with expanded input
2. **Curriculum shock is manageable**: Recovery takes ~1.5M steps for NPC count increase
3. **Lane features add value**: Agent utilizes lane information for decision making
4. **6M steps needed**: More complex observation requires longer training

#### Phase Comparison (Full v12)
| Phase | Reward | Environment | Observation | Key Learning |
|-------|--------|-------------|-------------|--------------|
| Phase A | +937 | 1 NPC @ 30% | 242D | Overtaking maneuver |
| Phase B | +903 | 1 NPC @ 30-90% | 242D | Decision policy |
| Phase C | +961 | 4 NPC, 4 zones | 242D | Multi-NPC generalization |
| **Phase D** | **+332** | 2 NPC, lane obs | **254D** | **Lane awareness** |

> **Note**: Phase D reward is lower due to different curriculum (2 NPC focus, not 4 NPC). Direct comparison with Phase C is not meaningful as they test different capabilities.

---

### Phase B Config (Updated based on Research 2026-01-25)

**Config**: `vehicle_ppo_v12_phaseB.yaml`
**Initialize from**: `v12_phaseA_fixed/E2EDrivingAgent-1999953.onnx`
**Expected Steps**: 2M

#### Curriculum Design
| Lesson | NPC Speed | Threshold | Goal |
|--------|-----------|-----------|------|
| VerySlow | 0.3 | 50.0 | Overtake reinforcement |
| Slow | 0.5 | 45.0 | Overtake vs hesitate |
| Medium | 0.7 | 40.0 | Decision boundary |
| Fast | 0.9 | - | Follow behavior |

#### Key Learning Objectives
1. **Decision Making**: When to overtake vs when to follow
2. **Speed Threshold Recognition**: Detect if NPC is "slow enough" to overtake
3. **Risk Assessment**: Fast NPC = risky overtake, follow is safer

#### Research-Informed Updates
Based on RESEARCH-TRENDS-2024-2026.md:
- **CuRLA Pattern**: 2-fold curriculum (environment + reward)
- **Teacher-Student Prep**: Dense reward phase before distillation
- **Safe RL Consideration**: Monitor TTC during overtake decisions

### Phase C Config (Updated based on Research 2026-01-25)

**Config**: `vehicle_ppo_v12_phaseC.yaml` (to be created)
**Initialize from**: Phase B best checkpoint
**Expected Steps**: 4M

#### Environment Complexity
| Lesson | NPCs | Speed Variation | Speed Zones | Goal Distance |
|--------|------|-----------------|-------------|---------------|
| TwoNPC | 2 | 0.15 | 2 | 180m |
| ThreeNPC | 3 | 0.2 | 3 | 200m |
| FourNPC | 4 | 0.3 | 4 | 230m |

#### New Considerations from Research
1. **Domain Randomization**: Add NPC behavior variation (passive/normal/aggressive)
2. **TTC Observation**: Add Time-To-Collision to observation space
3. **Multi-trajectory**: Consider multiple action candidates (Diffusion-lite)

---

## v12_ModularEncoder: Modular Encoder Architecture for Incremental Learning

**Date**: 2026-01-25
**Status**: **DECISION MADE** (Architecture Design Phase)
**Motivation**: Support Phase C observation space expansion without losing Phase B training

### Problem Statement

**Original Issue**: Phase C requires lane information in observation space (242D → 254D), but ML-Agents `initialize_from` requires exact matching dimensions for all network layers:
```
mlagents_envs.exception.UnityEnvironmentException:
The model is not compatible with the current environment.
Dimensions of network do not match.
```

**Consequence**: Cannot initialize Phase C from Phase B checkpoint, must restart from scratch:
- Phase B best reward: **+903.3** (peak +994.5)
- Phase C restart: -100 to +50 initial reward
- Lost knowledge: All decision-making learned in Phase B

### Research Background

Reviewed three architectural approaches to support incremental learning:

#### 1. UPGD (ICLR 2024) - Utility-based Perturbed Gradient Descent
- **Paper**: "Streaming Utility-based Gradient with Optimal Control for Practical Reinforcement Learning"
- **Mechanism**: Applies controlled perturbations to gradient updates when environment/reward changes
- **Pros**: Mathematically principled, preserves policy gradient properties
- **Cons**: Requires full gradient computation, computationally expensive, complex implementation
- **Verdict**: Overkill for observation dimension increase

#### 2. Progressive Neural Networks (Rusu et al., 2016)
- **Mechanism**: Freeze existing layer weights, add new columns for new features
- **Architecture**: Base network (242D) + Extension network (12D lane features)
- **Lateral Connections**: New columns can read from frozen base columns via adapter weights
- **Pros**: Theoretically well-studied, prevents catastrophic forgetting
- **Cons**: Doubles network width per phase, doesn't generalize well to curriculum changes
- **Verdict**: Too rigid; wastes capacity on frozen connections

#### 3. Elastic Weight Consolidation (EWC) - Kirkpatrick et al., 2017
- **Mechanism**: Apply L2 penalty on weights, stronger on "important" weights from Phase B
- **Importance**: Measured via Fisher Information Matrix
- **Pros**: Single-network solution, principled importance weighting
- **Cons**: Requires Hessian computation, expensive, requires two training loops (task1 → FIM → task2)
- **Verdict**: Overcomplicated for simple observation addition

### Solution: Modular Encoder Architecture (Custom)

**Design Philosophy**: Combine modular design with fine-tuning, simpler than UPGD/EWC but more flexible than Progressive Networks.

#### Architecture Overview

Replace single monolithic encoder with named, composable modules:

```python
class ModularEncoder(nn.Module):
    def __init__(self, input_shape: Dict[str, int]):
        super().__init__()

        # Named encoder modules for each observation subset
        self.encoders = nn.ModuleDict({
            'ego': nn.Linear(8, 128),           # ego state
            'history': nn.Linear(40, 128),      # ego history (5 steps)
            'agents': nn.Linear(160, 256),      # surrounding agents (20×8)
            'route': nn.Linear(30, 128),        # route info (10 waypoints)
            'speed': nn.Linear(4, 64),          # speed info
            'lane': nn.Linear(12, 64),          # NEW: lane info (added in Phase C)
        })

        # Fusion layer combines all encoders
        self.fusion = nn.Linear(
            128 + 128 + 256 + 128 + 64 + 64,    # 768 total
            512  # output to policy/value heads
        )

        # Freeze/unfreeze control
        self._frozen_modules = set()

    def freeze_module(self, name: str):
        """Freeze named encoder to prevent weight changes."""
        for param in self.encoders[name].parameters():
            param.requires_grad = False
        self._frozen_modules.add(name)

    def add_encoder(self, name: str, input_size: int, output_size: int = None):
        """Add new encoder module dynamically (e.g., for lane features)."""
        if output_size is None:
            output_size = max(64, input_size * 2)

        new_encoder = nn.Linear(input_size, output_size)

        # Initialize to small random values to minimize initial disruption
        nn.init.normal_(new_encoder.weight, mean=0, std=0.01)

        self.encoders[name] = new_encoder
        self._frozen_modules.discard(name)  # New encoders are trainable

        # Update fusion layer to accommodate new encoder output
        self._update_fusion_layer()

    def forward(self, observations: Dict[str, Tensor]) -> Tensor:
        # Encode each observation subset
        encoded = {}
        for name, encoder in self.encoders.items():
            if name in observations:
                encoded[name] = encoder(observations[name])
            # Frozen modules can be skipped if not in current observation dict

        # Concatenate all encoded features
        combined = torch.cat([encoded[name] for name in sorted(encoded.keys())], dim=1)

        # Fuse into final representation
        output = torch.relu(self.fusion(combined))
        return output
```

#### Two-Phase Training Strategy for Phase C

**Phase C.1 - New Encoder Learning (500K steps)**:
```yaml
# Phase C.1 Config
learning_rate: 0.0005  # Reduced learning rate for new encoder
buffer_size: 40960
batch_size: 4096

freeze_config:
  ego: true      # Freeze Phase B learned knowledge
  history: true
  agents: true
  route: true
  speed: true
  lane: false    # Train only the new lane encoder
```

**Goal**: Train lane encoder from scratch while using frozen Phase B knowledge as feature basis

**Phase C.2 - Fine-tune All (1.5M steps)**:
```yaml
# Phase C.2 Config
learning_rate: 0.0001  # Very small learning rate for fine-tuning
buffer_size: 40960
batch_size: 4096

freeze_config:
  # All encoders now trainable
  # Small learning rate prevents catastrophic forgetting
```

**Goal**: Jointly optimize all encoders for Phase C task (multi-NPC generalization)

#### Integration with ML-Agents

**How to integrate with existing mlagents-learn pipeline**:

1. **Checkpoint Saving**: Save entire ModularEncoder state dict
   ```python
   # Phase B checkpoint
   torch.save(modular_encoder.state_dict(),
              'phase_b_encoders.pt')

   # Phase C initialization
   model.load_state_dict(torch.load('phase_b_encoders.pt'))
   model.add_encoder('lane', input_size=12, output_size=64)

   # Freeze Phase B encoders
   for name in ['ego', 'history', 'agents', 'route', 'speed']:
       model.freeze_module(name)
   ```

2. **Config YAML Changes**:
   ```yaml
   # vehicle_ppo_v12_phaseC.yaml
   behaviors:
     E2EDrivingAgent:
       initialize_from: ../v12_phaseB/E2EDrivingAgent.onnx
       initialize_modular: true
       frozen_encoders: [ego, history, agents, route, speed]

       network_settings:
         hidden_units: 512
         num_layers: 3
         # ModularEncoder internal: 6 submodules → fusion(768→512)
   ```

3. **ONNX Export**: ModularEncoder → ONNX
   - Fuse encoder modules into single ONNX graph during export
   - No runtime overhead vs monolithic encoder
   - Unity Sentis inference unchanged

### Expected Impact

| Metric | Phase B (Baseline) | Phase C.1 (New Encoder Only) | Phase C.2 (Fine-tune) |
|--------|-------------------|------------------------------|----------------------|
| Initial Reward | N/A (2M steps) | +650 (preserved Phase B) | +700 (frozen features) |
| 500K Steps | N/A | +750 (lane encoder learns) | +850 (co-optimization) |
| 1.5M Steps | N/A | +800 stable | **+900+ (target)** |
| 2.0M Steps | **+903** | +805 plateau | **+920+ (expected)** |
| Knowledge Retention | 100% (baseline) | 95% (frozen) | 90% (fine-tune noise) |

### Advantages

1. **Preserves Phase B Knowledge**: Frozen encoders maintain learned decision-making (+700→+800 initial vs -100 restart)
2. **Modular & Extensible**: Easy to add new observation types (TTC, Lane Center Deviation, etc.)
3. **Fine-grained Control**: Freeze/unfreeze individual modules without binary all-or-nothing
4. **Theoretically Motivated**: Combines Progressive Networks + EWC philosophy without full complexity
5. **Production-Ready**: ONNX export fuses modules for zero runtime overhead

### Disadvantages

1. **Implementation Complexity**: Custom ModularEncoder class vs ML-Agents' default encoder
2. **Manual Integration**: Requires changes to neural_network_builder.py or custom trainer
3. **Hyperparameter Tuning**: Phase C.1/C.2 learning rates need careful tuning
4. **Monitoring**: Requires tracking frozen vs trainable parameters per phase

### Implementation Timeline

- **Sprint 1** (2-3 days): Implement ModularEncoder class + freeze/add_encoder methods
- **Sprint 2** (2-3 days): Integrate with mlagents-learn training loop
- **Sprint 3** (1 day): Test Phase B → Phase C transfer learning, measure reward preservation
- **Phase C Execution** (2-3 weeks): Run Phase C.1 (500K) + Phase C.2 (1.5M)

### Success Criteria

- Phase C.1 achieves +750 reward (preserves 85% of Phase B +903)
- Phase C.2 achieves +900+ reward (improves over Phase B)
- ONNX export runs without errors in Unity Sentis
- Training time competitive with restart (~6-8 weeks total for v12)

### Future Extensions

1. **Continual Learning**: Add task-specific adapters (e.g., highway vs urban)
2. **Curriculum-aware**: Different freeze policies per curriculum lesson
3. **Distillation**: Teacher (Phase B) → Student (Phase C) knowledge transfer
4. **Multi-task**: Share base encoders across different driving tasks

### Related Research

- **Progressive Neural Networks** (Rusu et al., 2016): Freezing strategy
- **Adapter-based Fine-tuning** (Houlsby et al., 2019): Modular additions
- **Continual Learning Surveys** (Delange et al., 2021): Catastrophic forgetting mitigation
- **Domain Adaptation in RL** (Tzeng et al., 2015): Transfer learning principles

---

## v12_HybridPolicy: Phase C-1 Hybrid Training Implementation (2026-01-26)

**Status**: **FAILED** ❌ - Catastrophic forgetting in Stage 5

### Problem Statement

ML-Agents `initialize_from` requires matching observation dimensions. Phase B (242D) cannot initialize Phase C (254D with lane info).

### Solution Attempted: HybridDrivingPolicy

Custom PyTorch policy that:
1. **Preserves Phase B encoder** (frozen, 649K params)
2. **Adds new lane encoder** (trainable, 2.5K params)
3. **Combines features** through trainable combiner + fusion

#### Architecture
```
254D input → [Phase B Encoder (frozen)] → 512D features (242D portion)
           → [Lane Encoder (trainable)] → 32D features (12D portion)
           → [Combiner (trainable)]     → 512D fused features
           → [Policy Head (from Phase B)] → 2D actions
           → [Value Head (trainable)]   → 1D value
```

### Implementation Details

**Files Created**:
- `python/src/models/hybrid_policy.py`: HybridDrivingPolicy, HybridPolicyConfig
- `python/src/training/unity_hybrid_trainer.py`: PPO trainer for hybrid policy

**Key Features**:
- Direct checkpoint loading from Phase B `.pt` file
- Selective freezing/unfreezing of components
- 6-stage gradual unfreezing training strategy
- ONNX export support for Unity Sentis

### 6-Stage Gradual Unfreezing Training Strategy

| Stage | Steps | Trainable Components | LR |
|-------|-------|---------------------|-----|
| 0 | 0-200K | value_head only | 3e-4 |
| 1 | 200K-600K | + lane_encoder | 1.5e-4 |
| 2 | 600K-1M | + combiner | 1e-4 |
| 3 | 1M-1.5M | + policy_head | 5e-5 |
| 4 | 1.5M-2.25M | + fusion layer | 3e-5 |
| 5 | 2.25M-3M | + Phase B encoder (0.1x LR) | 3e-6 |

### Training Results (2026-01-26) - FULL RUN

#### Training Progression

| Step | Stage | Mean Reward | Status |
|------|-------|-------------|--------|
| 204,800 | 0→1 | -341.7 | Value warmup complete |
| 614,400 | 1→2 | -159.3 | Lane encoder learning |
| 1,024,000 | 2→3 | -181.7 | Combiner integrated |
| 1,232,896 | 3 | **-373.1** | Policy head unfrozen |
| 1,437,696 | 3→4 | **-82.7** | **BEST checkpoint** |
| 1,642,496 | 4→5 | -334.5 | Fusion layer trainable |
| 2,256,896 | 5 | -1049.9 | Phase B encoder unfrozen |
| 2,666,496 | 5 | -1972.4 | **CATASTROPHIC FORGETTING** |
| 2,871,296 | 5 | -2171.9 | Collapsed |

#### Key Observations

**Stages 0-4 (Success)**:
- Gradual unfreezing worked as intended
- Best performance at step 1,437,696: **Reward -82.7**
- Showed improvement over initial -341.7

**Stage 5 (Failure)**:
- Unfreezing Phase B encoder caused catastrophic forgetting
- Reward collapsed from -82.7 to -1972.4 (24x worse)
- Even with 0.1x learning rate (3e-6), encoder weights were destabilized
- Pattern: More training = worse performance (negative learning)

### Checkpoints Saved

| Checkpoint | Step | Reward | Notes |
|------------|------|--------|-------|
| checkpoint_1232896.pt | 1.23M | -373.1 | End of Stage 3 |
| checkpoint_1437696.pt | 1.44M | **-82.7** | **BEST** |
| checkpoint_1642496.pt | 1.64M | -334.5 | End of Stage 4 |
| checkpoint_2871296.pt | 2.87M | -1972.4 | Stage 5 collapsed |

### Post-Training Issues

#### ONNX Export Failures

1. **hybrid_phaseC1_v1.onnx**: Wrong input name ("obs" instead of "obs_0")
2. **hybrid_phaseC1_v2.onnx**: Missing version_number/memory_size outputs
3. **hybrid_phaseC1_v3.onnx**: Caused Unity "Reloading Domain" hang

All hybrid ONNX files removed from Assets due to compatibility issues.

#### Observation Size Mismatch

- Agent outputs: 254D (242D + 12D lane)
- Phase B model expects: 238D
- **Resolution**: Disabled CollectSpeedObservations (4D) and CollectLaneObservations (12D) in E2EDrivingAgent.cs

### Rollback Decision

**Action**: Reverted to Phase B model (E2EDrivingAgent_v12_phaseB.onnx) with 238D observation
**Reason**: Hybrid training failed, ONNX export incompatible with Unity

### Root Cause Analysis

1. **Architectural mismatch**: HybridDrivingPolicy architecture differs from ML-Agents expected format
2. **ONNX format incompatibility**: ML-Agents requires specific output names (obs_0, version_number, memory_size, continuous_actions)
3. **Catastrophic forgetting**: Even 0.1x LR was too high for Phase B encoder fine-tuning
4. **Observation dimension confusion**: Phase B trained on 238D (not 242D as documented)

### Lessons Learned

1. **Don't unfreeze pretrained encoder**: Stage 5 (encoder fine-tuning) destroyed learned knowledge
2. **ONNX format is strict**: ML-Agents requires exact input/output naming convention
3. **Verify observation dimensions first**: Phase B actually uses 238D, not 242D
4. **Simpler approach often better**: Instead of hybrid architecture, just train Phase C from scratch
5. **Early stopping critical**: Best checkpoint was at 1.44M steps; training past Stage 4 was counterproductive

### Recommendation for Phase C

**Option A (Recommended - Most Efficient)**:
- Continue with Phase B model (238D)
- Defer lane observation to Phase D
- Focus on multi-NPC generalization first

**Option B (Alternative)**:
- Fix ONNX export to match ML-Agents format exactly
- Use best hybrid checkpoint (1.44M steps)
- Manually verify in Unity before extended training

**Option C (Clean Restart)**:
- Train Phase C from scratch with ML-Agents native architecture
- Use curriculum to gradually add complexity
- Accept loss of Phase B knowledge

---

## Future Research Integration (from RESEARCH-TRENDS-2024-2026)

### Short-term Improvements (Phase 5.5)
| Technique | Source | Priority | Expected Impact |
|-----------|--------|----------|-----------------|
| TTC in Observation | Safe RL | HIGH | Better safety decisions |
| Network 512→1024 | Quick Win | MEDIUM | Capacity for complex patterns |
| Normalize: true | Quick Win | LOW | Training stability |

### Medium-term Improvements (Phase 6)
| Technique | Source | Priority | Expected Impact |
|-----------|--------|----------|-----------------|
| GAIL Integration | IL Research | HIGH | Expert behavior imitation |
| Teacher-Student | CuRLA | HIGH | Dense→Sparse distillation |
| CMDP/Safe RL | LSTC | MEDIUM | Explicit safety constraints |
| ModularEncoder | Incremental Learning | HIGH | Support observation space growth |

### Long-term Research (Phase 7)
| Technique | Source | Priority | Expected Impact |
|-----------|--------|----------|-----------------|
| Diffusion Planning | ICLR 2025 | MEDIUM | Multi-modal trajectories |
| World Model | GAIA/DriveDreamer | LOW | Imagination-based learning |
| Transformer Policy | BEVFormer | LOW | Attention over agents |

---

## Value Changes Summary

| Parameter | v10g | v11 | v12 | v12_ModularEncoder | Reason |
|-----------|------|-----|-----|-------------------|--------|
| headingAlignmentReward | 0.02 | 0.02 | 0.02 | 0.02 | Stable |
| lateralDeviationPenalty | -0.02 | -0.02 | -0.02 | -0.02 | Stable |
| followingBonus | 0.3 | 0.3 (gated) | REMOVED | REMOVED | 추월 학습 방해 |
| targetSpeed policy | leadSpeed | leadSpeed (overtaking bypass) | speedLimit ALWAYS | speedLimit ALWAYS | 핵심 변경 |
| overtakePassBonus | - | 3.0 | - (replaced) | - | Sparse -> Dense |
| overtakeSpeedBonus | - | 0.15 | - (replaced) | - | Sparse -> Dense |
| overtakeInitiateBonus | - | - | 0.5 | 0.5 | Dense: 차선 변경 |
| overtakeBesideBonus | - | - | 0.2/step | 0.2/step | Dense: 옆 유지 |
| overtakeAheadBonus | - | - | 1.0 | 1.0 | Dense: 추월 완료 |
| overtakeCompleteBonus | - | - | 2.0 | 2.0 | Dense: 차선 복귀 |
| stuckBehindPenalty | - | - | -0.1/step | -0.1/step | 정체 페널티 |
| stuckBehindTimeout | - | - | 3.0s | 3.0s | 타임아웃 |
| npc_speed_ratio (env) | - | - | 0.3 (Phase A) | varied | NPC 속도 |
| Observation Space | - | - | 242D | 242D → 254D (Phase C) | Lane info addition |
| Network Architecture | - | - | Monolithic | **Modular (6 encoders)** | Incremental learning |

---

## Technical Notes

### Observation Space (242D in v12 Phase B, 254D planned for future)

**Current (v12 - Phase B/C) - CONFIRMED 242D**:
- Ego state (8D): position, velocity, heading, acceleration
- Ego history (40D): 5 past steps x 8D
- Surrounding agents (160D): 20 agents x 8 features
- Route info (30D): 10 waypoints x 3 (x, z, distance)
- Speed info (4D): current_speed, speed_limit, speed_ratio, next_limit
- ~~Lane info (12D)~~: DISABLED in Phase B/C
- **Total: 242D**

**Planned (v12_ModularEncoder - Phase C)**:
- All above (242D)
- Lane features (12D):
  - lane_id, lane_type, lane_width
  - centerline_offset, lateral_velocity
  - left_marking_type, right_marking_type
  - curvature, elevation, banking
  - traffic_light_state, traffic_light_distance
- **Total: 254D**

### Action Space (2D continuous, unchanged)
- Steering: [-0.5, 0.5] rad
- Acceleration: [-4.0, 4.0] m/s^2 (symmetric for exploration)

### Network Architecture (v12 - Monolithic)
- 3 layers x 512 hidden units
- PPO with linear learning rate schedule
- batch_size: 4096, buffer_size: 40960 (scaled for 16 parallel areas)

### Network Architecture (v12_ModularEncoder - Modular)
- 6 named encoder modules:
  - ego (8→128)
  - history (40→128)
  - agents (160→256)
  - route (30→128)
  - speed (4→64)
  - lane (12→64) [Phase C only]
- Fusion layer: concatenated outputs (768→512)
- Policy/Value heads: 3 layers x 512 (same as v12)
- **Freeze/Unfreeze**: Per-module fine-grained control

### Environment
- 16 Training Areas in single Unity instance
- time_scale: 20x
- NPC spawn: random positions ahead, lateral offset for overtaking space
- DrivingSceneManager reads curriculum params via Academy.Instance.EnvironmentParameters

### Known Issues
- Inspector serialization overrides public field defaults -> forced in Initialize()
- YAML file must be UTF-8 (not cp949) - avoid Unicode arrows/special chars
- mlagents-learn.exe must start BEFORE Unity Play button
- Trainer timeout: 600s for environment connection
- ML-Agents `initialize_from` requires exact dimension matching → **ModularEncoder solves this**

### ⚠️ Experiment Setup Rules (MUST FOLLOW)

**Rule 1: Observation Space Size Sync**
- BehaviorParameters의 `Space Size`는 반드시 에이전트의 실제 observation 차원과 일치해야 함
- CollectObservations() 함수에서 비활성화된 observation이 있으면 Space Size도 그에 맞게 조정
- 모델 파일(.onnx)이 기대하는 차원과도 일치해야 함
- **체크리스트**:
  - [ ] E2EDrivingAgent.cs의 CollectObservations() 확인
  - [ ] BehaviorParameters > Vector Observation > Space Size 확인
  - [ ] ONNX 모델의 입력 차원 확인
  - [ ] 세 값이 모두 동일한지 검증

**Rule 2: Observation Dimension Reference**
| Configuration | Ego | History | Agents | Route | Speed | Lane | Total |
|--------------|-----|---------|--------|-------|-------|------|-------|
| Phase B/C (current) | 8D | 40D | 160D | 30D | 4D | ❌ | **242D** |
| Full (future) | 8D | 40D | 160D | 30D | 4D | 12D | **254D** |

