# Training Log Archive - 2026-01-27

> **Note**: This file contains archived training logs from v10g through v12 Phase G.
> Archived because v10g training was restarted from scratch.
> Original file: docs/TRAINING-LOG.md

---

# Training Log - E2EDrivingAgent RL Training History

## Overview

| Version | Focus | Steps | Best Reward | Status |
|---------|-------|-------|-------------|--------|
| v10g | Lane Keeping + NPC 공존 | 8M (4.97M effective) | ~40 (NPC4) | Plateau |
| v11 | Overtaking Reward (Sparse) | 8M | ~51 peak (NPC4) | Marginal improvement |
| v12 Phase A | Dense Overtaking (Slow NPC) | 2M | **+937** | **COMPLETED** |
| v12 Phase B | Overtake vs Follow Decision | 2M | **+903** | **COMPLETED** |
| **v12 Phase C** | **Multi-NPC Generalization** | **4M** | **+961 (peak +1086)** | **COMPLETED** |
| **v12 Phase D** | **Lane Observation (254D)** | **6M** | **+332 (peak +402)** | (Phase E로 대체) |
| **v12 Phase E** | **Curved Roads** | **6M** | **+931 (peak +931)** | **COMPLETED** |
| **v12 Phase F** | **Multi-Lane Roads** | **6M** | **+988 (peak +988)** | **COMPLETED** |
| **v12 Phase G** | **Intersection Navigation** | **8M** | **+461 (340K)** | **IN PROGRESS** |
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

## v12: Dense Overtaking + Strategy 3

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

### Phase A Results (v12_phaseA_fixed) - COMPLETED 2026-01-25

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 2,000,209 |
| Training Time | ~30 minutes |
| Final Mean Reward | **+714** (peak: +935) |
| Std of Reward | 234 (stabilized) |

### Phase B Training Log (v12_phaseB) - COMPLETED 2026-01-25

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 2,000,150 |
| Training Time | ~16 minutes (950 seconds) |
| Final Mean Reward | **+903.3** |
| Peak Mean Reward | **+994.5** (at 1.64M steps) |

### Phase C Training Log (v12_phaseC_242D) - COMPLETED 2026-01-27

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 4,000,000 |
| Training Time | ~54 minutes (3269 seconds) |
| Final Mean Reward | **+961.8** |
| Peak Mean Reward | **+1086.0** (at 3.85M steps) |

### Phase D Training Log (v12_phaseD) - COMPLETED 2026-01-27

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 6,000,012 |
| Training Time | ~74 minutes (4448 seconds) |
| Final Mean Reward | **+332.4** |
| Peak Mean Reward | **+402.0** (at 5.86M steps) |

### Phase E Training Log (v12_phaseE_v2) - COMPLETED 2026-01-27

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 6,000,090 |
| Training Time | ~70 minutes (4231 seconds) |
| Final Mean Reward | **+931.1** |
| Peak Mean Reward | **+931.1** (at 6M steps) |

### Phase F Training Log (v12_phaseF) - COMPLETED 2026-01-27

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | 6,000,000 |
| Training Time | ~70 minutes |
| Final Mean Reward | **+988** |
| Peak Mean Reward | **+988** (at 6M steps) |

### Phase G Training Log (v12_phaseG) - IN PROGRESS

#### Training Summary (Current)
| Metric | Value |
|--------|-------|
| Current Steps | ~340,000 / 8,000,000 (4.3%) |
| Current Reward | **+461** |
| Target Steps | 8,000,000 |

---

## v12_HybridPolicy: Phase C-1 Hybrid Training Implementation (2026-01-26)

**Status**: **FAILED** - Catastrophic forgetting in Stage 5

### Training Results
- Best checkpoint at 1.44M steps: **-82.7**
- Stage 5 (encoder unfrozen): collapsed to -1972.4
- ONNX export incompatible with Unity

---

## v11 TensorBoard Enhancement Plan (2026-01-27)

### Implementation (E2EDrivingAgent.cs)
- Step-level StatsRecorder logging (every 100 steps)
- Episode-level logging (on episode end)
- summary_freq: 5000 in v11 config

### New TensorBoard Metrics
- Reward/*: Progress, Speed, LaneKeeping, Overtaking, LaneViolation, Jerk, Time
- Stats/*: Speed, SpeedLimit, SpeedRatio, Acceleration, Steering, DistanceTraveled, StuckTimer
- Episode/*: Length, TotalReward, OvertakeCount, CollisionCount, EndReason_*

---

## Technical Notes

### Observation Space
- Phase B/C: 242D (Speed enabled, Lane disabled)
- Phase D onwards: 254D (+12D lane features)
- Phase G: 260D (+6D intersection info)

### Action Space
- Steering: [-0.5, 0.5] rad
- Acceleration: [-4.0, 4.0] m/s^2

### Network Architecture
- 3 layers x 512 hidden units
- PPO with linear learning rate schedule
- batch_size: 4096, buffer_size: 40960

---

*Archive Date: 2026-01-27*
*Reason: v10g training restarted from scratch*
