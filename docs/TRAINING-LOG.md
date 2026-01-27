# Training Log - E2EDrivingAgent RL Training History

> **Note**: Previous training logs (v10g~v12 Phase G) archived to `docs/archives/TRAINING-LOG-ARCHIVE-2026-01-27.md`
> Fresh start from v10g (2026-01-27)

## Overview

| Version | Focus | Steps | Best Reward | Status |
|---------|-------|-------|-------------|--------|
| **v10g** | Lane Keeping + NPC Coexistence | 8M | - | **IN PROGRESS** |
| v11 | TensorBoard Enhanced + Sparse Overtaking | 8M | - | Planned |
| v12 | Dense Overtaking + Phased Training | - | - | Planned |

---

## v10g: Lane Keeping + NPC Coexistence (Fresh Start)

### Training Information
| Field | Value |
|-------|-------|
| Version | v10g (Fresh) |
| Start Date | 2026-01-27 |
| Status | **IN PROGRESS** |
| Config | `python/configs/planning/vehicle_ppo_v10g.yaml` |
| Command | `mlagents-learn python/configs/planning/vehicle_ppo_v10g.yaml --run-id=v10g_test --force` |

### Intent
- Speed policy 기반 주행 + 차선 유지
- NPC 0 -> 1 -> 2 -> 4 환경에서 안정적 공존
- Heading alignment + lateral deviation 보상
- 3-strike collision rule

### Key Parameters
```yaml
headingAlignmentReward: 0.02
lateralDeviationPenalty: -0.02
followingBonus: 0.3
collisionPenalty: -5.0
```

### Curriculum
| Parameter | Lessons | Threshold |
|-----------|---------|-----------|
| num_active_npcs | 0 -> 1 -> 2 -> 4 | 60 -> 40 -> 30 |
| goal_distance | 50 -> 100 -> 160 -> 230 | 60 -> 40 -> 30 |
| speed_zone_count | 1 -> 2 -> 3 -> 4 | 60 -> 40 -> 30 |

### Training Progress

| Step | Mean Reward | Std | Curriculum | Notes |
|------|-------------|-----|------------|-------|
| - | - | - | - | Training started |

### Expected Milestones
| Milestone | Steps | Expected Reward | Notes |
|-----------|-------|-----------------|-------|
| NPC 0 Mastery | ~500K | 80-95 | Free driving |
| NPC 1 Transition | ~850K | 60-85 | First NPC |
| NPC 2 Transition | ~1.2M | 45-65 | Two NPCs |
| NPC 4 Plateau | ~1.5M+ | 35-45 | Final curriculum |
| Training Complete | 8M | ~40 | Expected plateau |

### Checkpoints
| Checkpoint | Step | Reward | Notes |
|------------|------|--------|-------|
| - | - | - | - |

---

## v11: TensorBoard Enhanced + Sparse Overtaking (Planned)

### Intent
- v10g 완료 후 전환
- StatsRecorder 기반 TensorBoard 로깅 강화
- Sparse overtaking reward 도입

### Key Changes from v10g
| Aspect | v10g | v11 |
|--------|------|-----|
| TensorBoard | Basic | **StatsRecorder Enhanced** |
| summary_freq | 10000 | **5000** |
| Overtake Bonus | None | **3.0 (sparse)** |

### New TensorBoard Metrics
- `Reward/*`: Progress, Speed, LaneKeeping, Overtaking, LaneViolation
- `Stats/*`: Speed, SpeedLimit, Acceleration, Steering
- `Episode/*`: Length, OvertakeCount, CollisionCount, EndReason_*

---

## v12: Dense Overtaking (Planned)

### Intent
- Dense 5-phase overtaking reward
- Phased curriculum (A -> B -> C -> ...)
- targetSpeed = speedLimit ALWAYS

### Phase Plan
| Phase | Focus | Steps |
|-------|-------|-------|
| A | Dense Overtaking (Slow NPC) | 2M |
| B | Overtake vs Follow Decision | 2M |
| C | Multi-NPC Generalization | 4M |
| D+ | Advanced Features | TBD |

---

## Technical Reference

### Observation Space (242D)
| Component | Dimensions |
|-----------|------------|
| Ego state | 8D |
| Ego history | 40D (5 steps) |
| Surrounding agents | 160D (20 agents x 8) |
| Route info | 30D (10 waypoints) |
| Speed info | 4D |
| **Total** | **242D** |

### Action Space (2D Continuous)
- Steering: [-0.5, 0.5] rad
- Acceleration: [-4.0, 4.0] m/s^2

### Network Architecture
- 3 layers x 512 hidden units
- PPO with linear LR schedule
- batch_size: 4096, buffer_size: 40960

### Environment
- 16 Training Areas
- time_scale: 20x
- timeout: 600s

---

## Analysis Template

v10g 학습 결과 공유 시 사용:

```
=== v10g Training Report ===
[Config] steps=XXXXX | lr=3e-4 | batch=4096
[Reward] mean=XX.X | std=XX.X | min=XX | max=XX
[Curriculum] npc=X | goal=XXXm | zones=X
[Episode] len=XXX | goal=XX% | collision=XX% | offroad=XX%
[Trend] reward direction | curriculum progress
[Issue] any problems observed
```

---

*Last Updated: 2026-01-27*
