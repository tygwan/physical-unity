# Autonomous Driving ML Platform

Unity ML-Agents 기반 자율주행 Motion Planning AI 학습 플랫폼

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 1-2 | ✅ Complete | Foundation & Data Infrastructure |
| Phase 3-4 | ⏸️ On Hold | Ground Truth / Constant Velocity 사용 |
| **Phase 5** | 🔄 **In Progress** | Planning Models (RL/IL) - PRIMARY FOCUS |
| Phase 6-7 | 📋 Planned | Integration & Advanced Topics |

**Current Training**: Phase E Completed (+931), Phase F Next (N차선)

---

## Training History & Results

### Policy Evolution Summary

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│  Early Experiments (Jan 22-24)                                                   │
│  3dball_test → driving_ppo_v1 → curriculum_v1~v9                                │
│       │              │               │                                           │
│       │              │               └─ Curriculum learning 기초 (reward shaping)│
│       │              └─ 첫 자율주행 시도 (실패: reward -4.9)                      │
│       └─ ML-Agents 환경 검증 (3D Ball: +100)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Main Training (Jan 24-27)                                                       │
│  v10g → v11 → v12 Phase A → Phase B → Phase C → Phase D → Phase E              │
│   │      │        │           │          │          │          │                 │
│   │      │        │           │          │          │          └─ 곡선 도로      │
│   │      │        │           │          │          └─ Lane Observation (254D)  │
│   │      │        │           │          └─ Multi-NPC Generalization (4 NPCs)   │
│   │      │        │           └─ Overtake vs Follow Decision                    │
│   │      │        └─ Dense Overtaking (Slow NPC)                                │
│   │      └─ Sparse Overtaking Reward                                            │
│   └─ Lane Keeping + NPC Coexistence                                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Early Experiments (Pre-v10g)

| Run ID | Date | Steps | Reward | Purpose | Outcome |
|--------|------|-------|--------|---------|---------|
| 3dball_test5 | Jan 22 | 500K | **+100** | ML-Agents 환경 검증 | ✅ 튜토리얼 성공 |
| driving_ppo_v1 | Jan 23 | 87K | -4.9 | 첫 자율주행 시도 | ❌ 기본 주행 불가 |
| curriculum_v1~v3 | Jan 24 | ~17K | - | Curriculum 구조 테스트 | ⚠️ 설정 조정 |
| curriculum_v4 | Jan 24 | 25K | - | Reward shaping 개선 | ⚠️ 수렴 불안정 |
| curriculum_v5 | Jan 24 | 290K | **+275** | 첫 성공적 학습 | ✅ 기본 주행 성공 |
| curriculum_v6_parallel | Jan 24 | 2M | - | 병렬 환경 테스트 | ⚠️ 속도 향상 확인 |
| curriculum_v7_speed | Jan 24 | 3.5M | -12 | Speed zone 도입 | ❌ 속도 적응 실패 |
| curriculum_v8_gradual | Jan 24 | 285K | -3.4 | 점진적 난이도 | ⚠️ 개선 필요 |
| curriculum_v9_speed | Jan 24 | - | - | Speed policy 개선 | ⚠️ v10 시리즈로 이어짐 |
| curriculum_v10a~f | Jan 24 | - | - | Traffic + NPC 시리즈 | ⚠️ 반복 개선 |

### Main Training Results

| Phase | Steps | Best Reward | Final Reward | Status | Key Achievement |
|-------|-------|-------------|--------------|--------|-----------------|
| v10g | 8M | +95 (NPC0) | +40 (NPC4) | ✅ | Lane keeping, NPC avoidance |
| v11 | 8M | +51 | +41 | ⚠️ | Sparse reward insufficient |
| **v12 Phase A** | 2M | **+937** | +714 | ✅ | Learned overtaking maneuver |
| **v12 Phase B** | 2M | **+994** | +903 | ✅ | Overtake/follow decision |
| **v12 Phase C** | 4M | **+1086** | +961 | ✅ | 4-NPC generalization |
| **v12 Phase D** | 6M | **+402** | +332 | ✅ | Lane observation (254D) |
| **v12 Phase E** | 6M | **+931** | +931 | ✅ | Curved roads, 2 NPCs |
| v12_HybridPolicy | 3M | -82 | -2172 | ❌ | Catastrophic forgetting |

### Phase Details

#### v10g: Lane Keeping + NPC Coexistence
- **Intent**: Speed policy 기반 주행 + 차선 유지
- **Problem**: Agent "follows" slow NPCs indefinitely (no overtaking)
- **Lesson**: `followingBonus` rewards "not crashing" - 추월 동기 부재

#### v11: Sparse Overtaking Reward
- **Intent**: 느린 NPC 추월 학습 (sparse reward)
- **Problem**: `targetSpeed = leadSpeed` 구조적 문제
- **Lesson**: Sparse reward만으로는 추월 학습 불가

#### v12 Phase A: Dense Overtaking (Slow NPC)
- **Changes**:
  - `targetSpeed = speedLimit ALWAYS`
  - `followingBonus` 제거
  - Dense 5-phase overtaking reward
- **Result**: +937 peak, 추월 동작 학습 성공
- **Bug Fix**: Speed penalty 조건문 버그 수정

#### v12 Phase B: Overtake vs Follow Decision
- **Curriculum**: NPC speed 0.3 → 0.5 → 0.7 → 0.9
- **Result**: +994 peak, 조건부 추월/따라가기 판단 학습
- **Improvement**: +26% over Phase A

#### v12 Phase C: Multi-NPC Generalization
- **Environment**: 1→2→3→4 NPCs, 230m goal, 4 speed zones
- **Curriculum Shock**: +766 → -814 → +1086 (recovery success)
- **Result**: +6% improvement in 4x complexity

#### v12 Phase D: Lane Observation (254D)
- **Changes**: 242D → 254D (12D lane features added)
- **Curriculum**: 1→2 NPCs with curriculum shock recovery
- **Training**: 6M steps, +402 peak, +332 final
- **Result**: Successfully learned with expanded observation space

#### v12 Phase E: Curved Roads (Completed ✅)
- **Goal**: 곡선 도로에서 안정적 주행 학습
- **Results**: 6M steps, **+931 reward** (all curriculum passed)
- **Achievements**:
  - Sharp curves (curvature 1.0) 마스터
  - Mixed left/right curve directions
  - 2 NPCs on curved roads
  - 200m goal distance on curves
- **Curriculum Completed**: Straight → Gentle → Moderate → Sharp curves ✅

#### v12_HybridPolicy: Incremental Learning Attempt (FAILED)
- **Goal**: Preserve Phase B knowledge while adding lane encoder
- **Method**: Freeze Phase B encoder, train new lane encoder
- **Failure**: Stage 5 (encoder fine-tuning) caused catastrophic forgetting
- **Lesson**: Don't unfreeze pretrained encoder even with low LR

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   AUTONOMOUS DRIVING ML PLATFORM                     │
├─────────────────────────────────────────────────────────────────────┤
│                        Windows 11 Native                             │
│  ┌────────────────────┐   ┌────────────────────┐                    │
│  │   Unity 6 (6000.x) │   │    Python 3.10.11  │                    │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │                    │
│  │  │ ML-Agents    │  │◄─►│  │ PyTorch 2.3  │  │                    │
│  │  │ 4.0.1        │  │   │  │ mlagents 1.2 │  │                    │
│  │  └──────────────┘  │   │  └──────────────┘  │                    │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │                    │
│  │  │ Sentis 2.4.1 │  │   │  │ TensorBoard  │  │                    │
│  │  │ ONNX Infer.  │  │   │  │ MLflow       │  │                    │
│  │  └──────────────┘  │   │  └──────────────┘  │                    │
│  └────────────────────┘   └────────────────────┘                    │
├─────────────────────────────────────────────────────────────────────┤
│                       OBSERVATION SPACE                              │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Ego (8D) │ History (40D) │ Agents (160D) │ Route (30D) │        ││
│  │ Speed (4D) │ Lane (12D) │ → Total: 254D                         ││
│  └─────────────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────────────┤
│                         ACTION SPACE                                 │
│  Steering: [-0.5, +0.5] rad   │   Acceleration: [-4.0, +4.0] m/s²   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Environment

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Windows 11 | Native (WSL 미사용) |
| Unity | 6000.3.4f1 | Unity 6 LTS |
| ML-Agents | 4.0.1 | Unity Package |
| Sentis | 2.4.1 | ONNX Inference |
| Python | 3.10.11 | Windows Native |
| PyTorch | 2.3.1 | CUDA 12.x |
| mlagents | 1.2.0 | Python Package |
| GPU | RTX 4090 | 24GB VRAM |

---

## Project Structure

```
physical-unity/
├── .claude/                    # Claude Code 설정
├── Assets/                     # Unity 프로젝트
│   ├── Scripts/
│   │   ├── Agents/            # E2EDrivingAgent.cs
│   │   ├── Environment/       # DrivingSceneManager.cs
│   │   └── Sensors/           # CameraSensor, LiDARSensor
│   └── Resources/Models/      # ONNX 모델 파일
├── docs/
│   ├── PRD.md                 # 제품 요구사항
│   ├── TRAINING-LOG.md        # 학습 실험 기록 (상세)
│   ├── LEARNING-ROADMAP.md    # RL/IL 학습 로드맵
│   └── phases/                # Phase별 기술 문서
├── python/
│   ├── configs/planning/      # 학습 설정 YAML
│   └── src/
│       ├── models/            # PyTorch 모델
│       └── training/          # 학습 스크립트
├── results/                   # 학습 결과 (TensorBoard)
│   ├── v12_phaseA/
│   ├── v12_phaseB/
│   ├── v12_phaseC/
│   └── v12_phaseD/
└── models/planning/           # 최종 ONNX 모델
```

---

## Quick Start

### Training

```powershell
# Windows PowerShell
cd C:\Users\user\Desktop\dev\physical-unity

# Phase E 학습 (현재 진행중)
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseE.yaml --run-id=v12_phaseE

# Unity Editor에서 Play 버튼 클릭
```

### Monitoring

```powershell
# TensorBoard
tensorboard --logdir=results

# 브라우저에서 http://localhost:6006 접속
```

### Inference (Unity)

1. `results/<run-id>/E2EDrivingAgent.onnx` → `Assets/Resources/Models/` 복사
2. BehaviorParameters > Model에 할당
3. BehaviorType을 "Inference Only"로 변경
4. Play

---

## Reward Design (v12)

```yaml
# Per-step rewards
speed_compliance:     +0.3   # 80-100% of speed limit
speed_over_limit:     -0.5 ~ -3.0  # Progressive penalty
stuck_behind_npc:     -0.1   # After 3 seconds

# Overtaking rewards (one-time)
overtake_initiate:    +0.5   # Lane change started
overtake_beside:      +0.2/step  # Maintaining speed beside NPC
overtake_ahead:       +1.0   # Passed NPC
overtake_complete:    +2.0   # Returned to lane

# Penalties (one-time)
collision:            -5.0   # 3-strike rule
off_road:             -5.0   # Episode end
```

---

## Key Lessons Learned

### What Worked
1. **Dense Reward > Sparse Reward**: 5-phase overtaking reward enabled learning
2. **targetSpeed = speedLimit ALWAYS**: Critical for overtaking behavior
3. **Curriculum Learning**: Gradual complexity increase (NPC count, speed variation)
4. **Curriculum Shock Recovery**: Temporary drops are normal and recoverable
5. **Iterative Improvement**: v1 → v10g 과정에서 수십 번의 시행착오가 필수

### What Failed
1. **followingBonus**: Discouraged overtaking attempts
2. **Sparse overtakePassBonus**: Insufficient learning signal
3. **Hybrid Policy Encoder Fine-tuning**: Catastrophic forgetting at Stage 5
4. **ONNX Custom Format**: ML-Agents requires specific output names
5. **급격한 환경 변화**: curriculum_v7에서 speed zone 갑자기 도입 → 학습 붕괴

### Early Phase Insights (Pre-v10g)
| Problem | Attempted | Result |
|---------|-----------|--------|
| "Agent doesn't move" | driving_ppo_v1 | Observation/Action 연결 문제 |
| "Reward가 수렴 안됨" | curriculum_v1~v4 | Reward shaping 필요 |
| "학습이 너무 느림" | curriculum_v6_parallel | 병렬 환경으로 3x 속도 향상 |
| "Speed zone 적응 실패" | curriculum_v7_speed | 점진적 도입 필요 (v10 시리즈로 해결) |

### Best Practices
1. **Always verify observation dimensions**: BehaviorParameters Space Size = Agent output = ONNX input
2. **Monitor TensorBoard in real-time**: Catch issues early
3. **Save checkpoints frequently**: Best model may not be final model
4. **Don't unfreeze pretrained encoders**: Use very low LR or keep frozen

---

## Next Steps (Phase F+)

| Phase | Focus | Status |
|-------|-------|--------|
| **E** | 곡선 도로 + 비정형 각도 | ✅ **Completed (+931)** |
| **F** | N차선 + 중앙선 규칙 | 🔄 **Next** |
| **G** | 교차로 (T자/십자/Y자) | 📋 Planned |
| **H** | 신호등 + 정지선 | 📋 Planned |
| **I** | U턴 + 특수 기동 | 📋 Planned |
| **J** | 횡단보도 + 보행자 | 📋 Planned |
| **K** | 장애물 + 긴급 상황 | 📋 Planned |
| **L** | 복합 시나리오 통합 | 📋 Planned |

---

## Documentation

- [PRD (Product Requirements)](docs/PRD.md)
- [Training Log (Detailed)](docs/TRAINING-LOG.md)
- [Learning Roadmap](docs/LEARNING-ROADMAP.md)
- [Phase Documents](docs/phases/README.md)
- [Progress Tracking](docs/PROGRESS.md)

---

## References

- [Unity ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

---

**Last Updated**: 2026-01-27 | **Phase E Completed** | Phase E Reward: +931
