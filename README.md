# Autonomous Driving ML Platform

Unity ML-Agents 기반 자율주행 Motion Planning AI 학습 플랫폼

> **Development Infrastructure**: This project uses [cc-initializer](https://github.com/tygwan/cc-initializer) for Claude Code workflow automation, including custom agents, skills, hooks, and development lifecycle management.

## Project Status

| Component | Status | Notes |
|-----------|--------|-------|
| Phase 1-2 | ✅ Complete | Foundation & Data Infrastructure |
| Phase 3-4 | ⏸️ On Hold | Ground Truth / Constant Velocity 사용 |
| **Phase 5** | 🔄 **In Progress** | Planning Models (RL/IL) - PRIMARY FOCUS |
| Phase 6-7 | 📋 Planned | Integration & Advanced Topics |

**Current Training**: Phase G (Intersection) 학습 중 - 340K steps, +461 reward

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

| Phase | Base Checkpoint | Steps | Best Reward | Final Reward | Status | Key Achievement |
|-------|-----------------|-------|-------------|--------------|--------|-----------------|
| **Phase 0** | From scratch | 8M | **+1018** | +1018 | ✅ | Lane keeping, NPC coexistence |
| **Phase A** | Phase 0 | 2.5M | **+2113** | +2113 | ✅ | Overtaking mastery |
| **Phase B v1** | Phase 0 ⚠️ | 3M | -108 | -108 | ❌ | **FAILED** - Wrong checkpoint + reward bug |
| **Phase B v2** | Phase A ✅ | 1M | **+877** | +877 | ✅ | Decision learning (recovery) |
| **Phase C** | Phase B v2 | 3.6M | **+1,372** | +1,372 | ✅ | Multi-NPC (8 NPCs), perfect safety |
| **Phase D v1** | Phase C | 6M | +406 | -2,156 | ❌ | **FAILED** - Curriculum collapse (3 params simultaneous) |
| **Phase D v2** | Phase C | TBD | TBD | TBD | 📋 | Lane observation retry (single-param progression) |

**Legacy Results (Old Naming)**:
| Phase | Steps | Best Reward | Final Reward | Status | Key Achievement |
|-------|-------|-------------|--------------|--------|-----------------|
| v10g | 8M | +95 (NPC0) | +40 (NPC4) | ✅ | Lane keeping, NPC avoidance |
| v11 | 8M | +51 | +41 | ⚠️ | Sparse reward insufficient |
| v12 Phase A (old) | 2M | +937 | +714 | ✅ | Learned overtaking maneuver |
| v12 Phase B (old) | 2M | +994 | +903 | ✅ | Overtake/follow decision |
| v12 Phase C (old) | 4M | +1086 | +961 | ✅ | 4-NPC generalization |
| v12 Phase D (old) | 6M | +402 | +332 | ⏭️ | (Phase E로 대체) |
| v12 Phase E (old) | 6M | +931 | +931 | ✅ | Curved roads, 2 NPCs |
| v12 Phase F (old) | 6M | +988 | +988 | ✅ | Multi-lane roads |
| v12 Phase G (old) | 8M | +461 | 🔄 | 🔄 | Intersection navigation |
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

#### Phase B v1: Decision Learning (FAILED)
- **Start**: Phase 0 checkpoint (+1018) ⚠️ Wrong choice
- **Environment**: 2 NPCs, decision-making curriculum
- **Duration**: 39.4 minutes, 3M steps
- **Result**: **-108 reward** (catastrophic failure)
- **Root Cause**:
  1. Reward bug: `followingPenalty` too harsh (-0.5/step)
  2. Wrong checkpoint: Phase 0 lacks overtaking capability
  3. Curriculum shock: 0 NPC → 2 NPC too abrupt
- **Lesson**: Always resume from most capable checkpoint

#### Phase B v2: Decision Learning (Recovery SUCCESS)
- **Start**: Phase A checkpoint (2.5M steps, +2113) ✅ Correct
- **Curriculum**: 1→2→3→4 NPCs (gradual increase)
- **Duration**: ~1 hour, 1M additional steps (total 3.5M)
- **Result**: **+877 peak** (146% of target +600)
- **Improvement**:
  - Fixed reward function (removed harsh penalty)
  - Leveraged Phase A's overtaking capability
  - 4-stage curriculum prevented shock
- **Success Rate**: 100% goal completion, 0% collision

#### Phase C: Multi-NPC Generalization (4-8 NPCs)
- **Start**: Phase B v2 checkpoint (+877)
- **Environment**: 8 NPCs, complex multi-agent scenarios
- **Duration**: ~50 minutes, 3.6M steps
- **Result**: **+1,372 reward** (228% of target +600)
- **Achievement**:
  - Perfect safety (0% collision)
  - Robust generalization to 8 concurrent NPCs
  - Maintained high performance across complexity
- **Innovation**: Multi-agent decision-making at scale

#### Phase D v1: Lane Observation (FAILED)
- **Start**: Phase C checkpoint (+1,372)
- **Innovation**: Added 12D lane observation (242D → 254D)
  - Explicit lane boundaries (left/right distances at 4 positions)
  - Faster convergence for lane-keeping
  - Preparation for curved roads (Phase E)
- **Duration**: 100 minutes, 6M steps
- **Peak**: **+406 at 4.6M steps** (promising start)
- **Collapse**: **-2,156 final** (catastrophic failure)
- **Root Cause**:
  1. 3 curriculum parameters transitioned simultaneously at 4.68M:
     - num_active_npcs: 1 → 2
     - speed_zone_count: 1 → 2
     - npc_speed_variation: 0 → 0.3
  2. Agent's scenario-specific policies became invalid
  3. Collapse: +406 → -4,825 in <20K steps (-5,231 points)
- **Lessons**:
  - Curriculum parameters are NOT independent
  - Peak reward ≠ robust learning
  - Simultaneous transitions = exponential complexity
- **Recovery**: Phase D v2 planned with single-parameter progression

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

#### v12 Phase F: Multi-Lane Roads (Completed ✅)
- **Goal**: 다중 차선 도로에서 주행 학습
- **Results**: 6M steps, **+988 reward** (all curriculum passed)
- **Achievements**:
  - 1→2 차선 도로 마스터
  - 중앙선 규칙 학습
  - 곡선 + 다차선 복합 환경
  - 3 NPCs on multi-lane roads
- **Curriculum Completed**: SingleLane → TwoLanes → CenterLine ✅

#### v12 Phase G: Intersection Navigation (In Progress 🔄)
- **Goal**: 교차로 (T자/십자/Y자) 주행 학습
- **Current**: 340K steps, **+461 reward**
- **Target**: 8M steps
- **Curriculum**: NoIntersection → T-Junction → Cross → Y-Junction
- **Turn Direction**: Straight → Left → Right

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
│   ├── phase-A/
│   ├── phase-B/
│   ├── phase-C/
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
mlagents-learn python/configs/planning/vehicle_ppo_phase-E.yaml --run-id=phase-E

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

## Next Steps (Phase H+)

| Phase | Focus | Status |
|-------|-------|--------|
| **E** | 곡선 도로 + 비정형 각도 | ✅ **Completed (+931)** |
| **F** | N차선 + 중앙선 규칙 | ✅ **Completed (+988)** |
| **G** | 교차로 (T자/십자/Y자) | 🔄 **In Progress (340K, +461)** |
| **H** | 신호등 + 정지선 | 📋 Next |
| **I** | U턴 + 특수 기동 | 📋 Planned |
| **J** | 횡단보도 + 보행자 | 📋 Planned |
| **K** | 장애물 + 긴급 상황 | 📋 Planned |
| **L** | 복합 시나리오 통합 | 📋 Planned |

---

## Development Infrastructure

This project uses [cc-initializer](https://github.com/tygwan/cc-initializer) for automated development workflows with Claude Code.

### Agents (38)

**Core Framework Agents (26)**
| Category | Agents | Purpose |
|----------|--------|---------|
| **Documentation** | dev-docs-writer, doc-generator, doc-splitter, doc-validator, prd-writer, tech-spec-writer, readme-helper | 문서 생성 및 검증 |
| **Project Management** | progress-tracker, phase-tracker, project-analyzer, project-discovery, work-unit-manager | 프로젝트 추적 및 분석 |
| **Code Quality** | code-reviewer, refactor-assistant, test-helper | 코드 리뷰 및 테스트 |
| **Git/GitHub** | branch-manager, commit-helper, git-troubleshooter, github-manager, pr-creator | Git 워크플로우 자동화 |
| **Analytics** | analytics-reporter, obsidian-sync | 통계 및 지식 관리 |
| **Infrastructure** | config-validator, file-explorer, google-searcher, agent-writer | 인프라 지원 |

**ML/AD-Specific Agents (12)**
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| ad-experiment-manager | AD 실험 생성, 실행, 비교, 추적 | "experiment", "실험", "training run", "학습 실행" |
| benchmark-evaluator | nuPlan 벤치마크 실행, 메트릭 계산 | "evaluate", "평가", "benchmark", "metrics" |
| dataset-curator | 데이터셋 다운로드, 전처리, 큐레이션 | "dataset", "데이터셋", "nuPlan", "Waymo" |
| experiment-documenter | 자동 실험 문서화 및 결과 기록 | "실험 문서화", "학습 완료", "결과 기록", "update docs" |
| forensic-analyst | 학습 실패 근본 원인 분석 (수학적 검증) | "근본 원인", "root cause", "forensic", "왜 실패" |
| model-trainer | RL/IL 학습 시작 및 관리 | "train", "학습", "PPO", "SAC", "GAIL" |
| training-analyst | 학습 결과 분석, 성공/실패 판정 | "결과 분석", "리포트", "왜 실패", "원인 분석" |
| training-doc-manager | 학습 문서 동기화, 아카이브 관리 | "문서 동기화", "아카이브", "로그 정리" |
| training-monitor | 실시간 학습 상태 모니터링 | "학습 상태", "진행률", "모니터링", "현재 reward" |
| training-orchestrator | 학습 워크플로우 총괄 조율 | "다음 단계", "워크플로우", "전체 상태" |
| training-planner | 실험 설계 및 Config 생성 | "실험 설계", "다음 버전", "config 생성" |
| training-site-publisher | GitHub Pages 사이트 발행 | "사이트 업데이트", "gh-pages", "웹 발행" |

### Skills (22)

> **Note**: All core framework skills (18) from cc-initializer plus 4 ML-specific skills.

**Core Skills (18)**
| Skill | Description | Keywords |
|-------|-------------|----------|
| agile-sync | CHANGELOG, README, 진행상황 동기화 | "동기화", "sync", "changelog" |
| analytics | Tool/Agent 사용 통계 시각화 | "통계", "사용량", "analytics", "metrics" |
| brainstorming | 아이디어 구체화 및 대안 탐색 | "brainstorm", "아이디어", "alternative" |
| context-optimizer | 컨텍스트 로딩 최적화 | "context", "token", "optimize", "summarize" |
| dev-doc-system | 개발 문서 통합 관리 | "문서 시스템", "개발 기록", "방향 설정" |
| feedback-loop | 피드백 수집 및 ADR 생성 | "feedback", "learning", "retrospective" |
| gh | GitHub CLI 통합 | "github", "issue", "CI", "workflow" |
| hook-creator | Claude Code Hook 생성 | "create hook", "configure hook" |
| obsidian | Obsidian vault 동기화 | "obsidian", "vault", "지식 동기화" |
| prompt-enhancer | 프로젝트 컨텍스트 기반 프롬프트 향상 | "enhance prompt", "context-aware" |
| quality-gate | 개발 lifecycle Quality Gate | "pre-commit", "pre-merge", "quality" |
| readme-sync | README 자동 동기화 | "readme sync", "update readme" |
| repair | cc-initializer 자동 복구 | "repair", "fix", "troubleshoot" |
| skill-creator | 새로운 Skill 생성 가이드 | "create skill", "new skill" |
| sprint | Sprint lifecycle 관리 | "sprint", "velocity", "burndown" |
| subagent-creator | 커스텀 Sub-agent 생성 | "create agent", "new agent" |
| sync-fix | Phase/Sprint/문서 동기화 문제 해결 | "sync fix", "불일치", "동기화 문제" |
| validate | cc-initializer 설정 검증 | "validate", "검증", "check config" |

**ML-Specific Skills (4)**
| Skill | Description | Command |
|-------|-------------|---------|
| dataset | 데이터셋 다운로드, 전처리, 분할 | `/dataset` |
| experiment | ML 실험 생성, 실행, 비교, 추적 | `/experiment` |
| evaluate | 모델 평가 및 벤치마크 실행 | `/evaluate` |
| train | RL/IL 학습 시작 및 모니터링 | `/train` |

### Commands (6)

| Command | Purpose | Integration |
|---------|---------|-------------|
| /bugfix | 버그 수정 워크플로우 (이슈 분석→PR) | Git + Phase + Sprint |
| /dev-doc-planner | PRD, 기술 설계서, 진행상황 문서 작성 | Templates (PRD/TECH-SPEC/PROGRESS) |
| /feature | 기능 개발 워크플로우 (Phase→Sprint→Git→Doc) | Phase + Sprint + Git + Docs |
| /git-workflow | Git 워크플로우 관리 (브랜치, 커밋, PR) | GitHub Flow + Conventional Commits |
| /phase | Phase 상태 확인, 전환, 진행률 업데이트 | Phase 시스템 |
| /release | 릴리스 워크플로우 (버전→문서→배포) | Git + Docs + Archive |

### Hooks (6)

**Pre-Tool Hooks**
- `pre-tool-use-safety.sh`: Bash/Write/Edit 안전성 검사 (위험 명령어 차단)

**Post-Tool Hooks**
- `auto-doc-sync.sh`: Bash/Write/Edit 후 문서 자동 동기화
- `phase-progress.sh`: Write/Edit 후 Phase 진행 상황 업데이트
- `post-tool-use-tracker.sh`: Bash/Write/Edit 사용 추적 (analytics)

**Notification Hooks**
- `notification-handler.sh`: 모든 알림 처리

**Utility Hooks**
- `error-recovery.sh`: Hook 실패 시 자동 복구

### Key Features

**Automation**
- Phase/Sprint 자동 진행 추적
- Git 워크플로우 자동화 (Conventional Commits)
- 문서 자동 동기화 (CHANGELOG, README, PROGRESS)
- Quality Gate (pre-commit, pre-merge, pre-release)

**ML/AD Specific**
- 실험 추적 및 비교 (MLflow/W&B 통합)
- TensorBoard 모니터링
- 학습 문서 자동 생성 및 아카이브
- GitHub Pages 자동 발행

**Safety & Recovery**
- 위험 명령어 차단 (rm -rf, git reset --hard 등)
- Hook 실패 시 자동 복구
- 문서 손상 감지 및 복구

**Analytics**
- Tool/Agent/Skill 사용 통계 (JSONL)
- CLI 차트 시각화
- 30일 데이터 보관

---

## Documentation

- [PRD (Product Requirements)](docs/PRD.md)
- [Training Log (Detailed)](docs/TRAINING-LOG.md)
- [Learning Roadmap](docs/LEARNING-ROADMAP.md)
- [Phase Documents](docs/phases/README.md)
- [Progress Tracking](docs/PROGRESS.md)
- [Workflow Diagrams (Mermaid)](docs/WORKFLOW-DIAGRAMS.md)
- [cc-initializer Components](.claude/docs/CC-INITIALIZER-COMPONENTS.md)

---

## References

- [Unity ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)

---

**Last Updated**: 2026-01-29 | **Phase G In Progress** | Phase F: +988, Phase G: +461 (340K)

**Development Infrastructure**: [cc-initializer](https://github.com/tygwan/cc-initializer) - 38 Agents, 29 Skills, 6 Hooks, 12 Commands
