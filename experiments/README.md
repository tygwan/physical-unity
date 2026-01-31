# Training Experiments Archive

This directory contains all training experiments for the E2E Driving Agent project.

## Directory Structure

```
experiments/
├── early_experiments/      # Initial experiments (3DBot, curriculum v1-v9)
├── v10g_lane_keeping/      # v10g: Lane keeping focus
├── v11_sparse_overtake/    # v11: Sparse overtaking
├── phase-A-overtaking/  # Phase A: Dense traffic overtaking
├── phase-B_decision_learning/ # Phase B: Decision learning
├── phase-C_multi_npc/   # Phase C: Multi-NPC interaction
├── phase-E_curved_roads/ # Phase E: Curved road handling
├── phase-F_multi_lane/  # Phase F: Multi-lane navigation
├── phase-G_intersection/ # Phase G: Intersection navigation
└── failed_experiments/     # Archived failed experiments
```

## Parallel Training Environment

모든 Phase에서 **16개의 병렬 Training Area**를 사용하여 학습 효율을 극대화합니다.

### 병렬 환경 구성

| 항목 | 값 |
|------|-----|
| Training Areas | 16개 (일렬 배치) |
| Unity Instances | 1개 |
| Batch Size | 4096 |
| Buffer Size | 40960 |
| Time Scale | 20x |

### Scene 구성

각 Phase Scene에는 16개의 독립적인 도로 환경이 X축을 따라 일렬로 설치됩니다:
- 각 Training Area는 독립적인 도로, NPC, 에이전트를 포함
- Area 간격: 100m (X축 방향)
- 모든 Area가 동시에 학습 데이터 수집

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           Phase Scene Layout (1x16)                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│  [Area 0] - [Area 1] - [Area 2] - ... - [Area 14] - [Area 15]                    │
│     0m       100m       200m              1400m       1500m                       │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## Training Flow

```
Phase A (2M) ──> Phase B (2M) ──> Phase C (4M) ──> Phase E (6M) ──> Phase F (6M) ──> Phase G (6M)
   │                │                 │                │               │               │
   └─ Dense traffic └─ Decisions     └─ Multi-NPC    └─ Curves      └─ Lanes       └─ Intersections
```

## Scene-Phase Matching Rule (P-011)

Each training phase MUST use its designated Unity scene. Wrong scene = training failure.

| Phase | Required Scene | Road Width | Features |
|-------|---------------|------------|----------|
| A | PhaseA_DenseOvertaking | 4.5m (1 lane) | Dense NPC traffic |
| B | PhaseB_DecisionLearning | 4.5m (1 lane) | Decision making |
| C | PhaseC_MultiNPC | 4.5m (1 lane) | Multi-NPC |
| E | PhaseE_CurvedRoads | 4.5m (1 lane) | Curved waypoints |
| F | PhaseF_MultiLane | 11.5m (3 lane) | Multi-lane, center line |
| G | PhaseG_Intersection | 14m (4 lane) | Intersections, turn logic |

**Pre-training checklist**:
1. Open Unity Editor
2. Verify active scene name matches the phase (top of hierarchy)
3. DrivingSceneManager will log a warning if scene-phase mismatch is detected
4. Start mlagents-learn, THEN press Play in Unity

**Failure case (Phase F v1)**: PhaseE scene (4.5m road) was active during Phase F training.
num_lanes=2 transition generated 7m waypoints on 4.5m road -> off-road -> -8 reward -> 4.27M wasted steps.

## Quick Commands

### Start Training
```bash
# Phase A (start from scratch)
mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml --run-id=phase-A

# Subsequent phases (initialize from previous)
mlagents-learn python/configs/planning/vehicle_ppo_phase-B.yaml \
  --run-id=phase-B --initialize-from=results/phase-A/E2EDrivingAgent
```

### Monitor Training
```bash
tensorboard --logdir=experiments/
```

## Backup Policy

- **NEVER run `git clean -fd`** in this repository
- Commit checkpoints after each phase completion
- Use `git lfs` for large .onnx files if needed
