# Training Experiments Archive

This directory contains all training experiments for the E2E Driving Agent project.

## Directory Structure

```
experiments/
├── early_experiments/             # Initial experiments (3DBot, curriculum v1-v9)
├── failed_experiments/            # Archived failed experiments
├── phase-0-foundation/            # Phase 0: Lane keeping + NPC coexistence (+1018)
├── phase-A-overtaking/            # Phase A: Dense traffic overtaking (+2114)
├── phase-B-decision/              # Phase B v1: Decision learning (FAILED -108)
├── phase-B-decision-v2/           # Phase B v2: Decision learning recovery (+877)
├── phase-C-multi-npc/             # Phase C: Multi-NPC interaction (+1372)
├── phase-D-lane-observation/      # Phase D v1: Lane obs 254D (FAILED, collapse)
├── phase-D-lane-observation-v2/   # Phase D v2: Redesigned curriculum (stuck at -690)
├── phase-D-lane-observation-v3/   # Phase D v3: Speed zones + 254D (+904)
├── phase-E-curved-roads/          # Phase E: Curved road handling (+924)
├── phase-F-multi-lane/            # Phase F v1: Multi-lane (scene mismatch)
├── phase-F-multi-lane-v2/         # Phase F v2: Collapse to -14
├── phase-F-multi-lane-v3/         # Phase F v3: Shared thresholds, collapse to 0
├── phase-F-multi-lane-v4/         # Phase F v4: Degraded to 106
├── phase-F-multi-lane-v5/         # Phase F v5: Multi-lane highway (+643)
├── phase-G-intersection/          # Phase G v1/v2: Intersection navigation (+628)
├── phase-H-npc-intersection/      # Phase H v1: NPC at intersections (crashed)
├── phase-H-npc-intersection-v2/   # Phase H v2: Gradual variation (9/11)
├── phase-H-npc-intersection-v3/   # Phase H v3: Lowered thresholds (+701)
├── phase-I-curved-npc/            # Phase I v1: Curved roads + NPC (triple crash)
├── phase-I-curved-npc-v2/         # Phase I v2: Recovery training (+770 record)
├── phase-J-traffic-signals/       # Phase J v1: Tensor mismatch (FAILED)
├── phase-J-traffic-signals-v2/    # Phase J v2: From scratch 268D (9/13, +632)
├── phase-J-traffic-signals-v3/    # Phase J v3: Signal ordering issue (12/13, +477)
├── phase-J-traffic-signals-v4/    # Phase J v4: Signal-first green_ratio (3/4, +497)
└── phase-J-traffic-signals-v5/    # Phase J v5: Decel reward + lower thresholds (5/5, +537)
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
Phase A (2M) ──> Phase B (2M) ──> Phase C (4M) ──> Phase E (6M) ──> Phase F (6M) ──> Phase G (6M) ──> Phase H (13.5M) ──> Phase I (10M) ──> Phase J (25M)
   │                │                 │                │               │               │               │                    │                    │
   └─ Dense traffic └─ Decisions     └─ Multi-NPC    └─ Curves      └─ Lanes       └─ Intersections └─ NPC+Intersections └─ Curves+NPCs (+770) └─ Traffic signals (v5: +537, green=0.4, COMPLETE)
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
| H | PhaseH_NPCIntersection | 14m (4 lane) | NPC traffic at intersections |
| I | PhaseH_NPCIntersection* | 14m (4 lane) | Curved roads + NPC traffic |
| J | PhaseJ_TrafficSignals | 14m (4 lane) | Traffic signals + stop lines |

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
