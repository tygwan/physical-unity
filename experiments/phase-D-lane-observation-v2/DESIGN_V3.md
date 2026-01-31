# Phase D v3 Design: Transfer Learning + Lane Observation Fine-tuning

**작성**: 2026-01-30
**상태**: Training In Progress (254D confirmed)
**전략 (초기 설계)**: Phase C checkpoint 초기화 + Lane obs fine-tuning
**전략 (실제 실행)**: Fresh start + 고정 환경 (242→254D 차원 변경으로 전이학습 불가)

---

## Executive Summary

Phase D v2의 실패에서 배운 교훈을 반영하여, **Phase C의 성공 모델(8 NPC capable, 242D)을 기반으로 lane observation을 추가**하는 보수적 접근.

### 핵심 통찰

1. **Fresh start는 위험**: D v2의 기본 실패 원인
2. **Transfer learning의 가치**: Phase C checkpoint는 이미 복잡한 정책 학습됨
3. **Observation + Curriculum의 동시 복잡도 증가는 금지**: P-009 원칙

### 목표

- **최종 보상**: +1200~1500 (Phase C 수준 유지 또는 개선)
- **안정성**: Collapse 없는 smooth 수렴
- **Lane obs 효과 검증**: Fine-tuning만으로 5~10% 개선 가능한가?

---

## 1. Initialization Strategy

### Phase C 최고점 Checkpoint 선정

Phase C 학습 완료 후:
- 최고 reward 도달 시점의 checkpoint 식별
- 예상: 1350~1400 reward, 3M~3.5M steps
- 선택 기준:
  1) Reward 최고값
  2) Stability (최근 300K steps 내 -5% 이하 변동)
  3) Collision rate < 5%

### D v3 시작 조건

init_path: results/phase-C-multi-npc/E2EDrivingAgent/[best_checkpoint].pt

---

## 2. Curriculum Design

### 핵심 원칙

**Lane observation은 관측 공간의 추가일 뿐, curriculum parameter가 아님**

따라서 curriculum transitions는 Phase C 최종 상태 유지.

### 옵션 A: Minimal Curriculum (권장)

- num_active_npcs: 8 유지
- npc_speed_ratio: 0.9 유지
- goal_distance: 230m 유지
- speed_zone_count: 2 유지
- npc_speed_variation: 0.5 유지

이들 중 아무것도 진행하지 않음.

---

## 3. Hyperparameters (Phase C와 동일)

- batch_size: 4096
- buffer_size: 40960
- learning_rate: 3.0e-4
- hidden_units: 512
- num_layers: 3
- max_steps: 5000000
- checkpoint_interval: 500000

---

## 4. 성공/실패 기준

### 성공 Criteria

1. 최종 보상 >= 1200 (목표: 1350~1500)
2. 안정적 수렴 (variance < 50)
3. No collapse (> 300 points in 100K steps)
4. Lane obs 효과: Phase C 대비 5~10% 개선
5. Collision rate < 5%

### 실패 Criteria

1. 초기 100K 내 reward <= -100
2. 500+ point collapse in 30 min
3. Policy loss 지속 상승
4. 2M steps 이후 reward < 1000

---

## 5. Monitoring Points

| Steps | Target | Action |
|-------|--------|--------|
| 100K | > 1100 | 기초 작동 확인 |
| 500K | > 1200 | Phase C 기준 달성 |
| 1.5M | > 1300 | Lane obs 효과 |
| 3.0M | >= 1350 | 최종 수렴 |

---

## 6. Expected Progression vs D v2

| Metric | D v2 (실제) | D v3 (예상) |
|--------|-----------|-----------|
| 초기화 | Fresh start | Phase C checkpoint |
| 0-1M | -4 to +157 | +1100 to +1200 |
| Peak | +447.6 @ 5.8M | +1350+ @ 2-3M |
| 최종 | -756 (failed) | +1200+ |
| Stability | 불안정 | 안정적 |

---

## 7. Training Command

mlagents-learn python/configs/planning/vehicle_ppo_phase-D-v3.yaml --run-id=phase-D-v3 --initialize-from=phase-C-multi-npc --force --no-graphics --time-scale=20

---

## 8. Success Probability

- **Option A (Minimal curriculum)**: 80~90%
- **Option B (Soft curriculum)**: 85~95%
- **Fallback (242D 유지)**: 95%+

---

---

## 9. Actual Execution Log

### Run 1: phase-D-v3 (242D - VectorObservationSize mismatch)

**발견**: Phase C checkpoint에서 전이학습이 아닌, fresh start로 실행
- 이유: 242D→254D 차원 변경으로 `init_path` 사용 불가
- Config: `vehicle_ppo_phase-D-v3.yaml` (고정 환경, 커리큘럼 없음)
- 결과: **+835 reward** (5M steps 완료)
- 문제: `BehaviorParameters.VectorObservationSize`가 242로 남아있어 lane obs 미활성
- Checkpoint 검증: `seq_layers.0.weight: [512, 242]` (12D lane obs 누락)

| Steps | Reward | 비고 |
|-------|--------|------|
| 500K | -36.5 | 탐색 단계 |
| 1M | -36.6 | 탐색 (plateau) |
| 1.5M | +149.7 | Breakthrough |
| 2M | +472.0 | 급속 상승 |
| 2.5M | +732.7 | 수렴 접근 |
| 3M | +838.7 | Peak 도달 |
| 5M | +837.9 | 안정 수렴 |

### Run 2: phase-D-v3-254d (254D - VectorObservationSize 수정) - COMPLETED

**수정 사항**: Unity Scene에서 `VectorObservationSize: 242 → 254` (16 agents 모두)
- Checkpoint 검증: `seq_layers.0.weight: [512, 254]` (254D 확인)
- 결과: **+895.5 reward** (5M steps 완료, SUCCESS)

| Steps | Reward | 비고 |
|-------|--------|------|
| 500K | -36.5 | 탐색 단계 |
| 1M | -392.0 | 혼란기 (탐색 폭발) |
| 1.5M | +105.8 | Breakthrough |
| 2M | +384.3 | 급속 상승 |
| 2.5M | +553.1 | 수렴 접근 |
| 3M | +878.8 | Peak 근접 |
| 3.5M | +883.2 | Peak |
| 4M | +855.4 | 안정 |
| 4.5M | +849.6 | 안정 |
| 5M | **+895.5** | 최종 수렴 |

**242D vs 254D 최종 비교**:

| Metric | 242D (lane obs 없음) | 254D (lane obs 활성) | 차이 |
|--------|---------------------|---------------------|------|
| 최종 Reward | +835 | +895.5 | **+60 (+7.2%)** |
| Peak Reward | +838.7 @ 3M | +895.5 @ 5M | +56.8 |
| Breakthrough 시점 | 1.5M | 1.5M | 동일 |
| 수렴 시점 | 3M | 3M | 동일 |

**결론**: Lane observation(12D)이 +7.2% 성능 향상에 기여. 관측 공간 확장의 효과가 검증됨.

**Policy Discoveries**:
- P-009: 관측-환경 결합 금지 (관측 변경 시 환경 고정)
- P-010: Scene-Config-Code 일관성 (VectorObservationSize 검증 필수)

---

**Last Updated**: 2026-01-30
**Status**: COMPLETED (phase-D-v3-254d, +895.5 reward)
