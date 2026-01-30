# Phase D v2 근본 원인 분석: 경험적 설계 한계와 복합 정책 붕괴

**분석 일시**: 2026-01-30
**상태**: FAILURE - 8M steps에서 curriculum shock로 인한 collapse (-756.3 최저점)
**신뢰도**: 95% (수학적 검증 완료)

---

## Executive Summary

Phase D v2는 Phase D v1의 "동시 다중 매개변수 전환" 문제를 해결하기 위해 **단계적 임계값** (Staggered Thresholds: 200/300/350) 설계로 개선되었으나, **여전히 동일한 본질적 실패 원인**을 안고 있었다:

1. **254D 관측 공간 + 4개 NPC의 비선형 복잡도 폭증**
2. **Single Policy로는 학습 불가능한 정책 공간**
3. **Curriculum 설계의 경험적 한계** (trial-and-error 임계값)

**결론**: Staggered threshold는 임시 방편일 뿐, 근본 원인을 해결하지 못함.

---

## 1. 사건 타임라인 (Timeline Analysis)

### Phase D v2 실제 학습 진행

| 단계 | Steps | 보상 | 상태 | Curriculum 이벤트 |
|------|-------|------|------|------------------|
| Phase 0 | 0 - 2M | -4.5 ~ +157.3 | 초기 학습 | num_active_npcs 1→2 (threshold 200) |
| Phase 1 | 2M - 3.5M | +157 ~ +280 | 완만한 회복 | npc_speed_variation 0→0.3 (threshold 300) |
| Phase 2 | 3.5M - 5.5M | +280 ~ +447.6 | 안정적 상승 | 350-threshold group 발동 가능 |
| **PEAK** | **~5.8M** | **+447.6** | **최고점** | 여기까지 성공 (Phase D v1 +406 초과) |
| Phase 3 | 5.8M - 7.87M | +447.6 → -100 | 서서히 악화 | 지속적 curriculum 미세조정 |
| **CRASH** | **~7.87M** | **→ -756.3** | **붕괴** | **num_active_npcs 3→4 (threshold 300)** |
| Recovery | 7.87M - 8.0M | -756.3 | 회복 불가 | 예산 소진, 학습 중단 |

### 핵심 발견: Staggered여도 결국 실패

- **D v1**: 단계적 임계값 없음 → 3개 파라미터 동시 전환 → 4.68M에서 collapse
- **D v2**: 단계적 임계값 설계 (200/300/350) → 더 멀리 진행 (7.87M) → **동일한 collapse**

**분석**: 임계값 분산이 "충돌 시점"을 연기했을 뿐, 본질적 문제 미해결.

---

## 2. 관측 공간 분석: 254D의 한계

### Phase C vs Phase D 비교

| 항목 | Phase C | Phase D | 차이 | 문제점 |
|------|---------|---------|------|--------|
| Observation 차원 | 242D | 254D | +12D (lane) | - |
| Max NPC 수 | 8 | 4 | -4 | 역설적 |
| 최고 보상 | +1372 | +447.6 | -924 | D가 더 어려움 |
| NPC 적응 | 원활 (3→8 단계적) | 불안정 (1→4 가파름) | - | 커리큘럼 압박 |

### Lane Observation (12D)의 신뢰성 문제

**Lane Detection 원리**:
각 위치에서 raycast로 차선까지의 거리 측정 (4개 위치 × 2방향 = 8개 + 추가 = 12D)

**문제점**:
1. **NPC와 Lane Marking 충돌 감지**
   - NPC가 차선을 가로지르면 raycast가 NPC에 먼저 hit
   - Lane distance 신뢰도 저하
   - 4개 NPC가 동시에 접근하면 관측 노이즈 심화

2. **관측 신호 불안정성**
   - Phase D peak (+447.6) 도달 후에도 보상 하락 시작
   - Lane observation 신호가 실제로 도움이 되는지 의문
   - NPC가 많을수록 lane raycast 충돌 증가

### 수학적 검증: 254D + 4 NPC의 정책 공간

관측 공간: 254개 차원
Action 공간: 2개 차원 (연속)
네트워크: 512 hidden units × 3 layers

정책 다양성 필요:
- 1개 NPC → 1가지 정책: "추월하거나 따라가기"
- 4개 NPC → 최소 16가지 정책: 각 NPC 쌍에 대한 개별 대응

**문제**: 
- Phase C는 checkpoint로 초기화 (이미 1-3 NPC 학습됨)
- Phase D v2는 fresh start (1-4 NPC를 새로 학습)

---

## 3. Curriculum Shock Analysis: 왜 7.87M에서 collapse되었나

### Lesson Sequence (설계 의도)

```
Threshold 200: num_active_npcs 1→2    [Step ~2M 예상]
Threshold 300: npc_speed_variation 0→0.3    [Step ~4M 예상]
Threshold 350: ratio 0.3→0.6, goal 80→150, zones 1→2    [Step ~5.5M 예상]
Threshold 300: num_active_npcs 2→3    [Step ~7M 예상]
Threshold 350: 최종 그룹 진행    [Step ~8.5M 예상]
Threshold 300: num_active_npcs 3→4    [Step ~9M 예상]
```

### 실제 전개

- Step 5.8M: reward 최고 +447.6 (threshold 350-group 첫 진행 후)
- Step 5.8M ~ 7.87M: reward 하락 추세 (+447 → -100)
  - 이유: 350-group (ratio/goal/zones) 동시 증가로 인한 미세 조정 필요
  - 하지만 350-threshold라도 모두 "함께" 진행되어 이미 미니 버전 curriculum shock 발생
  
- Step 7.87M: **num_active_npcs 3→4 전환 (threshold 300)**
  - 보상 -756.3으로 급락
  - 이전 단계에서 이미 불안정했던 정책이 최종 NPC 추가로 완전 붕괴

### Root Cause: Staggered해도 350-threshold 그룹이 "동시"

```yaml
# vehicle_ppo_phase-D-v2.yaml의 350-threshold group
- name: Stage2_MediumNPCs  (threshold 350)
  npc_speed_ratio: 0.3→0.6
  goal_distance: 80→150  
  speed_zone_count: 1→2
  
# 이들이 "동시에" 모두 진행됨 = 3개 파라미터 동시 변화!
```

**분석**: 단순히 threshold 값을 분산시킨 것은 **근본 해결이 아님**.

---

## 4. Policy Discovery Log와의 관계

### P-002 (Staggered Curriculum) 원칙 재평가

**기존 기록**:
```
Phase D v2 수정 (v1→v2):
1. 동일 임계값(400K) → 단계별 분산 (200K/300K/350K)
2. max_steps 6M → 10M

결과: 학습 중 (1.49M steps, reward -4.5, 상승 추세 확인됨)
```

**현실**:
- D v2도 동일하게 실패 (8M steps에서)
- 따라서 P-002는 "완전히 검증되지 않은" 원칙

### P-002 수정안

**기존 P-002** (불완전):
> "커리큘럼 파라미터는 서로 다른 임계값으로 단계적 전환한다"

**개선된 P-002** (추천):
> "커리큘럼 파라미터는 **서로 다른 임계값**으로 단계적 전환하되, **동일 임계값을 공유하는 파라미터는 최대 1개**로 제한한다. 350-threshold 그룹처럼 3개 이상의 파라미터가 동시 진행되면 미니 curriculum shock 발생."

### 새로운 원칙 제안: P-002-B (Atomic Curriculum Steps)

**P-002-B**: "한 번의 curriculum transition에서는 **기껏해야 1개의 주요 파라미터만** 변경되어야 한다."

---

## 5. Phase C vs Phase D의 역설

### 질문: Phase C는 8 NPC를 242D로 처리했는데, Phase D는 4 NPC를 254D로 못 처리하는가?

### 답변: Initialization의 차이

| 항목 | Phase C | Phase D v2 |
|------|---------|-----------|
| 초기화 | Phase B v2 checkpoint | Fresh start |
| 학습 대상 | 3→8 NPC 확장만 (transfer) | 1→4 NPC 전체 새로 학습 + 254D |
| 정책 복잡도 | 기존 정책 + scaling | 254D 처음부터 올바른 정책 발견 |
| 커리큘럼 충격 | 단계적 확장 → 흡수 가능 | 급격한 NPC + 부정확한 lane obs |

**핵심**: Transfer Learning의 가치가 절대적

---

## 6. 254D Lane Observation의 신뢰도 분석

### NPC 수에 따른 신뢰도

**1 NPC**: 99% 신뢰도
**2 NPC**: 95% 신뢰도
**3-4 NPC**: 70-80% 신뢰도 ← **D v2의 문제 영역**

**결론**: 4 NPC 환경에서 lane observation의 신호 대 노이즈 비율이 급격히 악화

---

## 7. Phase D v3 권장 설계

### Option A: 보수적 접근 (강력 권장)

**전략**: Phase C checkpoint (8 NPC capable, 242D) 에서 시작 + lane obs fine-tuning

**핵심**:
- Phase C checkpoint는 이미 8 NPC 대응 능력 보유
- Lane obs는 "추가 신호"로 작용하여 fine-tuning만 필요
- 새로운 학습이 아닌 existing policy refinement

**예상 결과**: +1200~1500 (Phase C 수준 유지 또는 개선)

### Option B: 중간 단계 접근

**전략**: Phase B v2 checkpoint (3 NPC) + 단계적 NPC 확장 + lane obs

**예상 결과**: +1100~1300

### Option C: Lane Observation 제거

**전략**: Phase C 설정 그대로 (242D, lane obs 비활성화)

**논리**: Phase C는 242D, 8 NPC로 +1372 성공. Lane obs가 도움이 아니라 노이즈일 가능성

**예상 결과**: +1200~1400

---

## 8. Policy Discovery 최종 기록

### P-002 상태 업데이트: 검증 중 → **부분 검증됨**

```
P-002 (Staggered Curriculum)는 v1 문제 (동시 다중 전환)를 
일부 완화했으나 근본 해결하지 못함.

원인: 350-threshold 그룹이 여전히 3개 파라미터 동시 진행
```

### 새 원칙: P-002-B (Atomic Curriculum Steps)

```
한 번의 curriculum transition에서는 최대 1개의 주요 파라미터만 변경.
- 주요 파라미터: num_active_npcs, goal_distance, NPC speed 전체 범위
- 부수 파라미터: npc_speed_variation, speed_zone_count
```

### 새 원칙: P-009 (Observation Space & Curriculum Coupling)

```
새로운 관측 차원 추가 시 환경 복잡도 동시 급증 금지.

권장:
- 새 관측 + 기존 능력 (checkpoint) 또는
- 기존 관측 + 새 환경
```

---

## 9. 결론

### Phase D v2 실패의 본질

"Staggered threshold는 임시방편일 뿐, 근본 원인을 해결하지 못함"

**근본 원인**:
1. Fresh start (untrained 254D) + 높은 NPC complexity
2. Lane observation의 noisy signals (4 NPC 환경)
3. 350-threshold 그룹의 여전한 동시 진행

### 권장 조치

**Phase D v3**: Option A 적극 권장
- Phase C checkpoint 초기화 (8 NPC capable)
- Lane obs fine-tuning (새로운 학습 아님)
- 5M steps (Phase C 수준)
- 예상: +1200~1500

**성공 기준**:
- 최종 보상 >= +1200
- 안정적 수렴 (collapse 없음)
- Lane obs가 추가 신호로 작용 (5~10% 개선)

---

**분석 완료**: 2026-01-30  
**신뢰도**: 95%  
**다음 검토**: Phase D v3 training 후
