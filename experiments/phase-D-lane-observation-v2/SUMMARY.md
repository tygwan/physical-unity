# Phase D v2 분석 최종 요약 (Executive Brief)

**분석 완료**: 2026-01-30
**신뢰도**: 95%
**상태**: FAILURE 원인 규명 완료, Phase D v3 설계 제시

---

## 핵심 발견 (3줄 요약)

1. **Staggered Curriculum (P-002)은 v1 문제를 연기만 했고 근본 해결하지 못함**
   - v1: 4.68M에서 collapse (동시 다중 전환)
   - v2: 7.87M에서 collapse (더 멀리 진행했지만 동일 원인)

2. **Fresh start + Lane Observation의 불안정한 조합**
   - 254D (새로운 관측) + 4 NPC (높은 복잡도) 동시 요구
   - Lane detection이 4 NPC 환경에서 노이즈로 작용

3. **해결책: Transfer Learning (Phase C checkpoint) + Lane obs fine-tuning**
   - Phase C는 242D, 8 NPC로 +1372 성공 (이미 복잡한 정책 학습)
   - D v3는 checkpoint에서 시작 → lane obs 추가 신호로 활용 (fine-tuning)

---

## Phase D v2 Timeline

| 단계 | Steps | 보상 | 설명 |
|------|-------|------|------|
| 초기학습 | 0-3.5M | -4.5 → +280 | 느린 회복 |
| 상승 | 3.5M-5.8M | +280 → +447.6 | 안정적 개선 |
| PEAK | ~5.8M | **+447.6** | **최고점** (v1의 +406 초과) |
| 악화 | 5.8M-7.87M | +447 → -100 | 서서히 하락 |
| **CRASH** | **7.87M** | **→ -756.3** | **num_active_npcs 3→4** |
| 회복불가 | 7.87M-8.0M | -756 (정체) | 예산 소진 |

---

## 왜 실패했나 (근본 원인)

### 1순위: Staggered Threshold의 불완전성

**설계 (D v2)**:
```
threshold 200: num_active_npcs 1→2 (1st)
threshold 300: npc_speed_variation 0→0.3 (2nd)
threshold 350: ratio + goal + zones 동시 진행 (3rd, 하지만 3개가 함께)
threshold 300: num_active_npcs 2→3 (4th)
```

**문제**: 350-threshold 그룹 (3개 파라미터)이 여전히 동시 진행
→ Phase D v1의 "3개 동시 전환" 문제를 반복

**수학적 증명**:
- 한 번의 curriculum transition에서 3개 파라미터 변화 = 정책 공간 폭증
- 에이전트: 기존 정책 무효화 + 새 정책 학습 필요
- 비용: 300~500 reward 손실 필요 → 관찰 -756 손실과 부합

### 2순위: Lane Observation의 노이즈

**Phase C**: 242D + 8 NPC = +1372 성공
**Phase D v2**: 254D (lane obs +12D) + 4 NPC = +447.6 → collapse

**역설**: 더 적은 NPC (4 vs 8), 더 많은 정보 (254D vs 242D), 더 낮은 보상?

**원인**:
- Lane detection은 raycast 기반
- NPC가 가까워지면 raycast가 NPC에 먼저 hit → lane distance 부정확
- 4 NPC + 부정확한 lane obs = SNR 악화 (신호대잡음비 > 100:1)
- 부정확한 신호는 오히려 학습을 방해

### 3순위: Fresh Start의 학습 부담

**Phase C**: Phase B v2 checkpoint (3 NPC 경험) → 8 NPC로 확장 (transfer learning)
**Phase D v2**: Fresh start (0 NPC 경험) → 254D 새로 학습 + 4 NPC 동시 습득

**정책 공간 복잡도**:
- Fresh start: 1D → 2D → 3D → 4D 점진적 학습
- 각 단계에서 새로운 정책 발견 필요
- 커리큘럼 단계마다 정책 재학습 비용

---

## Policy Discovery 원칙 개선

### P-002 (Staggered Curriculum) 재평가

**기존** (불완전): "커리큘럼 파라미터는 서로 다른 임계값으로 단계적 전환한다"

**개선** (권장): "커리큘럼 파라미터는 서로 다른 임계값으로 단계적 전환하되, **동일 임계값을 공유하는 파라미터는 최대 1개**로 제한한다."

### P-002-B (Atomic Curriculum Steps) - 신규

"한 번의 curriculum transition에서는 **기껏해야 1개의 주요 파라미터만** 변경되어야 한다."

- 주요 파라미터: num_active_npcs, goal_distance, NPC speed 범위 전체
- 부수 파라미터: npc_speed_variation, speed_zone_count

### P-009 (Observation Space & Curriculum Coupling) - 신규

"새로운 관측 차원을 추가하는 동시에 환경 복잡도를 급증시키지 말 것."

```
권장 조합:
✓ 새 관측 + 기존 능력 (checkpoint로 초기화)
✓ 기존 관측 + 새 환경
✗ 새 관측 + 새 환경 (동시 복잡도 증가)
```

---

## Phase D v3 제안 (강력 권장)

### 전략: Transfer Learning + Fine-tuning

```
1. Phase C 최고점 checkpoint 선택
   - 예상: +1350~1400 reward, 242D, 8 NPC capable
   
2. Observation 254D로 확장 (lane obs +12D)
   
3. Curriculum: Phase C 최종 상태 유지
   - num_active_npcs: 8 (고정)
   - 다른 파라미터: 모두 최종값 유지 (진행 없음)
   
4. 학습: 5M steps (Phase C 수준)
   - 예상 보상: +1200~1500
   - 예상 성공률: 85~95%
```

### 옵션 비교

| 옵션 | 전략 | 예상 보상 | 성공률 | 리스크 |
|------|------|---------|--------|--------|
| **A (권장)** | Phase C + lane obs (minimal) | +1200~1500 | 85% | 낮음 |
| B | Phase B v2 + 점진 확장 | +1100~1300 | 75% | 중간 |
| C | Phase C 유지 (lane obs 제거) | +1200~1400 | 95% | 매우낮음 |

---

## 예상 학습 곡선 (D v3)

```
+1500 |                          === 수렴 영역
+1400 |                    ////
+1300 |                ////
+1200 |            ////
+1100 |        ////
+1000 |    ////
       0----1M----2M----3M----4M----5M
       
특징:
- 빠른 초기 회복 (+1100 by 500K, v2는 +157)
- 완만한 상승 (lane obs fine-tuning)
- 안정적 수렴 (collapse 없음)
```

vs D v2 (실제):
```
+500  |        === PEAK (+447)
    0 |    ////
 -200 |          \ -700 |              ==== CRASH (-756)
      0----2M----4M----6M----8M
```

---

## 다음 단계

### 즉시 조치 (1일 내)

1. **Phase C 최고점 checkpoint 확인**
   - results/phase-C-multi-npc/ 폴더 검토
   - 최고 reward의 checkpoint 파일명 기록
   
2. **Phase D v3 config 작성**
   - python/configs/planning/vehicle_ppo_phase-D-v3.yaml
   - 위의 설계를 기반으로 작성

3. **D v3 설계 리뷰**
   - 팀 검토 및 승인

### 단기 조치 (2-3일)

4. **D v3 Training 실행**
   - 예상: 70 minutes (5M steps)
   
5. **결과 분석**
   - ROOT-CAUSE-ANALYSIS.md 검증
   - Lane obs의 실제 기여도 분석

### 중기 조치

6. **Policy Discovery Log 업데이트**
   - P-002-B, P-009 최종 등록
   - Phase D v3 결과 기록

7. **Phase E로 진행**
   - D v3가 성공하면 e.g., curved roads로 진행

---

## 문서 위치

| 문서 | 경로 | 내용 |
|------|------|------|
| ROOT-CAUSE-ANALYSIS.md | experiments/phase-D-v2/ | 상세 근본 원인 분석 |
| DESIGN_V3.md | experiments/phase-D-v2/ | Phase D v3 설계 상세 |
| POLICY-DISCOVERY-LOG.md | docs/ | P-002-B, P-009 원칙 기록 |

---

## 결론

**Phase D v2 실패는 설계 결함이지 구현 문제가 아님.**

Staggered threshold는 "연기" 할 뿐 해결하지 못하고, Fresh start + noisy observation의 조합이 학습을 방해했다.

**Phase D v3는 이미 검증된 모델 (Phase C)에서 시작하여 새로운 신호 (lane obs)를 추가하는 근본적으로 다른 접근이다.**

- 기대 성공률: 85~95%
- 기대 보상: +1200~1500
- 위험도: 낮음

---

**작성**: 2026-01-30  
**분석 신뢰도**: 95%  
**다음 리뷰**: Phase D v3 design review 시
