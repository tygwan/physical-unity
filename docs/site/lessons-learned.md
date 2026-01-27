---
layout: default
title: Lessons Learned
---

# Lessons Learned

학습 과정에서 얻은 성공과 실패의 교훈

---

## Success Stories ✅

### 1. Dense Reward가 Sparse Reward를 압도 (v11 → Phase A)

**상황**: v11에서 추월 완료 시에만 +3.0 보상 (sparse)
**결과**: 8M steps 동안 reward +51로 정체

**해결**: Phase A에서 5단계 dense reward 설계
```
Approaching → Beside (+0.5) → Beside유지 (+0.2/step) → Ahead (+1.0) → LaneReturn (+2.0)
```
**결과**: 2M steps만에 +937 달성

**교훈**:
> 복잡한 행동은 전 과정에 걸쳐 보상해야 학습 가능

---

### 2. targetSpeed 고정이 핵심 (v10g → v12)

**상황**: v10g/v11에서 `targetSpeed = leadSpeed` (느린 NPC 속도에 맞춤)
**결과**: 따라가기만 해도 속도 보상 최대, 추월 동기 없음

**해결**: `targetSpeed = speedLimit ALWAYS`
```csharp
// Before: 추월 불가
float targetSpeed = hasSlowLead ? leadSpeed : speedLimit;

// After: 추월 학습 성공
float targetSpeed = speedLimit;  // 항상 제한속도
```

**교훈**:
> 의도한 행동과 보상 구조가 정렬되어야 함

---

### 3. Curriculum Shock은 정상이고 회복 가능 (Phase C)

**상황**: Phase C에서 NPC 1대 → 4대 전환 시 reward +766 → -814 급락
**우려**: 학습 실패인가?

**결과**: 1M steps 후 +1086까지 회복 (이전 최고 기록)

**교훈**:
> 커리큘럼 전환 시 일시적 하락은 정상, 충분한 학습 시간 부여

---

### 4. 점진적 복잡도 증가 (Phase A → B → C → ...)

**패턴**:
- Phase A: 1 NPC @ 30% speed
- Phase B: 1 NPC @ 30-90% speed (속도 다양화)
- Phase C: 4 NPC @ 다양한 speed (수량 다양화)
- Phase E: 곡선 도로 추가
- Phase F: 다차선 추가
- Phase G: 교차로 추가

**교훈**:
> 한 번에 하나의 새로운 요소만 추가

---

## Failure Stories ❌

### 1. followingBonus가 추월을 막음 (v10g)

**설계 의도**: NPC와 안전 거리 유지 시 +0.3 보상
**실제 결과**: 느린 NPC 뒤에서 무한 대기가 "최적 정책"

**분석**:
```
추월 시도: collision risk + lane deviation penalty + 불확실한 보상
따라가기: followingBonus + 안전 = 확실한 보상
```

**수정**: followingBonus 완전 제거, stuckBehindPenalty (-0.1/step) 추가

**교훈**:
> 의도치 않은 "꼼수"가 가능한지 검토 필수

---

### 2. HybridPolicy Catastrophic Forgetting (v12_HybridPolicy)

**설계 의도**: Phase B encoder 유지 + lane encoder 추가
**접근 방식**: 6단계 점진적 unfreeze (value → lane → combiner → policy → fusion → encoder)

**결과**:
| Stage | Reward | 상태 |
|-------|--------|------|
| Stage 4 | -82.7 | **최고점** |
| Stage 5 (encoder unfreeze) | -2171.9 | **붕괴** |

**분석**: 0.1x learning rate로도 encoder weights 불안정화

**교훈**:
> 사전학습 encoder는 unfreeze하지 말 것, 차라리 처음부터 재학습

---

### 3. Sparse Reward의 한계 (v11)

**설계**: 추월 완료 시에만 +3.0 (sparse reward)
**결과**: 8M steps 동안 +51로 정체 (v10g +40 대비 미미한 개선)

**분석**:
- 추월 과정: 100+ steps
- 보상 신호: 1번 (완료 시)
- Credit assignment problem: 어떤 행동이 좋았는지 알 수 없음

**교훈**:
> 복잡한 행동은 dense reward 필수

---

### 4. 급격한 환경 변화 (curriculum_v7)

**상황**: speed zone을 갑자기 도입
**결과**: 학습 붕괴, reward -12로 추락

**분석**:
- 이전: 단일 속도 환경
- 이후: 4개 속도 구간 + 구간별 다른 목표 속도
- Agent: 완전히 새로운 환경으로 인식

**수정**: 점진적 도입 (1 zone → 2 → 4)

**교훈**:
> 환경 변화는 커리큘럼으로 점진적 도입

---

## Best Practices

### Reward Design

```yaml
DO:
  - Dense reward (과정 전체에 보상)
  - Progressive penalty (점진적 패널티)
  - Rate-independent (× deltaTime)
  - 의도한 행동 = 높은 보상 검증

DON'T:
  - Sparse reward만 사용
  - 꼼수 가능한 보상 구조
  - 프레임 종속적 보상
```

### Curriculum Design

```yaml
DO:
  - 한 번에 하나의 새 요소
  - 점진적 난이도 (2배 이하)
  - min_lesson_length > 300
  - 충격 후 회복 시간 고려

DON'T:
  - 동시에 여러 요소 추가
  - 급격한 난이도 점프
  - 너무 짧은 lesson
```

### Observation Space

```yaml
DO:
  - 정규화 (-1~1 또는 0~1)
  - 모듈별 분리
  - Space Size = 실제 차원 검증

DON'T:
  - 정규화 안 된 raw 값
  - 중복 정보
  - 차원 불일치
```

---

## Key Metrics to Monitor

| Metric | 정상 | 주의 | 위험 |
|--------|------|------|------|
| Reward Trend | 상승/안정 | 정체 >500K | 하락 |
| Std of Reward | <100 | 100-300 | >300 (장기) |
| Curriculum Progress | 진행 중 | 정체 | 역행 |
| Policy Entropy | 감소 추세 | 급락 | 0 수렴 |

---

[← Back to Home](./)
