# TensorBoard Metrics Guide - ML-Agents PPO Training

## Overview

ML-Agents PPO 학습 중 TensorBoard에 기록되는 모든 성능 지표에 대한 상세 가이드입니다.

**목적**: 학습 진행 상황을 정확히 이해하고, 문제 진단 및 하이퍼파라미터 튜닝에 활용

---

## Table of Contents

1. [Environment Metrics](#1-environment-metrics) - 환경 성능 지표
2. [Policy Metrics](#2-policy-metrics) - 정책 네트워크 지표
3. [Value Function Metrics](#3-value-function-metrics) - 가치 함수 지표
4. [Loss Metrics](#4-loss-metrics) - 손실 함수 지표
5. [Learning Statistics](#5-learning-statistics) - 학습 통계
6. [Self-Play Metrics](#6-self-play-metrics) - 자기 대전 지표 (해당 시)

---

## 1. Environment Metrics

### 1.1 Cumulative Reward

**경로**: `Environment/Cumulative Reward`

**정의**: 에피소드 동안 누적된 총 보상

**수식**:
```
R_cumulative = Σ(r_t) for t=0 to T
```
where:
- `r_t`: 시간 t에서 받은 보상
- `T`: 에피소드 종료 시점

**해석**:
- **상승 추세**: 학습이 잘 진행되고 있음 (목표 달성)
- **정체**:
  - Plateau 도달 (수렴)
  - 또는 학습 정체 (하이퍼파라미터 조정 필요)
- **하락 추세**:
  - Catastrophic forgetting (학습률 너무 높음)
  - 환경 변화 (curriculum 전환)
- **진동(Oscillation)**:
  - 학습 불안정 (batch size 증가, learning rate 감소 고려)

**목표 값**:
- Phase 0 (Foundation): +1000
- Phase A (Overtaking): +900-1100
- Phase B (Decision): +1500-1800

**관련 지표**: Episode Length와 함께 분석 필요

---

### 1.2 Episode Length

**경로**: `Environment/Episode Length`

**정의**: 에피소드가 종료될 때까지 걸린 스텝 수

**수식**:
```
Episode_Length = T (number of steps until done=True)
```

**해석**:
- **증가 추세**:
  - 에이전트가 더 오래 생존 (충돌/실패 감소)
  - 목표 지점이 더 멀어짐
- **감소 추세**:
  - 빠른 실패 (충돌 증가)
  - 또는 목표를 더 빨리 달성 (효율성 증가)
- **안정적 유지**: 환경이 일정한 난이도 유지

**주의사항**:
- Episode Length가 max_steps에 도달하면 강제 종료
- Cumulative Reward와 함께 봐야 정확한 해석 가능
  - 긴 에피소드 + 낮은 보상 = 비효율적
  - 짧은 에피소드 + 높은 보상 = 효율적

---

### 1.3 Lesson (Curriculum)

**경로**: `Environment/Lesson`

**정의**: 현재 curriculum learning의 단계

**수식**:
```
Lesson_num ∈ {0, 1, 2, ..., N}
```

**해석**:
- 숫자가 증가하면 다음 난이도로 전환
- Lesson 전환 시점에서 Reward 일시 하락은 정상
- 전환 기준: `completion_criteria` 충족 시

**Phase B Curriculum**:
```
Lesson 0: Baseline (0 NPCs)            → 750K steps
Lesson 1: Single NPC (forced overtake) → 1.5M steps
Lesson 2: Mixed NPCs (selective)       → 2.25M steps
Lesson 3: Dense traffic (complex)      → 3.0M steps
```

---

## 2. Policy Metrics

### 2.1 Policy/Entropy

**경로**: `Policy/Entropy`

**정의**: 정책 분포의 엔트로피 (행동 다양성)

**수식**:
```
H(π) = -Σ π(a|s) log π(a|s)
```
where:
- `π(a|s)`: 상태 s에서 행동 a를 선택할 확률
- 높은 H(π): 행동이 균등하게 분산 (탐험 많음)
- 낮은 H(π): 특정 행동에 집중 (탐험 적음)

**해석**:
- **학습 초기 (High Entropy)**:
  - 값: 1.0-2.0 (연속 행동 공간)
  - 에이전트가 다양한 행동 시도 (Exploration)
- **학습 중기 (Decreasing)**:
  - 서서히 감소 → 정책 수렴 중
- **학습 후기 (Low Entropy)**:
  - 값: 0.1-0.5
  - 확정적 정책으로 수렴 (Exploitation)
- **너무 빠른 감소 (< 0.05)**:
  - Premature convergence (조기 수렴)
  - β (entropy coefficient) 증가 고려

**연관 하이퍼파라미터**:
```yaml
beta: 5e-3  # 엔트로피 계수 (높을수록 탐험 증가)
```

**목표 범위**:
- 초기: > 1.0
- 중기: 0.3-0.8
- 후기: 0.1-0.5

---

### 2.2 Policy/Learning Rate

**경로**: `Policy/Learning Rate`

**정의**: 정책 네트워크의 현재 학습률

**수식**:
```
# Constant Schedule
lr(t) = lr_initial

# Linear Decay
lr(t) = lr_initial × (1 - t/T_max)

# Exponential Decay
lr(t) = lr_initial × exp(-decay_rate × t)
```

**해석**:
- **Constant**: 전체 학습 기간 동안 동일 (안정적, 단순)
- **Linear Decay**: 시간에 따라 선형 감소 (후반부 미세 조정)
- **Exponential Decay**: 빠르게 감소 (빠른 수렴)

**설정값** (Phase B):
```yaml
learning_rate: 3e-4
learning_rate_schedule: constant
```

**트러블슈팅**:
- Reward 진동 심함 → learning rate 낮추기
- 학습 너무 느림 → learning rate 높이기
- 일반적 범위: 1e-5 ~ 1e-3

---

### 2.3 Policy/Extrinsic Reward

**경로**: `Policy/Extrinsic Reward`

**정의**: 외부 환경에서 받은 실제 보상 (내재적 보상 제외)

**수식**:
```
r_extrinsic(s, a) = reward from environment
```

**구성 요소** (Phase B):
```python
r_extrinsic = progress_reward          # +1.0
            + goal_bonus                # +10.0
            + speed_compliance          # +0.3
            + overtaking_bonus          # +5.0
            + lane_center_reward        # +0.2
            - collision_penalty         # -10.0
            - near_collision_penalty    # -2.0
            - following_penalty         # -0.5
```

**해석**:
- Cumulative Reward와 유사하지만 내재적 보상(Curiosity 등) 제외
- 대부분의 경우 Cumulative Reward = Extrinsic Reward

---

### 2.4 Policy/Extrinsic Value Estimate

**경로**: `Policy/Extrinsic Value Estimate`

**정의**: 가치 함수가 예측한 미래 누적 보상 기댓값

**수식**:
```
V(s) = E[Σ γ^t r_{t+k} | s_t = s]
```
where:
- `γ`: 할인율 (discount factor) = 0.995
- `r_{t+k}`: 미래 보상

**해석**:
- **Extrinsic Reward와 비교**:
  - Value가 Reward보다 높음 → 낙관적 예측 (과대평가)
  - Value가 Reward보다 낮음 → 비관적 예측 (과소평가)
  - 두 값이 수렴 → 정확한 가치 함수 학습
- **일관성**:
  - Reward 상승 시 Value도 상승 → 정상
  - Reward 상승하는데 Value 하락 → 가치 함수 문제

**목표**: Value Estimate ≈ Cumulative Reward

---

## 3. Value Function Metrics

### 3.1 Losses/Value Loss

**경로**: `Losses/Value Loss`

**정의**: 가치 함수의 예측 오차 (MSE)

**수식**:
```
L_value = (1/N) Σ (V_θ(s_t) - R_t)^2
```
where:
- `V_θ(s_t)`: 가치 함수의 예측값
- `R_t`: 실제 누적 보상 (return)
- `N`: 배치 크기

**해석**:
- **감소 추세**: 가치 함수가 정확해지고 있음 (정상)
- **높은 값 (> 1000)**:
  - 가치 함수가 큰 오차로 예측
  - 환경 보상 스케일이 크거나 분산이 큼
- **진동**:
  - 학습 불안정
  - Learning rate 낮추기 or batch size 증가
- **증가 추세**:
  - Catastrophic forgetting
  - 하이퍼파라미터 재검토 필요

**정상 범위**:
- 초기: 10-100
- 중기: 1-10
- 후기: 0.1-5

---

### 3.2 Value/Mean Value

**경로**: `Value/Mean Value`

**정의**: 전체 상태의 평균 가치 함수 값

**수식**:
```
V_mean = (1/N) Σ V(s_i)
```

**해석**:
- Extrinsic Value Estimate와 유사
- 학습이 진행될수록 Cumulative Reward에 수렴
- Reward 대비 Value가 지속적으로 높으면 과적합 우려

---

## 4. Loss Metrics

### 4.1 Losses/Policy Loss

**경로**: `Losses/Policy Loss`

**정의**: PPO의 clipped surrogate objective 손실

**수식**:
```
L_CLIP(θ) = -E[min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]
```
where:
- `r_t(θ) = π_θ(a|s) / π_θ_old(a|s)`: probability ratio
- `A_t`: advantage function (얼마나 현재 행동이 평균보다 좋은지)
- `ε`: clipping parameter (0.2)
- `clip()`: r_t를 [1-ε, 1+ε] 범위로 제한

**해석**:
- **작은 값 (< 0.1)**:
  - 정책이 거의 변하지 않음 (안정적)
  - 너무 작으면 학습 정체
- **큰 값 (> 1.0)**:
  - 정책이 크게 변함 (불안정)
  - Learning rate 낮추기
- **진동**:
  - 정상 (매 업데이트마다 정책 조정)
  - 진폭이 크면 batch_size 증가

**목표 범위**: 0.01-0.5

---

### 4.2 Policy/Epsilon (PPO Clip Range)

**경로**: `Policy/Epsilon`

**정의**: PPO clipping 범위 (정책 업데이트 제한)

**수식**:
```
clip_range = [1 - ε, 1 + ε]
```

**설정값**:
```yaml
epsilon: 0.2  # 기본값
```

**해석**:
- ε = 0.2 → 정책 확률비를 [0.8, 1.2]로 제한
- 너무 작으면 (< 0.1): 학습 느림
- 너무 크면 (> 0.3): 불안정

**관련 지표**: Policy Loss와 함께 분석

---

### 4.3 Policy/Beta (Entropy Coefficient)

**경로**: `Policy/Beta`

**정의**: 엔트로피 보너스 계수 (탐험 장려)

**수식**:
```
L_total = L_CLIP + c_1 × L_value - β × H(π)
```

**설정값**:
```yaml
beta: 5e-3  # 0.005
```

**해석**:
- 높은 β (> 0.01): 탐험 많음 (다양한 행동 시도)
- 낮은 β (< 0.001): 탐험 적음 (확정적 정책)
- **Entropy가 너무 빨리 감소**하면 β 증가

---

## 5. Learning Statistics

### 5.1 Policy/Gradient Norm

**경로**: `Policy/Gradient Norm`

**정의**: 정책 gradient의 L2 norm (업데이트 크기)

**수식**:
```
||∇θ|| = sqrt(Σ (∂L/∂θ_i)^2)
```

**해석**:
- **큰 값 (> 10)**:
  - Gradient exploding 위험
  - Learning rate 낮추기 또는 gradient clipping 적용
- **작은 값 (< 0.01)**:
  - Vanishing gradient
  - Learning rate 높이기 또는 네트워크 구조 변경
- **정상 범위**: 0.1-5.0

**관련 설정**:
```yaml
# Gradient clipping (ML-Agents 기본값)
max_grad_norm: 0.5
```

---

### 5.2 Policy/Approx KL Divergence

**경로**: `Policy/Approx KL Divergence`

**정의**: 이전 정책과 현재 정책 간의 KL divergence (변화량)

**수식**:
```
D_KL(π_old || π_new) ≈ E[(log π_old(a|s) - log π_new(a|s))]
```

**해석**:
- **작은 값 (< 0.01)**: 정책이 거의 안 변함 (안전, 느림)
- **적정 값 (0.01-0.05)**: 학습 진행 중
- **큰 값 (> 0.1)**:
  - 정책이 급격히 변함 (불안정)
  - Learning rate 낮추기
- **목표**: 0.01-0.03 (안정적 학습)

**관련 개념**: Early stopping in PPO
- ML-Agents는 KL divergence가 너무 크면 조기 종료 가능

---

### 5.3 Is Training

**경로**: `Is Training`

**정의**: 현재 학습 중인지 여부 (binary)

**값**:
- `1`: 학습 중 (buffer가 충분히 쌓여 업데이트 중)
- `0`: 경험 수집 중 (학습 대기)

**해석**:
- 대부분 1이어야 정상
- 0이 길게 유지되면:
  - Buffer size가 너무 큼
  - Batch size가 너무 큼
  - 환경이 너무 느림

---

## 6. Self-Play Metrics

(Phase B에서는 사용하지 않음, 추후 Multi-Agent 시 참고)

### 6.1 Self-Play/ELO Rating

**정의**: 에이전트의 상대적 실력 (체스 레이팅 방식)

**사용 시점**: Multi-agent competitive 환경

---

## 7. 지표 해석 실전 가이드

### 7.1 건강한 학습 패턴

```
✅ Cumulative Reward: 꾸준히 상승
✅ Episode Length: 안정적 또는 증가
✅ Entropy: 서서히 감소 (1.5 → 0.3)
✅ Value Loss: 감소 추세
✅ Policy Loss: 작고 안정적 (< 0.5)
✅ Value Estimate ≈ Cumulative Reward
✅ KL Divergence: 0.01-0.03
```

### 7.2 문제 패턴 및 해결책

| 문제 | 증상 | 원인 | 해결책 |
|------|------|------|--------|
| **학습 불안정** | Reward 심하게 진동 | Learning rate 너무 높음 | `learning_rate: 3e-4 → 1e-4` |
| **학습 정체** | Reward 오랫동안 평평 | Learning rate 너무 낮음 or plateau | `learning_rate` 증가 or curriculum 전환 |
| **조기 수렴** | Entropy < 0.05 너무 빨리 | Beta 너무 낮음 | `beta: 5e-3 → 1e-2` |
| **가치 함수 오차** | Value Loss > 100 지속 | 보상 스케일 너무 큼 | Reward normalization 활성화 |
| **Gradient Exploding** | Gradient Norm > 10 | Learning rate 높음 | `learning_rate` 감소 or gradient clipping |
| **KL Too High** | KL Divergence > 0.1 | 정책 변화 너무 급격 | `learning_rate` 감소 or `epsilon` 감소 |

### 7.3 Phase별 목표 지표

#### Phase 0 (Foundation)
```
Cumulative Reward: +1000
Episode Length: 800-1200 steps
Entropy: 0.2-0.5 (수렴)
Value Loss: < 10
```

#### Phase A (Overtaking)
```
Cumulative Reward: +900-1100
Episode Length: 700-1000 steps
Entropy: 0.3-0.6
Overtaking Events: > 100
```

#### Phase B (Decision Learning) - 현재
```
Cumulative Reward: +1500-1800 (목표)
Episode Length: 800-1200 steps
Entropy: 0.2-0.5
Overtaking Success Rate: > 70%
Value Loss: < 20
```

---

## 8. TensorBoard 사용법

### 8.1 실행

```bash
# Phase B 학습 모니터링
tensorboard --logdir results/phase-B-decision

# 여러 Phase 비교
tensorboard --logdir_spec=\
  "Phase 0:results/phase-0-foundation,\
   Phase A:results/phase-A-overtaking,\
   Phase B:results/phase-B-decision"
```

### 8.2 주요 탭

1. **SCALARS**: 위의 모든 수치 지표
2. **GRAPHS**: 네트워크 구조 시각화
3. **DISTRIBUTIONS**: 가중치 분포
4. **HISTOGRAMS**: 활성화 값 분포

### 8.3 유용한 기능

- **Smoothing**: 노이즈 제거 (0.6-0.9 추천)
- **Run Comparison**: 여러 실험 비교
- **Download CSV**: 지표 데이터 내보내기

---

## 9. 체크리스트: 학습 모니터링

학습 중 주기적으로 확인할 사항:

**매 1시간마다**:
- [ ] Cumulative Reward가 상승하는가?
- [ ] Entropy가 적절히 감소하는가? (너무 빠르지 않은가?)
- [ ] Value Loss가 감소하는가?

**매 500K steps마다**:
- [ ] Checkpoint 저장되었는가?
- [ ] Episode Length 안정적인가?
- [ ] KL Divergence < 0.05인가?
- [ ] Curriculum 전환이 필요한가?

**문제 발생 시**:
- [ ] Training log 확인 (`training.log`)
- [ ] Unity Console 에러 확인
- [ ] GPU 메모리 사용량 확인
- [ ] 디스크 공간 확인

---

## 10. 참고 자료

### 공식 문서
- [ML-Agents TensorBoard Guide](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Using-Tensorboard.md)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

### 내부 문서
- `TRAINING-LOG.md`: 학습 기록
- `PROGRESS.md`: Phase별 진행 상황
- `LEARNING-ROADMAP.md`: 학습 전략

### 수식 참고
- `γ` (gamma): discount factor
- `λ` (lambda): GAE parameter
- `ε` (epsilon): PPO clip range
- `β` (beta): entropy coefficient

---

**마지막 업데이트**: 2026-01-29
**작성자**: Claude Code
**대상**: Phase B Decision Learning Training
