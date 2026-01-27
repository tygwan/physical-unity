---
layout: default
title: Phase A - Dense Overtaking
---

# Phase A: Dense Overtaking

느린 NPC 추월 기동 학습

---

## Training Summary

| Item | Value |
|------|-------|
| **Run ID** | v12_phaseA_fixed |
| **Status** | ✅ Completed |
| **Date** | 2026-01-24 ~ 2026-01-25 |
| **Total Steps** | 2,000,209 |
| **Training Time** | ~30 minutes |
| **Final Reward** | +714 |
| **Peak Reward** | **+937** (at 1.37M steps) |
| **Observation** | 242D |

---

## Objective

v10g/v11에서 실패한 추월 학습을 Dense Reward로 해결

### Problem (v10g/v11)
- Agent가 느린 NPC 뒤에서 무한 대기
- Sparse reward (+3.0 추월 완료 시)로는 학습 신호 부족
- `followingBonus`가 따라가기를 보상하여 추월 동기 제거

### Solution (v12 Phase A)
- **Dense 5-phase overtaking reward** 설계
- `targetSpeed = speedLimit` 항상 고정
- `followingBonus` 완전 제거
- `stuckBehindPenalty` 추가

---

## Core Changes (v11 → v12)

```yaml
v11 (실패):
  targetSpeed: leadSpeed (NPC 속도에 맞춤)
  followingBonus: +0.3 (따라가기 보상)
  overtakeBonus: +3.0 (완료 시 1회, sparse)

v12 Phase A (성공):
  targetSpeed: speedLimit (항상 제한속도)
  followingBonus: REMOVED
  stuckBehindPenalty: -0.1/step (3초 후)
  overtaking_rewards:
    - initiate: +0.5 (차선 변경 시작)
    - beside: +0.2/step (NPC 옆 유지)
    - ahead: +1.0 (추월 완료)
    - lane_return: +2.0 (차선 복귀)
```

---

## Training Progress

| Step | Reward | Std | Phase |
|------|--------|-----|-------|
| 5K | -127 | 48 | Initial penalty |
| 200K | -244 | 127 | Heavy penalty for 0 speed |
| **460K** | **+7.3** | 162 | **First positive!** |
| 580K | +77.6 | 234 | Rapid improvement |
| 940K | +190.9 | 340 | Consistent positive |
| 1.2M | +502.9 | 298 | Breakthrough |
| **1.37M** | **+937.0** | 324 | **Peak performance** |
| 1.66M | +823.0 | 275 | Stabilizing |
| 2.0M | +714.7 | 234 | Converged |

---

## Key Bug Fix

### Speed Penalty Loophole

```csharp
// Before: 버그 - 0 속도에서 패널티 회피 가능
else if (speedRatio < 0.5f && speed > 1f)  // ← speed > 1f 조건이 허점
{
    reward += speedUnderPenalty;
}

// After: 수정 - 무조건 + 점진적 패널티
else if (speedRatio < 0.5f)
{
    float progressivePenalty = speedUnderPenalty * (2f - speedRatio * 2f);
    reward += progressivePenalty;  // 느릴수록 더 큰 패널티
}
```

**Impact**: Agent가 0-1 m/s로 정지하여 패널티 회피하던 문제 해결

---

## Reward Curve

```
Reward
  +937 │                    ★ Peak
       │                   ╱╲
  +700 │                  ╱  ╲___
       │                 ╱
  +500 │               _╱
       │              ╱
  +200 │            _╱
       │          _╱
    0  │_________╱
       │
 -244  │    ★ Worst
       └────────────────────────────
        0   0.5   1.0   1.5   2.0 M steps
```

---

## Environment Settings

| Parameter | Value |
|-----------|-------|
| NPC Count | 1 |
| NPC Speed | 30% of limit (very slow) |
| Goal Distance | 80m → 150m |
| Speed Zones | 1 |
| Training Areas | 16 parallel |

---

## Checkpoints

| Checkpoint | Step | Reward | Notes |
|------------|------|--------|-------|
| E2EDrivingAgent-499972.onnx | 500K | ~+100 | Early learning |
| E2EDrivingAgent-999988.onnx | 1M | ~+400 | Good progress |
| E2EDrivingAgent-1499899.onnx | 1.5M | ~+800 | Near peak |
| **E2EDrivingAgent-1999953.onnx** | 2M | +714 | **Final (recommended)** |

---

## Success Indicators

- ✅ Agent successfully overtakes slow NPCs (30% speed)
- ✅ Lane change → pass → return behavior observed
- ✅ Reward stabilized in 600-900 range
- ✅ Dense reward eliminated credit assignment problem

---

## Lessons Learned

1. **Dense Reward > Sparse Reward**: 복잡한 행동은 과정 전체에 보상 필요
2. **Speed penalty must be unconditional**: 조건문은 허점 생성
3. **Progressive penalty > Binary**: 부드러운 gradient로 학습 용이
4. **High variance during learning is normal**: Exploration 중 Std > 300 정상

---

## Next Phase

**Phase B**: 추월 vs 따라가기 판단
- NPC 속도 다양화 (30% → 90%)
- 빠른 NPC는 따라가기, 느린 NPC는 추월

---

[← Back to Phases](./index) | [Phase B →](./phase-b) | [Home](../)
