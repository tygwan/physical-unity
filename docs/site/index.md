---
layout: default
title: Home
---

# Autonomous Driving ML Platform

Unity ML-Agents 기반 자율주행 Motion Planning AI 학습 프로젝트

---

## Project Overview

강화학습(RL)을 활용하여 자율주행 차량의 의사결정 능력을 학습시키는 프로젝트입니다.

### Tech Stack

| Component | Technology |
|-----------|------------|
| Simulation | Unity 6 (6000.x) |
| ML Framework | ML-Agents 4.0, PyTorch 2.3 |
| Inference | Unity Sentis 2.4 |
| Algorithm | PPO (Proximal Policy Optimization) |

### Training Environment

- **16 Parallel Training Areas**: 동시 학습으로 데이터 수집 가속화
- **Curriculum Learning**: 단계별 난이도 증가로 점진적 학습
- **254D Observation Space**: 자차 상태, 주변 차량, 경로 정보 등

---

## Training Progress

### Completed Phases

| Phase | Focus | Best Reward | Key Achievement |
|-------|-------|-------------|-----------------|
| [Phase A](./phases/phase-a) | 기본 추월 | **+937** | Dense reward로 추월 기동 학습 |
| [Phase B](./phases/phase-b) | 추월 판단 | **+994** | 추월 vs 따라가기 의사결정 |
| [Phase C](./phases/phase-c) | 다중 NPC | **+1086** | 4대 NPC 환경 일반화 |
| [Phase E](./phases/phase-e) | 곡선 도로 | **+931** | 곡률 1.0까지 주행 |
| [Phase F](./phases/phase-f) | 다중 차선 | **+988** | 2차선 + 중앙선 규칙 |

### Current Training

**Phase G: Intersection Navigation** 🔄

교차로 (T자/십자/Y자) 주행 학습 중

```
Progress: ████░░░░░░░░░░░░░░░░ 10%
Reward:   +492 → target: +800
```

[Phase G 상세 보기](./phases/phase-g)

---

## Reward Evolution

```
v10g:    ████░░░░░░░░░░░░░░░░░░░░░░░░░░  +40  (추월 불가)
v11:     █████░░░░░░░░░░░░░░░░░░░░░░░░░  +51  (Sparse 실패)
Phase A: ███████████████████████░░░░░░░  +937 (추월 성공!)
Phase B: ████████████████████████░░░░░░  +994 (판단력)
Phase C: █████████████████████████░░░░░  +1086 (일반화)
Phase E: ███████████████████████░░░░░░░  +931 (곡선 도로)
Phase F: ████████████████████████░░░░░░  +988 (다중 차선)
Phase G: ███████████░░░░░░░░░░░░░░░░░░░  +492 (교차로 학습중)
```

---

## Gallery

### Training Screenshots

| Phase E: Curved Road | Phase F: Multi-Lane | Phase G: Intersection |
|---------------------|---------------------|----------------------|
| ![Curved](./gallery/screenshots/phase-e-curved.png) | ![Multi-lane](./gallery/screenshots/phase-f-multilane.png) | ![Intersection](./gallery/screenshots/phase-g-intersection.png) |

### Demo Videos

- [Phase A: 첫 추월 성공](./gallery/videos/phase-a-overtake.mp4)
- [Phase F: 다차선 주행](./gallery/videos/phase-f-demo.mp4)

---

## Key Insights

### What Worked ✅

1. **Dense Reward > Sparse Reward**: 추월 과정 전체에 보상 필요
2. **Curriculum Learning**: 점진적 난이도 증가가 핵심
3. **targetSpeed = speedLimit**: 절대 NPC 속도로 낮추면 안 됨

### What Failed ❌

1. **followingBonus**: 따라가기를 보상하면 추월 학습 불가
2. **Encoder Fine-tuning**: Catastrophic forgetting 발생
3. **급격한 환경 변화**: 커리큘럼 충격으로 학습 붕괴

[전체 교훈 보기](./lessons-learned)

---

## Resources

- [GitHub Repository](https://github.com/[username]/physical-unity)
- [Training Log (Detailed)](./training-log)
- [Learning Roadmap](./roadmap)

---

*Last Updated: 2026-01-27*
