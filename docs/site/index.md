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
- **268D Observation Space**: 자차 상태, 주변 차량, 경로, 교차로, 신호등 정보
- **Build Training**: 헤드리스 빌드 기반 병렬 학습 (3x 환경)

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
| [Phase G](./phases/phase-g) | 교차로 | **+628** | T/Cross/Y-junction 7/7 커리큘럼 |
| [Phase H](./phases/phase-h) | NPC 교차로 | **+701** | 3 NPCs + speed variation 11/11 커리큘럼 |
| [Phase I](./phases/phase-i) | 곡선+NPC | **+770** | 곡선도로 + 3 NPCs, 프로젝트 최고 기록 |
| [Phase J](./phases/phase-j) | 신호등 | **+497** | 268D obs, signal-first 3/4 (v4) |

### Next Up

Phase J v5 (lower threshold for green_ratio 0.4) or Phase K (U-turn + special maneuvers)

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
Phase G: ███████████████░░░░░░░░░░░░░░░  +628 (교차로 완료)
Phase H: █████████████████░░░░░░░░░░░░░  +701 (NPC 교차로 완료)
Phase I: ███████████████████░░░░░░░░░░░  +770 (곡선+NPC 완료, 최고기록)
Phase J: ████████████░░░░░░░░░░░░░░░░░░  +497 (신호등 v4, green_ratio=0.5)
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

### What Worked

1. **Dense Reward > Sparse Reward**: 추월 과정 전체에 보상 필요
2. **Curriculum Learning**: 점진적 난이도 증가가 핵심
3. **Warm Start**: 관측 차원 변경 시에도 기존 체크포인트 활용 가능
4. **WrongWay Multi-axis Detection**: 교차로에서는 단일 축 감지 실패
5. **Build Training**: 헤드리스 빌드 + num_envs=3으로 ~3x 학습 속도 향상

### What Failed

1. **followingBonus**: 따라가기를 보상하면 추월 학습 불가
2. **Encoder Fine-tuning**: Catastrophic forgetting 발생
3. **급격한 환경 변화**: 커리큘럼 충격으로 학습 붕괴
4. **Fresh Start with New Obs**: 260D fresh start -> 2M steps 낭비 (Phase G v1)
5. **도달 불가 Threshold**: 변동 활성 시 평균 reward 하락 미반영 (Phase H v2)
6. **Threshold 간격 부족**: 700/702/705 -> 3개 동시 전환 -> 760점 급락 (Phase I v1)
7. **Obs 차원 불일치**: 260D checkpoint -> 268D scene = Adam tensor crash (Phase J v1)
8. **커리큘럼 순서 충돌**: 독립 파라미터 간 순서 보장 불가 -- green_ratio가 signal ON 전에 변경 (Phase J v3)
9. **신호 보상 범위 축소**: green_ratio 낮아질수록 reward 범위 좁아져 threshold 도달 불가 (Phase J v4)

[전체 교훈 보기](./lessons-learned)

---

## Resources

- [GitHub Repository](https://github.com/[username]/physical-unity)
- [Training Log (Detailed)](./training-log)
- [Learning Roadmap](./roadmap)

---

*Last Updated: 2026-02-02 (Phase J v4 Partial 3/4 green_ratio)*
