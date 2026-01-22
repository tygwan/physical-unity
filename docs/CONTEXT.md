# Project Context

## Quick Summary

**Physical AI with Unity** - Unity ML-Agents를 활용한 물리 기반 AI 에이전트 학습 프로젝트

## Key Information

### Environment
- **OS**: Windows 11 + WSL2 (Ubuntu)
- **Unity**: Windows에 설치 (2023.2+)
- **Python**: WSL2에서 실행 (3.10.x)
- **Communication**: localhost:5004 소켓 통신

### Tech Stack
```
Unity (C#)          Python (WSL)
├── ML-Agents 3.0   ├── PyTorch 2.0+
├── Physics         ├── mlagents
└── Sensors         └── TensorBoard
```

### Current Focus
- Phase 1: 환경 설정
- 학습 로드맵 따라 기초부터 진행

## Important Paths

| 항목 | 경로 |
|-----|------|
| WSL Project | `/home/coffin/dev/physical-unity` |
| Unity Project | `unity/` (Windows에서 생성 예정) |
| Python venv | `.venv/` |
| Models | `models/` |
| Logs | `logs/` |

## Domain Glossary

| 용어 | 설명 |
|-----|------|
| Agent | ML-Agents의 학습 가능한 에이전트 |
| Episode | 하나의 학습 사이클 (시작→목표/실패) |
| Observation | 에이전트가 환경에서 수집하는 상태 정보 |
| Action Space | 에이전트가 취할 수 있는 행동의 집합 |
| Reward | 행동의 좋고 나쁨을 나타내는 신호 |
| PPO | Proximal Policy Optimization, ML-Agents 기본 알고리즘 |
| SAC | Soft Actor-Critic, 연속 행동에 효과적 |

## Decisions Made

1. **WSL + Windows 분리 구조** 선택
   - 이유: Unity는 Windows GUI 필요, ML 학습은 Linux가 효율적

2. **Phase 기반 학습** 채택
   - 이유: 체계적인 지식 축적 필요

## Useful Commands

```bash
# 환경 활성화
source .venv/bin/activate

# 학습 시작
mlagents-learn python/configs/trainer_config.yaml --run-id=test

# TensorBoard
tensorboard --logdir=logs

# 모델 내보내기 후 Unity에서 사용
# models/*.onnx → Unity Assets/
```

## Links

- [ML-Agents Docs](https://unity-technologies.github.io/ml-agents/)
- [학습 로드맵](./LEARNING_ROADMAP.md)
