# Project Context

## Quick Summary

**Autonomous Driving ML Platform** - Unity ML-Agents를 활용한 자율주행 Motion Planning AI 학습 프로젝트

## Key Information

### Environment
- **OS**: Windows 11 (Native - WSL 미사용)
- **Unity**: 6000.x (Unity 6) - Windows
- **Python**: Windows Native (3.10.11)
- **Communication**: ML-Agents 4.0 (localhost:5004 gRPC)

### Tech Stack
```
Unity (C#)              Python (Windows Native)
├── ML-Agents 4.0.1     ├── PyTorch 2.1+ (CUDA 12.x)
├── Sentis 2.4.1        ├── mlagents 1.2.0
├── Physics             ├── mlagents-envs 1.2.0
└── Sensors             └── TensorBoard / MLflow
```

### Current Focus
- **Phase 5**: Planning Models (RL/IL) - PRIMARY FOCUS
- **Phase D**: Lane Observation (254D) 학습 진행중

## Important Paths

| 항목 | 경로 |
|-----|------|
| Project Root | `C:\Users\user\Desktop\dev\physical-unity` |
| Unity Assets | `Assets/` |
| Python Code | `python/` |
| Python venv | `python/.venv/` |
| ML Configs | `python/configs/planning/` |
| Models (ONNX) | `models/planning/` |
| Results | `results/` |
| Logs | `results/<run-id>/` |

## Domain Glossary

| 용어 | 설명 |
|-----|------|
| Agent | ML-Agents의 학습 가능한 에이전트 |
| Episode | 하나의 학습 사이클 (시작→목표/실패) |
| Observation | 에이전트가 환경에서 수집하는 상태 정보 (242D/254D) |
| Action Space | 에이전트가 취할 수 있는 행동 (steering, acceleration) |
| Reward | 행동의 좋고 나쁨을 나타내는 신호 |
| PPO | Proximal Policy Optimization, ML-Agents 기본 알고리즘 |
| SAC | Soft Actor-Critic, 연속 행동에 효과적 |
| Curriculum | 점진적 난이도 증가 학습 전략 |

## Decisions Made

1. **Windows Native 환경** 선택
   - 이유: WSL2 대비 Unity-Python 통신 안정성, GPU 직접 접근
   - ML-Agents가 Windows에서 완전 지원

2. **Phase 기반 학습** 채택
   - Phase 1-2: Foundation & Data (완료)
   - Phase 3-4: Perception/Prediction (보류 - GT/CV 사용)
   - Phase 5: Planning (핵심 - RL/IL 모션 플래닝)

3. **Curriculum Learning** 채택
   - 단일 NPC → Multi-NPC 점진적 학습
   - Phase A → B → C → D 순차 진행

## Useful Commands

```powershell
# Windows PowerShell에서 실행

# 학습 시작 (Unity Editor Play 버튼 필요)
mlagents-learn python/configs/planning/vehicle_ppo_phase-D.yaml --run-id=phase-D

# 이전 학습에서 이어서
mlagents-learn python/configs/planning/vehicle_ppo.yaml --run-id=v12 --resume

# TensorBoard 모니터링
tensorboard --logdir=results

# ONNX 모델 → Unity에서 Sentis로 추론
# models/planning/*.onnx → Unity Assets/Resources/
```

## Hardware

| Component | Spec | Notes |
|-----------|------|-------|
| GPU | RTX 4090 (24GB VRAM) | CUDA 12.x |
| RAM | 128GB | 대용량 데이터 처리 |
| Storage | 4TB SSD | 데이터셋 + 체크포인트 |

## Links

- [ML-Agents Docs](https://unity-technologies.github.io/ml-agents/)
- [학습 로드맵](./LEARNING-ROADMAP.md)
- [Phase별 문서](./phases/README.md)
- [학습 기록](./TRAINING-LOG.md)
