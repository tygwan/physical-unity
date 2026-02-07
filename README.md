# Autonomous Driving ML Platform

[![Training Progress](https://img.shields.io/badge/Phase-N%20v5b-blue)](https://tygwan.github.io/physical-unity/)
[![Unity](https://img.shields.io/badge/Unity-6000.x-black)](https://unity.com/)
[![ML-Agents](https://img.shields.io/badge/ML--Agents-4.0.1-green)](https://github.com/Unity-Technologies/ml-agents)

Unity ML-Agents 기반 자율주행 모션 플래닝 학습 플랫폼.

> **[📊 Live Training Dashboard →](https://tygwan.github.io/physical-unity/)**
>
> 전체 Phase별 학습 결과, Policy Discovery, 실시간 진행 상황을 확인하세요.

---

## Overview

| Component | Technology |
|-----------|-----------|
| Simulation | Unity 6 (6000.x) |
| ML Framework | ML-Agents 4.0.1, PyTorch 2.3.1 |
| Hardware | RTX 4090 (24GB VRAM), 128GB RAM |

### Current Status

- **Latest Phase**: N v5b (ProceduralRoadBuilder, +521.8 reward)
- **Observation Space**: 280D vector
- **Completed Phases**: 0 → A → B v2 → C → D v3 → E → F v4 → G → H v3 → I v2 → J v5 → K v1 → L v5 → N v5b
- **Policy Discoveries**: P-001 ~ P-029 (29 verified principles)

---

## Quick Start

### Training

```powershell
# TensorBoard 모니터링
tensorboard --logdir=results

# 학습 시작
mlagents-learn python/configs/planning/vehicle_ppo_phase-N-v1.yaml --run-id=phase-N-v1

# Unity Editor에서 Play 버튼 클릭
```

### Inference

1. 모델 복사: `results/<run-id>/E2EDrivingAgent.onnx` → `Assets/ML-Agents/Models/`
2. Unity Inspector에서:
   - `BehaviorParameters > Model`에 ONNX 파일 할당
   - `BehaviorType`을 **Inference Only**로 변경
3. Play 모드 실행

---

## Project Structure

```
physical-unity/
├── Assets/Scripts/Agents/     # E2EDrivingAgent
├── python/configs/planning/   # Training YAML configs
├── results/                   # TensorBoard logs & models
├── docs/                      # Detailed documentation
│   ├── TRAINING-LOG.md       # Full training history
│   └── phases/               # Phase-specific docs
└── site/portfolio/            # Training dashboard (Astro)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  280D Observation  →  MLP Policy (PPO)  →  2D Action       │
│  ├─ Ego State (8D)     ├─ 3-layer NN       ├─ Steering     │
│  ├─ Route (30D)        └─ 512 hidden       └─ Accel        │
│  ├─ NPCs (152D)                                             │
│  ├─ Lane (12D)                                              │
│  ├─ Traffic Signal (8D)                                     │
│  ├─ Intersection (18D)                                      │
│  ├─ Pedestrian (12D)                                        │
│  └─ Goal (12D)                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Documentation

| Resource | Description |
|----------|-------------|
| [📊 Training Dashboard](https://tygwan.github.io/physical-unity/) | Live phase progress & policy discoveries |
| [TRAINING-LOG.md](docs/TRAINING-LOG.md) | Detailed training experiment logs |
| [CLAUDE.md](.claude/CLAUDE.md) | Development workflow & conventions |

---

## Tech Stack

| Component | Version |
|-----------|---------|
| Unity | 6000.x (Unity 6) |
| ML-Agents | 4.0.1 (Unity Package) |
| Sentis | 2.4.1 (ONNX inference) |
| Python | 3.10.11 |
| PyTorch | 2.3.1 |
| CUDA | 12.x |

---

## Development

This project uses [cc-initializer](https://github.com/tygwan/cc-initializer) for Claude Code automation:
- 38 AI agents (training-analyst, training-monitor, forensic-analyst, etc.)
- 22 skills (/experiment, /train, /evaluate, /phase)
- Automated hooks for quality gates

---

## License

[MIT License](LICENSE)

---

**Last Updated**: 2026-02-07 | **[View Full Progress →](https://tygwan.github.io/physical-unity/)**
