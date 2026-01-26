# Development Phases

Autonomous Driving ML Platform의 개발 단계별 기술 설계서입니다.

## Phase Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    AUTONOMOUS DRIVING ML PLATFORM - PHASES                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐                  │
│  │ Phase 1  │ → │ Phase 2  │ → │ Phase 3  │ → │ Phase 4  │                  │
│  │Foundation│   │  Data    │   │Perception│   │Prediction│                  │
│  │    ✅    │   │    ✅    │   │    ⏸️    │   │    ⏸️    │                  │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘                  │
│                                      │              │                        │
│                                      └──────┬───────┘                        │
│                                             │ (Simplified)                   │
│                                             ▼                                │
│                                      ┌──────────┐                            │
│                                      │ Phase 5  │ ← PRIMARY FOCUS            │
│                                      │ Planning │                            │
│                                      │    🔄    │                            │
│                                      └────┬─────┘                            │
│                                           │                                  │
│                                           ▼                                  │
│                   ┌──────────┐     ┌──────────┐                              │
│                   │ Phase 6  │  →  │ Phase 7  │                              │
│                   │Integration│     │ Advanced │                              │
│                   │    📋    │     │    📋    │                              │
│                   └──────────┘     └──────────┘                              │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Legend: ✅ 완료  🔄 진행중  ⏸️ 보류  📋 계획
```

## Phase Status Summary

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| **Phase 1** | Foundation & Architecture | ✅ 완료 | Unity 6, ML-Agents 4.0, Windows Native 환경 구축 |
| **Phase 2** | Data Infrastructure | ✅ 완료 | nuPlan/Waymo 데이터 파이프라인 |
| **Phase 3** | Perception Models | ⏸️ 보류 | Ground Truth 방식으로 단순화 |
| **Phase 4** | Prediction Models | ⏸️ 보류 | Constant Velocity 방식으로 단순화 |
| **Phase 5** | Planning Models | 🔄 **진행중** | RL/IL 기반 Motion Planner (핵심) |
| **Phase 6** | Integration & Evaluation | 📋 계획 | E2E 통합, nuPlan 벤치마크 |
| **Phase 7** | Advanced Topics | 📋 계획 | World Model, LLM Planning, Sim-to-Real |

## Phase 5: Planning - 세부 진행 상황

Phase 5는 프로젝트의 **핵심 Phase**로, 별도의 학습 로드맵으로 관리됩니다.

> 📖 **상세 학습 계획**: [LEARNING-ROADMAP.md](../LEARNING-ROADMAP.md)

### Planning Sub-Phases (RL/IL Training)

| Sub-Phase | Focus | Steps | Best Reward | Status |
|-----------|-------|-------|-------------|--------|
| **Foundation (v10-v11)** | 기본 주행 + 추월 시도 | 16M | +51 | ✅ 완료 |
| **Phase A** | Dense Overtaking (느린 NPC) | 2M | **+937** | ✅ 완료 |
| **Phase B** | Overtake vs Follow 판단 | 2M | **+903** | ✅ 완료 |
| **Phase C** | Multi-NPC 일반화 (4대) | 4M | **+961** | ✅ 완료 |
| **Phase D** | Lane Observation (254D) | 6M | **+332** | ✅ 완료 |
| **Phase E** | 곡선 도로 + 비정형 각도 | 4-6M | - | 📋 계획 |
| **Phase F** | N차선 + 중앙선 규칙 | 4-6M | - | 📋 계획 |
| **Phase G** | 교차로 (T자/십자/Y자) | 6-8M | - | 📋 계획 |
| **Phase H** | 신호등 + 정지선 | 4-6M | - | 📋 계획 |
| **Phase I** | U턴 + 특수 기동 | 4-6M | - | 📋 계획 |
| **Phase J** | 횡단보도 + 보행자 | 6-8M | - | 📋 계획 |
| **Phase K** | 장애물 + 긴급 상황 | 6-8M | - | 📋 계획 |
| **Phase L** | 복합 시나리오 통합 | 10-15M | - | 📋 계획 |

## Directory Structure

```
docs/phases/
├── README.md              # This file
├── phase-1/
│   └── SPEC.md           # Foundation & Architecture
├── phase-2/
│   └── SPEC.md           # Data Infrastructure
├── phase-3/
│   └── SPEC.md           # Perception Models (Simplified)
├── phase-4/
│   └── SPEC.md           # Prediction Models (Simplified)
├── phase-5/
│   └── SPEC.md           # Planning Models (PRIMARY FOCUS)
├── phase-6/
│   └── SPEC.md           # Integration & Evaluation
└── phase-7/
    └── SPEC.md           # Advanced Topics
```

## Document Relationship

```
PRD.md                    # 전체 제품 요구사항
    │
    ├── docs/phases/      # 인프라 + 기술 설계 (Phase 1-7)
    │   ├── phase-1/      # Foundation
    │   ├── phase-2/      # Data
    │   ├── phase-3/      # Perception (보류)
    │   ├── phase-4/      # Prediction (보류)
    │   ├── phase-5/      # Planning (기술 설계)
    │   ├── phase-6/      # Integration
    │   └── phase-7/      # Advanced
    │
    └── LEARNING-ROADMAP.md  # RL/IL 학습 로드맵 (Phase A-L)
                              # Phase 5의 세부 학습 계획
```

## Key Milestones

| Milestone | Target | Status |
|-----------|--------|--------|
| M1 | Unity-ML-Agents 연동 | ✅ 완료 |
| M2 | 데이터셋 파이프라인 | ✅ 완료 |
| M3 | Perception MVP | ⏸️ 보류 (GT 사용) |
| M4 | Prediction MVP | ⏸️ 보류 (CV 사용) |
| M5 | Planning MVP (RL Planner) | 🔄 진행중 |
| M6 | E2E 통합 시스템 | 📋 계획 |

## Success Criteria (Overall)

| Category | Metric | Target | Current |
|----------|--------|--------|---------|
| Safety | Collision Rate | < 5% | 🔄 개선중 |
| Comfort | Jerk | < 2 m/s³ | ✅ 달성 |
| Progress | Route Completion | > 85% | 🔄 개선중 |
| Latency | End-to-end | < 200ms | ✅ 달성 |

## Related Documents

- [PRD.md](../PRD.md) - 제품 요구사항 문서
- [LEARNING-ROADMAP.md](../LEARNING-ROADMAP.md) - RL/IL 학습 로드맵
- [TRAINING-LOG.md](../TRAINING-LOG.md) - 학습 실험 기록
- [PROGRESS.md](../PROGRESS.md) - 전체 진행 상황

## Current Environment

| Component | Version | Notes |
|-----------|---------|-------|
| OS | Windows 11 | Native (WSL 미사용) |
| Unity | 6000.x (Unity 6) | LTS |
| ML-Agents | 4.0.1 | Unity Package |
| Sentis | 2.4.1 | ONNX Inference |
| Python | 3.10.11 | Windows Native |
| PyTorch | 2.1+ | CUDA 12.x |
| GPU | RTX 4090 | 24GB VRAM |

### Quick Start (Training)

```powershell
# Windows PowerShell
cd C:\Users\user\Desktop\dev\physical-unity

# ML-Agents 학습 실행 (Phase E 예정)
mlagents-learn python/configs/planning/vehicle_ppo_v12_phaseE.yaml --run-id=v12_phaseE

# Unity Editor에서 Play 버튼 클릭
```

---

**Last Updated**: 2026-01-27
