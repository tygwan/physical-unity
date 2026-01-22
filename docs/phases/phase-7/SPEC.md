# Phase 7: Advanced Topics (Ongoing)

## Overview

최신 자율주행 기술 연구 및 실험 단계입니다. 지속적으로 새로운 기술을 탐색하고 적용합니다.

## Goals

1. **World Model**: 자율주행을 위한 World Model 연구
2. **LLM-based Planning**: DriveGPT 등 LLM 기반 플래닝
3. **VLA Framework**: Vision-Language-Action 통합
4. **Sim-to-Real Transfer**: 시뮬레이션에서 실차로 전이

## Research Areas

### 1. World Model for Driving

**개요**: 환경의 미래 상태를 예측하는 생성 모델을 활용한 자율주행

**Key Papers**:
- GAIA-1 (Wayve, 2023)
- DriveDreamer (NIO, 2023)
- MILE (Wayve, 2022)

**Research Questions**:
- World Model이 Planning에 어떻게 도움이 되는가?
- 시뮬레이션 데이터로 World Model을 학습할 수 있는가?
- World Model 기반 Planning의 안전성은?

**Implementation Ideas**:
```
1. Video Prediction Model 학습
   └─ Unity에서 생성한 주행 영상으로 학습

2. World Model + Planning 통합
   └─ Imagination을 활용한 Planning

3. Model Predictive Control with World Model
   └─ World Model 예측을 활용한 MPC
```

### 2. LLM-based Planning

**개요**: Large Language Model을 자율주행 Planning에 활용

**Key Papers/Projects**:
- DriveGPT (Waymo)
- GPT-Driver (Shanghai AI Lab)
- LMDrive (Huawei)

**Research Questions**:
- LLM이 주행 의사결정에 어떻게 활용될 수 있는가?
- Chain-of-Thought Reasoning이 Planning에 도움이 되는가?
- LLM의 추론 속도가 실시간 요구사항을 충족하는가?

**Implementation Ideas**:
```
1. LLM as High-level Planner
   └─ LLM이 전략적 결정, Low-level은 기존 방식

2. Language-conditioned Planning
   └─ 자연어 지시에 따른 주행 (예: "안전하게 주행해줘")

3. LLM for Scenario Understanding
   └─ 복잡한 상황 해석에 LLM 활용
```

### 3. VLA (Vision-Language-Action) Framework

**개요**: Vision, Language, Action을 통합한 End-to-end 프레임워크

**Key Projects**:
- RT-2 (Google DeepMind)
- PaLM-E (Google)
- Embodied GPT

**Research Questions**:
- VLA가 자율주행에 어떻게 적용될 수 있는가?
- Vision-Language 사전학습이 Planning에 도움이 되는가?
- Transfer Learning 가능성은?

### 4. Sim-to-Real Transfer

**개요**: 시뮬레이션에서 학습한 모델을 실제 환경으로 전이

**Key Techniques**:
- Domain Randomization
- Domain Adaptation
- Differentiable Simulation

**Research Questions**:
- Unity 시뮬레이션과 실제 환경의 Gap은 얼마나 큰가?
- 어떤 Domain Randomization이 효과적인가?
- 실차 데이터 없이 Sim-to-Real이 가능한가?

**Implementation Ideas**:
```
1. Visual Domain Randomization
   └─ 조명, 텍스처, 날씨 변화

2. Dynamics Randomization
   └─ 차량 물리 파라미터 변화

3. Sensor Noise Modeling
   └─ 실제 센서 노이즈 모방
```

## Scope

### In Scope
- 논문 서베이 및 기술 조사
- Proof-of-Concept 구현
- 성능 비교 실험
- 기술 문서화

### Out of Scope (시작 단계)
- 프로덕션 배포
- 대규모 학습
- 실차 테스트

## Task Breakdown (Initial)

| ID | Task | Priority | Est. Time |
|----|------|----------|-----------|
| **World Model** |
| P7-01 | 관련 논문 서베이 | Medium | 1주 |
| P7-02 | Simple Video Prediction PoC | Low | 2주 |
| **LLM Planning** |
| P7-03 | DriveGPT 논문 분석 | Medium | 1주 |
| P7-04 | LLM High-level Planner PoC | Low | 2주 |
| **VLA** |
| P7-05 | VLA 프레임워크 조사 | Low | 1주 |
| **Sim-to-Real** |
| P7-06 | Domain Randomization 구현 | Medium | 2주 |
| P7-07 | Sim-to-Real Gap 분석 | Medium | 2주 |

## Success Criteria

- [ ] 각 연구 영역별 서베이 문서 작성
- [ ] 최소 1개 영역에서 PoC 구현
- [ ] Phase 5 모델 대비 성능 향상 또는 새로운 기능 추가
- [ ] 기술 보고서 작성

## Timeline

**예상 소요**: Ongoing (지속적 연구)

초기 목표: 각 영역별 PoC 구현 (각 2-4주)

## Dependencies

- Phase 1-6 완료
- 최신 논문 접근
- 추가 컴퓨팅 리소스 (LLM 등)

## Resources

### Papers
- [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)
- [DriveGPT: A Large Language Model for Autonomous Driving](https://arxiv.org/abs/2310.01415)
- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://robotics-transformer2.github.io/)

### Codebases
- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [nuPlan-devkit](https://github.com/motional/nuplan-devkit)
- [DriveGAN](https://research.nvidia.com/labs/toronto-ai/DriveGAN/)

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| 기술 복잡도 | High | High | 단계별 접근 |
| 컴퓨팅 리소스 부족 | Medium | Medium | 경량 모델 우선 |
| 실제 적용 어려움 | High | High | PoC 수준 유지 |

## Deliverables

1. **Survey Documents**: 각 연구 영역별 서베이
2. **PoC Implementations**: Proof-of-Concept 코드
3. **Experiment Results**: 실험 결과 및 분석
4. **Technical Reports**: 기술 보고서
5. **Future Roadmap**: 후속 연구 방향

## Notes

이 Phase는 명확한 종료 시점이 없는 지속적인 연구 단계입니다.
주요 목표는:
1. 최신 기술 동향 파악
2. 새로운 아이디어 실험
3. 기존 시스템 개선 가능성 탐색
4. 장기적 기술 로드맵 수립

실제 구현보다는 연구와 탐색에 초점을 맞추며,
유망한 기술이 발견되면 별도 Phase로 분리하여 진행할 수 있습니다.
