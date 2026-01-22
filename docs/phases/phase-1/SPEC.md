# Phase 1: Foundation & Architecture

## Overview

자율주행 ML 플랫폼의 첫 번째 단계로, Windows 네이티브 환경 구축과 Unity-ROS2 연동을 목표로 합니다.

## Goals

1. **Windows ROS2 환경 구축**: ROS2 Humble 설치
2. **Unity-ROS2 연동 확립**: ros2-for-unity 또는 Unity Robotics Hub 선택
3. **ML-Agents RL 환경 구축**: 강화학습 학습 파이프라인
4. **기본 주행 환경 Scene 생성**: 차량, 도로, 센서

## Scope

### In Scope
- ROS2 Humble Windows 설치
- Unity Robotics Hub 테스트
- ros2-for-unity 테스트 및 비교
- ML-Agents 3.0 RL 환경 구축
- 기본 주행 Scene 생성 (차량, 도로)
- AWSIM 센서 통합 (LiDAR, Camera)
- MLflow/W&B 실험 추적 설정

### Out of Scope
- 데이터셋 구축 (Phase 2)
- Perception/Prediction 모델 (Phase 3-4)
- Planning 모델 개발 (Phase 5)

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Phase 1: Foundation Architecture                 │
├─────────────────────────────────────────────────────────────────────┤
│                        Windows Native                                │
│  ┌────────────────────┐   ┌────────────────────┐   ┌─────────────┐  │
│  │   Unity 2023.2+    │   │    Python 3.10+    │   │  ROS2 Humble│  │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │   │  (Windows)  │  │
│  │  │ ML-Agents    │  │◄─►│  │ PyTorch 2.0+ │  │◄─►│  Topics     │  │
│  │  │ Package 3.0  │  │   │  │ mlagents     │  │   │  & Services │  │
│  │  └──────────────┘  │   │  └──────────────┘  │   └─────────────┘  │
│  │  ┌──────────────┐  │   │  ┌──────────────┐  │                    │
│  │  │ AWSIM Sensors│  │   │  │ MLflow/W&B   │  │                    │
│  │  │ (LiDAR/Cam)  │  │   │  │ Tracking     │  │                    │
│  │  └──────────────┘  │   │  └──────────────┘  │                    │
│  └────────────────────┘   └────────────────────┘                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Task Breakdown

| ID | Task | Priority | Est. Time |
|----|------|----------|-----------|
| P1-01 | Windows에 ROS2 Humble 설치 | High | 0.5일 |
| P1-02 | Unity Robotics Hub 테스트 | High | 1일 |
| P1-03 | ros2-for-unity 테스트 | High | 1일 |
| P1-04 | 두 방식 성능 비교 후 선택 | High | 0.5일 |
| P1-05 | ML-Agents 3.0 RL 환경 구축 | High | 1일 |
| P1-06 | 기본 주행 환경 Scene 생성 | High | 2일 |
| P1-07 | AWSIM 센서 통합 | Medium | 2일 |
| P1-08 | 실험 추적 설정 (MLflow/W&B) | Medium | 0.5일 |

## Success Criteria

- [ ] ROS2 Humble Windows에서 정상 작동
- [ ] Unity-ROS2 토픽 송수신 확인
- [ ] ML-Agents로 간단한 RL 학습 가능
- [ ] 차량이 도로에서 주행하는 기본 Scene
- [ ] LiDAR/Camera 센서 데이터 수집
- [ ] MLflow에서 실험 추적 가능

## Timeline

**예상 소요**: 2-3주

## Dependencies

### Software
- Windows 10/11
- Unity Hub & Unity 2023.2+
- ROS2 Humble (Windows build)
- Python 3.10+
- CUDA 12.x / cuDNN 8.x

### Unity Packages
- com.unity.ml-agents (3.0+)
- ros2-for-unity 또는 ROS-TCP-Connector
- AWSIM sensor components

## ROS2 Bridge Comparison

### ros2-for-unity
| Aspect | Rating | Notes |
|--------|--------|-------|
| Latency | ★★★★★ | Native DDS, ~1ms |
| Setup | ★★☆☆☆ | Complex initial setup |
| AWSIM Compatible | ★★★★★ | Full support |
| Documentation | ★★★☆☆ | Limited |

### Unity Robotics Hub
| Aspect | Rating | Notes |
|--------|--------|-------|
| Latency | ★★★☆☆ | TCP-based, ~5-10ms |
| Setup | ★★★★★ | Easy, official support |
| AWSIM Compatible | ★★★☆☆ | Requires adaptation |
| Documentation | ★★★★★ | Extensive |

## Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| ROS2 Windows 빌드 문제 | High | Medium | Docker 대안 준비 |
| Unity-ROS2 통신 지연 | Medium | Medium | 로컬호스트 사용 |
| AWSIM 호환성 | Medium | Low | 자체 센서 구현 대안 |
| GPU 메모리 부족 | Low | Low | 배치 크기 조절 |

## Deliverables

1. **Environment Setup Guide**: ROS2/Unity 설치 가이드
2. **Unity-ROS2 Bridge Selection Report**: 비교 분석 결과
3. **Basic Driving Scene**: 차량, 도로, 센서 포함
4. **RL Training Pipeline**: ML-Agents 기반 학습 환경
5. **Experiment Tracking Setup**: MLflow 구성
