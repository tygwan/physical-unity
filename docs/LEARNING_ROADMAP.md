# Physical AI 학습 로드맵

Unity ML-Agents를 활용한 Physical AI 개발을 위한 단계별 학습 가이드

## Phase 1: 기초 다지기 (1-2주)

### 1.1 Python 기초
- [ ] Python 문법 복습 (클래스, 데코레이터, 타입힌트)
- [ ] NumPy 기본 연산
- [ ] PyTorch 텐서 조작

**추천 자료:**
- [PyTorch 60분 블리츠](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)

### 1.2 Unity 기초
- [ ] Unity Editor 인터페이스 익히기
- [ ] GameObject, Component 개념
- [ ] C# 스크립팅 기초
- [ ] **Rigidbody 물리 시스템** (중요!)
  - AddForce, velocity, angularVelocity
  - FixedUpdate vs Update
- [ ] Collider와 충돌 감지

**추천 자료:**
- [Unity Learn - Beginner Scripting](https://learn.unity.com/project/beginner-scripting)

### 1.3 수학 기초
- [ ] 벡터 연산 (내적, 외적, 정규화)
- [ ] 회전 (Quaternion 기초)
- [ ] 확률 분포 (정규분포, 샘플링)

---

## Phase 2: 강화학습 이론 (2-3주)

### 2.1 핵심 개념
- [ ] MDP (Markov Decision Process)
- [ ] 상태(State), 행동(Action), 보상(Reward)
- [ ] 정책(Policy)과 가치함수(Value Function)
- [ ] 탐험 vs 활용 (Exploration vs Exploitation)

### 2.2 알고리즘
- [ ] **PPO (Proximal Policy Optimization)** - ML-Agents 기본
  - Clipped objective
  - Actor-Critic 구조
- [ ] **SAC (Soft Actor-Critic)** - 연속 행동 공간에 적합
- [ ] Reward Shaping 기법

**추천 자료:**
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Lilian Weng's RL Blog](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)

---

## Phase 3: ML-Agents 실습 (2-3주)

### 3.1 기본 예제
- [ ] 3D Ball 예제 실행 및 분석
- [ ] Push Block 예제
- [ ] Walker 예제 (로봇 보행)

### 3.2 핵심 컴포넌트 이해
```
Agent (C#)
├── CollectObservations()   # 상태 수집
├── OnActionReceived()      # 행동 실행
├── Heuristic()             # 수동 제어
└── OnEpisodeBegin()        # 에피소드 초기화
```

- [ ] Observation 설계
  - Vector Observation (위치, 속도 등)
  - Ray Perception Sensor (거리 감지)
  - Camera Sensor (이미지 입력)
- [ ] Action Space 설계
  - Discrete (이산) vs Continuous (연속)
- [ ] Reward 설계
  - Sparse vs Dense Reward
  - 보상 스케일링

### 3.3 학습 설정
- [ ] trainer_config.yaml 이해
- [ ] 하이퍼파라미터 튜닝
  - batch_size, buffer_size
  - learning_rate
  - beta (엔트로피 계수)
- [ ] TensorBoard로 학습 모니터링

---

## Phase 4: Physical AI 심화 (3-4주)

### 4.1 로봇 제어
- [ ] 관절(Joint) 기반 로봇 모델링
  - Hinge Joint, Configurable Joint
- [ ] 토크 제어 vs 위치 제어
- [ ] 역기구학(IK) 기초

### 4.2 고급 학습 기법
- [ ] **Curriculum Learning**
  - 쉬운 환경 → 어려운 환경
- [ ] **Imitation Learning**
  - GAIL (Generative Adversarial Imitation Learning)
  - Behavioral Cloning
- [ ] **Self-Play**
  - 경쟁적 환경 학습

### 4.3 Sim-to-Real
- [ ] Domain Randomization
  - 물리 파라미터 랜덤화
  - 시각적 랜덤화
- [ ] 시뮬레이션 정확도 향상

---

## Phase 5: 프로젝트 적용 (진행 중)

### 5.1 목표 설정
- [ ] 구체적인 Physical AI 목표 정의
- [ ] 환경 설계
- [ ] 보상 함수 설계

### 5.2 반복 개발
```
설계 → 구현 → 학습 → 평가 → 개선 (반복)
```

---

## 참고 문서

### 공식 문서
- [Unity ML-Agents Documentation](https://unity-technologies.github.io/ml-agents/)
- [ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)

### 논문
- [PPO 논문](https://arxiv.org/abs/1707.06347)
- [SAC 논문](https://arxiv.org/abs/1801.01290)
- [Domain Randomization](https://arxiv.org/abs/1703.06907)

### 강의
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/)
- [CS285 Deep RL (Berkeley)](http://rail.eecs.berkeley.edu/deeprlcourse/)

---

## 진행 상황 체크

| Phase | 상태 | 시작일 | 완료일 |
|-------|------|--------|--------|
| Phase 1: 기초 | 🔄 진행중 | - | - |
| Phase 2: RL 이론 | ⏳ 대기 | - | - |
| Phase 3: ML-Agents | ⏳ 대기 | - | - |
| Phase 4: 심화 | ⏳ 대기 | - | - |
| Phase 5: 프로젝트 | ⏳ 대기 | - | - |
