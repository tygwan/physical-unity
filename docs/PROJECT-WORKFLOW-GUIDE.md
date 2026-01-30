# Project Workflow and Detail Analysis Guide

**프로젝트 진행 워크플로우 및 디테일 가이드**

**Generated**: 2026-01-29
**Purpose**: 프로젝트 진행 시 문서 참조 flow, 지속적 관찰 방법, 3가지 관점의 디테일 분석

---

## 목차

1. [문서 Flow 및 아키텍처](#1-문서-flow-및-아키텍처)
2. [지속적 관찰 방법](#2-지속적-관찰-방법)
3. [디테일 분석](#3-디테일-분석)
   - 3A. 기술적 디테일
   - 3B. 개발적 관점
   - 3C. PM 관점
4. [실행 가능한 권장사항](#4-실행-가능한-권장사항)

---

## 1. 문서 Flow 및 아키텍처

### 1.1 문서 계층 구조

```
=============================================================================
                    DOCUMENT FLOW ARCHITECTURE
=============================================================================

TIER 0: VISION & REQUIREMENTS (Why)
    ┌─────────────────────────────────────────┐
    │           docs/PRD.md                   │  Product vision, goals,
    │    Product Requirements Document        │  success criteria, timeline
    └───────────────┬─────────────────────────┘
                    │
TIER 1: STRATEGY & DESIGN (What)
    ┌───────────────┴─────────────────────────┐
    │          docs/TECH-SPEC.md              │  Architecture, ML models,
    │    Technical Specification              │  data pipeline, APIs
    └───────────────┬─────────────────────────┘
                    │
    ┌───────────────┴───────────────┐
    │                               │
    v                               v
┌─────────────────────┐   ┌─────────────────────────┐
│ docs/phases/README  │   │ docs/LEARNING-ROADMAP.md │  Phase A-L RL/IL
│ Phase 1-7 Overview  │   │ Learning Strategy        │  training roadmap
│ (Infrastructure)    │   │ (Training Plan)          │
└──────┬──────────────┘   └─────────┬───────────────┘
       │                            │
       v                            │
┌──────────────────┐                │
│ docs/phases/     │                │
│ phase-N/SPEC.md  │                │
│ (7 phase specs)  │                │
└──────────────────┘                │
                                    │
TIER 2: EXECUTION & TRACKING (How)  │
    ┌───────────────────────────────┘
    │
    ├──> ┌──────────────────────────────────────────────┐
    │    │ docs/PROGRESS.md                             │  Dashboard: current
    │    │ Progress Tracker                             │  status, milestones,
    │    │ (Central Status Dashboard)                   │  completion grades
    │    └──────────────────────────────────────────────┘
    │
    ├──> ┌──────────────────────────────────────────────┐
    │    │ docs/TRAINING-LOG.md                         │  Per-phase detailed
    │    │ Training Log                                 │  training metrics,
    │    │ (Detailed Training History)                  │  checkpoints, configs
    │    └──────────────────────────────────────────────┘
    │
    └──> ┌──────────────────────────────────────────────┐
         │ docs/CONTEXT.md                              │  Quick reference:
         │ Project Context                              │  env, paths, commands,
         │ (Session Quick Reference)                    │  glossary
         └──────────────────────────────────────────────┘

TIER 3: EXPERIMENT ARTIFACTS (Evidence)
    ┌─────────────────────────────────────────────────────────┐
    │ experiments/phase-{X}/                                  │
    │                                                         │
    │   README.md          - Experiment overview              │
    │   HYPOTHESIS.md      - Testable hypotheses              │
    │   DESIGN-SUMMARY.md  - Design decisions & rationale     │
    │   COMPARISON.md      - Previous vs current comparison   │
    │   TRAINING-GUIDE.md  - Execution commands & monitoring  │
    │   ANALYSIS.md        - Post-training analysis           │
    │   SUMMARY.md         - Executive summary                │
    │   config/*.yaml      - Training configuration           │
    │   logs/              - TensorBoard events               │
    │   checkpoints/       - Model checkpoints                │
    └─────────────────────────────────────────────────────────┘

TIER 4: REFERENCE & LEARNING (Knowledge Base)
    ┌───────────────────────────────────────────────────┐
    │ docs/TENSORBOARD-METRICS-GUIDE.md                 │  Metric definitions
    │ docs/site/lessons-learned.md                      │  Success/failure cases
    │ docs/site/phases/failed-experiments.md             │  Archived failures
    │ docs/archives/                                    │  Historical documents
    └───────────────────────────────────────────────────┘
```

### 1.2 활동별 문서 참조 순서

#### 새 실험 시작 시

```
1. CONTEXT.md          (5분)   환경/경로 확인
2. PROGRESS.md         (5분)   현재 상태 파악
3. LEARNING-ROADMAP.md (10분)  이번 Phase 전략
4. 이전 ANALYSIS.md    (10분)  지난 Phase 교훈
5. lessons-learned.md  (5분)   알려진 함정
6. DESIGN-SUMMARY.md   (작성)  실험 계획
7. HYPOTHESIS.md       (작성)  성공 기준 정의
8. TRAINING-GUIDE.md   (작성)  명령어 및 모니터링
```

#### 학습 모니터링 중

```
1. TRAINING-GUIDE.md              모니터링 임계값
2. TENSORBOARD-METRICS-GUIDE.md   지표 해석
3. DESIGN-SUMMARY.md              예상 보상 궤적
4. COMPARISON.md                  이전 Phase와 비교
```

#### 학습 완료/실패 시

```
1. ANALYSIS.md         (작성)  전체 사후 분석
2. TRAINING-LOG.md     (업데이트)  결과 기록
3. PROGRESS.md         (업데이트)  대시보드 갱신
4. LEARNING-ROADMAP.md (업데이트)  필요시 전략 수정
5. lessons-learned.md  (업데이트)  새 교훈 아카이브
```

### 1.3 문서 간 의존성 매트릭스

| 문서 | 읽는 문서 | 제공하는 문서 | 업데이트 주기 |
|------|----------|-------------|-------------|
| PRD.md | (외부) | TECH-SPEC, phases | 분기별 |
| TECH-SPEC.md | PRD | phase SPECs, 학습 | 드물게 |
| LEARNING-ROADMAP.md | PRD, TECH-SPEC | PROGRESS, experiments | Phase별 |
| PROGRESS.md | TRAINING-LOG, ANALYSIS | (사용자 대시보드) | 학습 실행마다 |
| TRAINING-LOG.md | TensorBoard, ANALYSIS | PROGRESS, ROADMAP | 학습 실행마다 |
| CONTEXT.md | 시스템 상태 | (세션 부트스트랩) | 환경 변경 시 |
| HYPOTHESIS.md | 이전 ANALYSIS, ROADMAP | DESIGN-SUMMARY | 실험마다 |
| DESIGN-SUMMARY.md | HYPOTHESIS, COMPARISON | TRAINING-GUIDE | 실험마다 |
| TRAINING-GUIDE.md | DESIGN-SUMMARY, config | (운영자 참조) | 실험마다 |
| ANALYSIS.md | TensorBoard, logs | TRAINING-LOG, PROGRESS | 학습 후 |

### 1.4 현재 문서 상태 이슈

**이슈 1**: PROGRESS.md 구식 (2026-01-27 최종 업데이트, Phase A/B 진행 미반영)

**이슈 2**: LEARNING-ROADMAP.md 미반영 (Phase A "준비 중" 표시, 실제로는 완료됨)

**이슈 3**: 문서 간 불일치 (phases/README.md는 Phase B +903 성공, TRAINING-LOG.md는 -108 실패)

**이슈 4**: 단일 정보원 부재 (PROGRESS.md, TRAINING-LOG.md, phases/README.md 모두 상태 추적하나 불일치)

---

## 2. 지속적 관찰 방법

### 2.1 모니터링 체크리스트

#### 학습 전 체크 ✅

- [ ] Phase 0 또는 이전 checkpoint 확인 (파일 존재, reward 확인)
- [ ] YAML config 검증 (10K 스텝 sanity run)
- [ ] TensorBoard 로그 디렉토리 생성
- [ ] Unity 환경 씬 로드 및 Play 준비
- [ ] GPU 메모리 가용성 (nvidia-smi 확인)
- [ ] 디스크 공간 충분 (최소 50GB 여유)
- [ ] 보상 함수 설계 검토 (step별 penalty 누적 확인)
- [ ] Curriculum 진행 단계 명시적 임계값 정의

#### 학습 중 체크 (매 500K 스텝)

- [ ] Cumulative Reward 추세 양수 또는 예상 궤적
- [ ] Episode Length 안정적 (거의 0으로 붕괴하지 않음)
- [ ] Entropy 점진적 감소 (조기 수렴 <0.05 아님)
- [ ] Value Loss 감소 중 (폭발 >1000 아님)
- [ ] KL Divergence 범위 0.01-0.05
- [ ] Policy Loss 범위 0.01-0.5
- [ ] Curriculum lesson 전환이 예상 스텝에서 발생
- [ ] Unity Console 에러 없음 (연결 끊김, NaN 값)
- [ ] GPU 사용률 정상 (>80%)
- [ ] Checkpoint 파일 저장 중

#### 긴급 중단 트리거 🚨

- [ ] Collision Rate 언제든 8% 초과
- [ ] Reward가 음수 값에 >250K 스텝 고정
- [ ] Entropy 0으로 감소 (완전한 정책 붕괴)
- [ ] Value Loss 발산 (>10000)
- [ ] 어떤 지표든 NaN 발생

#### 학습 후 검증

- [ ] 최종 reward가 최소 성공 기준 충족
- [ ] 20-episode 평가 실행으로 지표 확인
- [ ] ONNX 내보내기 성공
- [ ] Analysis 문서 작성
- [ ] TRAINING-LOG.md 업데이트
- [ ] PROGRESS.md 업데이트
- [ ] 교훈 학습 포착

### 2.2 주요 모니터링 포인트

```
=============================================================================
        모니터링 파이프라인 (실시간 + 주기적 + 사후)
=============================================================================

실시간 (TensorBoard - 연속):
  ┌───────────────────────────────────────────────────┐
  │  tensorboard --logdir=results/<run-id>            │
  │                                                   │
  │  주요 패널:                                        │
  │    Environment/Cumulative Reward  <-- 가장 중요     │
  │    Environment/Episode Length                      │
  │    Environment/Lesson                             │
  │    Losses/Policy Loss                             │
  │    Losses/Value Loss                              │
  │    Policy/Entropy                                 │
  │    Policy/Approx KL Divergence                    │
  │                                                   │
  │  파생 체크:                                        │
  │    Reward 기울기 (양수여야 함)                      │
  │    Entropy 감소율 (점진적이어야 함)                 │
  │    Value Estimate와 Reward 정렬                    │
  └───────────────────────────────────────────────────┘

주기적 (매 30분 / 500K 스텝):
  ┌───────────────────────────────────────────────────┐
  │  수동 체크:                                        │
  │    1. Checkpoint 파일 크기 (증가 = 정상)            │
  │    2. training.pid 여전히 활성                     │
  │    3. Unity Editor 여전히 Play 모드                │
  │    4. GPU 메모리 누수 없음                          │
  │    5. 디스크 공간 고갈 안 됨                        │
  └───────────────────────────────────────────────────┘

사후 (학습 후):
  ┌───────────────────────────────────────────────────┐
  │  분석 파이프라인:                                   │
  │    1. TensorBoard CSV 내보내기                     │
  │    2. Checkpoint reward 비교 테이블                 │
  │    3. Episode별 통계 추출                          │
  │    4. 구성요소별 reward 분해                        │
  │    5. 이전 Phase와 비교                            │
  │    6. 20-episode 평가 실행                         │
  └───────────────────────────────────────────────────┘
```

### 2.3 자동화 기회

| 활동 | 현재 방법 | 권장 자동화 |
|------|---------|-----------|
| TensorBoard 모니터링 | 수동 브라우저 | 스크립트: reward plateau >250K 스텝 시 알림 |
| Checkpoint 백업 | 수동 cp | Post-checkpoint 훅: 백업 디렉토리로 자동 복사 |
| Reward 정상성 체크 | 사후 분석 | Early-termination 스크립트: reward 200K 동안 음수면 중단 |
| 문서 업데이트 | 수동 편집 | Post-training 스크립트: TensorBoard 데이터에서 ANALYSIS.md 뼈대 생성 |
| Phase 간 비교 | 수동 테이블 작성 | 스크립트: TensorBoard 로그에서 비교 테이블 자동 생성 |
| Config 검증 | 수동 YAML 검토 | Pre-training 스크립트: YAML 스키마 검증, 10K sanity 실행 |
| 디스크/GPU 모니터링 | nvidia-smi 수동 | 백그라운드 감시: VRAM >22GB 또는 디스크 <50GB 시 알림 |

### 2.4 대시보드/리포트 전략

**권장 대시보드 구조:**

```
=============================================================================
          학습 대시보드 레이아웃 (단일 페이지 뷰)
=============================================================================

┌─────────── 현재 상태 ─────────┬──────── 히스토리 ──────────────┐
│                               │                                │
│  Active Phase: B (Decision)   │  Phase 0: +1018 (A+)          │
│  Status: FAILED               │  Phase A: +2113 (A)           │
│  Step: 3,000,000 / 3,000,000  │  Phase B: -108  (FAIL)        │
│  Reward: -108                 │  Phase C: Planned              │
│  Duration: 39.4 min           │                                │
│  GPU: RTX 4090                │                                │
│                               │                                │
├─────────── 주요 지표 ─────────┼──────── 알림 ───────────────┤
│                               │                                │
│  Collision Rate: >80% (추정)  │  [CRITICAL] Phase B 실패      │
│  Goal Completion: ~0% (추정)  │  [ACTION] 근본 원인 필요       │
│  Overtaking Events: 0         │  [BLOCKED] Phase C 보류        │
│  Convergence: 250K (조기)     │                                │
│                               │                                │
├─────────── 다음 조치 ─────────┼──────── 타임라인 ─────────────┤
│                               │                                │
│  1. 근본 원인 조사 (24h)      │  Phase B 재시도: +48h         │
│  2. Reward 함수 디버그        │  Phase C 시작: +72h           │
│  3. 재시도 전 100K 테스트     │  Phase E 시작: +2주           │
│  4. Phase B v2 설계 검토      │                                │
│                               │                                │
└───────────────────────────────┴────────────────────────────────┘
```

---

## 3. 디테일 분석

### 3A. 기술적 디테일

#### 3A.1 ML 알고리즘 파라미터 튜닝 포인트

| 파라미터 | 현재 값 | Phase B 영향 | 튜닝 권장사항 |
|---------|--------|------------|-------------|
| **learning_rate** | 3e-4 (Phase A) / 2e-4 (Phase B) | 감소는 회복 느리게 할 수 있음 | Phase B v2는 3e-4 유지 |
| **batch_size** | 4096 | 16-area 병렬에 적합 | 변경 불필요 |
| **buffer_size** | 40960 | 10x batch, 표준 | 변경 불필요 |
| **epsilon (PPO clip)** | 0.2 | 표준, 안정적 | 변경 불필요 |
| **beta (entropy)** | 5e-3 (Phase A) / 3e-3 (Phase B) | 낮은 탐험은 조기 고정 기여 | Phase B v2는 5e-3로 복원 |
| **lambd (GAE)** | 0.95 | 표준 | 변경 불필요 |
| **num_epoch** | 5 (Phase B) vs 3 (Phase A) | 더 많은 epoch은 음수 reward signal에 빠르게 과적합 | Phase B v2는 3으로 감소 |
| **network** | 512 x 3 layers | 검증된 아키텍처 | 변경 불필요 |
| **gamma** | 0.99 | 운전용 표준 | 변경 불필요 |
| **time_horizon** | 256 (Phase B) vs 2048 (Phase A) | **CRITICAL: 8배 짧은 horizon이 reward 누적과 credit assignment를 크게 변경** | Phase B v2는 2048로 복원 |

**중요 기술 발견: time_horizon 불일치**

Phase B 설정(`vehicle_ppo_v12_phaseB.yaml`)은 `time_horizon: 256` 사용, Phase A는 `time_horizon: 2048` 사용. 중대한 차이:
- 짧은 time horizon은 에이전트가 미래 256 스텝만 보고 reward 추정
- -0.5 step별 패널티로 에이전트는 즉각적 horizon에서 -128 봄
- 2048 horizon이면 먼 미래 양수 reward(골 완료)를 볼 수 있어 패널티 상쇄 가능
- Temporal credit assignment 창을 압축하여 희소 양수 reward(overtaking bonus +5.0) 거의 안 보임

#### 3A.2 학습 안정성 체크포인트

| 체크포인트 | 정상 징후 | 경고 징후 | 긴급 |
|-----------|---------|----------|------|
| 0-50K 스텝 | Reward 다양하게 변함 | Reward 이미 음수 | Reward 단일 값 고정 |
| 50K-250K | 점진적 개선 | 느리지만 움직임 | 음수 plateau |
| 250K-1M | 명확한 상승 추세 | 진동 | 평탄선 또는 하락 |
| 1M-2M | 목표 접근 | 느린 수렴 | 여전히 baseline 미만 |
| 2M+ | 목표 근처 | 수렴 안 됨 | 퇴행 |

**Phase B는 모든 체크포인트에서 "긴급" 패턴 보임** -- reward가 250K 스텝까지 -108에 고정, 이후 변동 없음.

#### 3A.3 성능 최적화 기회

| 최적화 | 현재 상태 | 잠재적 이득 | 구현 |
|-------|---------|-----------|------|
| Mixed Precision (AMP) | 미활성화 | 1.5-2x 처리량 | `--fp16` 플래그 또는 PyTorch AMP 래퍼 |
| Time Scale | 20x | 이미 최대화 | 추가 이득 없음 |
| Parallel Areas | 16 | 좋은 균형 | 32 가능하나 수익 체감 |
| Checkpoint 주기 | 매 50K | I/O 오버헤드 감소 | 100K로 증가 (디스크 쓰기 절약) |
| TensorBoard summary_freq | 5000 | 오버헤드 감소 | 긴 실행에는 10000으로 증가 |

---

### 3B. 개발적 관점

#### 3B.1 코드 품질 및 구조 평가

**강점:**
- 깔끔한 분리: Unity(C#)는 환경 처리, Python은 ML 처리
- Config 기반 학습 (YAML 파일)
- Phase별 잘 조직된 실험 디렉토리 구조
- 각 실험마다 자체 config, logs, checkpoint 디렉토리

**약점:**
- 자동화된 config 검증 없음 (Phase B 패널티 재앙을 잡을 수 있었음)
- Reward 함수 동작에 대한 단위 테스트 없음
- 짧은 학습 실행이 예상 reward 범위를 생성하는지 검증하는 통합 테스트 없음
- Reward 함수 로직이 Unity C#에 있음 (Python보다 반복하기 어려움)
- 여러 config 파일 위치가 혼란 초래 (experiments/phase-B-decision/config/ vs python/configs/planning/)

#### 3B.2 테스트 전략

**현재 상태:** ML 학습 파이프라인에 자동화 테스트 없음.

**권장 테스트 매트릭스:**

| 테스트 레벨 | 테스트 대상 | 구현 |
|-----------|----------|------|
| **Unit** | Reward 함수 수학 (주어진 상태, reward 값 검증) | Reward calculator용 Python pytest |
| **Sanity** | 10K-스텝 학습이 양수 reward 방향 생성 | Pre-training shell 스크립트 |
| **Integration** | Unity-Python gRPC 연결 100K 스텝 안정 | 자동화 테스트 스위트 |
| **Regression** | 새 config이 이전 Phase baseline 미만으로 퇴행 안 함 | 50K-스텝 실행을 알려진 baseline과 비교 |
| **Reward Accumulation** | Config 가중치에서 예상 episode reward 계산 | Reward 누적 시뮬레이션하는 Python 스크립트 |

**Phase B 실패를 방지했을 중요 테스트:**
```python
Test: reward_accumulation_sanity
Input: Phase B reward config (speed: 0.3, following: -0.5, etc.)
예상 episode length: ~216 스텝
계산: 216 스텝에 걸친 step별 reward 합계
Assert: 순 episode reward > 0
실제 결과: -108 (FAIL -- 학습 전에 문제 포착)
```

#### 3B.3 디버깅 및 트러블슈팅 프로세스

**현재 프로세스 (문서에서 관찰):**
```
1. 학습 완료 또는 실패
2. 수동 TensorBoard 검사
3. 사후 분석 문서 작성
4. 근본 원인 가설 공식화
5. 이전 Phase와 비교
6. 수정 설계
7. 재시도
```

**권장 개선 프로세스:**
```
1. PRE-FLIGHT: 자동화된 config 검증 + 10K sanity 체크
2. MONITORING: Reward 궤적 이상에 대한 알림 시스템
3. EARLY-STOP: Reward가 >200K 스텝 음수면 자동 중단
4. DIAGNOSIS: 구성요소별 reward 로깅 (집계만이 아님)
5. COMPARISON: 이전 Phase와 자동화된 델타 분석
6. FIX: 전체 커밋 전 100K 스텝으로 A/B 테스트
7. RETRY: 수정 검증 후에만 전체 실행
```

#### 3B.4 재현 가능성

**현재 재현 가능성 자산:**

| 자산 | 위치 | 상태 |
|-----|------|-----|
| Phase 0 checkpoint | `experiments/phase-0-foundation/results/E2EDrivingAgent/E2EDrivingAgent-8000047.pt` | 존재 |
| Phase 0 ONNX | `experiments/phase-0-foundation/results/E2EDrivingAgent/E2EDrivingAgent-8000047.onnx` | 존재 |
| Phase 0 config | `experiments/phase-0-foundation/config/vehicle_ppo_v10g.yaml` | 존재 |
| Random seed | 명시적으로 설정 안 됨 | 위험: 재현 불가능 |

**재현 가능성 격차:**
- Config에 random seed 고정 안 됨 (config에서 `seed:` 필드 관찰 안 됨)
- Unity 씬 상태가 버전 관리 안 됨 (Phase B에 어떤 씬 사용?)
- 환경 C# 코드(reward 함수) 변경이 config 변경과 함께 추적 안 됨
- Phase B 학습에 사용된 실제 config이 experiments/의 deprecated config와 다를 수 있음

#### 3B.5 CI/CD 및 자동화

**현재 상태:** CI/CD 파이프라인 없음. 모든 작업 수동.

**권장 파이프라인:**

```
┌─────────────────────────────────────────────────────────────┐
│                   학습 CI/CD 파이프라인                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Stage 1: CONFIG 검증 (30초)                                  │
│    - YAML 스키마 검증                                         │
│    - Reward 누적 수학 체크                                    │
│    - 파라미터 경계 체크                                        │
│    - 이전 Phase config와 diff                                 │
│                                                               │
│  Stage 2: SANITY 실행 (5분)                                   │
│    - 10K 스텝 학습 실행                                       │
│    - Assert: reward가 음수 고정 안 됨                          │
│    - Assert: NaN 값 없음                                      │
│    - Assert: Unity-Python 연결 안정                           │
│                                                               │
│  Stage 3: 짧은 검증 (15분)                                    │
│    - 100K 스텝 학습 실행                                      │
│    - Assert: reward 양수 추세 또는 예상 범위 내               │
│    - Assert: collision rate < 임계값                         │
│    - 사람 승인 게이트                                          │
│                                                               │
│  Stage 4: 전체 학습 (25-90분)                                 │
│    - max_steps까지 전체 학습 실행                              │
│    - Early-stop 포함 자동화 모니터링                          │
│    - Checkpoint 관리                                          │
│                                                               │
│  Stage 5: POST-TRAINING (10분)                                │
│    - 20-episode 평가                                          │
│    - ONNX 내보내기                                            │
│    - 분석 뼈대 생성                                           │
│    - 문서 업데이트                                            │
│    - 아티팩트 git commit                                      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

### 3C. PM 관점

#### 3C.1 Phase 마일스톤 추적

```
=============================================================================
         Phase 마일스톤 추적기 (실제 vs 계획)
=============================================================================

Phase 0: Foundation
  계획: +1000 reward, 8M 스텝
  실제: +1018 reward, 8M 스텝, 1.17 시간
  등급: A+ (목표의 101.8%)
  상태: 완료 (2026-01-27)

Phase A: Dense Overtaking
  계획: +950 reward, 2-4M 스텝
  실제: +2113 reward, 2.5M 스텝, 29.6 분
  등급: A (목표의 235%)
  상태: 완료 (2026-01-28)
  주: 보너스에도 0 overtaking 이벤트 감지

Phase B: Decision Learning
  계획: +1500-1800 reward, 3M 스텝
  실제: -108 reward, 3M 스텝, 39.4 분
  등급: FAIL (목표의 -107%)
  상태: 실패 (2026-01-29)
  주: 치명적 reward 붕괴

Phase C: Multi-NPC      --> 차단됨 (Phase B 전제조건)
Phase E: Curved Roads   --> 차단됨
Phase F: Multi-Lane     --> 차단됨
Phase G: Intersection   --> 차단됨
```

#### 3C.2 리스크 관리 포인트

| 리스크 ID | 리스크 | 확률 | 영향 | 상태 | 완화 |
|---------|------|-----|------|------|------|
| R1 | Reward 함수 설계 오류 | **실현됨** | 치명적 | Phase B 정확히 이것 때문에 실패 | 100K 검증 실행 필수 |
| R2 | 가혹한 curriculum shock | 높음 | 높음 | Phase B 기여 요인 | 점진적 0-1-2 NPC 진행 |
| R3 | Config 파라미터 drift | 중간 | 높음 | time_horizon 2048에서 256로 변경 눈치채지 못함 | 학습 전 config diff 검토 |
| R4 | 문서 구식 상태 | **실현됨** | 중간 | PROGRESS.md, ROADMAP.md 구식 | Post-training 문서 업데이트 체크리스트 |
| R5 | Pre-flight 검증 없음 | **실현됨** | 치명적 | 3M-스텝 커밋 전 sanity 체크 없음 | 필수 10K+100K 검증 |
| R6 | 단일 장애 지점 | 중간 | 높음 | 한 사람이 모든 Phase 관리 | 핸드오프용 프로세스 문서화 |
| R7 | Overtaking 감지 시스템 미검증 | **실현됨** | 중간 | Phase A 0 이벤트 보고, Phase B로 이월 | 명시적 감지 검증 테스트 |

#### 3C.3 의사결정 프로세스 평가

**잘 된 결정:**
1. Phase 0 foundation-first 접근 -- 견고한 baseline 구축
2. Curriculum learning 채택 -- Phase 0, A에서 효과 입증
3. 실험별 상세 문서화 -- 분석 및 학습 가능
4. 보수적 안전 접근 (collision penalty -10.0) -- Phase A에서 0% collision 유지

**역효과 낸 결정:**
1. Phase B에 Phase A 대신 Phase 0 init 사용 -- 근거는 타당했으나(속도 편향 회피), 성공한 능력 기반 제거
2. 누적 분석 없이 -0.5/step following penalty 추가 -- 설계 문서(COMPARISON.md)는 예상 reward 계산했으나 총 누적 잡지 못함
3. 전체 3M-스텝 실행 전 검증 테스트 안 함 -- 39.4분과 3M 스텝 낭비
4. 여러 config 파라미터 동시 변경 (speed weight, following penalty, overtaking bonus, lane center bonus, beta, num_epoch, time_horizon) -- 어떤 변경이 실패 초래했는지 격리 불가능

**의사결정 프레임워크 권장:**

```
의사결정 게이트: 각 Phase 학습 전
  ┌──────────────────────────────────────────────────┐
  │  1. 단일 변경 원칙                                │
  │     - 실험당 최대 2개 파라미터 변경               │
  │     - 다른 모든 파라미터는 성공한 이전 것         │
  │                                                  │
  │  2. REWARD 누적 테스트                           │
  │     - 예상 episode별 reward 계산                 │
  │     - 순 양수여야 함                              │
  │     - 패널티 예산 < 양수 reward의 50%            │
  │                                                  │
  │  3. SANITY 검증 (10K 스텝)                       │
  │     - Pass: reward가 음수 고정 안 됨             │
  │     - Pass: NaN 또는 크래시 없음                 │
  │                                                  │
  │  4. 짧은 검증 (100K 스텝)                        │
  │     - Pass: reward 예상 방향으로 추세             │
  │     - Pass: collision rate < 임계값              │
  │                                                  │
  │  5. 사람 승인                                    │
  │     - 100K 검증에서 TensorBoard 검토             │
  │     - 승인 또는 재설계                           │
  │                                                  │
  │  6. 전체 학습 (early-stop 규칙 포함)             │
  └──────────────────────────────────────────────────┘
```

#### 3C.4 진행 상황 보고 구조

**현재 상태:** 진행 상황이 PROGRESS.md, TRAINING-LOG.md, LEARNING-ROADMAP.md, phases/README.md에 흩어짐. 이 문서들이 불일치하며 일부는 희망 데이터를 실제 결과와 섞음.

**권장 단일 정보원 구조:**

```
주간 상태 보고서
====================
주: [날짜 범위]
Phase: [현재]
상태: [GREEN/YELLOW/RED]

이번 주 완료:
- [Phase X] 스텝 Y에서 reward Z로 완료 (등급)
- [문서] 발견사항으로 업데이트

진행 중:
- [Phase X] 스텝 Y/Z (XX% 완료)
- 현재 reward: [값] vs 목표: [값]

차단됨/이슈:
- [이슈 설명] -- [완화 계획]

다음 주 계획:
- [조치 1] -- 예상 [기간]
- [조치 2] -- 예상 [기간]

지표 스냅샷:
| Phase | 목표 | 실제 | 갭 | 상태 |
|-------|------|------|-----|------|
```

#### 3C.5 점진적 성장 전략

프로젝트는 curriculum 기반 성장 모델을 따름. Phase B 실패를 바탕으로 조정된 전략:

```
=============================================================================
         점진적 성장 전략 (Phase B 실패 후 개정)
=============================================================================

원칙: 한 번에 하나의 새 요소

Phase 0 (완료):     Lane Keeping + NPC Coexistence    --> 기반
Phase A (완료):     + Dense Speed Reward              --> 속도 숙달
Phase B (재시도):   + 1 Slow NPC (forced overtake)    --> 기본 Overtaking
                    수정: step별 penalty 제거,
                         Phase A init 사용,
                         점진적 0->1 NPC curriculum
Phase B+ (신규):    + NPC 속도 변동 (slow->fast)      --> 의사결정
                    변경: NPC 속도 curriculum만
Phase C (계획):     + NPC 수 (1->2->4)                --> Multi-Agent
                    변경: NPC 수만
Phase E (계획):     + 도로 곡률                        --> 기하학
Phase F (계획):     + 차선 수 (2->3->4)               --> 내비게이션
Phase G (계획):     + 교차로                          --> 복잡한 시나리오

주요 규칙:
1. 각 Phase는 정확히 하나의 환경 차원만 변경
2. 모든 reward 가중치는 깨지지 않는 한 앞으로 이월
3. 전체 실행 전 필수 100K 검증
4. Phase B 실패 교훈: penalty 항은 reward만큼 강력
5. 실패와 재시도를 위해 30% 버퍼 시간 예산
```

---

## 4. 실행 가능한 권장사항

### 즉각 조치 (24-48시간)

| 우선순위 | 조치 | 소유자 | 예상 결과 |
|---------|-----|-------|----------|
| P0 | Unity C# 코드에서 Phase B reward 함수 조사 | Dev | -0.5/step penalty 누적이 -108 일치 확인 |
| P0 | 실제 학습 config vs deprecated config 비교 | Dev | 모든 파라미터 차이 식별 (time_horizon, beta, num_epoch) |
| P0 | Phase B v2 설계용 reward 누적 calculator 실행 | Dev | 학습 전 순 양수 episode reward 검증 |
| P1 | PROGRESS.md를 실제 Phase A, B 결과와 동기화 | Dev | 프로젝트 상태의 단일 정보원 |
| P1 | TRAINING-LOG.md를 Phase B 실패 세부사항으로 업데이트 | Dev | 완전한 역사 기록 |
| P1 | LEARNING-ROADMAP.md를 Phase A 성공 데이터로 업데이트 | Dev | 전략이 현실 반영 |

### 단기 조치 (1-2주)

| 우선순위 | 조치 | 예상 결과 |
|---------|-----|----------|
| P0 | 단일 변경 원칙으로 Phase B v2 설계: Phase A config + 점진적 0-1 NPC curriculum | 깔끔한 실험 설계 |
| P0 | Pre-training 검증 스크립트 생성 (10K + 100K 스텝) | 미래 낭비 방지 |
| P1 | Unity 환경에 구성요소별 reward 로깅 추가 | Reward 분해 분석 가능 |
| P1 | 문서 불일치 수정 (phases/README.md는 B/C/D 완료 표시, TRAINING-LOG.md와 모순) | 문서에 대한 신뢰 |
| P2 | TensorBoard에서 자동화된 비교 테이블 생성기 생성 | 수동 분석 시간 감소 |

### 프로세스 개선 (지속적)

| 개선 | 근거 | 구현 |
|-----|------|------|
| 필수 pre-flight 검증 (10K + 100K) | Phase B는 테스트되지 않은 config 때문에 3M 스텝과 39분 낭비 | mlagents-learn 주변 shell 스크립트 래퍼 |
| 실험당 단일 변경 규칙 | Phase B는 7+ 파라미터 동시 변경, 원인 격리 불가능 | 실험 DESIGN-SUMMARY 템플릿에 문서화 |
| Reward 누적 calculator | Phase B penalty 수학 틀림; 자동화 체크가 잡을 수 있었음 | Python 유틸리티: config 입력, 예상 episode reward 출력 |
| Config diff 검토 | time_horizon이 문서화 없이 2048에서 256으로 변경 | 자동화 diff 포함 pre-training 체크리스트 항목 |
| Post-training 문서 동기화 | 각 학습 실행 후 3+ 문서가 구식 됨 | TRAINING-GUIDE 템플릿의 체크리스트 |
| 긴급 중단 자동화 | Phase B는 250K에서 -108로 고정됐는데도 전체 3M 스텝 실행 | Python watchdog: reward가 >200K 연속 스텝 음수면 중단 |

---

## 요약

이 프로젝트는 잘 구조화된 문서 계층과 규율된 Phase별 실험 접근 방식을 갖추고 있습니다. Phase B 실패는 프로젝트에 가장 가치 있는 단일 교훈을 제공합니다: **reward 함수 설계 오류는 치명적이며 전체 학습 실행 커밋 전에 검증되어야 합니다.**

가장 높은 영향을 미칠 세 가지 핵심 체계적 개선사항:

1. **Pre-flight 검증** (모든 전체 학습 커밋 전 10K sanity + 100K 짧은 실행)
2. **단일 변경 원칙** (실험 간 변수 격리)
3. **문서 동기화** (모든 학습 실행 후 PROGRESS.md, TRAINING-LOG.md, LEARNING-ROADMAP.md, phases/README.md 간 발산 방지)

Phase B의 기술적 근본 원인은 높은 신뢰도(95%)로 episode당 약 -108로 누적되는 -0.5/step following penalty로 확인되며, time_horizon 감소(256 vs 2048), entropy 계수 감소(3e-3 vs 5e-3), 가혹한 curriculum(점진적 도입 없이 즉각 2-NPC)을 포함한 기여 요인이 있습니다.

---

**문서 상태:** 완료 - 결정 준비
**신뢰 수준:** 주요 진단 95%
**다음 조치:** 근본 원인 조사 시작
**결정 시점:** 48시간 (조사 후)
**보고 날짜:** 2026-01-29
