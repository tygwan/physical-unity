# Autonomous Driving ML Learning Roadmap

자율주행 ML 학습 종합 로드맵. 지금까지 완료한 작업과 향후 모든 학습 계획을 포함합니다.

---

## Executive Summary

| Phase | 주제 | 상태 | 최고 성과 |
|-------|------|------|----------|
| **Foundation (v10-v11)** | 기본 주행 + 추월 시도 | ✅ 완료 | +51 (정체) |
| **Phase A** | Dense Overtaking (느린 NPC) | ✅ 완료 | **+937** |
| **Phase B** | Overtake vs Follow 판단 | ✅ 완료 | **+903** |
| **Phase C** | Multi-NPC 일반화 (4대) | ✅ 완료 | **+961** |
| **Phase D** | Lane Observation (254D) | 🔄 진행중 | -41 (830K) |
| **Phase E** | 곡선 도로 + 비정형 각도 | 📋 계획 | - |
| **Phase F** | N차선 + 중앙선 규칙 | 📋 계획 | - |
| **Phase G** | 교차로 (T자/십자) | 📋 계획 | - |
| **Phase H** | 신호등 + 정지선 | 📋 계획 | - |
| **Phase I** | U턴 + 특수 기동 | 📋 계획 | - |
| **Phase J** | 횡단보도 + 보행자 | 📋 계획 | - |
| **Phase K** | 장애물 + 긴급 상황 | 📋 계획 | - |
| **Phase L** | 복합 시나리오 통합 | 📋 계획 | - |

---

## Part 1: 완료된 학습 (Completed Training)

### Foundation Phase (v10-v11)

#### v10g: Lane Keeping + NPC Coexistence
- **기간**: 2026-01-20 ~ 2026-01-23
- **Steps**: 8M (4.97M effective)
- **결과**: Reward ~40 (NPC 4대 환경에서 정체)
- **성과**:
  - 차선 유지 (headingAlignment, lateralDeviation) 학습
  - NPC 충돌 회피 학습
  - 3-strike collision rule 적용
- **한계**: 느린 NPC 뒤에서 무한 대기 (추월 불가)
- **교훈**: followingBonus가 추월 학습을 방해

#### v11: Overtaking Reward (Sparse)
- **기간**: 2026-01-23 ~ 2026-01-24
- **Steps**: 8M
- **결과**: Reward ~51 (미미한 개선)
- **시도**:
  - overtakePassBonus (+3.0) 스파스 보상
  - SphereCast 기반 NPC 감지
- **실패 원인**:
  - Sparse reward로는 추월 학습 불가
  - targetSpeed = leadSpeed 구조가 추월 동기 제거
- **교훈**: Dense reward 필요, targetSpeed는 speedLimit으로 고정

---

### Phase A: Dense Overtaking (Single Slow NPC)

#### 학습 정보
| 항목 | 값 |
|------|-----|
| **Run ID** | v12_phaseA_fixed |
| **기간** | 2026-01-24 ~ 2026-01-25 |
| **Steps** | 2,000,209 |
| **Final Reward** | +714 |
| **Peak Reward** | **+937** |
| **Observation** | 242D |

#### 핵심 변경사항 (v11 → v12)
```yaml
핵심 변경:
  1. targetSpeed = speedLimit ALWAYS  # NPC 속도로 낮추지 않음
  2. followingBonus 완전 제거
  3. stuckBehindPenalty: -0.1/step (3초 후)
  4. Dense 5-phase overtaking reward:
     - Approaching: 감지 시작
     - Beside: +0.5 (차선 변경 시작)
     - Beside 유지: +0.2/step
     - Ahead: +1.0 (추월 완료)
     - LaneReturn: +2.0 (차선 복귀)
```

#### 버그 수정
```csharp
// 속도 0에서 패널티 회피 버그 수정
// Before: speed > 1f 조건으로 0속도 허점 존재
// After: 무조건 + 점진적 패널티
else if (speedRatio < 0.5f)
{
    float progressivePenalty = speedUnderPenalty * (2f - speedRatio * 2f);
    reward += progressivePenalty;
}
```

#### 학습 성과
- **첫 양수 보상**: 460K steps (+7.3)
- **Breakthrough**: 1.2M steps (+502.9)
- **Peak**: 1.37M steps (+937.0)
- **수렴**: 2.0M steps (+714.7)

---

### Phase B: Overtake vs Follow Decision

#### 학습 정보
| 항목 | 값 |
|------|-----|
| **Run ID** | v12_phaseB |
| **기간** | 2026-01-25 |
| **Steps** | 2,000,150 |
| **Final Reward** | **+903.3** |
| **Peak Reward** | **+994.5** |
| **Observation** | 242D |
| **Initialize From** | v12_phaseA_fixed |

#### 커리큘럼 설계
```yaml
NPC Speed Curriculum:
  - VerySlow: 30% of limit (threshold: 50.0)
  - Slow: 50% of limit (threshold: 45.0)
  - Medium: 70% of limit (threshold: 40.0)
  - Fast: 90% of limit (final)
```

#### 학습 목표 달성
- NPC < 70%: 추월 행동 관찰
- NPC > 85%: 적절한 따라가기/패싱
- Phase A 대비 +26% 성능 향상 (+714 → +903)

---

### Phase C: Multi-NPC Generalization

#### 학습 정보
| 항목 | 값 |
|------|-----|
| **Run ID** | v12_phaseC_242D |
| **기간** | 2026-01-26 ~ 2026-01-27 |
| **Steps** | 4,000,000 |
| **Final Reward** | **+961.8** |
| **Peak Reward** | **+1086.0** |
| **Observation** | 242D |
| **Initialize From** | v12_phaseB |

#### 커리큘럼 복잡도
```yaml
환경 복잡도:
  NPC 수: 1 → 2 → 3 → 4
  NPC 속도 변동: 40% → 60% → 80%
  목표 거리: 100m → 160m → 230m
  속도 구간: 1 → 2 → 4
```

#### 커리큘럼 충격 및 회복
- **90K**: +766 (peak before transition)
- **110K**: -814 (curriculum shock)
- **760K**: +11 (recovery)
- **4M**: +961 (final)

#### Phase 비교
| Phase | Reward | 환경 복잡도 | 개선 |
|-------|--------|-------------|------|
| Phase A | +937 | 1 NPC @ 30% | Baseline |
| Phase B | +903 | 1 NPC @ 30-90% | 판단력 |
| **Phase C** | **+961** | **4 NPC, 4 zones, 230m** | **+6% in 4x complexity** |

---

### 실패한 시도들

#### v12_ModularEncoder (Superseded)
- **목적**: Phase B → C 전환 시 observation 차원 변경 대응
- **설계**: 모듈별 encoder (ego, history, agents, route, speed, lane)
- **결과**: 구현 전 대안 발견으로 보류

#### v12_HybridPolicy (FAILED)
- **목적**: Phase B encoder 유지하며 lane observation 추가
- **Steps**: 3M
- **Best Reward**: -82.7 (step 1.44M)
- **실패**: Stage 5에서 catastrophic forgetting (-2171.9)
- **교훈**:
  - 사전학습 encoder를 unfreeze하면 안 됨
  - ONNX 형식이 ML-Agents와 호환되지 않음
  - 단순 재학습이 복잡한 아키텍처보다 효과적

---

### Phase D: Lane Observation (진행 중)

#### 학습 정보
| 항목 | 값 |
|------|-----|
| **Run ID** | v12_phaseD |
| **시작일** | 2026-01-27 |
| **목표 Steps** | 6,000,000 |
| **현재 Steps** | 830,000 (13.8%) |
| **현재 Reward** | **-41.0** |
| **Observation** | 254D (+12D Lane) |

#### Observation 구성
```yaml
Phase D (254D):
  ego_state: 8D        # position, velocity, heading, acceleration
  ego_history: 40D     # 5 past steps × 8D
  surrounding: 160D    # 20 agents × 8 features
  route_info: 30D      # 10 waypoints × 3
  speed_info: 4D       # current, limit, ratio, next_limit
  lane_info: 12D       # NEW: 3 lanes × 4D (dist, type, offset, heading)
```

#### 학습 진행
```
Step        Reward    Progress
────────────────────────────────
10K         -162.8    Start
210K        -106.5    ████
420K        -104.2    ████
630K         -87.4    ██████
830K         -41.0    ████████ ← Current Best
```

---

## Part 2: 향후 학습 계획 (Future Training Plan)

### Phase E: 곡선 도로 + 비정형 각도

#### 목표
- 곡률이 있는 도로에서 안정적 주행
- 90°/180°가 아닌 어중간한 각도의 도로 대응
- Trajectory planning 기초

#### Observation 추가 (+8D → 262D)
```yaml
curvature_info: 8D
  - current_curvature: 1D       # 현재 위치 곡률 (-1~+1)
  - lookahead_curvature: 5D     # 10m, 20m, 30m, 40m, 50m 앞 곡률
  - road_heading_delta: 1D      # 도로 방향과 차량 방향 차이
  - optimal_steering: 1D        # 곡률 기반 권장 조향각
```

#### 환경 구성
```yaml
Road Types:
  - Straight (기존)
  - Gentle Curve (R > 100m, 곡률 < 0.01)
  - Medium Curve (R = 50-100m)
  - Sharp Curve (R = 30-50m)
  - S-Curve (연속 곡선)
  - Variable Angle (15°, 30°, 45°, 60°, 75° 등)
```

#### 커리큘럼
```yaml
curriculum:
  road_curvature:
    - Lesson0_Straight: curvature=0
    - Lesson1_Gentle: curvature≤0.005
    - Lesson2_Medium: curvature≤0.01
    - Lesson3_Sharp: curvature≤0.02
    - Lesson4_SCurve: mixed
```

#### Reward 추가
```yaml
rewards:
  curvature_following: +0.3     # 곡률에 맞는 조향
  smooth_curve_navigation: +0.2 # 부드러운 곡선 주행
  curve_overspeed: -1.0         # 곡선에서 과속
  understeer: -0.5              # 곡선에서 미끄러짐
  oversteer: -0.5               # 오버스티어
```

#### 예상 Steps: 4-6M

---

### Phase F: N차선 왕복도로 + 중앙선 규칙

#### 목표
- 왕복 2차선, 4차선, 6차선 도로
- 중앙선 침범 절대 금지 (Hard Constraint)
- 차선 변경 규칙 학습

#### Observation 추가 (+6D → 268D)
```yaml
lane_structure_info: 6D
  - total_lanes: 1D             # 전체 차선 수 (정규화)
  - current_lane_index: 1D      # 현재 차선 번호
  - distance_to_centerline: 1D  # 중앙선까지 거리
  - centerline_type: 1D         # 0=없음, 0.5=점선, 1.0=실선/이중실선
  - can_change_left: 1D         # 좌측 차선변경 가능 여부
  - can_change_right: 1D        # 우측 차선변경 가능 여부
```

#### Hard Constraint (코드 레벨)
```csharp
// 중앙선 침범 시 강제 복귀 + 즉시 에피소드 종료
void FixedUpdate()
{
    if (IsCrossingCenterline() && !IsUTurnZone())
    {
        AddReward(-10f);  // 치명적 패널티
        EndEpisode();     // 즉시 종료
    }
}

// Action Masking으로 중앙선 방향 조향 차단
public override void WriteDiscreteActionMask(IDiscreteActionMask actionMask)
{
    if (IsNearCenterline() && !IsPermittedZone())
    {
        actionMask.SetActionEnabled(steeringBranch, leftSteerAction, false);
    }
}
```

#### 환경 구성
```yaml
Road Configurations:
  - 2-Lane Bidirectional (왕복 2차선)
  - 4-Lane Bidirectional (왕복 4차선, 중앙분리대 없음)
  - 4-Lane Divided (왕복 4차선, 중앙분리대)
  - 6-Lane Divided (왕복 6차선)

Centerline Types:
  - YELLOW_DASHED: 추월 가능 구간
  - YELLOW_SOLID: 추월 금지
  - DOUBLE_YELLOW: 절대 금지 (Hard Constraint)
  - BARRIER: 물리적 분리대
```

#### 예상 Steps: 4-6M

---

### Phase G: 교차로 (T자/십자/Y자)

#### 목표
- 교차로 인식 및 진입
- Navigation command에 따른 회전
- 우선권 규칙 학습

#### Observation 추가 (+14D → 282D)
```yaml
intersection_info: 14D
  - distance_to_intersection: 1D
  - intersection_type: 1D         # T=0.33, Cross=0.67, Y=1.0
  - num_exits: 1D                 # 출구 수
  - entry_angle: 1D               # 진입 각도
  - navigation_command: 6D        # one-hot [직진, 좌회전, 우회전, 유턴, 좌차선, 우차선]
  - has_priority: 1D              # 우선권 여부
  - oncoming_vehicles: 1D         # 대향 차량 유무
  - cross_traffic: 1D             # 교차 교통 유무
```

#### 환경 구성
```yaml
Intersection Types:
  T_Junction:
    - 3 exit edges
    - 좌/우회전 또는 직진

  Cross_Junction:
    - 4 exit edges
    - 모든 방향 가능

  Y_Junction:
    - 3 exit edges
    - 비대칭 각도 (30°, 45°, 60°)

Priority Rules:
  - 직진 우선
  - 우측 차량 우선 (무신호)
  - 회전 차량 양보
```

#### Reward 추가
```yaml
rewards:
  correct_turn: +5.0              # 올바른 방향 진입
  wrong_turn: -5.0                # 잘못된 방향
  missed_turn: -3.0               # 회전 실패
  yield_correctly: +2.0           # 올바른 양보
  failed_to_yield: -5.0           # 양보 실패 (위험)
  intersection_speed_compliance: +0.3  # 교차로 감속
```

#### 커리큘럼
```yaml
curriculum:
  intersection_complexity:
    - Lesson0: 직선만
    - Lesson1: T자 1개 (좌/우회전)
    - Lesson2: 십자 1개
    - Lesson3: Y자 (비정형 각도)
    - Lesson4: 복합 (2-3개 교차로)
```

#### 예상 Steps: 6-8M

---

### Phase H: 신호등 + 정지선

#### 목표
- 신호등 상태 인식 (적/황/녹)
- 정지선 준수
- 황색 신호 딜레마 해결

#### Observation 추가 (+8D → 290D)
```yaml
traffic_light_info: 8D
  - light_state: 3D               # one-hot [적, 황, 녹]
  - distance_to_light: 1D
  - time_to_change: 1D            # 신호 변경까지 시간 (추정)
  - distance_to_stop_line: 1D
  - can_stop_safely: 1D           # 안전 정지 가능 여부
  - should_proceed: 1D            # 진행 권장 여부
```

#### 의사결정 로직
```yaml
Yellow Light Dilemma:
  - 정지 가능 거리 내: 정지 권장
  - 정지 불가능 거리: 진행 권장
  - 계산: stopping_distance = v²/(2*max_decel) + reaction_distance
```

#### Reward 추가
```yaml
rewards:
  stop_at_red: +1.0               # 적색에서 정지
  run_red_light: -10.0            # 적색 신호 위반 (EndEpisode)
  yellow_safe_stop: +0.5          # 황색에서 안전 정지
  yellow_safe_proceed: +0.3       # 황색에서 안전 통과
  unnecessary_stop_green: -0.5    # 녹색에서 불필요 정지
  stop_line_compliance: +0.3      # 정지선 준수
  stop_line_overshoot: -1.0       # 정지선 초과
```

#### 예상 Steps: 4-6M

---

### Phase I: U턴 + 특수 기동

#### 목표
- U턴 가능 구간 인식
- U턴 기동 실행
- 3점 회전 등 특수 기동

#### Observation 추가 (+4D → 294D)
```yaml
special_maneuver_info: 4D
  - is_uturn_zone: 1D             # U턴 허용 구간
  - uturn_space_available: 1D     # U턴 공간 충분 여부
  - reverse_space: 1D             # 후진 공간
  - maneuver_command: 1D          # 특수 기동 명령
```

#### 기동 종류
```yaml
Special Maneuvers:
  U_Turn:
    - 조건: U턴 허용 구간 + 충분한 공간
    - 실행: 좌회전 180° (또는 우회전 180°)

  Three_Point_Turn:
    - 조건: 좁은 도로 + 회전 불가
    - 실행: 전진-후진-전진 3단계

  Parallel_Parking:
    - 조건: 주차 공간 탐지
    - 실행: 측면 주차 기동 (Future)
```

#### 예상 Steps: 4-6M

---

### Phase J: 횡단보도 + 보행자

#### 목표
- 횡단보도 인식 및 정지
- 보행자 탐지 및 양보
- 보행자 의도 예측

#### Observation 추가 (+12D → 306D)
```yaml
pedestrian_info: 12D
  - crosswalk_distance: 1D
  - crosswalk_occupied: 1D
  - pedestrian_count: 1D          # 감지된 보행자 수
  - nearest_pedestrian: 4D        # [x, z, vx, vz] 상대 위치/속도
  - pedestrian_intent: 2D         # [crossing, waiting]
  - pedestrian_priority: 1D       # 보행자 우선권
  - safe_to_proceed: 1D
```

#### Reward 추가
```yaml
rewards:
  yield_to_pedestrian: +2.0
  pedestrian_collision: -20.0     # 매우 높은 패널티
  crosswalk_stop: +1.0
  crosswalk_violation: -5.0       # 보행자 있을 때 통과
  unnecessary_stop: -0.2          # 보행자 없을 때 불필요 정지
```

#### 예상 Steps: 6-8M

---

### Phase K: 장애물 + 긴급 상황

#### 목표
- 정적/동적 장애물 회피
- 긴급 차량 대응
- 돌발 상황 대처

#### Observation 추가 (+10D → 316D)
```yaml
obstacle_info: 10D
  - obstacle_detected: 1D
  - obstacle_type: 3D             # [static, dynamic, emergency]
  - obstacle_distance: 1D
  - obstacle_size: 2D             # [width, length]
  - avoidance_direction: 1D       # 권장 회피 방향
  - emergency_vehicle: 1D         # 긴급 차량 감지
  - should_pull_over: 1D          # 갓길 정차 필요
```

#### 장애물 종류
```yaml
Obstacle Types:
  Static:
    - 낙하물
    - 공사 구간
    - 정지 차량

  Dynamic:
    - 급정거 차량
    - 끼어드는 차량
    - 자전거

  Emergency:
    - 구급차
    - 소방차
    - 경찰차
```

#### 예상 Steps: 6-8M

---

### Phase L: 복합 시나리오 통합

#### 목표
- 모든 요소 통합 테스트
- 일반화 능력 검증
- 실제 주행 시나리오 시뮬레이션

#### 통합 Observation (316D + 확장)
```yaml
Full Observation (~320D):
  ego_state: 8D
  ego_history: 40D
  surrounding_vehicles: 160D
  route_info: 30D
  speed_info: 4D
  lane_info: 12D
  curvature_info: 8D
  lane_structure: 6D
  intersection_info: 14D
  traffic_light_info: 8D
  special_maneuver: 4D
  pedestrian_info: 12D
  obstacle_info: 10D
  context_flags: 4D               # [urban, highway, residential, parking]
```

#### 시나리오 조합
```yaml
Integrated Scenarios:
  Urban_Complex:
    - 다차선 + 신호등 + 보행자 + 곡선

  Highway_Merge:
    - 고속 합류 + 차선 변경 + 추월

  Residential_Area:
    - 좁은 도로 + 보행자 + 주차 차량

  Construction_Zone:
    - 장애물 + 차선 변경 + 속도 제한
```

#### 예상 Steps: 10-15M

---

## Part 3: 학습 원칙 및 가이드라인

### 1. Observation 설계 원칙

```yaml
원칙:
  1. 점진적 확장: 한 Phase에 +10~20D 이내
  2. 모듈화: 각 기능별 분리된 observation 블록
  3. 정규화: 모든 값 -1~1 또는 0~1 범위
  4. Redundancy 회피: 중복 정보 최소화

Space Size 동기화:
  - BehaviorParameters Space Size = 실제 observation 차원
  - Unity Editor에서 Tools > ML-Agents > Update Observation Size
```

### 2. Reward 설계 원칙

```yaml
원칙:
  1. Dense > Sparse: 학습 신호 충분히 제공
  2. 즉시 종료: 치명적 실패는 EndEpisode()
  3. Rate-independent: 연속 패널티는 ×deltaTime
  4. 허점 없음: 모든 조건에서 적절한 보상/패널티
  5. 균형: 긍정 보상과 부정 패널티 균형

패널티 스케일:
  - 경미: -0.1 ~ -0.5
  - 일반: -1.0 ~ -3.0
  - 심각: -5.0
  - 치명적: -10.0 + EndEpisode
```

### 3. 커리큘럼 설계 원칙

```yaml
원칙:
  1. 한 번에 하나씩: 한 차원만 난이도 증가
  2. 점진적: 2배 이하 복잡도 증가
  3. 충분한 학습: min_lesson_length > 300
  4. 충격 대비: 커리큘럼 전환 후 회복 기간 예상

Threshold 설계:
  - 각 환경 변수별 분리된 threshold
  - 단계적 진행 (0 → 1 → 2 → 4, 한 번에 2 → 4 금지)
```

### 4. 규칙 vs 학습 분류

```yaml
Hard Constraints (코드로 강제):
  - 중앙선 침범 금지
  - 역주행 금지
  - 적색 신호 위반 금지
  - 보행자 충돌 금지

Soft Constraints (Reward로 학습):
  - 차선 유지
  - 속도 준수
  - 안전 거리 유지
  - 부드러운 주행

Pure Learning (순수 학습):
  - 최적 경로 선택
  - 추월 타이밍
  - 차선 변경 판단
  - 속도 조절
```

### 5. 실험 관리 원칙

```yaml
체크포인트:
  - 500K steps마다 자동 저장
  - Best reward 모델 별도 보관
  - Phase 완료 시 Unity 배포

모니터링:
  - TensorBoard 실시간 확인 (localhost:6006)
  - 500K마다 training-orchestrator 분석
  - 이상 징후 시 즉시 검토

문서화:
  - TRAINING-LOG.md에 모든 실험 기록
  - 성공/실패 원인 분석
  - 교훈 및 개선점 정리
```

---

## Part 4: 예상 일정

### 타임라인 (예상)

```
2026-01 ─────────────────────────────────────────────────
  Week 4: Phase D 완료 (Lane Observation)

2026-02 ─────────────────────────────────────────────────
  Week 1-2: Phase E (곡선 도로)
  Week 3-4: Phase F (N차선 + 중앙선)

2026-03 ─────────────────────────────────────────────────
  Week 1-3: Phase G (교차로)
  Week 4: Phase H (신호등)

2026-04 ─────────────────────────────────────────────────
  Week 1-2: Phase I (U턴)
  Week 3-4: Phase J (보행자)

2026-05 ─────────────────────────────────────────────────
  Week 1-2: Phase K (장애물)
  Week 3-4: Phase L (복합 통합)

2026-06+ ────────────────────────────────────────────────
  Camera Visual Integration
  Euro NCAP Benchmark
  Hybrid RL+IL (GAIL/CIMRL)
```

### 리소스 요구사항

| Phase | GPU 사용량 | 예상 학습 시간 | Training Areas |
|-------|-----------|---------------|----------------|
| D-F | ~4 GB | 4-6M steps × 10min/M | 16 |
| G-H | ~5 GB | 6-8M steps × 10min/M | 16 |
| I-K | ~6 GB | 4-8M steps × 10min/M | 16 |
| L | ~8 GB | 10-15M steps × 10min/M | 16-32 |
| Camera | ~12 GB | 10M+ steps | 8-16 |

---

## Part 5: 성공 기준

### Phase별 목표 Reward

| Phase | 목표 Reward | 핵심 지표 |
|-------|-------------|----------|
| D | > 500 | Lane keeping accuracy |
| E | > 400 | Curve navigation success |
| F | > 500 | Centerline violation = 0% |
| G | > 400 | Correct turn rate > 90% |
| H | > 450 | Red light violation = 0% |
| I | > 350 | U-turn success rate > 80% |
| J | > 400 | Pedestrian collision = 0% |
| K | > 350 | Obstacle avoidance > 95% |
| L | > 500 | Route completion > 85% |

### 최종 시스템 목표

```yaml
Safety Metrics:
  collision_rate: < 1%
  centerline_violation: 0%
  red_light_violation: 0%
  pedestrian_incident: 0%

Performance Metrics:
  route_completion: > 90%
  correct_turn_rate: > 95%
  speed_compliance: > 95%
  lane_keeping: > 98%

Comfort Metrics:
  avg_jerk: < 2 m/s³
  max_lateral_acc: < 3 m/s²
  smooth_steering: > 90%

Latency:
  inference_time: < 50ms
  end_to_end: < 200ms
```

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-27 | 1.0 | 초기 작성 - 완료된 Phase A-C, 진행중 Phase D, 향후 계획 E-L |

---

*이 문서는 학습 진행에 따라 지속적으로 업데이트됩니다.*
