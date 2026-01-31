# Policy Discovery Log

본 문서는 자율주행 ML 모델 학습 과정에서 시행착오를 통해 발견한 설계 원칙(Policy)을 기록합니다.
각 Phase의 실패와 성공에서 도출된 원칙이 국제 표준(SOTIF ISO 21448, UN R171/R157)과 어떻게 수렴하는지 추적합니다.

## 문서 목적

"Phase-X를 이렇게 학습했더니 이런 문제가 있었고, 로직을 수정(다음 Phase에 개선 적용)했더니 이런 정책을 지키게 되더라"
→ 경험적 발견이 국제 표준과 자연스럽게 수렴하는 과정을 기록

## 발견 원칙 요약 (Policy Registry)

| ID | 원칙명 | 발견 Phase | 관련 표준 | 상태 |
|----|--------|-----------|----------|------|
| P-001 | 변수 격리 원칙 (Single Variable Isolation) | Phase B v1→v2 | 실험설계 기본원칙 | 검증 완료 |
| P-002 | Staggered Curriculum 원칙 | Phase D v1→v2, E | SOTIF 점진적 복잡도 | 검증 완료 (위반 시 회복 가능하나 비효율) |
| P-003 | 능력 기반 체크포인트 선택 | Phase B v1→v2 | Transfer Learning Best Practice | 검증 완료 |
| P-004 | 보수적 페널티 설계 | Phase B v1→v2 | Reward Shaping Theory | 검증 완료 |
| P-005 | 횡가속도 제한 (예정) | Phase E (예정) | UN R157 (lat_accel < 0.3g) | 미검증 |
| P-006 | TTC 기반 반응 (예정) | Phase I (예정) | UN R171 (TTC 1.5-5.0s) | 미검증 |
| P-007 | 곡률 변화율 대응 (예정) | Phase H (예정) | SOTIF FI (dk/ds ≤ 0.1/m²) | 미검증 |
| P-008 | 센서 열화 대응 (예정) | Phase K (예정) | SOTIF TC (Triggering Condition) | 미검증 |
| P-009 | 관측-환경 결합 금지 (Observation Coupling) | Phase D v2→v3 | P-001 확장, SOTIF 점진적 복잡도 | 검증 완료 |
| P-010 | Scene-Config-Code 일관성 (Triple Consistency) | Phase D v3 254D | 시스템 무결성, Preflight Check | 검증 완료 |
| P-011 | Scene-Phase 매칭 (Scene-Phase Matching) | Phase F v1 | P-010 확장, 환경 무결성 | 검증 완료 |

## 상세 기록

### Entry #1: Phase B v1 → v2 (변수 격리 원칙)

**Phase**: Phase B v1 (Decision Learning)

**시도**: Phase 0 체크포인트 + 7개 하이퍼파라미터 동시 변경 + 즉시 2 NPC 투입

**문제**: 에이전트가 "정지"를 학습 (speedUnderPenalty -0.1/step이 과도)
- 초기 보상: +600 (Phase 0 전이 효과)
- 학습 진행 시: +600 → -108 (붕괴)
- 현상: 에이전트가 속도를 늦추거나 정지하여 페널티 회피
- 원인: 과도한 속도 페널티 + Phase 0 체크포인트(추월 능력 없음) + 즉시 복잡한 시나리오

**결과**: 최종 보상 -108 (FAILED)

**수정 (Phase B v2)**:
1. 7개 하이퍼파라미터 → Phase A 값으로 전부 복원 (변수 격리)
2. Phase 0 체크포인트 → Phase A 체크포인트-2500155.pt (추월 능력 보유)
3. 즉시 2 NPC → 0→1→2→3 단계적 커리큘럼
4. speedUnderPenalty -0.1 → -0.02 (80% 감소)
5. Blocked Detection 추가 (정지 상태 감지 및 페널티)

**결과**: +877 보상 (SUCCESS, 목표 +600 대비 146%)
- Stage 0 (Solo Warmup, 0 NPC): +1,340
- Stage 1 (Single Slow NPC, 1 NPC): -594 → +500
- Stage 2 (Two Mixed NPCs, 2 NPC): +630 → +845
- Stage 3 (Three Mixed NPCs, 3 NPC): +825 → +897 (최종)

**발견한 원칙**:
- **P-001 (변수 격리)**: 한 번에 하나의 변수만 변경한다. v1은 7개 파라미터를 동시에 변경하여 실패 원인 식별 불가. v2는 Phase A 설정을 유지하고 커리큘럼만 조정.
- **P-003 (능력 기반 체크포인트)**: 다음 Phase에 필요한 능력을 이미 보유한 체크포인트를 선택한다. Phase 0는 차선 유지만 가능, Phase A는 추월 능력 보유(+2113 보상).
- **P-004 (보수적 페널티)**: 페널티는 보수적으로 설계한다. 과도한 페널티는 학습을 방해하거나 역효과 발생. -0.1 → -0.02로 80% 감소 후 정상 학습.

**관련 표준**: 실험설계 기본원칙 (Controlled Experiment), Reward Shaping 이론

---

### Entry #2: Phase D v1 → v2 (Staggered Curriculum)

**Phase**: Phase D v1 (Lane Observation, 254D)

**시도**: 5개 커리큘럼 파라미터를 3-stage로 설계했으나, 3개가 동일 임계값(~400K steps)에서 동시 전환

**문제**: Step 4.68M에서 num_active_npcs, speed_zone_count, npc_speed_variation이 동시에 Stage 2로 전환
- 전환 직전 보상: +406 (Peak, 성공 직전)
- 전환 직후 보상: -4,825 (20K steps 내)
- 보상 붕괴: 5,231 points (-1,287% 변화)
- 회복 실패: -4,825 → -2,156 (부분 회복만 이루어짐)
- 시간 소진: 6M steps, 100분 학습 후 예산 고갈

**결과**: 최종 보상 -2,156 (FAILED)

**수정 (Phase D v2)**:
1. 동일 임계값(400K) → 단계별 분산 (200K/300K/350K)
   - num_active_npcs: 200K (가장 먼저)
   - speed_zone_count: 300K
   - npc_speed_variation: 350K (가장 나중에)
2. max_steps 6M → 10M (회복 시간 확보 + 여유 있는 학습)
3. Checkpoint Strategy: 각 stage 전환 시점에 체크포인트 저장

**결과**: 학습 중 (1.49M steps, reward -4.5, 상승 추세 확인됨)

**발견한 원칙**:
- **P-002 (Staggered Curriculum)**: 커리큘럼 파라미터는 서로 다른 임계값으로 단계적 전환한다. 동시 전환 시 에이전트는 여러 변화에 동시 대응해야 하며, 기존 정책이 무효화되면서 학습 붕괴 발생. 시간차를 두면 각 변화에 순차적으로 적응 가능.

**관련 표준**: SOTIF ISO 21448의 점진적 복잡도 접근 (Quadrant 2→3 이동 시 한 번에 하나의 FI/TC만 추가)

---

### Entry #3: Phase A (보상 신호 해석의 한계)

**Phase**: Phase A (Dense Overtaking)

**시도**: 추월 성공 시 +3.0 보너스 보상 설계
- 목표: 에이전트가 NPC를 추월하도록 유도
- 기대: 추월 이벤트 발생 시 명확한 보상 신호

**관찰**: 추월 감지 이벤트 0건. 에이전트는 추월 보너스가 아닌 속도 유지로 +2,113 달성
- 보상 구성: 속도 추적 1,985.30 (93.9%) + 진행 보상 284.91 (13.5%)
- 추월 보상: 0 (0건 감지)
- 최종 성과: +2,113 (목표 +900의 235%)
- 안전: 충돌률 0%, 목표 완료 100%

**문제**: 에이전트가 설계 의도(추월)가 아닌 다른 경로(속도)로 보상 최적화
- 추월 감지 시스템 작동 불확실
- 또는 에이전트가 차선 변경 없이 속도만으로 높은 보상 달성
- 실제 행동과 설계 의도의 불일치 (Reward Hacking의 경미한 형태)

**교훈**: 보상 함수의 의도와 에이전트의 실제 행동이 다를 수 있다
- 보상 신호만으로는 특정 행동 유도가 어려움
- 행동 검증(video logging, explicit event tracking) 필수
- Phase B에서 추월/추종 판단을 명시적 커리큘럼으로 유도하는 방향 채택

**후속 조치**: Phase B에서 추월 감지 시스템 검증 + 명시적 행동 유도 커리큘럼 설계

**관련 표준**: Reward Shaping Theory, Goodhart's Law ("측정 지표가 목표가 되면, 좋은 측정 지표가 아니게 된다")

---

### Entry #4: Phase D v2 → v3 (관측-환경 결합 금지)

**Phase**: Phase D v2 (Lane Observation, 254D)

**시도**: Phase D v2에서 254D 관측(+12D lane observation)과 4 NPC 커리큘럼을 동시에 적용
- 새로운 관측 공간(12D lane) 추가와 동시에 커리큘럼으로 NPC 수를 증가
- max_steps: 10M, staggered curriculum 적용 (P-002 적용)

**문제**: Step 7.87M에서 4 NPC 전환 시 보상 붕괴
- 전환 직전: +447 (Peak)
- 전환 직후: -756 (보상 급락, 1,203 points 하락)
- 근본 원인: 관측 공간 변경(242→254D)으로 아직 lane observation 학습이 불완전한 상태에서 환경 난이도까지 상승
- P-002(Staggered Curriculum)를 적용했음에도 실패 → P-001(변수 격리)의 확장 필요

**결과**: 최종 보상 -756 (FAILED)

**수정 (Phase D v3)**:
1. 커리큘럼 완전 제거 → 고정 환경 (3 NPC, speed_ratio 0.6, goal 150m, zones 2)
2. 관측 공간 변경(254D)만을 유일한 변수로 격리
3. max_steps: 5M (고정 환경이므로 충분)
4. init_path 없음 (242→254D 차원 변경으로 전이학습 불가)

**Phase D v3 결과 (242D 실수)**:
- VectorObservationSize 설정 오류로 실제 242D로 학습됨 (Entry #5 참조)
- 242D 기준 +835 달성 (Phase C와 동등, lane obs 미활성 상태)

**Phase D v3 결과 (254D 수정 후)**:
- **+895.5 reward** (5M steps 완료, SUCCESS)
- 254D 입력 확인: checkpoint `seq_layers.0.weight: [512, 254]`
- 242D 대비 **+7.2% 향상** (835 → 895.5)

**발견한 원칙**:
- **P-009 (관측-환경 결합 금지)**: 관측 공간 변경과 환경 난이도 변경을 동시에 수행하지 않는다. P-001(변수 격리)의 확장으로, 새로운 센서/관측을 추가할 때는 환경을 고정하고, 관측 학습이 완료된 후에만 커리큘럼을 적용한다.

**관련 표준**: P-001 확장, SOTIF ISO 21448의 점진적 복잡도 (한 번에 하나의 변수만)

---

### Entry #5: Phase D v3 VectorObservationSize 불일치 (Scene-Config-Code 일관성)

**Phase**: Phase D v3 (Lane Observation, 254D 시도)

**시도**: Phase D v3를 254D(242D + 12D lane)로 설계하여 학습 시작
- Config: `vehicle_ppo_phase-D-v3.yaml` (init_path 없음, 고정 환경)
- Scene: `PhaseB_DecisionLearning.unity` (16 agents)
- Agent code: `enableLaneObservation = true` (Unity Inspector에서 활성화)

**문제**: 학습 완료 후 checkpoint 검사에서 입력 차원 불일치 발견
- 기대: `seq_layers.0.weight: [512, 254]`
- 실제: `seq_layers.0.weight: [512, 242]`
- 원인: Unity Scene 파일의 `BehaviorParameters.VectorObservationSize`가 `242`로 남아있었음
- `enableLaneObservation = true`로 설정했지만, `VectorObservationSize`는 별도 수동 설정 필요
- ML-Agents는 Scene의 BehaviorParameters 값을 우선 사용 → 12D lane observation 데이터가 자동 절삭(truncation)
- 결과적으로 lane observation이 전혀 학습에 반영되지 않음 (12D all zeros와 동일)

**발견 경위**:
- 학습 로그에 `hidden_units: 128, num_layers: 2` 표시 → 조사 시작
- 확인: 해당 값은 `reward_signals.extrinsic.network_settings` (보상 신호 네트워크)로 정상
- 추가 확인: PyTorch checkpoint의 첫 번째 레이어 weight shape 검사에서 242D 발견
- 242D → 254D가 되어야 하는데 12D(lane obs)가 누락됨

**수정**:
1. Scene 파일에서 `VectorObservationSize: 242` → `254`로 변경 (16개 에이전트 모두)
2. `phase-D-v3-254d`로 재학습 시작
3. 500K checkpoint에서 `[512, 254]` 확인 → 수정 성공

**발견한 원칙**:
- **P-010 (Scene-Config-Code 일관성)**: 학습 시작 전 3가지 소스의 일관성을 검증한다:
  1. **Scene**: BehaviorParameters.VectorObservationSize
  2. **Config**: YAML의 network_settings (implicit observation size)
  3. **Code**: Agent 코드의 CollectObservations()가 실제 write하는 dimension
  - 셋 중 하나라도 불일치하면 학습이 잘못된 차원으로 진행되며, 성공적으로 완료되더라도 의도한 관측을 사용하지 않음
  - **Preflight Check 필수**: `.claude/agents/training-preflight.md` 에이전트로 자동 검증

**관련 표준**: 시스템 무결성 검증, Pre-flight Check (항공/우주 분야에서 차용)

---

### Entry #6: Phase E (Curved Roads) - P-002 위반과 회복

**Phase**: Phase E (Curved Roads, 254D)

**시도**: Phase D v3 체크포인트에서 초기화, 7개 커리큘럼 파라미터로 곡선 도로 학습
- road_curvature: 0→0.3→0.6→1.0 (4단계)
- curve_direction_variation: 0→1.0
- num_active_npcs: 0→1→2
- npc_speed_ratio: 0.4→0.7
- goal_distance: 100→150→200
- speed_zone_count: 1→2
- npc_speed_variation: 0→0.2

**문제**: Step 1.68M에서 4개 파라미터가 동시 전환 (P-002 위반)
- **동시 전환**: npc_speed_ratio(→0.7), goal_distance(→150), speed_zone_count(→2), npc_speed_variation(→0.2)
- 전환 직전 보상: +362
- 전환 직후 보상: -3,863 (4,225 points 폭락)
- **원인**: 4개 커리큘럼의 completion_criteria threshold가 유사 (280-300 범위)하여 거의 동시에 도달
- 이전 Phase D v1/v2와 동일한 패턴의 커리큘럼 붕괴

**회복**: Step 2.44M에서 양수 보상 복귀 (~800K steps 회복기간)
- 2.44M: -158 → 양수 전환
- 2.63M: goal_distance → LongGoal (200m)
- 2.88M: num_active_npcs → OneNPC
- 3.14M: num_active_npcs → TwoNPCs
- 3.47M: road_curvature → GentleCurves (0.3)
- 3.81M: road_curvature → ModerateCurves (0.6)
- 4.15M: road_curvature → SharpCurves (1.0)
- 3.58M: Peak +956, 이후 안정적 +920-940 유지

**결과**: 최종 보상 +892.6, Peak +938.2 (SUCCESS)
- 모든 7개 커리큘럼 최종 레슨 완료
- Sharp curves (1.0) + Mixed directions + 2 NPCs + 200m goal

**분석**: Phase D v1/v2와 Phase E의 차이
- Phase D v1: 동시 전환 후 회복 실패 (+406 → -2,156)
- Phase D v2: 동시 전환 후 회복 실패 (+447 → -756)
- Phase E: 동시 전환 후 회복 성공 (+362 → -3,863 → +938)
- **가설**: Phase E는 Phase D v3(254D, 고정 환경)에서 충분히 안정화된 정책을 기반으로 학습하여, 커리큘럼 충격에서 회복할 수 있는 기반이 존재했음
- **결론**: P-002 위반은 여전히 위험하지만, 강건한 기반 정책이 있으면 회복 가능. 그러나 800K steps(~13%)의 학습 예산이 낭비되므로 P-002 준수를 권장

**관련 표준**: P-002 (Staggered Curriculum) 보강, P-003 (능력 기반 체크포인트)의 유효성 재확인

---

### Entry #7: Phase F v1 (Multi-Lane) - Scene-Phase 불일치로 즉사

**Phase**: Phase F v1 (Multi-Lane, 254D)

**시도**: Phase E 체크포인트에서 초기화, 9개 커리큘럼으로 다차선 도로 학습
- num_lanes: 1 -> 2 -> 3 -> 4 (핵심 신규 기능)
- center_line_enabled: 0 -> 1
- road_curvature, curve_direction_variation (Phase E 유지)
- num_active_npcs, npc_speed_ratio, npc_speed_variation
- goal_distance, speed_zone_count
- P-002 준수: 임계값 300/350/400/450/500 분산 배치

**문제**: Step 1520K에서 `num_lanes` 1->2 전환 시 즉사
- 전환 직전 보상: +303 (안정적 학습 중)
- 전환 직후 보상: -8.19 (20K steps 내 붕괴)
- **4.27M steps (1520K~5820K) 동안 -8.15에 고착, 회복 불가**
- Phase D v1/v2, E의 커리큘럼 붕괴와 달리, 완전히 다른 패턴 (구조적 불일치)

**근본 원인**: **잘못된 Unity Scene 로딩**
- 학습 시 로딩된 Scene: `PhaseE_CurvedRoads.unity` (단일 차선, 도로 폭 4.5m)
- 필요한 Scene: `PhaseF_MultiLane.unity` (3차선 기본, 도로 폭 11.5m)
- `num_lanes` 파라미터가 2로 전환 → WaypointManager가 2차선 waypoint 생성
- 그러나 물리적 도로 표면은 4.5m (1차선) → Agent가 즉시 도로 이탈
- 도로 표면 크기는 Scene 생성 시 결정 (`numLanes * 3.5f + 1.0f`), 런타임 변경 불가

**결과**: 최종 보상 -8.15 (FAILED, 5.82M steps에서 수동 중단)

**수정 (Phase F v2)**:
1. Unity Editor에서 `PhaseF_MultiLane.unity` scene 로드 (build_index=4)
2. Scene 검증: 도로 폭 11.5m (RoadSurface scale 1.15), 3차선 지원 확인
3. Agent 검증: VectorObservationSize=254, enableLaneObservation=true 확인
4. 이전 학습 결과 삭제 후 fresh start

**발견한 원칙**:
- **P-011 (Scene-Phase Matching)**: 학습 시작 전 반드시 해당 Phase의 전용 Scene이 로딩되어 있는지 확인한다.
  - 각 Phase는 전용 Scene을 가짐 (PhaseA_DenseOvertaking, PhaseE_CurvedRoads, PhaseF_MultiLane 등)
  - Scene의 물리적 도로 구조 (폭, 곡률, 교차로)는 Scene 생성 시 결정되며, 런타임에 변경 불가
  - P-010 (Triple Consistency)의 확장: Scene-Config-Code에 더해 **Scene 파일 자체**의 일치도 검증 필요
  - **Preflight Check 필수**: 학습 시작 전 `get_active` scene name과 config의 phase가 일치하는지 확인

**관련 표준**: P-010 확장 (Scene-Config-Code 일관성 → Scene File 일관성), 환경 무결성 검증

---

### Entry #8-11: 예정 항목 (Phase F v2 ~ K)

(향후 학습 완료 시 추가 예정 - 아래는 예상 시나리오)

#### Entry #7: Phase E (예정) - 횡가속도 제한

**예상 시도**: 곡선 도로에서 고속 주행

**예상 문제**: 횡가속도 초과로 차선 이탈
- 현상: 곡선 진입 시 속도 유지 → 과도한 횡가속도 → 차선 이탈
- 원인: 속도-곡률 관계 미학습, lateral acceleration 제약 없음

**예상 수정**: 속도-곡률 관계 학습, lateral_accel penalty 추가
- 관측 공간: curvature (1D), curvature_lookahead (1D) 추가
- 보상: R_lateral_stability = -0.3 * max(0, a_lat - 7.0)
- 커리큘럼: 완만한 곡선(k=0.05) → 급곡선(k=0.15) 단계적 학습

**매칭 표준**: UN R157 (lateral acceleration < 0.3g = 2.94 m/s²)

#### Entry #5: Phase H (예정) - 곡률 전환점 대응

**예상 시도**: 클로소이드(완화곡선) 진입 구간에서 주행

**예상 문제**: 직선→곡선 전환 시 조향 지연, 차선 이탈
- 현상: 곡률 변화 구간에서 steering lag 발생
- 원인: 곡률 변화율(dk/ds) 미관측, 전방 예측 부족

**예상 수정**: curvature rate (dk/ds) 관측 추가, 전방 곡률 예측
- 관측 공간: curvature_rate (1D), curvature_type (1D) 추가
- 보상: R_curvature_response = +0.4 (steering_lag < 0.3s) else -0.2 * lag
- 검증: Lane deviation < 0.3m, steering rate < 15 deg/s

**매칭 표준**: SOTIF FI (Functional Insufficiency), UN R157 dk/ds ≤ 0.1/m²

#### Entry #6: Phase I (예정) - Cut-in/Cut-out 대응

**예상 시도**: NPC의 갑작스러운 차선 진입

**예상 문제**: TTC 부족, 급제동 시 jerk 초과
- 현상: Cut-in 감지 지연 → 급제동 → jerk > 3.0 m/s³
- 원인: TTC 관측 없음, 반응 시간 부족

**예상 수정**: TTC 기반 반응 학습, 비대칭 action space [-7, +2]
- 관측 공간: nearest_ttc (1D), lateral_intrusion (1D) 추가
- 행동 공간: deceleration limit [-7.0, +2.0] m/s²
- 커리큘럼: Gentle cut-in (TTC 4s) → Rapid cut-in (TTC 1.5s)

**매칭 표준**: UN R171 (max decel -7.0 m/s², jerk ≤ 3.0 m/s³, TTC 1.5-5.0s)

#### Entry #7: Phase K (예정) - 센서 열화 조건

**예상 시도**: 노이즈가 추가된 관측에서 주행

**예상 문제**: 관측 품질 저하 시 성능 급락
- 현상: Lane confidence < 0.5 시 성능 -30% 이상 하락
- 원인: 노이즈 환경 미학습, 열화 대응 전략 부재

**예상 수정**: degradation-aware 학습, 보수적 행동 전략
- 커리큘럼: Clean → 30% noise → 70% noise
- 보상: Confidence-weighted reward, fallback penalty
- 검증: 노이즈 환경에서 성능 유지 > 80%

**매칭 표준**: SOTIF TC (Triggering Condition), ISO 21448 Quadrant 3 축소

---

## 표준 수렴 맵

```
경험적 발견                          국제 표준
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
변수 격리 원칙 (P-001)        ←→  실험설계 기본원칙
Staggered Curriculum (P-002)  ←→  SOTIF 점진적 복잡도
능력 기반 체크포인트 (P-003)   ←→  Transfer Learning
보수적 페널티 설계 (P-004)     ←→  Reward Shaping Theory
관측-환경 결합 금지 (P-009)   ←→  P-001 확장 + SOTIF 점진적 복잡도
Scene-Config-Code 일관성(P-010)←→  Preflight Check (시스템 무결성)
Scene-Phase 매칭 (P-011)    ←→  P-010 확장 (환경 무결성)
횡가속도 제한 (P-005)         ←→  UN R157 (lat_accel < 0.3g)
TTC 기반 반응 (P-006)        ←→  UN R171 (TTC 1.5-5.0s)
곡률 변화율 대응 (P-007)      ←→  SOTIF FI + UN R157 dk/ds
센서 열화 대응 (P-008)        ←→  SOTIF TC (Triggering Condition)
```

## 기록 가이드라인

### 새 Entry 작성 시 필수 항목
1. **Phase**: 해당 Phase 이름과 버전
2. **시도**: 무엇을 어떻게 시도했는지
3. **문제**: 어떤 문제가 발생했는지 (수치 포함)
4. **수정**: 어떻게 고쳤는지 (구체적 변경 사항)
5. **결과**: 수정 후 결과 (보상, 성공률 등)
6. **발견한 원칙**: Policy Registry에 등록할 원칙 (P-XXX)
7. **관련 표준**: 매칭되는 국제 표준이나 이론

### Policy ID 규칙
- P-001 ~ P-099: 학습 설계 원칙 (실험설계, 커리큘럼, 체크포인트)
- P-100 ~ P-199: 안전 관련 원칙 (UN R157, R171, collision avoidance)
- P-200 ~ P-299: SOTIF 관련 원칙 (FI, TC, Quadrant 관리)
- P-300 ~ P-399: 성능/편의 원칙 (jerk, comfort, efficiency)

## 참조 문서
- SOTIF 전략: `docs/SOTIF-EDGE-CASE-STRATEGY.md`
- 기술 설계서: `docs/TECH-SPEC.md`
- Phase 로드맵: `docs/phases/README.md`
- 학습 로드맵: `docs/LEARNING-ROADMAP.md`
- 각 Phase 실험: `experiments/phase-*/README.md`
