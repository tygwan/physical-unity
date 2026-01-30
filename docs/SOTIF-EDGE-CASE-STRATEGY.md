# SOTIF & Edge Case Strategy for ML Training Orchestration

**Document Version**: 1.0
**Created**: 2026-01-29
**Status**: Active
**Scope**: ISO 21448 (SOTIF), UN R171 (DCAS), UN R157 (ALKS) 기반 ML 학습 커리큘럼 설계

---

## 1. Background: Feedback Integration

### 1.1 Received Feedback

프로젝트 리뷰에서 다음 3가지 피드백을 수신:

1. **도로 타입 변경 노드 지점 대응**
   - 직선->곡선, 곡선->직선 전환점(node)에서의 edge case 대응 필요
   - 곡률 변화 지점에서의 제어 불안정성 우려

2. **UN R171 Physical Test Extended Conditions**
   - UN Regulation 171의 물리적 테스트 조건을 edge case로 활용 가능
   - DCAS(Driver Control Assistance Systems) 규제의 테스트 시나리오 참조

3. **SOTIF (ISO 21448) 참조 제안**
   - ISO 표준에서 edge case 정의/핸들링 방법론 참조
   - 체계적인 안전성 검증 프레임워크 도입

### 1.2 Feedback -> Action Mapping

| Feedback | Action | Impact |
|----------|--------|--------|
| 도로 노드 지점 | Phase H: Curvature Nodes 신설 | 곡률 전환 특화 학습 |
| UN R171 | Phase I: Cut-in/Cut-out 시나리오 | 규제 파라미터 기반 보상 설계 |
| SOTIF | 전체 Phase에 FI/TC 프레임워크 적용 | 체계적 검증/문서화 |

---

## 2. SOTIF (ISO 21448) Framework

### 2.1 Overview

ISO 21448은 **Safety of the Intended Functionality**을 정의하며, 시스템 결함이 아닌 **기능적 한계(Functional Insufficiency)**로 인한 위험을 다룬다. ISO 26262(기능 안전)와 상호 보완적이며, 특히 ML/AI 기반 시스템에 필수적이다.

### 2.2 4-Quadrant Safety Model

SOTIF의 핵심 분류 체계:

```
              KNOWN                    UNKNOWN
         +--------------+         +--------------+
  SAFE   | Quadrant 1   |         | Quadrant 4   |
         | Known Safe   |         | Unknown Safe |
         | - 직선도로   |         | - 학습 중    |
         | - 일정 속도  |         |   자연 발견  |
         +--------------+         +--------------+
         +--------------+         +--------------+
 UNSAFE  | Quadrant 2   |         | Quadrant 3   |
         | Known Unsafe |         | Unknown Unsafe|
         | - 급곡선     |         | - 미발견     |
         | - Cut-in     |         |   위험 상황  |
         +--------------+         +--------------+
```

| Quadrant | 검증 방법 | ML 학습 대응 | 프로젝트 Phase |
|----------|----------|-------------|--------------|
| **1. Known Safe** | Requirements-Based Testing | 기본 커리큘럼 | Phase A-C |
| **2. Known Unsafe** | Targeted Verification | 특화 시나리오 학습 | Phase D-G |
| **3. Unknown Unsafe** | Scenario-Based Exploration | Adversarial + SOTIF 분석 | Phase H-L |
| **4. Unknown Safe** | Field Monitoring | 실제 배포 후 데이터 수집 | Phase M |

**축소 전략**: Quadrant 2/3의 면적을 줄이고, Quadrant 1을 확대하는 것이 목표.

### 2.3 Functional Insufficiency (FI)

FI는 시스템이 의도한 기능을 수행할 수 없는 성능 한계 또는 사양 부족을 의미한다.

**FI vs HARA (ISO 26262)**:

| 측면 | SOTIF FI | HARA (ISO 26262) |
|------|----------|------------------|
| 초점 | Specification 부족, 성능 한계 | HW/SW 결함, 고장 모드 |
| 원인 | 설계 한계, 데이터 부족, 일반화 실패 | Systematic fault, Random error |
| ML/AI | 핵심 (모델 학습 한계) | 부차적 |

**FI 분류 체계**:

```
Sensor Level FI
  - 객체 감지 실패 (역광, 포화)
  - 차선 인식 불가 (마모, 혼동)
  - LiDAR 오염 (안개, 비)

Algorithm Level FI
  - Distribution Shift (학습 != 배포 환경)
  - OOD 샘플 처리 불가
  - 극단 상황 일반화 실패

System Level FI
  - Sensor Fusion 실패
  - 액추에이터 응답 지연
  - 예측 신뢰도 저하
```

**ML 특수 FI**:
- **데이터 부족**: 특정 시나리오 학습 데이터 미포함
- **일반화 실패**: Covariate shift, label shift
- **Overconfidence**: OOD 데이터에 대한 높은 확신도

### 2.4 Triggering Condition (TC)

TC는 FI를 활성화시키는 특정 시나리오 조건이다.

**TC 식별 방법론**:

1. **시나리오 기반**: ODD 파라미터 공간 탐색
   - v_ego x v_other x distance x weather x curvature
   - Criticality = f(TTC, Headway, Lateral_Deviation)

2. **데이터 기반**: 모델 실패 케이스 분석
   - False Negative/Positive 패턴 클러스터링
   - 공통 파라미터 범위 식별

3. **전문가 기반**: FMEA, 도메인 지식

**FI x TC Matrix 예시**:

| | TC: 급곡선 진입 | TC: Cut-in TTC<2s | TC: 센서 열화 |
|---|---|---|---|
| **FI: 곡률 예측 오류** | HIGH | LOW | MEDIUM |
| **FI: 반응 지연** | MEDIUM | HIGH | MEDIUM |
| **FI: 조향 포화** | HIGH | LOW | LOW |

### 2.5 Residual Risk 허용 수준

SOTIF GAMAB 원칙: 새 시스템은 기존보다 위험하지 않아야 함.

| 기준 | 값 |
|------|-----|
| 인간 운전자 사고율 | 4.1-4.85 /M miles |
| AV 목표 (ASIL D) | < 0.05 harmful events /M miles |
| ODD 커버리지 목표 | > 90% |
| 센서 가용성 | > 99.9% |

---

## 3. UN R171 (DCAS) & UN R157 (ALKS) Regulations

### 3.1 Regulation Overview

| 항목 | UN R171 (DCAS) | UN R157 (ALKS) |
|------|---------------|----------------|
| SAE Level | Level 2 | Level 3 |
| 발효 | 2024년 9월 | 2021년 (확장 2024) |
| 최대 속도 | 130 km/h | 60 km/h (확장: 130 km/h) |
| 운전자 역할 | 항상 감시, 개입 가능 | 시스템 운전 담당 |
| 주요 테스트 | Cut-in, Cut-out, Decel | Lane keeping, Change |
| 성공 기준 | 90% 시나리오 통과 | Annex 3 시나리오 통과 |

### 3.2 Cut-in Test Parameters (Annex 4)

선행/인접 차량이 에고 차선으로 침입하는 시나리오:

| Parameter | Range | Note |
|-----------|-------|------|
| TTC (Time-to-Collision) | 1.5 - 5.0 s | 충돌 회피 마진 |
| Relative Speed (dv) | -20 ~ +20 km/h | 상대 접근 속도 |
| Initial Distance | 50 - 150 m | 시인 범위 |
| Cut-in Angle | 15 - 30 deg | 차선 변경 급격함 |
| Max Response Accel | -7.0 ~ +2.0 m/s^2 | 급제동 제한 |
| Lateral Accel | +/- 3.0 m/s^2 | 안정성 |
| Reaction Time | < 500 ms | 인지 |

### 3.3 Deceleration Test Parameters

| Parameter | Value | Note |
|-----------|-------|------|
| Lead Vehicle Decel | -3.0 ~ -7.0 m/s^2 | 선행차 급제동 |
| Max Ego Decel | -7.0 m/s^2 | 안전 한계 |
| Jerk Limit | <= 3.0 m/s^3 | 승객 편의 |
| Start Distance | 40 - 100 m | 초기 차간 거리 |
| Ego Speed | 60 - 130 km/h | 운영 범위 |

### 3.4 Lane Keeping Test Parameters (UN R157 Annex 3)

| Parameter | Value | Note |
|-----------|-------|------|
| Curvature Rate (dk/ds) | <= 0.1 /m^2 | 조향 부드러움 |
| Max Lateral Accel | <= 2.94 m/s^2 (0.3g) | 안전/편의 |
| Crosstrack Error | <= 0.3 m (곡선) | 진입 정확도 |
| Steering Angle Rate | <= 15 deg/s | 액추에이터 한계 |
| Lane Width | >= 3.5 m | 테스트 도로 |

### 3.5 Extended Operating Conditions

```
Speed:        40 - 130 km/h (고속도로)
Weather:      Clear, Rain(40mm/h), Fog(vis 200m)
Road Surface: Dry(mu=1.0), Wet(mu=0.7), Icy(mu=0.3)
Lane Marking: High contrast, Faded, Partial
Lighting:     Day(>500 lux), Night(headlight)
Temperature:  -10C ~ +50C
```

---

## 4. Curvature Transition Node Dynamics

### 4.1 Clothoid/Euler Spiral Theory

도로 설계에서 직선->곡선 전환은 Clothoid (Euler Spiral)로 구현된다.

**핵심 수식**:
```
R * L = A^2        (Clothoid 본질 방정식)
k = L / A^2        (곡률)
dk/dL = 1 / A^2    (곡률 변화율, 선형)
```

- R: 곡선 반경, L: 곡선 길이, A: Clothoid 파라미터
- A가 작을수록 곡률 변화 급격, 클수록 완만
- AASHTO 기준: 400대/일 이상 도로에서 반경 기준 이하 시 의무 적용

### 4.2 Vehicle Dynamics at Transition Points

**기본 관계식**:
```
Lateral Acceleration:  a_lat = v^2 * k
Lateral Jerk:          jerk_lat = v^3 * dk/ds
Safe Speed:            v_safe = sqrt(mu * g * R)
Required Steering:     d(delta)/dt = (v/L) * dk/ds
```

**수치 예시**:
```
v = 20 m/s (72 km/h), k = 0.01 (R=100m)
  a_lat = 400 * 0.01 = 4.0 m/s^2

v = 20 m/s, dk/ds = 0.05 /m^2
  jerk_lat = 8000 * 0.05 = 400 m/s^3  (!!!)
  -> 안전 기준 3.0 m/s^3 대비 100x 이상 초과

v_safe at R=100m, mu=0.9:
  v_safe = sqrt(0.9 * 9.81 * 100) = 29.7 m/s = 107 km/h
```

### 4.3 Friction Circle Constraint

타이어의 횡력과 종력은 마찰원 안에서만 동시 발생 가능:
```
F_lateral^2 + F_longitudinal^2 <= (mu * N)^2
```

곡선 진입 시 감속(종방향) + 조향(횡방향) 동시 필요 -> 커플링 문제 발생.
독립적인 두 제어기로는 차량 안정성 상실 위험.

### 4.4 LKA Failure at Curvature Transitions

실제 LKA 시스템 성능 데이터 (2025 연구):

| Metric | Value | Source |
|--------|-------|--------|
| LKA 곡률 임계값 | 0.006 /m (R~167m) | OpenLKA dataset |
| 안전 편차 한계 | 0.25 m | anomaly threshold |
| 중대 고장 한계 | 0.65 m 또는 해제 | critical failure |
| 최악 사례 편차 | 1.2 m | Hyundai Ioniq 5, sharp curve |

회귀 모델: `Lane_Deviation = -8.327 * Curvature + 0.214` (R^2 = 0.673)

**실패 요인 순위**: (1) 페이드된 차선 표시, (2) 낮은 명암 대비, (3) 날카로운 곡률

### 4.5 MPC vs RL for Curvature Handling

| Aspect | MPC | RL (DRL) |
|--------|-----|----------|
| Prediction | 차량 모델 기반 | 신경망 정책 |
| Curvature Response | 고정 Horizon (N) | 학습된 적응 |
| Computation | 높음 (N 증가 시) | 78% 절감 (추론 시) |
| Model Error | 작은 범위 내 정확 | 큰 오차에 강건 |
| Strong Curves | Horizon 적응 필요 | 자동 적응 |

**Hybrid 권장**: MPC-PID 시연 기반 DRL (초기 모방 -> 후기 자율 학습)

---

## 5. Vehicle Dynamics Reference Values

### 5.1 Safety & Comfort Limits

| Parameter | Safety Limit | Comfort Limit | Unit |
|-----------|-------------|---------------|------|
| Lateral Acceleration | 3.0 | 1.5 | m/s^2 |
| Longitudinal Deceleration | 7.0 | 2.0 | m/s^2 |
| Jerk (lateral) | 3.0 | 1.0 | m/s^3 |
| Jerk (longitudinal) | 3.0 | 1.0 | m/s^3 |
| Steering Angle | +/-30 | +/-15 | deg |
| Steering Rate | 15 | 10 | deg/s |
| CTE (straight) | 0.5 | 0.2 | m |
| CTE (curve) | 0.3 | 0.1 | m |

### 5.2 Tire Friction Coefficients

| Surface | mu (dry) | mu (wet) | mu (icy) |
|---------|----------|----------|----------|
| Asphalt | 0.8-1.0 | 0.5-0.7 | 0.1-0.3 |
| Concrete | 0.7-0.9 | 0.5-0.7 | 0.1-0.3 |
| Gravel | 0.5-0.7 | 0.4-0.6 | - |

### 5.3 Scenario Parameter Ranges

| Parameter | Range | Unit |
|-----------|-------|------|
| Ego Speed | 40 - 130 | km/h |
| Relative Speed | -50 ~ +50 | km/h |
| TTC | 1.5 - 10.0 | s |
| Lane Width | 3.0 - 4.5 | m |
| Curvature (1/R) | 0.0 - 0.02 | 1/m |
| Curvature Rate (dk/ds) | 0.0 - 0.1 | 1/m^2 |
| Road Friction (mu) | 0.1 - 1.0 | - |

---

## 6. Application to Training Orchestration

### 6.1 Current Phase Structure (A-G) mapped to SOTIF

| Phase | Description | SOTIF Quadrant | FI Focus |
|-------|-------------|---------------|----------|
| A | Lane Keeping (Overtaking) | 1: Known Safe | Baseline |
| B | Decision Learning (NPC) | 1->2 | Algorithm FI: NPC interaction |
| C | Multi-NPC Generalization | 2: Known Unsafe | Generalization |
| D | Lane Observation (254D) | 2: Known Unsafe | Sensor FI: lane detection |
| E | Curved Roads | 2: Known Unsafe | Control FI: curvature |
| F | Multi-lane | 2: Known Unsafe | System FI: lane change |
| G | Intersection | 2: Known Unsafe | Complex FI: turning |

### 6.2 Proposed Extension (H-M) based on Feedback

| Phase | Name | SOTIF Quadrant | Key FI/TC | Source |
|-------|------|---------------|-----------|--------|
| **H** | Curvature Nodes | 3: Unknown Unsafe | FI: curvature prediction, speed-steering mismatch | Feedback #1 |
| **I** | Cut-in/Cut-out | 3: Unknown Unsafe | FI: reaction latency, decel limit | Feedback #2, UN R171 |
| **J** | Sensor Degradation | 3: Unknown Unsafe | FI: lane confidence drop, sensor noise | SOTIF standard |
| **K** | Boundary Violations | 3: Unknown Unsafe | FI: ODD edge detection | SOTIF standard |
| **L** | Integrated Complex | 3: Unknown Unsafe | Multi-FI x Multi-TC | Combined |
| **M** | SOTIF Validation | 1: Known Safe (target) | All FI/TC verified | ISO 21448 |

### 6.3 Phase H: Curvature Nodes (Detail Design)

**Problem Statement**: 직선->곡선 전환점에서 속도-조향 부정합이 주요 FI.

**Functional Insufficiencies**:
- FI_H1: Curvature Prediction Error (threshold: 0.05 /m)
- FI_H2: Speed Reduction Latency (threshold: 0.5s)
- FI_H3: Steering Rate Limitation (threshold: 0.3 rad/s = 17 deg/s)

**Triggering Conditions**:
- TC_H1: dk/ds > 0.15 /m^2 (rapid curvature change)
- TC_H2: v > v_safe for curvature (speed exceeds optimal)
- TC_H3: Steering saturation (delta > 0.4 rad)

**Curriculum Stages**:

| Stage | Scenario | Speed | Curvature | NPCs | Target |
|-------|----------|-------|-----------|------|--------|
| 1 | Straight -> Gentle Curve | 5 m/s | k=0.05 (R=200m) | 0 | +300 |
| 2 | Straight -> Sharp Curve | 10 m/s | k=0.15 (R=67m) | 1 | +600 |
| 3 | S-Curve (L+R) | 12 m/s | k=0.10 | 2 | +800 |
| 4 | Multi-node Complex | 15 m/s | k=0.05-0.20 | 3 | +1000 |

**Reward Design (SOTIF-aware)**:
```
R_total = w_speed * R_speed_compliance
        + w_curve * R_curvature_response
        + w_lane * R_lateral_stability
        + w_jerk * R_jerk_penalty

where:
  R_speed_compliance:  v within [0.8*v_opt, v_opt] -> +0.3
                       v > v_opt -> -0.5*(v - v_opt)

  R_curvature_response: steering_lag < 0.3s AND |delta - delta_opt| < 0.1 -> +0.4
                        else -> -0.2 * steering_lag

  R_lateral_stability:  a_lat < 7.0 AND lane_dev < 0.3 -> +0.2
                        else -> -0.3*(a_lat - 7.0) - 0.2*lane_dev

  R_jerk_penalty:       |jerk| > 2.0 -> -0.1 * |jerk|
```

**Validation Criteria**:
- Lane deviation < 0.3 m (UN R157 compliance)
- Steering rate < 15 deg/s (actuator limit)
- Lateral accel < 2.94 m/s^2 (0.3g, UN R157)
- Jerk < 2.0 m/s^3 (comfort)

### 6.4 Phase I: Cut-in/Cut-out (Detail Design)

**Problem Statement**: UN R171 규제의 Cut-in/Cut-out 시나리오 대응.

**Functional Insufficiencies**:
- FI_I1: Cut-in Detection Latency (threshold: 500ms)
- FI_I2: Deceleration Command Delay (threshold: 300ms)
- FI_I3: Steering-Braking Interference (threshold: 200ms)

**Triggering Conditions**:
- TC_I1: Lateral intrusion (0.5-1.5m from lane center)
- TC_I2: Sudden deceleration (-1 ~ -3 m/s^2, duration 0.5-2s)
- TC_I3: Lateral accel limit (max 8 m/s^2 combined)

**Curriculum Stages**:

| Stage | Scenario | TTC | dv | Decel | NPCs |
|-------|----------|-----|-----|-------|------|
| 1 | Gentle Cut-in | 4s | same | - | 1 |
| 2 | Rapid Cut-in | 2-3s | -10 km/h | - | 2 |
| 3 | Cut-in + Deceleration | 1.5-2s | -20 km/h | -2 m/s^2 | 3 |
| 4 | Multiple Simultaneous | 1.5s | mixed | -3 m/s^2 | 3+ |

**UN R171 Compliance Check**:
- Max deceleration: -7.0 m/s^2
- Jerk: <= 3.0 m/s^3
- TTC maintenance: > 1.5s after response
- Collision avoidance: 100%

### 6.5 Phase J-M: Overview

**Phase J: Sensor Degradation**
- Weather noise injection: visibility -30%~-70%
- Lane confidence degradation
- Graceful degradation + fallback control

**Phase K: Boundary Violations**
- Extreme curvature (k > 0.015)
- High speed (v > 25 m/s)
- Extreme traffic density
- Safe degradation learning

**Phase L: Integrated Complex**
- Multi-factor: Curves + Cut-in + Weather
- Combined FI x TC matrix coverage
- End-to-end validation

**Phase M: SOTIF Validation**
- Systematic FI x TC test matrix
- Quantitative residual risk assessment
- Safety case documentation

---

## 7. Observation Space Extensions

### 7.1 Current: 254D (Phase D)

```
ego_state:           8D  (position, velocity, heading, acceleration)
route_info:         30D  (waypoints, distances)
surrounding:        40D  (8 vehicles x 5 features)
NPC_observations:  152D  (detailed NPC state)
lane_observations:  12D  (left/right lane markings)
goal_info:          12D  (goal distance, direction)
```

### 7.2 Proposed: +5D for Curvature (Phase H)

```
curvature_current:    1D  (current road curvature k)
curvature_rate:       1D  (dk/ds, curvature change rate)
curvature_lookahead:  1D  (k at 20m ahead)
optimal_speed:        1D  (v_safe = sqrt(mu*g*R))
curvature_type:       1D  (0=straight, 1=entry, 2=curve, 3=exit)
```

Total: 259D

### 7.3 Proposed: +3D for Cut-in Detection (Phase I)

```
nearest_ttc:          1D  (minimum TTC to any vehicle)
lateral_intrusion:    1D  (nearest lateral approach distance)
cut_in_probability:   1D  (estimated cut-in likelihood)
```

Total: 262D (Phase I)

---

## 8. Key Metrics and Monitoring

### 8.1 SOTIF Metrics (per Phase)

| Metric | Formula | Target |
|--------|---------|--------|
| FI Coverage | tested_FIs / total_FIs | > 95% |
| TC Coverage | tested_TCs / total_TCs | > 90% |
| Residual Risk | P(hazard) x Severity | < threshold |
| ODD Coverage | tested_scenarios / ODD_scenarios | > 90% |

### 8.2 Training Metrics (TensorBoard)

| Metric | Phase H Target | Phase I Target |
|--------|---------------|---------------|
| Final Reward | +1000 | +900 |
| Collision Rate | 0% | 0% |
| Lane Deviation (CTE) | < 0.3m | < 0.3m |
| Jerk | < 2.0 m/s^3 | < 3.0 m/s^3 |
| Speed Compliance | 95% | 90% |
| Steering Smoothness | < 15 deg/s | < 15 deg/s |

### 8.3 Regulatory Compliance

| Regulation | Parameter | Limit | Phase |
|------------|-----------|-------|-------|
| UN R157 | Lateral Accel | <= 0.3g | H |
| UN R157 | CTE (curve) | <= 0.3m | H |
| UN R157 | dk/ds | <= 0.1 /m^2 | H |
| UN R171 | Max Decel | -7.0 m/s^2 | I |
| UN R171 | Jerk | <= 3.0 m/s^3 | I |
| UN R171 | Reaction Time | < 500ms | I |
| UN R171 | TTC (post-response) | > 1.5s | I |

---

## 9. Implementation Roadmap

### 9.1 Priority Order

```
Current:  Phase D v2 training (10M steps, staggered curriculum)
          -> Lane Observation (254D)

Next:     Phase E: Curved Roads
          Phase F: Multi-lane
          Phase G: Intersection

Proposed: Phase H: Curvature Nodes       (Feedback #1)
          Phase I: Cut-in/Cut-out        (Feedback #2, UN R171)
          Phase J: Sensor Degradation    (SOTIF)
          Phase K: Boundary Violations   (SOTIF)
          Phase L: Integrated Complex    (Combined)
          Phase M: SOTIF Validation      (ISO 21448)
```

### 9.2 Per-Phase FI/TC Documentation

Each experiment folder will include:

```
experiments/phase-{X}/
  +-- SOTIF-ANALYSIS.md    # FI/TC identification and results
```

Template:
```markdown
# Phase {X}: SOTIF Analysis

## Functional Insufficiencies
| ID | Name | Threshold | Consequence | Mitigation |
|----|------|-----------|-------------|------------|

## Triggering Conditions
| ID | Description | Detection | Criticality |
|----|-------------|-----------|-------------|

## FI x TC Matrix
| | TC_1 | TC_2 | TC_3 |
|---|---|---|---|
| FI_1 | result | result | result |

## Residual Risk
- Risk Level: {HIGH/MEDIUM/LOW}
- Acceptance: {ACCEPTED/REQUIRES_ACTION}
```

---

## 10. References

### Standards
- [ISO 21448:2022 - SOTIF](https://www.iso.org/standard/77490.html)
- [ISO/PAS 8800:2024 - Safety and AI](https://www.iso.org/standard/83303.html)
- [UN Regulation No. 171 - DCAS](https://unece.org/sites/default/files/2025-03/R171e.pdf)
- [UN Regulation No. 157 - ALKS](https://unece.org/transport/documents/2021/03/standards/un-regulation-no-157-automated-lane-keeping-systems-alks)

### Technical Analysis
- [Navigating SOTIF (ISO 21448) | Automotive IQ](https://www.automotive-iq.com/functional-safety/articles/navigating-sotif-iso-21448-and-ensuring-safety-in-autonomous-driving)
- [Understanding DCAS and UN R171 | Applied Intuition](https://www.appliedintuition.com/blog/navigating-dcas-regulations)
- [SOTIF Acceptance Criteria | SRES](https://sres.ai/autonomous-systems/demystifying-sotif-acceptance-criteria-and-validation-targets-part-2/)
- [ATIC DCAS Analysis](https://www.atic-ts.com/un-r171-dcas-01-series-and-00-series-comparison-and-analysis/)

### Academic Papers
- [Analysis of FI & TC for MPC Trajectory Planner (arXiv 2407.21569)](https://arxiv.org/html/2407.21569v1)
- [Systematization of Triggering Conditions (ResearchGate)](https://www.researchgate.net/publication/362121834)
- [AV vs Human Accident Analysis (Nature Communications)](https://www.nature.com/articles/s41467-024-48526-4)
- [LKA Empirical Performance Evaluation (arXiv 2505.11534)](https://arxiv.org/html/2505.11534v1)

### Control Theory
- [Path Planning with Clothoid Curves (DIVA Portal)](https://www.diva-portal.org/smash/get/diva2:1150741/FULLTEXT01.pdf)
- [Clothoid-Based Lateral Controller (MDPI)](https://www.mdpi.com/2076-3417/14/5/1817)
- [Adaptive MPC for Lane Keeping (arXiv 1806.04335)](https://arxiv.org/pdf/1806.04335)
- [MPC-PID Demonstration DRL (arXiv 2506.04040)](https://arxiv.org/html/2506.04040v1)

### Curriculum Learning
- [CuRLA: Curriculum DRL for Autonomous Driving (arXiv 2501.04982)](https://arxiv.org/html/2501.04982v1)
- [Value of Curriculum Learning for Self-Driving (Frontiers)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9905678/)
- [Automatic Curriculum Learning for Driving (arXiv 2505.08264)](https://arxiv.org/html/2505.08264v1)
