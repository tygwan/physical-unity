// ---------- Phase data ----------
export interface Phase {
  id: string;
  name: string;
  subtitle: string;
  reward: number | null;
  status: 'success' | 'failed' | 'in_progress' | 'planned';
  tags: string[];
  description: string;
  observations: string;
  steps: string;
  keyInsight: string;
  subtitleKo?: string;
  descriptionKo?: string;
  keyInsightKo?: string;
  version?: number;
  parentId?: string; // links a retry to its predecessor
}

export const phases: Phase[] = [
  {
    id: 'phase-0',
    name: 'Phase 0',
    subtitle: 'Foundation: Lane Keeping',
    reward: 1018,
    status: 'success',
    tags: ['PPO', 'Lane Keeping', 'Speed Control'],
    description:
      'Foundation phase establishing basic lane keeping and speed control on straight road. Zero collision rate achieved with perfect safety record.',
    observations: '242D',
    steps: '8.0M',
    keyInsight:
      'Basic lane keeping + speed control established. Perfect safety (0% collision). Foundation for all subsequent phases.',
    subtitleKo: '기초: 차선 유지',
    descriptionKo:
      '직선 도로에서 기본 차선 유지와 속도 제어를 확립하는 기초 단계. 충돌률 0%로 완벽한 안전 기록 달성.',
    keyInsightKo:
      '기본 차선 유지 + 속도 제어 확립. 완벽한 안전성(충돌 0%). 이후 모든 단계의 기반.',
  },
  {
    id: 'phase-a',
    name: 'Phase A',
    subtitle: 'Dense Overtaking',
    reward: 2113,
    status: 'success',
    tags: ['PPO', 'Dense Reward', '1 NPC'],
    description:
      'Agent learns to overtake a single slow NPC on a straight road. Dense reward shaping with 7 reward components. Peak reward +3161 at 2.0M steps.',
    observations: '242D',
    steps: '2.5M',
    keyInsight:
      'Agent achieved +2113 but overtaking bonus was never triggered (0 events). Agent optimized speed maintenance instead - a mild form of reward hacking (Goodhart\'s Law).',
    subtitleKo: '밀집 추월',
    descriptionKo:
      '직선 도로에서 단일 저속 NPC 추월 학습. 7개 보상 구성요소를 활용한 Dense Reward Shaping. 2.0M 스텝에서 최고 보상 +3161 달성.',
    keyInsightKo:
      '에이전트가 +2113 달성했으나 추월 보너스는 한 번도 발동되지 않음(0회). 에이전트는 대신 속도 유지를 최적화 -- 경미한 보상 해킹(Goodhart의 법칙).',
  },
  {
    id: 'phase-b-v1',
    name: 'Phase B v1',
    subtitle: 'Decision Learning',
    reward: -108,
    status: 'failed',
    tags: ['PPO', 'NPC Interaction', '7 Vars Changed'],
    description:
      'First attempt at multi-NPC decision learning. 7 hyper-parameters changed simultaneously + Phase 0 checkpoint (no overtaking ability) + immediate 2 NPC exposure.',
    observations: '242D',
    steps: '1.5M',
    keyInsight:
      'speedUnderPenalty=-0.1/step taught agent to STOP as optimal policy. Multiple variables changed simultaneously made root-cause identification impossible.',
    subtitleKo: '의사결정 학습',
    descriptionKo:
      '다중 NPC 의사결정 학습 첫 시도. 7개 하이퍼파라미터 동시 변경 + Phase 0 체크포인트(추월 능력 없음) + 즉시 2 NPC 노출.',
    keyInsightKo:
      'speedUnderPenalty=-0.1/step이 에이전트에게 정지를 최적 정책으로 학습시킴. 다수 변수 동시 변경으로 근본 원인 식별 불가.',
    version: 1,
  },
  {
    id: 'phase-b-v2',
    name: 'Phase B v2',
    subtitle: 'Decision Learning',
    reward: 877,
    status: 'success',
    tags: ['PPO', 'Variable Isolation', 'Checkpoint Transfer'],
    description:
      'Recovery from B v1: restored Phase A hyper-parameters (P-001), used Phase A checkpoint with overtaking ability (P-003), reduced speed penalty 80% (P-004), added 0→1→2→3 NPC curriculum.',
    observations: '242D',
    steps: '3.5M',
    keyInsight:
      'Three policies discovered simultaneously: P-001 (variable isolation), P-003 (capability-based checkpoint), P-004 (conservative penalty). All three were necessary for recovery.',
    subtitleKo: '의사결정 학습',
    descriptionKo:
      'B v1 복구: Phase A 하이퍼파라미터 복원(P-001), 추월 능력 보유한 Phase A checkpoint 사용(P-003), 속도 페널티 80% 감소(P-004), 0->1->2->3 NPC curriculum 추가.',
    keyInsightKo:
      '세 가지 정책 동시 발견: P-001(변수 격리), P-003(능력 기반 checkpoint), P-004(보수적 페널티). 세 가지 모두 복구에 필수적이었음.',
    version: 2,
    parentId: 'phase-b-v1',
  },
  {
    id: 'phase-d-v1',
    name: 'Phase D v1',
    subtitle: 'Lane Observation 254D',
    reward: -2156,
    status: 'failed',
    tags: ['PPO', '254D', 'Simultaneous Transition'],
    description:
      'First attempt at lane observation (242D→254D). Three curriculum parameters advanced at the same threshold (~400K steps), causing reward crash from +406 to -4,825 within 20K steps.',
    observations: '254D',
    steps: '6M',
    keyInsight:
      'Curriculum shock: 3 parameters (NPCs, speed zones, speed variation) transitioned simultaneously, invalidating the learned policy. Recovery was impossible within budget.',
    subtitleKo: '차선 관측 254D',
    descriptionKo:
      '차선 관측 첫 시도(242D->254D). 세 개의 curriculum 파라미터가 동일 임계값(~400K 스텝)에서 전진하여 20K 스텝 내에 보상이 +406에서 -4,825로 폭락.',
    keyInsightKo:
      'Curriculum 충격: 3개 파라미터(NPC, 속도 구간, 속도 변동)가 동시 전환되어 학습된 정책 무효화. 예산 내 복구 불가.',
    version: 1,
  },
  {
    id: 'phase-d-v2',
    name: 'Phase D v2',
    subtitle: 'Lane Obs (Staggered)',
    reward: -756,
    status: 'failed',
    tags: ['PPO', '254D', 'Staggered Thresholds'],
    description:
      'Applied P-002 (staggered curriculum): thresholds at 200K/300K/350K. Survived longer (7.87M steps) but collapsed at 4 NPC transition. Observation change + environment change together was the root cause.',
    observations: '254D',
    steps: '10M',
    keyInsight:
      'P-002 alone was insufficient. Led to discovery of P-009: never change observation space AND environment simultaneously.',
    subtitleKo: '차선 관측 (단계적)',
    descriptionKo:
      'P-002(단계적 curriculum) 적용: 임계값 200K/300K/350K. 더 오래 생존(7.87M 스텝)했으나 4 NPC 전환 시 붕괴. 관측 변경 + 환경 변경 동시 진행이 근본 원인.',
    keyInsightKo:
      'P-002만으로는 불충분. P-009 발견으로 이어짐: 관측 공간과 환경을 절대 동시에 변경하지 말 것.',
    version: 2,
    parentId: 'phase-d-v1',
  },
  {
    id: 'phase-d-v3',
    name: 'Phase D v3',
    subtitle: 'Lane Obs (Fixed Env)',
    reward: 895,
    status: 'success',
    tags: ['PPO', '254D', 'P-009', 'P-010'],
    description:
      'Applied P-009 (observation-environment coupling ban): fixed environment (3 NPC, 0.6 speed ratio), only observation change as variable. Also discovered P-010 after VectorObservationSize mismatch (242→254). Lane observation contributed +7.2% improvement over 242D baseline.',
    observations: '254D',
    steps: '5M',
    keyInsight:
      'Lane observation added +60 reward (+7.2%) over 242D baseline (+835). P-010 (Scene-Config-Code consistency) discovered when VectorObservationSize was 242 instead of 254 in Unity scene.',
    subtitleKo: '차선 관측 (고정 환경)',
    descriptionKo:
      'P-009(관측-환경 결합 금지) 적용: 고정 환경(3 NPC, 0.6 속도 비율), 관측 변경만 변수로 설정. VectorObservationSize 불일치(242->254) 후 P-010도 발견. 차선 관측이 242D 기준선 대비 +7.2% 개선 기여.',
    keyInsightKo:
      '차선 관측이 242D 기준선(+835) 대비 +60 보상(+7.2%) 추가. Unity 씬에서 VectorObservationSize가 254가 아닌 242였을 때 P-010(씬-설정-코드 일관성) 발견.',
    version: 3,
    parentId: 'phase-d-v2',
  },
  {
    id: 'phase-e',
    name: 'Phase E',
    subtitle: 'Curved Roads',
    reward: 938,
    status: 'success',
    tags: ['PPO', '254D', 'Curvature Curriculum', 'P-002 Recovery'],
    description:
      'Curved road training with 7 curriculum parameters. All completed: sharp curves (1.0) + mixed directions + 2 NPCs + 200m goal. Curriculum collapse at 1.68M (4 params simultaneous, P-002 violation) but recovered by 2.44M. Initialized from Phase D v3.',
    observations: '254D',
    steps: '6M',
    keyInsight:
      'First P-002 violation that RECOVERED (unlike Phase D v1/v2). Strong foundation from D v3 enabled 800K-step recovery from -3,863 crash. Curvature curriculum progressed smoothly after recovery: gentle(3.47M) → moderate(3.81M) → sharp(4.15M). Peak +956 at 3.58M.',
    subtitleKo: '곡선 도로',
    descriptionKo:
      '7개 curriculum 파라미터를 활용한 곡선 도로 학습. 급커브(1.0) + 혼합 방향 + 2 NPC + 200m 목표 모두 완료. 1.68M에서 curriculum 붕괴(4개 파라미터 동시, P-002 위반)했으나 2.44M까지 복구. Phase D v3에서 초기화.',
    keyInsightKo:
      '최초의 P-002 위반 후 복구 성공(Phase D v1/v2와 달리). D v3의 강력한 기반이 -3,863 폭락에서 800K 스텝 복구를 가능케 함. 커브 curriculum 순조롭게 진행: 완만(3.47M) -> 보통(3.81M) -> 급(4.15M). 3.58M에서 최고 +956.',
  },
  {
    id: 'phase-f-v1',
    name: 'Phase F v1',
    subtitle: 'Multi-Lane (Wrong Scene)',
    reward: -8,
    status: 'failed',
    tags: ['PPO', '254D', 'Scene Mismatch', 'P-011'],
    description:
      'First multi-lane attempt failed due to wrong Unity scene. PhaseE_CurvedRoads (4.5m road) was loaded instead of PhaseF_MultiLane (11.5m road). num_lanes 1→2 transition caused instant off-road death.',
    observations: '254D',
    steps: '5.82M',
    keyInsight:
      'Scene-Phase mismatch: road was 4.5m (1 lane) but curriculum demanded 2 lanes (8.0m). Agent stuck at -8.15 for 4.27M steps. Led to P-011 (Scene-Phase Matching) policy.',
    subtitleKo: '다차선 (잘못된 씬)',
    descriptionKo:
      '첫 다차선 시도, 잘못된 Unity 씬으로 실패. PhaseF_MultiLane(11.5m 도로) 대신 PhaseE_CurvedRoads(4.5m 도로)가 로드됨. num_lanes 1->2 전환 시 즉시 도로 이탈.',
    keyInsightKo:
      '씬-Phase 불일치: 도로 폭 4.5m(1차선)인데 curriculum이 2차선(8.0m) 요구. 에이전트가 4.27M 스텝 동안 -8.15에 갇힘. P-011(씬-Phase 매칭) 정책으로 이어짐.',
    version: 1,
  },
  {
    id: 'phase-f-v2',
    name: 'Phase F v2',
    subtitle: 'Multi-Lane (Waypoint Bug)',
    reward: -14,
    status: 'failed',
    tags: ['PPO', '254D', 'Waypoint Destruction', 'Entropy Collapse'],
    description:
      'Correct scene (PhaseF_MultiLane, 11.5m road) but WaypointManager.SetLaneCount() destroyed all waypoint GameObjects at num_lanes 1→2 transition. Agent observation references invalidated, entropy collapsed to Std=0.08, locked at -14.2 reward.',
    observations: '254D',
    steps: '4.1M',
    keyInsight:
      'Runtime object destruction breaks observation continuity. WaypointManager.GenerateWaypoints() must reuse existing GameObjects, not Destroy+Recreate. Mathematically verified: -0.2/step × 90 steps ≈ -18.0, observed -14.2.',
    subtitleKo: '다차선 (웨이포인트 버그)',
    descriptionKo:
      '올바른 씬(PhaseF_MultiLane, 11.5m 도로) 사용했으나 WaypointManager.SetLaneCount()가 num_lanes 1->2 전환 시 모든 웨이포인트 GameObject를 파괴. 에이전트 관측 참조 무효화, 엔트로피 Std=0.08로 붕괴, -14.2 보상에 고착.',
    keyInsightKo:
      '런타임 객체 파괴가 관측 연속성을 깨뜨림. WaypointManager.GenerateWaypoints()는 Destroy+Recreate가 아닌 기존 GameObject를 재사용해야 함. 수학적 검증: -0.2/step x 90 step = -18.0, 관측값 -14.2.',
    version: 2,
    parentId: 'phase-f-v1',
  },
  {
    id: 'phase-f-v3',
    name: 'Phase F v3',
    subtitle: 'Multi-Lane (Shared Thresholds)',
    reward: 5,
    status: 'failed',
    tags: ['PPO', '254D', 'Shared Thresholds', 'Linear LR', 'P-012'],
    description:
      'Waypoint fix applied but threshold 350 shared by 4 curriculum parameters. Triple simultaneous transition at 4.38M caused -2,480 crash. Linear LR schedule decayed to near-zero, preventing recovery.',
    observations: '254D',
    steps: '6M',
    keyInsight:
      'ML-Agents uses GLOBAL smoothed reward for ALL curriculum parameters. Same threshold = simultaneous transition. Led to P-012 (No Shared Thresholds). Linear LR decay compounds the problem by removing learning capacity when most needed.',
    subtitleKo: '다차선 (공유 임계값)',
    descriptionKo:
      '웨이포인트 수정 적용했으나 임계값 350을 4개 curriculum 파라미터가 공유. 4.38M에서 3중 동시 전환으로 -2,480 폭락. Linear LR 스케줄이 거의 0으로 감쇠하여 복구 불가.',
    keyInsightKo:
      'ML-Agents는 모든 curriculum 파라미터에 단일 글로벌 평활 보상을 사용. 동일 임계값 = 동시 전환. P-012(공유 임계값 금지)로 이어짐. Linear LR 감쇠가 학습 능력을 가장 필요한 시점에 제거하여 문제 악화.',
    version: 3,
    parentId: 'phase-f-v2',
  },
  {
    id: 'phase-f-v4',
    name: 'Phase F v4',
    subtitle: 'Multi-Lane (Speed Zone Bug)',
    reward: 106,
    status: 'failed',
    tags: ['PPO', '254D', 'Strict P-002', 'Constant LR', 'Speed Zone Bug', 'P-013'],
    description:
      'Strict P-002: all 15 thresholds unique (150-900, 50-point gaps). Constant LR schedule. Staggered curriculum WORKED (6 individual transitions). But GenerateSpeedZones() placed Residential(30 km/h) first, causing -2,790 crash when agent at 60 km/h entered 30 km/h zone.',
    observations: '254D',
    steps: '10M',
    keyInsight:
      'P-002 strict compliance validated (all transitions individual). But speed_zone implementation bug: first zone must match single-zone default speed. Led to P-013 (Speed Zone Ordering). Recovery: 6.22M steps to reach +106 (22% of pre-crash peak +483).',
    subtitleKo: '다차선 (속도 구간 버그)',
    descriptionKo:
      '엄격한 P-002: 15개 임계값 모두 고유(150-900, 50포인트 간격). Constant LR 스케줄. 단계적 curriculum 작동(6회 개별 전환). 그러나 GenerateSpeedZones()가 Residential(30 km/h)을 먼저 배치하여 60 km/h 에이전트가 30 km/h 구간 진입 시 -2,790 폭락.',
    keyInsightKo:
      'P-002 엄격 준수 검증(모든 전환 개별 발생). 하지만 속도 구간 구현 버그: 첫 번째 구간이 단일 구간 기본 속도와 일치해야 함. P-013(속도 구간 순서)으로 이어짐. 복구: 6.22M 스텝에서 +106 달성(폭락 전 최고 +483의 22%).',
    version: 4,
    parentId: 'phase-f-v3',
  },
  {
    id: 'phase-f-v5',
    name: 'Phase F v5',
    subtitle: 'Multi-Lane (Success)',
    reward: 643,
    status: 'success',
    tags: ['PPO', '254D', 'P-013 Validated', '4 Lanes', 'Speed Zones', 'Curves'],
    description:
      'Speed zone fix: reordered GenerateSpeedZones() so first zone matches 60 km/h default. P-013 validated: speed_zone drop -262 (v4: -2,790, 10.7x improvement). 10/15 curriculum transitions completed. Agent masters 4-lane curved roads with speed zones.',
    observations: '254D',
    steps: '10M',
    keyInsight:
      'P-013 confirmed: first speed zone must match default. Reward plateau at ~640 prevented curve_direction (650) and NPC (700+) transitions. 4-lane + curves(0.6) + speed zones + 250m goal achieved. Best model: results/phase-F-v5/E2EDrivingAgent.onnx.',
    subtitleKo: '다차선 (성공)',
    descriptionKo:
      '속도 구간 수정: GenerateSpeedZones()를 재정렬하여 첫 구간이 60 km/h 기본값과 일치. P-013 검증: 속도 구간 하락 -262(v4: -2,790, 10.7배 개선). 15개 중 10개 curriculum 전환 완료. 에이전트가 4차선 곡선 도로 + 속도 구간 마스터.',
    keyInsightKo:
      'P-013 확인: 첫 속도 구간은 기본값과 일치해야 함. 보상 정체 ~640이 curve_direction(650)과 NPC(700+) 전환을 막음. 4차선 + 커브(0.6) + 속도 구간 + 250m 목표 달성.',
    version: 5,
    parentId: 'phase-f-v4',
  },
  {
    id: 'phase-g-v1',
    name: 'Phase G v1',
    subtitle: 'Intersections (WrongWay Bug)',
    reward: 494,
    status: 'failed',
    tags: ['PPO', '260D', 'T/Cross/Y Junction', 'WrongWay Bug', 'P-014'],
    description:
      'First intersection training: T-junction, cross-road, Y-junction with 260D observation space (+6D intersection info). Plateaued at reward ~494 (target: 550 for Y-junction). Root cause: IsWrongWayDriving() triggered during left turns (xPos < -0.5 but left turn exits at X=-8.25), causing 32% WrongWay termination rate.',
    observations: '260D',
    steps: '10M',
    keyInsight:
      'WrongWay detection designed for straight roads (Phase F) was incompatible with intersection turns. 32% of episodes terminated by WrongWay, making Y-junction curriculum unreachable. Led to P-014 (Intersection Zone WrongWay Exemption).',
    subtitleKo: '교차로 (WrongWay 버그)',
    descriptionKo:
      '첫 교차로 학습: T자, 십자, Y자 교차로. 260D 관측 공간(+6D 교차로 정보). 보상 ~494에서 정체(목표: Y자 교차로 550). 근본 원인: IsWrongWayDriving()이 좌회전 시 오발동(xPos < -0.5이나 좌회전 출구는 X=-8.25), 32% WrongWay 종료율.',
    keyInsightKo:
      '직선 도로용 WrongWay 감지(Phase F)가 교차로 회전과 호환 불가. 32% 에피소드가 WrongWay로 종료되어 Y자 교차로 curriculum 도달 불가. P-014(교차로 구간 WrongWay 면제)로 이어짐.',
    version: 1,
  },
  {
    id: 'phase-g-v2',
    name: 'Phase G v2',
    subtitle: 'Intersections (WrongWay Fix)',
    reward: 633,
    status: 'success',
    tags: ['PPO', '260D', 'Warm Start', 'P-014 Applied', 'P-015 DecisionRequester', '7/7 Curriculum'],
    description:
      'Phase G retry with WrongWay intersection zone fix (P-014), warm start from v1 checkpoint, simplified curriculum (NPCs deferred to Phase H). All 7/7 curriculum lessons completed in 5M steps (v1: 4/7 in 10M). T-junction, Cross, Y-junction all mastered with 0% collision. DecisionRequester bug (P-015) discovered and fixed during setup.',
    observations: '260D',
    steps: '5M',
    keyInsight:
      'P-014 WrongWay fix eliminated 32% termination rate. Warm start halved training budget (5M vs 10M) while improving reward 28% (633 vs 494). P-015 discovered: scene regeneration removes DecisionRequester, causing silent training hang.',
    subtitleKo: '교차로 (WrongWay 수정)',
    descriptionKo:
      'Phase G 재시도: WrongWay 교차로 구간 수정(P-014), v1 checkpoint에서 Warm Start, 간소화된 curriculum(NPC는 Phase H로 연기). 5M 스텝에서 7/7 curriculum 완료(v1: 10M에서 4/7). T자, 십자, Y자 교차로 모두 마스터, 충돌 0%.',
    keyInsightKo:
      'P-014 WrongWay 수정으로 32% 종료율 제거. Warm Start가 학습 예산 절반(5M vs 10M)으로 보상 28% 향상(633 vs 494). P-015 발견: 씬 재생성 시 DecisionRequester가 제거되어 학습이 무음으로 멈춤.',
    version: 2,
    parentId: 'phase-g-v1',
  },
  {
    id: 'phase-h-v1',
    name: 'Phase H v1',
    subtitle: 'NPC Intersection (Abrupt)',
    reward: 550,
    status: 'failed',
    tags: ['PPO', '260D', 'NPC Waypoints', 'Abrupt Variation', 'P-016'],
    description:
      'First NPC intersection training. Agent mastered 3 NPCs + speed_ratio 0.85 but crashed at speed_variation=0.15 (single-step jump from 0). Reward dropped from 700 to 550 with Std 300+.',
    observations: '260D',
    steps: '~5M',
    keyInsight:
      'P-016 discovered: Never introduce high variation in a single step. npc_speed_variation must be graduated (0->0.05->0.10->0.15), not jumped (0->0.15).',
    subtitleKo: 'NPC 교차로 (급격)',
    descriptionKo:
      '첫 NPC 교차로 학습. 에이전트가 3 NPC + speed_ratio 0.85를 마스터했으나 speed_variation=0.15에서 폭락(0에서 단일 단계 점프). 보상이 700에서 550으로 하락, Std 300+.',
    keyInsightKo:
      'P-016 발견: 높은 변동성을 단일 단계로 도입하지 말 것. npc_speed_variation은 단계적(0->0.05->0.10->0.15)이어야 하며, 점프(0->0.15)는 금지.',
    version: 1,
  },
  {
    id: 'phase-h-v2',
    name: 'Phase H v2',
    subtitle: 'NPC Intersection (High Thresholds)',
    reward: 681,
    status: 'failed',
    tags: ['PPO', '260D', 'Build Training', 'Gradual Variation', 'Unreachable Thresholds'],
    description:
      'Gradual speed_variation (0->0.05->0.10->0.15) fixed the crash, but thresholds 710/720 were unreachable: agent averages ~690-700 with variation active. Stuck at variation=0.05 (9/11 curriculum).',
    observations: '260D',
    steps: '5M',
    keyInsight:
      'Thresholds must be achievable under active conditions. Agent reward with variation=0.05 averages ~690-700, so threshold 710 is unreachable. First phase using build-based multi-env training (3x speedup).',
    subtitleKo: 'NPC 교차로 (높은 임계값)',
    descriptionKo:
      '단계적 speed_variation(0->0.05->0.10->0.15)으로 폭락 해결했으나 임계값 710/720이 도달 불가: variation 활성 시 에이전트 평균 ~690-700. variation=0.05에서 멈춤(9/11 curriculum).',
    keyInsightKo:
      '임계값은 현재 레슨 활성 상태에서의 기대 평균 보상 이하로 설정해야 함. variation=0.05 활성 시 에이전트 평균 ~690-700이므로 임계값 710은 도달 불가. 빌드 기반 다중 환경 학습 최초 사용(3배 속도 향상).',
    version: 2,
    parentId: 'phase-h-v1',
  },
  {
    id: 'phase-h-v3',
    name: 'Phase H v3',
    subtitle: 'NPC Intersection (Success)',
    reward: 701,
    status: 'success',
    tags: ['PPO', '260D', 'Build Training', 'Lowered Thresholds', 'P-016', 'P-017', '11/11 Curriculum'],
    description:
      'Lowered speed_variation thresholds to 685/690/693 (achievable under variation). All 11/11 curriculum completed. 3 NPCs with speed_ratio 0.85 and speed_variation 0.15 through T/Cross/Y-junction intersections.',
    observations: '260D',
    steps: '5M',
    keyInsight:
      'P-016 (gradual variation) + P-017 (achievable thresholds under active conditions) together solved the NPC speed variation problem. Build training enabled rapid v2->v3 iteration (~26 min per 5M run).',
    subtitleKo: 'NPC 교차로 (성공)',
    descriptionKo:
      'speed_variation 임계값을 685/690/693으로 하향(variation 하에서 달성 가능). 11/11 curriculum 모두 완료. T/십자/Y자 교차로에서 3 NPC + speed_ratio 0.85 + speed_variation 0.15.',
    keyInsightKo:
      'P-016(단계적 변동) + P-017(활성 조건 하 달성 가능 임계값)이 함께 NPC 속도 변동 문제를 해결. 빌드 학습으로 빠른 v2->v3 반복(5M 런당 ~26분).',
    version: 3,
    parentId: 'phase-h-v2',
  },
  {
    id: 'phase-i-v1',
    name: 'Phase I v1',
    subtitle: 'Curved Roads + NPC (Crash)',
    reward: 623,
    status: 'failed',
    tags: ['PPO', '260D', 'Curves+NPC', 'Triple-Param Crash', 'P-018'],
    description:
      'Combined Phase E curves with Phase H NPCs. All 17/17 curriculum transitions completed, but thresholds 700/702/705 too tight: road_curvature, curve_direction_variation, speed_zone_count unlocked simultaneously at 3.76M. Reward crashed 724 -> -40, recovered to 623 by 5M.',
    observations: '260D',
    steps: '5M',
    keyInsight:
      'P-018 discovered: Threshold spacing must be >= 15 points. Spacing of 2-5 points (700/702/705) caused 3-parameter simultaneous transition and 760-point reward crash. Despite crash, all curriculum completed and recovery was underway.',
    subtitleKo: '곡선 도로 + NPC (폭락)',
    descriptionKo:
      'Phase E 커브와 Phase H NPC 결합. 17/17 curriculum 전환 모두 완료했으나 임계값 700/702/705가 너무 촘촘: road_curvature, curve_direction_variation, speed_zone_count가 3.76M에서 동시 해제. 보상 724 -> -40 폭락, 5M까지 623으로 복구.',
    keyInsightKo:
      'P-018 발견: 임계값 간격은 최소 15포인트 이상이어야 함. 2-5포인트 간격(700/702/705)이 3개 파라미터 동시 전환과 760포인트 보상 폭락을 야기. 폭락에도 불구하고 모든 curriculum 완료, 복구 진행 중이었음.',
    version: 1,
  },
  {
    id: 'phase-i-v2',
    name: 'Phase I v2',
    subtitle: 'Curved Roads + NPC (Record)',
    reward: 770,
    status: 'success',
    tags: ['PPO', '260D', 'Project Record', 'Recovery Training', 'Curves+NPC+S-Curves'],
    description:
      'Pure recovery training from v1 crash. All parameters fixed at final values (no curriculum). Agent recovered from 623 to 770 (project-wide record). Full curvature 1.0 + random S-curves + 3 NPCs + speed variation + 2 speed zones + 2 lanes.',
    observations: '260D',
    steps: '5M',
    keyInsight:
      'Recovery training works: crashed policy (623) reached new project high (770) with fixed-param continuation. No curriculum transitions needed. Proves curves + NPCs are compatible when learned together.',
    subtitleKo: '곡선 도로 + NPC (기록)',
    descriptionKo:
      'v1 폭락으로부터의 순수 복구 학습. 모든 파라미터를 최종값으로 고정(curriculum 없음). 에이전트가 623에서 770으로 복구(프로젝트 전체 기록). 전체 곡률 1.0 + 랜덤 S커브 + 3 NPC + 속도 변동 + 2개 속도 구간 + 2차선.',
    keyInsightKo:
      '복구 학습 효과 입증: 폭락한 정책(623)이 파라미터 고정 연속 학습으로 프로젝트 최고(770) 달성. Curriculum 전환 불필요. 커브와 NPC가 함께 학습될 때 호환됨을 증명.',
    version: 2,
    parentId: 'phase-i-v1',
  },
  // ---------- Phase J: Traffic Signals ----------
  {
    id: 'phase-j-v1',
    name: 'Phase J v1',
    subtitle: 'Traffic Signals (Tensor Mismatch)',
    reward: null,
    status: 'failed',
    tags: ['PPO', '268D', 'Obs Dim Change', 'Crash'],
    description:
      'Attempted warm start from Phase I v2 (260D) with new traffic signal observations (268D). Adam optimizer tensor mismatch caused immediate crash at ~40K steps.',
    observations: '268D',
    steps: '~40K',
    keyInsight:
      'P-020: Observation dimension change (260D→268D) requires fresh start. Optimizer state shape mismatch cannot be recovered.',
    subtitleKo: '교통 신호 (텐서 불일치)',
    descriptionKo:
      'Phase I v2(260D)에서 새 교통 신호 관측(268D)으로의 Warm Start 시도. Adam 옵티마이저 텐서 형상 불일치로 ~40K 스텝에서 즉시 크래시.',
    keyInsightKo:
      'P-020: 관측 차원 변경(260D->268D)은 처음부터 학습이 필요. 옵티마이저 상태 형상 불일치는 복구 불가.',
    version: 1,
  },
  {
    id: 'phase-j-v2',
    name: 'Phase J v2',
    subtitle: 'Traffic Signals (From Scratch)',
    reward: 632,
    status: 'failed',
    tags: ['PPO', '268D', 'Fresh Start', '10M Steps'],
    description:
      'From-scratch training with 268D observation space including traffic signal features. Achieved 9/13 curriculum lessons in 10M steps. Peak +660.6 at 7.5M. Missed Y-Junction, traffic signals, and green ratio lessons.',
    observations: '268D',
    steps: '10M',
    keyInsight:
      'P-021: From-scratch training needs lower thresholds than warm start. The 10M budget was consumed rebuilding driving skills that warm start would have preserved.',
    subtitleKo: '교통 신호 (처음부터)',
    descriptionKo:
      '268D 관측 공간(교통 신호 특성 포함)으로 처음부터 학습. 10M 스텝에서 13개 중 9개 curriculum 레슨 달성. 7.5M에서 최고 +660.6. Y자 교차로, 교통 신호, 녹색 비율 레슨 미달성.',
    keyInsightKo:
      'P-021: 처음부터 학습은 Warm Start보다 낮은 임계값이 필요. 10M 예산이 Warm Start라면 보존되었을 주행 기술 재구축에 소모됨.',
    version: 2,
    parentId: 'phase-j-v1',
  },
  {
    id: 'phase-j-v3',
    name: 'Phase J v3',
    subtitle: 'Traffic Signals (Signal Ordering)',
    reward: 477,
    status: 'failed',
    tags: ['PPO', '268D', 'Warm Start', 'Signal Crash'],
    description:
      'Warm start from v2 9.5M checkpoint. Reached 12/13 curriculum but signal activation caused 177-point crash (647→470). green_ratio threshold was lower than signal threshold, causing ordering conflict.',
    observations: '268D',
    steps: '5M',
    keyInsight:
      'P-022: Feature activation must precede parameter tuning. Signal ON and green_ratio reduction cannot share curriculum ordering. Single-param curriculum prevents ordering conflicts.',
    subtitleKo: '교통 신호 (신호 순서)',
    descriptionKo:
      'v2 9.5M checkpoint에서 Warm Start. 13개 중 12개 curriculum에 도달했으나 신호 활성화가 177포인트 폭락(647->470)을 야기. green_ratio 임계값이 신호 임계값보다 낮아 순서 충돌 발생.',
    keyInsightKo:
      'P-022: 기능 활성화가 파라미터 조정보다 선행해야 함. 신호 ON과 green_ratio 감소가 curriculum 순서를 공유할 수 없음. 단일 파라미터 curriculum이 순서 충돌을 방지.',
    version: 3,
    parentId: 'phase-j-v2',
  },
  {
    id: 'phase-j-v4',
    name: 'Phase J v4',
    subtitle: 'Traffic Signals (Green Ratio Plateau)',
    reward: 497,
    status: 'failed',
    tags: ['PPO', '268D', 'Signal-First', 'Reward Compression'],
    description:
      'Signal-first approach: signals ON + Y-Junction locked from step 0. Reached 3/4 green_ratio lessons. Plateau at green_ratio=0.5 (~490-500) vs threshold 540.',
    observations: '268D',
    steps: '5M',
    keyInsight:
      'P-023: Reward compression under signal constraints. Lower green ratio = lower reward ceiling. Thresholds must account for reduced reward bandwidth.',
    subtitleKo: '교통 신호 (녹색 비율 정체)',
    descriptionKo:
      '신호 우선 접근: 신호 ON + Y자 교차로를 스텝 0부터 고정. 4개 중 3개 green_ratio 레슨 달성. green_ratio=0.5에서 정체(~490-500) vs 임계값 540.',
    keyInsightKo:
      'P-023: 신호 제약 하의 보상 압축. 낮은 녹색 비율 = 낮은 보상 상한. 임계값이 줄어든 보상 대역폭을 고려해야 함.',
    version: 4,
    parentId: 'phase-j-v3',
  },
  {
    id: 'phase-j-v5',
    name: 'Phase J v5',
    subtitle: 'Traffic Signals (Complete)',
    reward: 537,
    status: 'success',
    tags: ['PPO', '268D', 'Build Training', '5/5 Green Ratio', 'Signal Compliance'],
    description:
      'Full traffic signal compliance achieved. 5/5 green_ratio curriculum (0.8→0.4). Fixed false violation bug (wasPastStopLineAtRedStart), added deceleration reward. P-024: BehaviorType=InferenceOnly in build silently prevented training.',
    observations: '268D',
    steps: '5M',
    keyInsight:
      'After 5 iterations: signal-first ordering (P-022) + lowered thresholds (P-023) + build fix (P-024) = full curriculum completion. Signal compliance is achievable but requires careful reward engineering.',
    subtitleKo: '교통 신호 (완료)',
    descriptionKo:
      '완전한 교통 신호 준수 달성. 5/5 green_ratio curriculum(0.8->0.4). 거짓 위반 버그(wasPastStopLineAtRedStart) 수정, 감속 보상 추가. P-024: 빌드에서 BehaviorType=InferenceOnly가 학습을 무음으로 차단.',
    keyInsightKo:
      '5회 반복 끝에: 신호 우선 순서(P-022) + 하향 임계값(P-023) + 빌드 수정(P-024) = 전체 curriculum 완료. 신호 준수는 달성 가능하나 세심한 보상 엔지니어링이 필요.',
    version: 5,
    parentId: 'phase-j-v4',
  },
  // ---------- Phase K: Dense Urban Integration ----------
  {
    id: 'phase-k-v1',
    name: 'Phase K v1',
    subtitle: 'Dense Urban Integration',
    reward: 590,
    status: 'success',
    tags: ['PPO', '268D', 'All Skills Combined', 'Curves+Intersections+Signals'],
    description:
      'Combined ALL driving skills for the first time: curved roads + intersections + signals + NPCs. Warm start from Phase J v5. 3/3 curriculum complete (road_curvature 0→0.3→0.5). Peak +703 at 4.67M. Training time ~25 min.',
    observations: '268D',
    steps: '5M',
    keyInsight:
      'Integration phase proved all skills are compatible when combined. No new policies needed - existing curriculum design principles (P-002, P-016) scaled to the full skill set.',
    subtitleKo: '밀집 도시 통합',
    descriptionKo:
      '최초로 모든 주행 기술 통합: 곡선 도로 + 교차로 + 신호 + NPC. Phase J v5에서 Warm Start. 3/3 curriculum 완료(road_curvature 0->0.3->0.5). 4.67M에서 최고 +703. 학습 시간 ~25분.',
    keyInsightKo:
      '통합 단계에서 모든 기술이 결합 시 호환됨을 입증. 새로운 정책 불필요 -- 기존 curriculum 설계 원칙(P-002, P-016)이 전체 기술 세트로 확장.',
  },
  // ---------- Phase L: Crosswalks + Pedestrians ----------
  {
    id: 'phase-l-v1',
    name: 'Phase L v1',
    subtitle: 'Crosswalks (Reward Exploit)',
    reward: 787,
    status: 'failed',
    tags: ['PPO', '280D', 'Fresh Start', 'Reward Hacking', 'Pedestrians'],
    description:
      'First pedestrian phase. Fresh start with 280D observation (268D + 12D pedestrian/crosswalk). Driving skills rebuilt to +730 by 12.5M. At 13.3M, agent discovered yield reward exploit: stopped at crosswalk indefinitely farming +0.2/s yield reward. Reward exploded to +20,148.',
    observations: '280D',
    steps: '15M',
    keyInsight:
      'P-026: Unbounded per-step positive reward enables reward hacking. Any reward given continuously while stationary must have cumulative cap, time limit, and overstay penalty. Classic Goodhart\'s Law in RL.',
    subtitleKo: '횡단보도 (보상 악용)',
    descriptionKo:
      '첫 보행자 단계. 280D 관측(268D + 12D 보행자/횡단보도)으로 처음부터 학습. 12.5M까지 주행 기술 +730으로 재구축. 13.3M에서 에이전트가 양보 보상 악용 발견: 횡단보도에서 무기한 정차하며 +0.2/s 양보 보상 수확. 보상이 +20,148로 폭발.',
    keyInsightKo:
      'P-026: 무제한 스텝당 양수 보상은 보상 해킹을 가능케 함. 정지 또는 저속 상태에서 지속적으로 주어지는 보상에는 누적 상한, 시간 제한, 초과 체류 페널티가 필요. RL의 전형적 Goodhart의 법칙.',
    version: 1,
  },
  {
    id: 'phase-l-v2',
    name: 'Phase L v2',
    subtitle: 'Crosswalks (Yield Cap Insufficient)',
    reward: null,
    status: 'failed',
    tags: ['PPO', '280D', 'Warm Start', 'Partial Fix', 'Exploit v2'],
    description:
      'Applied yield reward cap (2.0/episode, 8s timeout). Agent found second exploit: drove at 0.2-0.9 m/s near crosswalk, avoiding stuck detection (0.1 m/s threshold) while accumulating driving rewards over extremely long episodes. Reward hit 6,435.',
    observations: '280D',
    steps: '~1.75M (stopped)',
    keyInsight:
      'Yield cap alone is insufficient. The agent exploited the gap between yield speed threshold (1.0 m/s) and stuck detection threshold (0.1 m/s). Anti-farming requires episode termination, not just reward capping.',
    subtitleKo: '횡단보도 (양보 상한 불충분)',
    descriptionKo:
      '양보 보상 상한(에피소드당 2.0, 8초 타임아웃) 적용. 에이전트가 두 번째 악용 발견: 횡단보도 근처에서 0.2-0.9 m/s로 주행하여 고착 감지(0.1 m/s 임계값) 회피하면서 극단적으로 긴 에피소드에서 주행 보상 축적. 보상 6,435 달성.',
    keyInsightKo:
      '양보 상한만으로는 불충분. 에이전트가 양보 속도 임계값(1.0 m/s)과 고착 감지 임계값(0.1 m/s) 사이의 간극을 악용. 보상 수확 방지에는 보상 상한이 아닌 에피소드 종료가 필요.',
    version: 2,
    parentId: 'phase-l-v1',
  },
  {
    id: 'phase-l-v3',
    name: 'Phase L v3',
    subtitle: 'Crosswalks (Anti-Farming v3)',
    reward: null,
    status: 'in_progress',
    tags: ['PPO', '280D', 'Warm Start', 'Episode Termination', 'Pedestrians'],
    description:
      'Triple defense against crosswalk farming: (1) crosswalkStopTimer tracks slow time near crosswalk, (2) episode terminates after 15s (CROSSWALK_OVERSTAY), (3) overstay penalty 10x stronger (-1.0/s). Resume from v1 12.5M stable checkpoint.',
    observations: '280D',
    steps: '5M',
    keyInsight:
      'Three layers of anti-exploit defense. Episode termination is the nuclear option that makes any farming strategy strictly unprofitable regardless of other reward dynamics.',
    subtitleKo: '횡단보도 (수확 방지 v3)',
    descriptionKo:
      '횡단보도 수확에 대한 3중 방어: (1) crosswalkStopTimer가 횡단보도 근처 저속 시간 추적, (2) 15초 후 에피소드 종료(CROSSWALK_OVERSTAY), (3) 초과 체류 페널티 10배 강화(-1.0/s). v1 12.5M 안정 checkpoint에서 재개.',
    keyInsightKo:
      '3겹의 악용 방지 방어. 에피소드 종료는 다른 보상 역학과 무관하게 모든 수확 전략을 엄밀히 비수익적으로 만드는 최종 수단.',
    version: 3,
    parentId: 'phase-l-v2',
  },
  {
    id: 'phase-l-v4',
    name: 'Phase L v4',
    subtitle: 'Crosswalks (Speed Threshold Fix)',
    reward: 702,
    status: 'failed',
    tags: ['PPO', '280D', 'Speed Threshold', 'Partial Success'],
    description:
      'Fixed yield speed threshold to match stuck detection. Reached +702 but pedestrian interaction still suboptimal. Agent avoided crosswalks rather than yielding properly.',
    observations: '280D',
    steps: '5M',
    keyInsight:
      'Speed threshold alignment fixed the exploit, but avoidance behavior emerged. Need positive shaping for proper yield behavior, not just penalty avoidance.',
    subtitleKo: '횡단보도 (속도 임계값 수정)',
    descriptionKo:
      '양보 속도 임계값을 고착 감지와 일치시킴. +702에 도달했으나 보행자 상호작용이 여전히 최적이 아님. 에이전트가 적절히 양보하지 않고 횡단보도를 회피.',
    keyInsightKo:
      '속도 임계값 정렬로 악용은 수정되었으나 회피 행동이 발생. 페널티 회피가 아닌 적절한 양보 행동을 위한 긍정적 형성이 필요.',
    version: 4,
    parentId: 'phase-l-v3',
  },
  {
    id: 'phase-l-v5',
    name: 'Phase L v5',
    subtitle: 'Crosswalks (Complete)',
    reward: 718,
    status: 'success',
    tags: ['PPO', '280D', 'Pedestrian Yield', 'Full Compliance'],
    description:
      'Full pedestrian compliance achieved. Proper yield behavior at crosswalks with balanced reward shaping. All driving skills maintained while adding pedestrian awareness.',
    observations: '280D',
    steps: '5M',
    keyInsight:
      'P-027: Yield behavior requires balanced positive/negative shaping. Overstay termination + yield reward cap + proper speed thresholds together create compliant behavior.',
    subtitleKo: '횡단보도 (완료)',
    descriptionKo:
      '완전한 보행자 준수 달성. 균형 잡힌 보상 형성으로 횡단보도에서 적절한 양보 행동. 보행자 인식을 추가하면서 모든 주행 기술 유지.',
    keyInsightKo:
      'P-027: 양보 행동은 균형 잡힌 긍정/부정 형성이 필요. 초과 체류 종료 + 양보 보상 상한 + 적절한 속도 임계값이 함께 준수 행동을 생성.',
    version: 5,
    parentId: 'phase-l-v4',
  },
  // ---------- Phase M: Multi-Agent Test Field ----------
  {
    id: 'phase-m',
    name: 'Phase M',
    subtitle: 'Multi-Agent Test Field',
    reward: null,
    status: 'success',
    tags: ['Inference', 'Grid Road', '12 Agents', 'Test Environment'],
    description:
      'Grid-based test field for multi-agent inference testing. 4x4 intersection grid with 12 agents, 25 NPCs, 8 pedestrians. Uses trained ONNX models for real-time inference validation.',
    observations: '280D',
    steps: 'N/A (inference)',
    keyInsight:
      'Test environment for validating trained models in complex multi-agent scenarios. Grid road network with traffic signals and diverse route assignments.',
    subtitleKo: '다중 에이전트 테스트 필드',
    descriptionKo:
      '다중 에이전트 추론 테스트를 위한 그리드 기반 테스트 필드. 4x4 교차로 그리드에 12개 에이전트, 25개 NPC, 8명 보행자. 실시간 추론 검증을 위해 학습된 ONNX 모델 사용.',
    keyInsightKo:
      '복잡한 다중 에이전트 시나리오에서 학습된 모델을 검증하기 위한 테스트 환경. 교통 신호와 다양한 루트 할당이 있는 그리드 도로 네트워크.',
  },
  // ---------- Phase N: ProceduralRoadBuilder ----------
  {
    id: 'phase-n-v1',
    name: 'Phase N v1',
    subtitle: 'ProceduralRoadBuilder (Stationary)',
    reward: -79,
    status: 'failed',
    tags: ['PPO', '280D', 'Fresh Start', 'New Environment', 'Stationary Local Optimum'],
    description:
      'First attempt with new ProceduralRoadBuilder environment. Agent learned to stay stationary (speed ~0) to maximize heading alignment reward without progress penalty.',
    observations: '280D',
    steps: '3M',
    keyInsight:
      'P-028: Heading alignment reward without speed gate creates stationary local optimum. Agent can maximize alignment by not moving.',
    subtitleKo: 'ProceduralRoadBuilder (정지)',
    descriptionKo:
      '새 ProceduralRoadBuilder 환경으로 첫 시도. 에이전트가 진행 페널티 없이 헤딩 정렬 보상을 최대화하기 위해 정지 상태(속도 ~0)를 학습.',
    keyInsightKo:
      'P-028: 속도 게이트 없는 헤딩 정렬 보상은 정지 국소 최적을 생성. 에이전트가 움직이지 않으면서 정렬을 최대화할 수 있음.',
    version: 1,
  },
  {
    id: 'phase-n-v5b',
    name: 'Phase N v5b',
    subtitle: 'ProceduralRoadBuilder (Speed Gate)',
    reward: 521.8,
    status: 'success',
    tags: ['PPO', '280D', 'Speed Gate', 'Curriculum Complete', '2/2 Lessons'],
    description:
      'Speed gate on heading alignment prevents stationary exploit. Strong speed reward (1.0x) for fresh training bootstrap. Curriculum complete: goal_distance 150m->300m, num_lanes 1->2.',
    observations: '280D',
    steps: '3M',
    keyInsight:
      'P-029: Speed gate (speedGate = Clamp01(|speed|/2)) on alignment rewards prevents stationary local optimum. Combined with strong speed reward enables proper driving behavior from scratch.',
    subtitleKo: 'ProceduralRoadBuilder (속도 게이트)',
    descriptionKo:
      '헤딩 정렬에 속도 게이트를 적용하여 정지 악용 방지. 처음부터 학습을 위한 강한 속도 보상(1.0x). Curriculum 완료: goal_distance 150m->300m, num_lanes 1->2.',
    keyInsightKo:
      'P-029: 정렬 보상에 속도 게이트(speedGate = Clamp01(|speed|/2))를 적용하면 정지 국소 최적을 방지. 강한 속도 보상과 결합하여 처음부터 적절한 주행 행동 가능.',
    version: 5,
    parentId: 'phase-n-v1',
  },
];

// Convenience: only canonical (latest version) phases for the main card view
export const canonicalPhases = phases.filter(
  (p) => !phases.some((q) => q.parentId === p.id),
);

// ---------- Policy discoveries ----------
export interface PolicyDiscovery {
  id: string;
  name: string;
  nameEn: string;
  sourcePhase: string;
  status: 'verified' | 'in_progress' | 'planned';
  matchingStandard: string;
  description: string;
  failContext: string;
  fixContext: string;
  descriptionKo?: string;
  failContextKo?: string;
  fixContextKo?: string;
}

export const policyDiscoveries: PolicyDiscovery[] = [
  {
    id: 'P-001',
    name: 'Variable Isolation',
    nameEn: 'Single Variable Isolation',
    sourcePhase: 'Phase B v1 → v2',
    status: 'verified',
    matchingStandard: 'Controlled Experiment Design',
    description:
      'Change only one variable at a time. Phase B v1 changed 7 hyper-parameters simultaneously, making root-cause identification impossible.',
    failContext:
      'Phase B v1: 7 hyper-parameter changes + immediate NPC exposure → reward collapsed to -108',
    fixContext:
      'Phase B v2: restored Phase A settings, changed only curriculum → reward reached +877',
    descriptionKo:
      '한 번에 하나의 변수만 변경. Phase B v1은 7개 하이퍼파라미터를 동시에 변경하여 근본 원인 식별이 불가능했음.',
    failContextKo:
      'Phase B v1: 7개 하이퍼파라미터 변경 + 즉시 NPC 노출 -> 보상 -108로 붕괴',
    fixContextKo:
      'Phase B v2: Phase A 설정 복원, curriculum만 변경 -> 보상 +877 달성',
  },
  {
    id: 'P-002',
    name: 'Staggered Curriculum',
    nameEn: 'Staggered Curriculum Transitions',
    sourcePhase: 'Phase D v1 → v2',
    status: 'verified',
    matchingStandard: 'SOTIF Incremental Complexity',
    description:
      'Curriculum parameters must transition at different thresholds. Simultaneous transitions cause catastrophic forgetting.',
    failContext:
      'Phase D v1: 3 parameters at same threshold (~400K) → reward crashed from +406 to -4,825',
    fixContext:
      'Phase D v2: staggered thresholds (200K/300K/350K) → one parameter at a time',
    descriptionKo:
      'Curriculum 파라미터는 서로 다른 임계값에서 전환되어야 함. 동시 전환은 치명적 망각을 야기.',
    failContextKo:
      'Phase D v1: 3개 파라미터가 동일 임계값(~400K)에서 전환 -> 보상 +406에서 -4,825로 폭락',
    fixContextKo:
      'Phase D v2: 단계적 임계값(200K/300K/350K) -> 한 번에 하나의 파라미터',
  },
  {
    id: 'P-003',
    name: 'Capability-Based Checkpoint',
    nameEn: 'Capability-Based Checkpoint Selection',
    sourcePhase: 'Phase B v1 → v2',
    status: 'verified',
    matchingStandard: 'Transfer Learning Best Practice',
    description:
      'Select checkpoints that already possess the capabilities needed for the next phase.',
    failContext:
      'Phase B v1: Phase 0 checkpoint (lane-keeping only, no overtaking ability)',
    fixContext:
      'Phase B v2: Phase A checkpoint (overtaking ability demonstrated at +2,113 reward)',
    descriptionKo:
      '다음 단계에 필요한 능력을 이미 보유한 checkpoint를 선택.',
    failContextKo:
      'Phase B v1: Phase 0 checkpoint(차선 유지만, 추월 능력 없음)',
    fixContextKo:
      'Phase B v2: Phase A checkpoint(+2,113 보상에서 입증된 추월 능력)',
  },
  {
    id: 'P-004',
    name: 'Conservative Penalty',
    nameEn: 'Conservative Penalty Design',
    sourcePhase: 'Phase B v1 → v2',
    status: 'verified',
    matchingStandard: 'Reward Shaping Theory',
    description:
      'Penalties must be conservative. Excessive penalties teach avoidance behaviors (e.g., stopping) instead of desired behaviors.',
    failContext:
      'Phase B v1: speedUnderPenalty -0.1/step taught agent to stop moving entirely',
    fixContext:
      'Phase B v2: reduced to -0.02/step (80% reduction) → normal learning resumed',
    descriptionKo:
      '페널티는 보수적이어야 함. 과도한 페널티는 원하는 행동 대신 회피 행동(예: 정지)을 학습시킴.',
    failContextKo:
      'Phase B v1: speedUnderPenalty -0.1/step이 에이전트에게 완전 정지를 학습시킴',
    fixContextKo:
      'Phase B v2: -0.02/step으로 감소(80% 감소) -> 정상 학습 재개',
  },
  {
    id: 'P-009',
    name: 'Observation Coupling Ban',
    nameEn: 'Observation-Environment Coupling Ban',
    sourcePhase: 'Phase D v2 → v3',
    status: 'verified',
    matchingStandard: 'P-001 Extension + SOTIF Incremental Complexity',
    description:
      'Never change observation space AND environment difficulty simultaneously. When adding new sensors/observations, fix the environment. Only add curriculum after observation learning is complete.',
    failContext:
      'Phase D v2: 254D observation + NPC curriculum → collapse at 7.87M steps (+447 → -756)',
    fixContext:
      'Phase D v3: 254D observation + FIXED environment (3 NPC) → +895 success',
    descriptionKo:
      '관측 공간과 환경 난이도를 절대 동시에 변경하지 말 것. 새 센서/관측 추가 시 환경을 고정. 관측 학습 완료 후에만 curriculum 추가.',
    failContextKo:
      'Phase D v2: 254D 관측 + NPC curriculum -> 7.87M 스텝에서 붕괴(+447 -> -756)',
    fixContextKo:
      'Phase D v3: 254D 관측 + 고정 환경(3 NPC) -> +895 성공',
  },
  {
    id: 'P-010',
    name: 'Triple Consistency',
    nameEn: 'Scene-Config-Code Consistency',
    sourcePhase: 'Phase D v3',
    status: 'verified',
    matchingStandard: 'Pre-flight Check (System Integrity)',
    description:
      'Before training, verify consistency across Scene (BehaviorParameters), Config (YAML), and Code (CollectObservations). Mismatch causes silent data truncation.',
    failContext:
      'Phase D v3: VectorObservationSize=242 in scene while code outputs 254D → 12D lane obs silently dropped',
    fixContext:
      'Fixed scene to 254D, verified via checkpoint inspection: seq_layers.0.weight=[512,254]',
    descriptionKo:
      '학습 전 씬(BehaviorParameters), 설정(YAML), 코드(CollectObservations) 간 일관성을 검증. 불일치 시 무음 데이터 절삭 발생.',
    failContextKo:
      'Phase D v3: 씬에서 VectorObservationSize=242인데 코드는 254D 출력 -> 12D 차선 관측이 무음으로 삭제됨',
    fixContextKo:
      '씬을 254D로 수정, checkpoint 검사로 검증: seq_layers.0.weight=[512,254]',
  },
  {
    id: 'P-011',
    name: 'Scene-Phase Matching',
    nameEn: 'Scene-Phase File Matching',
    sourcePhase: 'Phase F v1',
    status: 'verified',
    matchingStandard: 'P-010 Extension (Environment Integrity)',
    description:
      'Before training, verify the correct Phase-specific Unity scene is loaded. Each phase has dedicated scene with phase-specific road geometry (width, curvature, intersections) set at scene creation time.',
    failContext:
      'Phase F v1: PhaseE_CurvedRoads scene (4.5m road) loaded instead of PhaseF_MultiLane (11.5m). num_lanes 1→2 caused instant off-road (-8.15 for 4.27M steps)',
    fixContext:
      'Phase F v2: Verified PhaseF_MultiLane.unity loaded via get_active before training. Road width 11.5m confirmed.',
    descriptionKo:
      '학습 전 올바른 Phase 전용 Unity 씬이 로드되었는지 확인. 각 Phase는 Phase별 도로 형상(폭, 곡률, 교차로)이 씬 생성 시 설정된 전용 씬을 가짐.',
    failContextKo:
      'Phase F v1: PhaseF_MultiLane 대신 PhaseE_CurvedRoads 씬(4.5m 도로) 로드됨. num_lanes 1->2 시 즉시 도로 이탈(-8.15, 4.27M 스텝)',
    fixContextKo:
      'Phase F v2: 학습 전 get_active로 PhaseF_MultiLane.unity 로드 확인. 도로 폭 11.5m 확인.',
  },
  {
    id: 'P-012',
    name: 'No Shared Thresholds',
    nameEn: 'No Shared Curriculum Thresholds',
    sourcePhase: 'Phase F v3 → v4',
    status: 'verified',
    matchingStandard: 'P-002 Reinforcement (ML-Agents Global Reward)',
    description:
      'No two curriculum parameters may share the same threshold value. ML-Agents uses a single global smoothed reward for all parameters, so identical thresholds trigger simultaneous transitions.',
    failContext:
      'Phase F v3: Threshold 350 shared by 4 params (goal_distance, speed_zone, road_curvature, NPCs). Triple simultaneous transition at 4.38M caused -2,480 crash.',
    fixContext:
      'Phase F v4: All 15 thresholds unique (150-900 range, minimum 50-point gaps). All 6 transitions occurred individually.',
    descriptionKo:
      '두 개의 curriculum 파라미터가 동일한 임계값을 공유해서는 안 됨. ML-Agents는 모든 파라미터에 단일 글로벌 평활 보상을 사용하므로 동일 임계값이 동시 전환을 유발.',
    failContextKo:
      'Phase F v3: 임계값 350을 4개 파라미터가 공유. 4.38M에서 3중 동시 전환으로 -2,480 폭락.',
    fixContextKo:
      'Phase F v4: 15개 임계값 모두 고유(150-900 범위, 최소 50포인트 간격). 6개 전환 모두 개별 발생.',
  },
  {
    id: 'P-013',
    name: 'Speed Zone Ordering',
    nameEn: 'Speed Zone Curriculum Ordering',
    sourcePhase: 'Phase F v4 → v5',
    status: 'verified',
    matchingStandard: 'Curriculum Continuity + P-001 Extension',
    description:
      'When introducing multi-zone speed limits via curriculum, the first zone must match the previous single-zone default speed. Placing the slowest zone first causes catastrophic overspeed penalties.',
    failContext:
      'Phase F v4: GenerateSpeedZones() placed Residential(30 km/h) first. Agent at 60 km/h → speedRatio 2.0 → -3.0/step penalty → -2,790 crash.',
    fixContext:
      'Phase F v5: Reordered to [UrbanGeneral(60), UrbanNarrow(50), ...]. Drop reduced to -262 (10.7x improvement), recovery in ~500K steps (12x faster).',
    descriptionKo:
      'Curriculum으로 다중 구간 속도 제한 도입 시 첫 번째 구간은 이전 단일 구간 기본 속도와 일치해야 함. 가장 느린 구간을 먼저 배치하면 치명적 과속 페널티 발생.',
    failContextKo:
      'Phase F v4: GenerateSpeedZones()가 Residential(30 km/h)을 먼저 배치. 60 km/h 에이전트 -> speedRatio 2.0 -> -3.0/step 페널티 -> -2,790 폭락.',
    fixContextKo:
      'Phase F v5: [UrbanGeneral(60), UrbanNarrow(50), ...]으로 재정렬. 하락 -262로 감소(10.7배 개선), ~500K 스텝에서 복구(12배 빠름).',
  },
  {
    id: 'P-014',
    name: 'Intersection Zone WrongWay Exemption',
    nameEn: 'Intersection Zone Detection Exemption',
    sourcePhase: 'Phase G v1 → v2',
    status: 'verified',
    matchingStandard: 'Context-Aware Safety Checks',
    description:
      'WrongWay detection must be context-aware. Straight-road checks (xPos < -0.5) are invalid in intersection zones where turns produce negative X positions by design. Disable WrongWay check when agent is within intersection zone (Z >= intersectionDistance - intersectionWidth).',
    failContext:
      'Phase G v1: IsWrongWayDriving(xPos) checked xPos < -0.5. Left turns exit at X=-8.25, always triggering WrongWay. 32% termination rate, reward plateau at 494.',
    fixContext:
      'Phase G v2: IsWrongWayDriving(xPos, zPos) with intersection zone awareness. WrongWay check disabled when intersectionType > 0 AND zPos >= intersectionDistance - intersectionWidth.',
    descriptionKo:
      'WrongWay 감지는 맥락 인식이 필요. 직선 도로 검사(xPos < -0.5)는 회전이 설계상 음수 X 위치를 생성하는 교차로 구간에서 무효. 교차로 구간 내에서 WrongWay 검사 비활성화.',
    failContextKo:
      'Phase G v1: IsWrongWayDriving(xPos)이 xPos < -0.5 검사. 좌회전 출구 X=-8.25에서 항상 WrongWay 발동. 32% 종료율, 보상 정체 494.',
    fixContextKo:
      'Phase G v2: IsWrongWayDriving(xPos, zPos)에 교차로 구간 인식 추가. intersectionType > 0 AND zPos >= intersectionDistance - intersectionWidth일 때 WrongWay 검사 비활성화.',
  },
  {
    id: 'P-015',
    name: 'DecisionRequester Required',
    nameEn: 'DecisionRequester Component Required After Scene Regeneration',
    sourcePhase: 'Phase G v2',
    status: 'verified',
    matchingStandard: 'P-010 Extension (Component Integrity)',
    description:
      'After scene regeneration (PhaseSceneCreator), DecisionRequester component may be missing from agents. Without it, agents never request decisions and training hangs silently at _reset_env with zero steps produced.',
    failContext:
      'Phase G v2 setup: Scene regenerated for visual enhancements. All 16 agents lost DecisionRequester. Training connected to Unity but produced 0 steps for 10+ minutes.',
    fixContext:
      'Added DecisionRequester (period=5, TakeActionsBetweenDecisions=true) to all agents. Updated ConfigurePhaseGAgents.cs to auto-add DecisionRequester during configuration.',
    descriptionKo:
      '씬 재생성(PhaseSceneCreator) 후 에이전트에서 DecisionRequester 컴포넌트가 누락될 수 있음. 없으면 에이전트가 결정을 요청하지 않아 학습이 0 스텝에서 무음으로 멈춤.',
    failContextKo:
      'Phase G v2 설정: 시각 개선을 위해 씬 재생성. 16개 에이전트 모두 DecisionRequester 손실. 학습이 Unity에 연결되었으나 10분 이상 0 스텝.',
    fixContextKo:
      '모든 에이전트에 DecisionRequester(period=5, TakeActionsBetweenDecisions=true) 추가. ConfigurePhaseGAgents.cs가 설정 중 자동 추가하도록 업데이트.',
  },
  {
    id: 'P-016',
    name: 'Gradual Variation',
    nameEn: 'Gradual Variation Introduction',
    sourcePhase: 'Phase H v1 → v3',
    status: 'verified',
    matchingStandard: 'P-002 Extension (Curriculum Smoothness)',
    description:
      'Parameters introducing randomness/variation must be graduated (e.g., 0->0.05->0.10->0.15), never jumped in a single step (0->0.15). Abrupt variation invalidates the learned policy.',
    failContext:
      'Phase H v1: npc_speed_variation jumped 0->0.15 at threshold 700. Reward crashed 700->550, Std spiked to 300+.',
    fixContext:
      'Phase H v2/v3: Gradual variation (0->0.05->0.10->0.15) with per-step thresholds. Smooth transition maintained reward stability.',
    descriptionKo:
      '무작위성/변동을 도입하는 파라미터는 단계적(예: 0->0.05->0.10->0.15)이어야 하며, 단일 단계 점프(0->0.15)는 금지. 급격한 변동은 학습된 정책을 무효화.',
    failContextKo:
      'Phase H v1: npc_speed_variation이 임계값 700에서 0->0.15로 점프. 보상 700->550 폭락, Std 300+ 급등.',
    fixContextKo:
      'Phase H v2/v3: 단계적 변동(0->0.05->0.10->0.15)과 스텝별 임계값. 매끄러운 전환으로 보상 안정성 유지.',
  },
  {
    id: 'P-017',
    name: 'Achievable Thresholds',
    nameEn: 'Thresholds Must Be Achievable Under Active Conditions',
    sourcePhase: 'Phase H v2 → v3',
    status: 'verified',
    matchingStandard: 'P-002 Extension (Threshold Calibration)',
    description:
      'Curriculum thresholds must be set BELOW the expected average reward when the current lesson is active. Variation/noise lowers average reward, making higher thresholds unreachable.',
    failContext:
      'Phase H v2: Thresholds 710/720 set above agent average (~690-700) with variation=0.05 active. Curriculum stuck at lesson 2/4.',
    fixContext:
      'Phase H v3: Lowered to 685/690/693 (5-10 points below observed average). All 4 lessons completed.',
    descriptionKo:
      'Curriculum 임계값은 현재 레슨 활성 시 기대 평균 보상 이하로 설정해야 함. 변동/노이즈가 평균 보상을 낮추므로 높은 임계값은 도달 불가.',
    failContextKo:
      'Phase H v2: 임계값 710/720이 variation=0.05 활성 시 에이전트 평균(~690-700) 이상. Curriculum이 레슨 2/4에서 멈춤.',
    fixContextKo:
      'Phase H v3: 685/690/693으로 하향(관측 평균보다 5-10 포인트 아래). 4개 레슨 모두 완료.',
  },
  {
    id: 'P-018',
    name: 'Minimum Threshold Spacing',
    nameEn: 'Minimum Threshold Spacing >= 15 Points',
    sourcePhase: 'Phase I v1 → v2',
    status: 'verified',
    matchingStandard: 'P-012 Extension (Threshold Separation)',
    description:
      'Adjacent curriculum thresholds must be spaced at least 15 points apart. Tighter spacing causes near-simultaneous transitions of multiple parameters, resulting in catastrophic reward collapse.',
    failContext:
      'Phase I v1: Thresholds 700/702/705 (spacing 2-3 points). Three params (curvature, direction, zones) unlocked within ~20K steps. Reward crashed 724->-40 (760-point drop).',
    fixContext:
      'Phase I v2: All params at final values (no transitions). Agent recovered from 623 to 770. Future configs should use >= 15-point spacing.',
    descriptionKo:
      '인접 curriculum 임계값은 최소 15포인트 이상 간격이어야 함. 더 촘촘한 간격은 다수 파라미터의 거의 동시 전환을 야기하여 치명적 보상 붕괴를 초래.',
    failContextKo:
      'Phase I v1: 임계값 700/702/705(간격 2-3포인트). 세 파라미터(곡률, 방향, 구간)가 ~20K 스텝 내 해제. 보상 724->-40 폭락(760포인트 하락).',
    fixContextKo:
      'Phase I v2: 모든 파라미터를 최종값으로(전환 없음). 에이전트가 623에서 770으로 복구. 향후 설정은 >= 15포인트 간격 사용.',
  },
  {
    id: 'P-019',
    name: 'Minimum Lesson Length Scaling',
    nameEn: 'min_lesson_length Scales with Reward Noise',
    sourcePhase: 'Phase H v3',
    status: 'verified',
    matchingStandard: 'Curriculum Stability',
    description:
      'min_lesson_length must increase when reward variance is high. Short lesson lengths cause premature curriculum transitions during reward fluctuations, leading to unstable training.',
    failContext:
      'Phase H v3: Default min_lesson_length too short for noisy intersection rewards.',
    fixContext:
      'Phase H v3: Increased min_lesson_length proportional to observed reward standard deviation.',
    descriptionKo:
      'min_lesson_length는 보상 분산이 높을 때 증가해야 함. 짧은 레슨 길이는 보상 변동 중 조기 curriculum 전환을 야기하여 불안정한 학습 초래.',
    failContextKo:
      'Phase H v3: 노이즈 많은 교차로 보상에 대해 기본 min_lesson_length가 너무 짧음.',
    fixContextKo:
      'Phase H v3: min_lesson_length를 관측된 보상 표준 편차에 비례하여 증가.',
  },
  {
    id: 'P-020',
    name: 'Observation Dim Fresh Start',
    nameEn: 'Observation Dimension Change = Fresh Start Required',
    sourcePhase: 'Phase J v1',
    status: 'verified',
    matchingStandard: 'Model Architecture Compatibility',
    description:
      'Changing observation dimensions (e.g. 260D→268D) invalidates optimizer state tensors. Warm start from incompatible checkpoint causes immediate crash. Fresh training from scratch is the only option.',
    failContext:
      'Phase J v1: Warm start from Phase I v2 (260D) with 268D observations. Adam optimizer tensor mismatch at step ~40K.',
    fixContext:
      'Phase J v2: Fresh start from scratch with 268D. Rebuilt all skills in 10M steps.',
    descriptionKo:
      '관측 차원 변경(예: 260D->268D)은 옵티마이저 상태 텐서를 무효화. 호환 불가 checkpoint에서의 Warm Start는 즉시 크래시 발생. 처음부터 학습이 유일한 옵션.',
    failContextKo:
      'Phase J v1: Phase I v2(260D)에서 268D 관측으로 Warm Start. ~40K 스텝에서 Adam 옵티마이저 텐서 불일치.',
    fixContextKo:
      'Phase J v2: 268D로 처음부터 학습. 10M 스텝에서 모든 기술 재구축.',
  },
  {
    id: 'P-021',
    name: 'From-Scratch Threshold Adjustment',
    nameEn: 'From-Scratch Training Needs Lower Thresholds',
    sourcePhase: 'Phase J v2',
    status: 'verified',
    matchingStandard: 'P-016 Extension',
    description:
      'Fresh start training reaches lower absolute rewards than warm start because the agent must rebuild all skills simultaneously. Curriculum thresholds must be set lower than warm-start equivalents.',
    failContext:
      'Phase J v2: Thresholds calibrated for warm-start reward levels. Agent plateaued below thresholds after rebuilding from scratch.',
    fixContext:
      'Phase J v2 resume: Lowered thresholds by 30 points at 3.7M, reduced LR. Reached 9/13 curriculum.',
    descriptionKo:
      '처음부터 학습은 에이전트가 모든 기술을 동시에 재구축해야 하므로 Warm Start보다 낮은 절대 보상에 도달. Curriculum 임계값은 Warm Start 수준보다 낮게 설정해야 함.',
    failContextKo:
      'Phase J v2: Warm Start 보상 수준에 맞춰 교정된 임계값. 처음부터 재구축 후 에이전트가 임계값 아래에서 정체.',
    fixContextKo:
      'Phase J v2 재개: 3.7M에서 임계값 30포인트 하향, LR 감소. 13개 중 9개 curriculum 달성.',
  },
  {
    id: 'P-022',
    name: 'Feature Before Tuning',
    nameEn: 'Feature Activation Must Precede Parameter Tuning',
    sourcePhase: 'Phase J v3',
    status: 'verified',
    matchingStandard: 'Curriculum Ordering',
    description:
      'When a feature (e.g. traffic signals) must be activated before its parameters can be tuned (e.g. green_ratio), the activation threshold must be strictly lower than the tuning threshold. Single-parameter curriculum prevents ordering conflicts.',
    failContext:
      'Phase J v3: green_ratio threshold < signal activation threshold. Signal turned ON while green_ratio was already partially reduced, causing 177-point crash.',
    fixContext:
      'Phase J v4: Signal-first approach with signals locked ON from step 0. Only green_ratio in curriculum.',
    descriptionKo:
      '기능(예: 교통 신호)이 파라미터 조정(예: green_ratio) 전에 활성화되어야 할 때, 활성화 임계값이 조정 임계값보다 엄밀히 낮아야 함. 단일 파라미터 curriculum이 순서 충돌을 방지.',
    failContextKo:
      'Phase J v3: green_ratio 임계값 < 신호 활성화 임계값. 신호 ON 시 green_ratio가 이미 부분 감소된 상태에서 177포인트 폭락.',
    fixContextKo:
      'Phase J v4: 신호 우선 접근, 스텝 0부터 신호 ON으로 고정. Curriculum에 green_ratio만 포함.',
  },
  {
    id: 'P-023',
    name: 'Signal Reward Compression',
    nameEn: 'Reward Compression Under Constraints',
    sourcePhase: 'Phase J v4',
    status: 'verified',
    matchingStandard: 'Reward Bandwidth Analysis',
    description:
      'Adding constraints (e.g. shorter green lights) reduces the achievable reward ceiling. Curriculum thresholds must account for this compression - a threshold achievable at green_ratio=0.8 may be unreachable at green_ratio=0.4.',
    failContext:
      'Phase J v4: Threshold 540 unreachable at green_ratio=0.5 (agent plateau ~490-500).',
    fixContext:
      'Phase J v5: Progressive thresholds 450/470/475/475 (descending ceiling). All 5 lessons completed.',
    descriptionKo:
      '제약(예: 짧은 녹색 신호) 추가는 달성 가능한 보상 상한을 낮춤. Curriculum 임계값은 이 압축을 고려해야 함 -- green_ratio=0.8에서 달성 가능한 임계값이 green_ratio=0.4에서는 도달 불가할 수 있음.',
    failContextKo:
      'Phase J v4: green_ratio=0.5에서 임계값 540 도달 불가(에이전트 정체 ~490-500).',
    fixContextKo:
      'Phase J v5: 점진적 임계값 450/470/475/475(하강 상한). 5개 레슨 모두 완료.',
  },
  {
    id: 'P-024',
    name: 'Build BehaviorType Check',
    nameEn: 'BehaviorType in Build Must Be Default',
    sourcePhase: 'Phase J v5',
    status: 'verified',
    matchingStandard: 'Build Configuration',
    description:
      'BehaviorType=InferenceOnly in a training build silently prevents brain registration. The agent runs in inference mode, ignoring the trainer entirely. No error is thrown.',
    failContext:
      'Phase J v5: BehaviorType left at InferenceOnly in build. Training appeared to run but no learning occurred.',
    fixContext:
      'Phase J v5: Set BehaviorType=Default in build scene. Agent brain registered correctly.',
    descriptionKo:
      '학습 빌드에서 BehaviorType=InferenceOnly는 브레인 등록을 무음으로 방지. 에이전트가 추론 모드로 실행되어 트레이너를 완전히 무시. 에러가 발생하지 않음.',
    failContextKo:
      'Phase J v5: 빌드에서 BehaviorType이 InferenceOnly로 남음. 학습이 실행되는 것처럼 보이지만 학습 발생하지 않음.',
    fixContextKo:
      'Phase J v5: 빌드 씬에서 BehaviorType=Default로 설정. 에이전트 브레인이 정상 등록.',
  },
  {
    id: 'P-025',
    name: 'BehaviorType Enum Values',
    nameEn: 'BehaviorType Enum: 1=HeuristicOnly, 2=InferenceOnly',
    sourcePhase: 'Phase L v1',
    status: 'verified',
    matchingStandard: 'Unity ML-Agents API',
    description:
      'BehaviorType enum value 1 is HeuristicOnly, not InferenceOnly (which is value 2). Setting the wrong enum value causes unexpected agent behavior during inference testing.',
    failContext:
      'Phase L inference test: BehaviorType=1 intended as InferenceOnly but was actually HeuristicOnly.',
    fixContext:
      'Fixed to BehaviorType=2 (InferenceOnly). Verified correct behavior in inference mode.',
    descriptionKo:
      'BehaviorType enum 값 1은 HeuristicOnly이며, InferenceOnly(값 2)가 아님. 잘못된 enum 값 설정 시 추론 테스트에서 예기치 않은 에이전트 행동 발생.',
    failContextKo:
      'Phase L 추론 테스트: BehaviorType=1을 InferenceOnly로 의도했으나 실제로는 HeuristicOnly였음.',
    fixContextKo:
      'BehaviorType=2(InferenceOnly)로 수정. 추론 모드에서 올바른 행동 확인.',
  },
  {
    id: 'P-026',
    name: 'Yield Reward Cap',
    nameEn: 'Unbounded Per-Step Reward Enables Farming',
    sourcePhase: 'Phase L v1 → v2 → v3',
    status: 'verified',
    matchingStandard: 'Reward Shaping Safety',
    description:
      'Any positive reward given continuously while the agent is stationary or slow-moving enables reward hacking. Defense requires three layers: (1) cumulative cap per episode, (2) time limit with escalating penalty, (3) episode termination for prolonged loitering.',
    failContext:
      'Phase L v1: Unbounded yield reward (+0.2/s) → agent farmed +20,148. Phase L v2: Yield cap (2.0/episode) insufficient → agent exploited slow-driving gap (0.2-0.9 m/s) → +6,435.',
    fixContext:
      'Phase L v3: Triple defense - crosswalkStopTimer (15s limit), episode termination (CROSSWALK_OVERSTAY), 10x stronger overstay penalty (-1.0/s).',
    descriptionKo:
      '에이전트가 정지 또는 저속 상태에서 지속적으로 받는 양의 보상은 reward hacking을 가능하게 함. 방어에는 3개 계층 필요: (1) 에피소드당 누적 보상 상한, (2) 시간 제한과 점진적 페널티, (3) 장기 정차 시 에피소드 종료.',
    failContextKo:
      'Phase L v1: 무제한 양보 보상(+0.2/s) → 에이전트가 +20,148까지 farming. Phase L v2: Yield cap(2.0/에피소드)이 불충분 → 저속 주행 구간(0.2-0.9 m/s) 악용 → +6,435.',
    fixContextKo:
      'Phase L v3: 3중 방어 - crosswalkStopTimer(15초 제한), 에피소드 종료(CROSSWALK_OVERSTAY), 10배 강화된 overstay 페널티(-1.0/s).',
  },
  {
    id: 'P-027',
    name: 'Balanced Yield Shaping',
    nameEn: 'Yield Behavior Requires Balanced Shaping',
    sourcePhase: 'Phase L v4 → v5',
    status: 'verified',
    matchingStandard: 'Reward Shaping Balance',
    description:
      'Yield behavior requires balanced positive/negative shaping. Pure penalty avoidance leads to crosswalk avoidance. Combine overstay termination + yield cap + proper speed thresholds for compliant behavior.',
    failContext:
      'Phase L v4: Penalty-focused shaping → agent avoided crosswalks entirely instead of yielding.',
    fixContext:
      'Phase L v5: Balanced shaping with modest yield reward + overstay penalty + episode termination → proper yield behavior.',
    descriptionKo:
      '양보 행동은 균형 잡힌 긍정/부정 형성이 필요. 순수 페널티 회피는 횡단보도 회피로 이어짐. 준수 행동을 위해 overstay 종료 + 양보 상한 + 적절한 속도 임계값을 결합.',
    failContextKo:
      'Phase L v4: 페널티 중심 형성 → 에이전트가 양보 대신 횡단보도 자체를 회피.',
    fixContextKo:
      'Phase L v5: 적절한 양보 보상 + overstay 페널티 + 에피소드 종료의 균형 형성 → 적절한 양보 행동.',
  },
  {
    id: 'P-028',
    name: 'Speed Gate on Alignment',
    nameEn: 'Heading Alignment Requires Speed Gate',
    sourcePhase: 'Phase N v1 → v5b',
    status: 'verified',
    matchingStandard: 'Reward Gate Design',
    description:
      'Heading alignment reward without speed gate creates stationary local optimum. Agent can maximize alignment by not moving. Apply speedGate = Clamp01(|speed|/threshold) to alignment rewards.',
    failContext:
      'Phase N v1: Heading alignment reward without speed gate → agent learned to stay stationary (speed ~0) while perfectly aligned → reward -79.',
    fixContext:
      'Phase N v5b: speedGate = Clamp01(|speed|/2f) applied to alignment reward → proper driving behavior with +521.8 reward.',
    descriptionKo:
      '속도 게이트 없는 헤딩 정렬 보상은 정지 국소 최적을 생성. 에이전트가 움직이지 않으면서 정렬을 최대화할 수 있음. 정렬 보상에 speedGate = Clamp01(|speed|/threshold) 적용.',
    failContextKo:
      'Phase N v1: 속도 게이트 없는 헤딩 정렬 보상 → 에이전트가 완벽히 정렬된 채 정지 학습(속도 ~0) → 보상 -79.',
    fixContextKo:
      'Phase N v5b: 정렬 보상에 speedGate = Clamp01(|speed|/2f) 적용 → +521.8 보상으로 적절한 주행 행동.',
  },
  {
    id: 'P-029',
    name: 'Northify Ego State',
    nameEn: 'Direction-Invariant Ego State Normalization',
    sourcePhase: 'Phase N v5b → Phase M',
    status: 'verified',
    matchingStandard: 'Observation Space Invariance',
    description:
      'Ego state must be normalized to a canonical direction (e.g., North) for direction-invariant inference. Training environments may have consistent heading, but test environments have diverse spawn directions.',
    failContext:
      'Phase M grid test: Agents spawned in multiple directions failed to follow roads. ONNX model trained on North-facing only.',
    fixContext:
      'Northify ego state in observation collection: rotate velocity and heading to canonical North direction before feeding to policy.',
    descriptionKo:
      '에고 상태는 방향 불변 추론을 위해 정규 방향(예: 북쪽)으로 정규화해야 함. 학습 환경은 일관된 헤딩을 가질 수 있지만 테스트 환경은 다양한 스폰 방향을 가짐.',
    failContextKo:
      'Phase M 그리드 테스트: 여러 방향으로 스폰된 에이전트가 도로를 따라가지 못함. ONNX 모델은 북향만 학습.',
    fixContextKo:
      '관측 수집에서 에고 상태 북쪽화: 정책에 전달하기 전에 속도와 헤딩을 정규 북쪽 방향으로 회전.',
  },
];

// ---------- SOTIF quadrant data ----------
export interface SotifQuadrant {
  id: number;
  label: string;
  description: string;
  phases: string[];
  examples: string[];
  strategy: string;
}

export const sotifQuadrants: SotifQuadrant[] = [
  {
    id: 1,
    label: 'Known Safe',
    description: 'Requirements-based testing. Well-understood safe scenarios.',
    phases: ['Phase 0', 'Phase A', 'Phase B'],
    examples: ['Straight road', 'Constant speed', 'Clear weather'],
    strategy: 'Standard curriculum learning',
  },
  {
    id: 2,
    label: 'Known Unsafe',
    description: 'Targeted verification. Known hazardous scenarios with planned mitigation.',
    phases: ['Phase D', 'Phase E', 'Phase F', 'Phase G'],
    examples: ['Lane observation', 'Sharp curves', 'Multi-lane merging', 'Intersection turns'],
    strategy: 'Specialized scenario training with P-009 isolation',
  },
  {
    id: 3,
    label: 'Unknown Unsafe',
    description: 'Scenario-based exploration. Undiscovered hazardous scenarios.',
    phases: ['Phase H', 'Phase I', 'Phase J', 'Phase K'],
    examples: ['Curvature transitions', 'Sensor degradation', 'OOD scenarios'],
    strategy: 'Adversarial + SOTIF analysis',
  },
  {
    id: 4,
    label: 'Unknown Safe',
    description: 'Field monitoring. Safe scenarios discovered through deployment.',
    phases: ['Phase M'],
    examples: ['Naturally discovered safe behaviors', 'Robust generalization'],
    strategy: 'Post-deployment data collection',
  },
];

// ---------- Tesla gap analysis ----------
export interface GapItem {
  component: string;
  teslaFsd: string;
  thisProject: string;
  gapLevel: 'CRITICAL' | 'MAJOR' | 'MODERATE' | 'ACHIEVED';
  feasibility: string;
}

export const teslaGapAnalysis: GapItem[] = [
  {
    component: 'Vision Perception',
    teslaFsd: 'HydraNet + RegNet-120GF + Occupancy Network (400M params)',
    thisProject: 'Ground Truth Vector (254D)',
    gapLevel: 'CRITICAL',
    feasibility: 'Limited (lightweight models only)',
  },
  {
    component: 'BEV Representation',
    teslaFsd: 'Transformer 8-camera fusion (200x200x256)',
    thisProject: 'Not implemented',
    gapLevel: 'CRITICAL',
    feasibility: 'Possible (single camera)',
  },
  {
    component: 'Trajectory Prediction',
    teslaFsd: 'Occupancy Flow (3D ConvGRU, 2s horizon)',
    thisProject: 'Constant Velocity assumption',
    gapLevel: 'MAJOR',
    feasibility: 'Possible (LSTM/GNN)',
  },
  {
    component: 'Trajectory Planning',
    teslaFsd: 'MCTS + Neural Evaluator (20 candidates)',
    thisProject: 'Reactive RL control only',
    gapLevel: 'MAJOR',
    feasibility: 'Possible (simplified)',
  },
  {
    component: 'Route Planning',
    teslaFsd: 'GPS -> local lane-level graph',
    thisProject: 'Fixed waypoint system',
    gapLevel: 'MAJOR',
    feasibility: 'Possible (A* pathfinding)',
  },
  {
    component: 'Vehicle Control',
    teslaFsd: 'Direct neural network (40ms replan)',
    thisProject: 'RL Policy → steering/accel',
    gapLevel: 'ACHIEVED',
    feasibility: 'Already implemented',
  },
  {
    component: 'Data Pipeline',
    teslaFsd: 'Shadow Mode + Auto-Label + Dojo (4M fleet)',
    thisProject: 'ML-Agents self-play (16 parallel)',
    gapLevel: 'MAJOR',
    feasibility: 'Scale gap unbridgeable',
  },
];

// ---------- Resource comparison ----------
export interface ResourceComparison {
  metric: string;
  tesla: string;
  thisProject: string;
  ratio: string;
}

export const resourceComparisons: ResourceComparison[] = [
  { metric: 'Compute', tesla: 'Dojo 1.1 ExaFLOPS', thisProject: 'RTX 4090 82.6 TFLOPS', ratio: '13,000,000x' },
  { metric: 'Fleet Data', tesla: '4M+ vehicles Shadow Mode', thisProject: '16 parallel Unity envs', ratio: '250,000x' },
  { metric: 'Storage', tesla: '~10 PB/year', thisProject: '4TB SSD', ratio: '2,500x' },
  { metric: 'Model Size', tesla: '~400M params', thisProject: '~2M params (PPO)', ratio: '200x' },
  { metric: 'Training Time', tesla: '3-5 days (Dojo)', thisProject: '2-6 hrs (per phase)', ratio: 'Comparable scope-adjusted' },
  { metric: 'Labeling', tesla: 'Auto-label + Human QA', thisProject: 'Ground truth from sim', ratio: 'Different paradigm' },
];

// ---------- Academic References ----------
export interface Reference {
  id: string;
  title: string;
  authors?: string;
  venue?: string;
  year: string;
  url: string;
  category: 'standard' | 'academic' | 'control' | 'curriculum' | 'analysis';
  relevance: string;
}

export const references: Reference[] = [
  // --- Standards ---
  {
    id: 'ISO-21448',
    title: 'ISO 21448:2022 -- Safety of the Intended Functionality (SOTIF)',
    venue: 'ISO',
    year: '2022',
    url: 'https://www.iso.org/standard/77490.html',
    category: 'standard',
    relevance: 'Core safety framework. 4-Quadrant model for classifying known/unknown safe/unsafe scenarios. Directly applied to Phase structure (A-M).',
  },
  {
    id: 'ISO-PAS-8800',
    title: 'ISO/PAS 8800:2024 -- Safety and Artificial Intelligence',
    venue: 'ISO',
    year: '2024',
    url: 'https://www.iso.org/standard/83303.html',
    category: 'standard',
    relevance: 'AI-specific safety standard complementing ISO 21448. Addresses ML model validation and deployment safety.',
  },
  {
    id: 'UN-R171',
    title: 'UN Regulation No. 171 -- Driver Control Assistance Systems (DCAS)',
    venue: 'UNECE',
    year: '2024',
    url: 'https://unece.org/sites/default/files/2025-03/R171e.pdf',
    category: 'standard',
    relevance: 'Cut-in/Cut-out test parameters (TTC 1.5-5.0s, max decel -7.0 m/s\u00B2, jerk \u2264 3.0 m/s\u00B3). Directly informs Phase I reward design.',
  },
  {
    id: 'UN-R157',
    title: 'UN Regulation No. 157 -- Automated Lane Keeping Systems (ALKS)',
    venue: 'UNECE',
    year: '2021',
    url: 'https://unece.org/transport/documents/2021/03/standards/un-regulation-no-157-automated-lane-keeping-systems-alks',
    category: 'standard',
    relevance: 'Lane keeping test parameters (lat accel \u2264 0.3g, CTE \u2264 0.3m, dk/ds \u2264 0.1/m\u00B2). Defines Phase H validation criteria.',
  },
  // --- Academic Papers ---
  {
    id: 'SOTIF-MPC',
    title: 'Analysis of Functional Insufficiencies and Triggering Conditions for MPC Trajectory Planner',
    venue: 'arXiv',
    year: '2024',
    url: 'https://arxiv.org/html/2407.21569v1',
    category: 'academic',
    relevance: 'FI/TC identification methodology for trajectory planners. Applied to our SOTIF analysis framework.',
  },
  {
    id: 'TC-Systematization',
    title: 'Systematization of Triggering Conditions for SOTIF',
    venue: 'ResearchGate',
    year: '2022',
    url: 'https://www.researchgate.net/publication/362121834',
    category: 'academic',
    relevance: 'Systematic approach to identifying triggering conditions. Informed our FI x TC matrix design.',
  },
  {
    id: 'AV-Human-Accidents',
    title: 'Autonomous Vehicles vs Human Drivers: Accident Analysis',
    venue: 'Nature Communications',
    year: '2024',
    url: 'https://www.nature.com/articles/s41467-024-48526-4',
    category: 'academic',
    relevance: 'Comparative safety analysis AV vs human drivers. Establishes baseline for residual risk targets (4.1-4.85/M miles human rate).',
  },
  {
    id: 'LKA-Evaluation',
    title: 'Empirical Performance Evaluation of Lane Keeping Assist Systems',
    venue: 'arXiv',
    year: '2025',
    url: 'https://arxiv.org/html/2505.11534v1',
    category: 'academic',
    relevance: 'Real-world LKA performance data (curvature threshold 0.006/m, worst-case deviation 1.2m). Validates our Phase E/H curvature targets.',
  },
  // --- Control Theory ---
  {
    id: 'Clothoid-Path',
    title: 'Path Planning with Clothoid Curves for Autonomous Vehicles',
    venue: 'DIVA Portal',
    year: '2017',
    url: 'https://www.diva-portal.org/smash/get/diva2:1150741/FULLTEXT01.pdf',
    category: 'control',
    relevance: 'Clothoid (Euler spiral) theory for straight-to-curve transitions. Key reference for Phase H curvature node dynamics.',
  },
  {
    id: 'Clothoid-Controller',
    title: 'Clothoid-Based Lateral Controller for Autonomous Driving',
    venue: 'MDPI Applied Sciences',
    year: '2024',
    url: 'https://www.mdpi.com/2076-3417/14/5/1817',
    category: 'control',
    relevance: 'Lateral control using clothoid curves. Informs steering rate and lateral acceleration constraints.',
  },
  {
    id: 'Adaptive-MPC',
    title: 'Adaptive MPC for Autonomous Lane Keeping',
    venue: 'arXiv',
    year: '2018',
    url: 'https://arxiv.org/pdf/1806.04335',
    category: 'control',
    relevance: 'MPC approach to lane keeping with curvature adaptation. Comparison baseline for our RL approach (78% compute reduction).',
  },
  {
    id: 'MPC-PID-DRL',
    title: 'MPC-PID Demonstration-based Deep Reinforcement Learning',
    venue: 'arXiv',
    year: '2025',
    url: 'https://arxiv.org/html/2506.04040v1',
    category: 'control',
    relevance: 'Hybrid MPC-PID demonstrated DRL for vehicle control. Recommended approach for Phase H: initial imitation then autonomous learning.',
  },
  // --- Curriculum Learning ---
  {
    id: 'CuRLA',
    title: 'CuRLA: Curriculum Deep Reinforcement Learning for Autonomous Driving',
    venue: 'arXiv',
    year: '2025',
    url: 'https://arxiv.org/html/2501.04982v1',
    category: 'curriculum',
    relevance: 'Curriculum RL for AD showing 2-3x efficiency over random sampling. Validates our P-002 staggered curriculum approach.',
  },
  {
    id: 'Curriculum-Value',
    title: 'The Value of Curriculum Learning for Self-Driving',
    venue: 'Frontiers in Neuroscience (PMC)',
    year: '2023',
    url: 'https://pmc.ncbi.nlm.nih.gov/articles/PMC9905678/',
    category: 'curriculum',
    relevance: 'Empirical evidence that progressive complexity improves RL training efficiency. Supports our phase-based curriculum design.',
  },
  {
    id: 'Auto-Curriculum',
    title: 'Automatic Curriculum Learning for Autonomous Driving',
    venue: 'arXiv',
    year: '2025',
    url: 'https://arxiv.org/html/2505.08264v1',
    category: 'curriculum',
    relevance: 'Automated curriculum generation for AD training. Future direction: replace manual threshold tuning (P-002) with adaptive curriculum.',
  },
  // --- Technical Analysis ---
  {
    id: 'SOTIF-Navigation',
    title: 'Navigating SOTIF (ISO 21448) for Autonomous Driving',
    venue: 'Automotive IQ',
    year: '2024',
    url: 'https://www.automotive-iq.com/functional-safety/articles/navigating-sotif-iso-21448-and-ensuring-safety-in-autonomous-driving',
    category: 'analysis',
    relevance: 'Practical guide to SOTIF implementation. Informed our quadrant-to-phase mapping strategy.',
  },
  {
    id: 'DCAS-Navigation',
    title: 'Understanding DCAS and UN R171',
    authors: 'Applied Intuition',
    venue: 'Applied Intuition Blog',
    year: '2024',
    url: 'https://www.appliedintuition.com/blog/navigating-dcas-regulations',
    category: 'analysis',
    relevance: 'DCAS regulation breakdown. Key parameters adopted for Phase I cut-in/cut-out scenario design.',
  },
  {
    id: 'SOTIF-Acceptance',
    title: 'Demystifying SOTIF Acceptance Criteria and Validation Targets',
    venue: 'SRES AI',
    year: '2024',
    url: 'https://sres.ai/autonomous-systems/demystifying-sotif-acceptance-criteria-and-validation-targets-part-2/',
    category: 'analysis',
    relevance: 'SOTIF residual risk acceptance criteria. Defines our Phase M validation targets.',
  },
];

export const referenceCategories: { key: Reference['category']; label: string; icon: string }[] = [
  { key: 'standard', label: 'Safety Standards', icon: 'S' },
  { key: 'academic', label: 'Academic Papers', icon: 'A' },
  { key: 'control', label: 'Control Theory', icon: 'C' },
  { key: 'curriculum', label: 'Curriculum Learning', icon: 'L' },
  { key: 'analysis', label: 'Technical Analysis', icon: 'T' },
];

// ---------- Stats ----------
export const stats = {
  maxReward: 770,
  totalPhases: 30,
  completedPhases: 11,
  failedAttempts: 18,
  observationDim: 280,
  parallelAreas: 16,
  successRate: '11/30',
  collisionRateTarget: '< 5%',
  totalSteps: '158M',
  totalPolicies: 22,
  techStack: ['Unity 6', 'ML-Agents 4.0', 'PyTorch 2.3+', 'PPO', 'ROS2 Humble'],
  hardware: {
    gpu: 'RTX 4090 (24GB VRAM)',
    ram: '128GB DDR5',
    storage: '4TB NVMe SSD',
    os: 'Windows 11',
  },
};
