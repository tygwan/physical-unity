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
    version: 2,
    parentId: 'phase-g-v1',
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
  maxReward: 2113,
  totalPhases: 16,
  completedPhases: 7,
  failedAttempts: 9,
  observationDim: 260,
  parallelAreas: 16,
  successRate: '7/16',
  collisionRateTarget: '< 5%',
  totalSteps: '91.4M',
  totalPolicies: 11,
  techStack: ['Unity 6', 'ML-Agents 4.0', 'PyTorch 2.3+', 'PPO', 'ROS2 Humble'],
  hardware: {
    gpu: 'RTX 4090 (24GB VRAM)',
    ram: '128GB DDR5',
    storage: '4TB NVMe SSD',
    os: 'Windows 11',
  },
};
