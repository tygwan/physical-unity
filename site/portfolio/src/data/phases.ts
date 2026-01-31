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
    id: 'phase-f',
    name: 'Phase F v2',
    subtitle: 'Multi-Lane',
    reward: null,
    status: 'in_progress',
    tags: ['PPO', '254D', 'Multi-Lane', 'Center Line', 'P-002 Staggered', 'P-011'],
    description:
      'Multi-lane (1→4 lanes) with center line enforcement on correct PhaseF_MultiLane scene (11.5m road). 9 curriculum parameters with P-002 staggered thresholds.',
    observations: '254D',
    steps: '6M',
    keyInsight:
      'Correct scene verified via P-011. Road width 11.5m supports up to 3 lanes. Lane changes via steering with Korean traffic rules (drive on right).',
    version: 2,
    parentId: 'phase-f-v1',
  },
  {
    id: 'phase-g',
    name: 'Phase G',
    subtitle: 'Intersections',
    reward: null,
    status: 'planned',
    tags: ['PPO', 'T/Cross Junction', 'Right-of-way'],
    description:
      'T-intersections, cross-roads, and Y-junctions. Agent must learn turn decisions, right-of-way rules, and intersection navigation.',
    observations: '254D+',
    steps: 'TBD',
    keyInsight:
      'Pending Phase F completion. Most complex planned phase requiring fundamentally different policy from highway driving.',
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
    examples: ['Lane observation', 'Sharp curves', 'Multi-lane merging'],
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
  totalPhases: 11,
  completedPhases: 5,
  failedAttempts: 5,
  observationDim: 254,
  parallelAreas: 16,
  successRate: '5/11',
  collisionRateTarget: '< 5%',
  totalSteps: '40.3M',
  totalPolicies: 7,
  techStack: ['Unity 6', 'ML-Agents 4.0', 'PyTorch 2.3+', 'PPO', 'ROS2 Humble'],
  hardware: {
    gpu: 'RTX 4090 (24GB VRAM)',
    ram: '128GB DDR5',
    storage: '4TB NVMe SSD',
    os: 'Windows 11',
  },
};
