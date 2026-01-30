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
    id: 'phase-a',
    name: 'Phase A',
    subtitle: 'Dense Overtaking',
    reward: 937,
    status: 'success',
    tags: ['PPO', 'Dense Reward', 'Curriculum'],
    description:
      'Foundation phase establishing core driving behavior with dense reward shaping. Agent learns lane keeping, speed control, and basic overtaking on a straight single-lane road.',
    observations: '242D',
    steps: '2.5M',
    keyInsight:
      'Dense reward shaping with 7 reward components enabled stable learning from scratch. Curriculum learning (1-4 NPCs) prevented early catastrophic failures.',
  },
  {
    id: 'phase-b-v1',
    name: 'Phase B v1',
    subtitle: 'Decision Learning',
    reward: -108,
    status: 'failed',
    tags: ['PPO', 'NPC Interaction', 'Fresh Start'],
    description:
      'First attempt at decision learning. Training from scratch with 7 simultaneous hyper-parameter changes and immediate 2-NPC exposure caused reward collapse at 1.8M steps.',
    observations: '242D',
    steps: '2.5M',
    keyInsight:
      'Excessive speed penalty (-0.1/step) taught agent to stop. Multiple variables changed simultaneously made root-cause identification impossible.',
    version: 1,
  },
  {
    id: 'phase-b-v2',
    name: 'Phase B v2',
    subtitle: 'Decision Learning',
    reward: 994,
    status: 'success',
    tags: ['PPO', 'Checkpoint Transfer', 'Variable Isolation'],
    description:
      'Fixed version: restored Phase A hyper-parameters (variable isolation), used Phase A checkpoint (capability-based selection), reduced speed penalty by 80%, and added staggered 0-1-2-3 NPC curriculum.',
    observations: '242D',
    steps: '3.0M',
    keyInsight:
      'Checkpoint transfer from Phase A provided critical bootstrap. Single-variable changes enabled clear cause-effect debugging.',
    version: 2,
    parentId: 'phase-b-v1',
  },
  {
    id: 'phase-c',
    name: 'Phase C',
    subtitle: 'Multi-NPC Generalization',
    reward: 1086,
    status: 'success',
    tags: ['PPO', '4-8 NPCs', 'Curriculum'],
    description:
      'Scaled from 4 to 8 NPCs with curriculum learning. Agent generalizes overtaking behavior to dense traffic scenarios with varying NPC speeds and behaviors.',
    observations: '242D',
    steps: '3.6M',
    keyInsight:
      'Staggered curriculum thresholds prevented simultaneous parameter transitions. Highest reward across all phases (+1,086).',
  },
  {
    id: 'phase-d-v1',
    name: 'Phase D v1',
    subtitle: 'Lane Observation 254D',
    reward: -2156,
    status: 'failed',
    tags: ['PPO', '254D', 'Simultaneous Transition'],
    description:
      'First attempt at lane observation integration. Three curriculum parameters advanced at the same threshold (~400K steps), causing reward crash from +406 to -4,825 within 20K steps.',
    observations: '254D',
    steps: '6M',
    keyInsight:
      'Simultaneous curriculum shock: 3 parameters transitioned at once, invalidating the learned policy. Recovery was impossible within training budget.',
    version: 1,
  },
  {
    id: 'phase-d-v2',
    name: 'Phase D v2',
    subtitle: 'Lane Observation 254D',
    reward: null,
    status: 'in_progress',
    tags: ['PPO', '254D', 'Staggered Thresholds'],
    description:
      'Fixed version: staggered thresholds (200K/300K/350K) ensuring one parameter advances at a time. Extended to 10M steps for recovery margin.',
    observations: '254D',
    steps: '10M',
    keyInsight:
      'Staggered curriculum (P-002) applied: num_active_npcs at 200K, speed_zone_count at 300K, npc_speed_variation at 350K.',
    version: 2,
    parentId: 'phase-d-v1',
  },
  {
    id: 'phase-e',
    name: 'Phase E',
    subtitle: 'Curved Roads',
    reward: 931,
    status: 'success',
    tags: ['PPO', 'Curvature', 'Banked Roads'],
    description:
      'Introduced curved road geometry with variable radius turns and banked surfaces. Agent learns speed adaptation for curves and maintains lane discipline through turns.',
    observations: '242D',
    steps: '3.0M',
    keyInsight:
      'Curvature observations critical for safe cornering. Agent learned to decelerate before curves and accelerate on exits, matching human driving patterns.',
  },
  {
    id: 'phase-f',
    name: 'Phase F',
    subtitle: 'Multi-Lane',
    reward: 988,
    status: 'success',
    tags: ['PPO', '2-Lane', 'Center Line Rules'],
    description:
      'Expanded road to 2 lanes with center line rules. Agent learns proper lane selection, lane change timing, and center line discipline while navigating traffic.',
    observations: '242D',
    steps: '3.5M',
    keyInsight:
      'Center line violation penalty was essential for teaching lane discipline. Without it, agent treated both lanes as single wide road.',
  },
  {
    id: 'phase-g',
    name: 'Phase G',
    subtitle: 'Intersection',
    reward: 492,
    status: 'in_progress',
    tags: ['PPO', 'T/Cross/Y Junction', 'Turn Decision'],
    description:
      'Most complex phase with T-intersections, cross-roads, and Y-junctions. Agent must learn turn decisions, right-of-way, and intersection navigation.',
    observations: '242D',
    steps: '5M',
    keyInsight:
      'Intersection navigation requires fundamentally different policy from highway driving. Lower reward reflects dramatically increased complexity.',
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
    sourcePhase: 'Phase B v1 -> v2',
    status: 'verified',
    matchingStandard: 'Controlled Experiment Design',
    description:
      'Change only one variable at a time. Phase B v1 changed 7 hyper-parameters simultaneously, making root-cause identification impossible.',
    failContext:
      'Phase B v1: 7 hyper-parameter changes + immediate NPC exposure -> reward collapsed to -108',
    fixContext:
      'Phase B v2: restored Phase A settings, changed only curriculum -> reward reached +994',
  },
  {
    id: 'P-002',
    name: 'Staggered Curriculum',
    nameEn: 'Staggered Curriculum Transitions',
    sourcePhase: 'Phase D v1 -> v2',
    status: 'in_progress',
    matchingStandard: 'SOTIF Incremental Complexity',
    description:
      'Curriculum parameters must transition at different thresholds. Simultaneous transitions cause catastrophic forgetting.',
    failContext:
      'Phase D v1: 3 parameters at same threshold (400K) -> reward crashed from +406 to -4,825',
    fixContext:
      'Phase D v2: staggered thresholds (200K/300K/350K) -> one parameter at a time',
  },
  {
    id: 'P-003',
    name: 'Capability-Based Checkpoint',
    nameEn: 'Capability-Based Checkpoint Selection',
    sourcePhase: 'Phase B v1 -> v2',
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
    sourcePhase: 'Phase B v1 -> v2',
    status: 'verified',
    matchingStandard: 'Reward Shaping Theory',
    description:
      'Penalties must be conservative. Excessive penalties teach avoidance behaviors (e.g., stopping) instead of desired behaviors.',
    failContext:
      'Phase B v1: speedUnderPenalty -0.1/step taught agent to stop moving entirely',
    fixContext:
      'Phase B v2: reduced to -0.02/step (80% reduction) -> normal learning resumed',
  },
  {
    id: 'P-009',
    name: 'Observation Coupling',
    nameEn: 'Observation Space Dimension Coupling',
    sourcePhase: 'Phase D v1',
    status: 'in_progress',
    matchingStandard: 'Neural Network Architecture Design',
    description:
      '242D to 254D observation space change breaks checkpoint compatibility. Fresh training required, losing accumulated knowledge.',
    failContext:
      'Phase D: dimension change from 242D to 254D required complete retraining',
    fixContext:
      'Must train from scratch with new observation space; cannot warm-start from 242D checkpoint',
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
    phases: ['Phase A', 'Phase B', 'Phase C'],
    examples: ['Straight road', 'Constant speed', 'Clear weather'],
    strategy: 'Standard curriculum learning',
  },
  {
    id: 2,
    label: 'Known Unsafe',
    description: 'Targeted verification. Known hazardous scenarios with planned mitigation.',
    phases: ['Phase D', 'Phase E', 'Phase F', 'Phase G'],
    examples: ['Sharp curves', 'Cut-in events', 'Multi-lane merging'],
    strategy: 'Specialized scenario training',
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
    thisProject: 'Ground Truth Vector (242D)',
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
    thisProject: 'RL Policy -> steering/accel',
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

// ---------- Stats ----------
export const stats = {
  maxReward: 1086,
  totalPhases: 7,
  completedPhases: 5,
  failedAttempts: 3,
  observationDim: 254,
  parallelAreas: 16,
  successRate: '5/7',
  collisionRateTarget: '< 5%',
  totalSteps: '24.1M',
  totalPolicies: 4,
  techStack: ['Unity 6', 'ML-Agents 3.0', 'PyTorch 2.0+', 'PPO', 'ROS2 Humble'],
  hardware: {
    gpu: 'RTX 4090 (24GB VRAM)',
    ram: '128GB DDR5',
    storage: '4TB NVMe SSD',
    os: 'Windows 11',
  },
};
