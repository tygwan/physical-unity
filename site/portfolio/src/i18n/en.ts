export const en = {
  meta: {
    title: 'AD ML Platform | Autonomous Driving Research Portfolio',
    description:
      'Autonomous Driving ML Platform - RL/IL Motion Planning Research with Unity + ML-Agents + PyTorch',
  },
  nav: {
    brand: 'ML Platform',
    research: 'Research',
    phases: 'Phases',
    architecture: 'Architecture',
    insights: 'Insights',
    github: 'GitHub',
    navigation: 'Navigation',
    builtWith: 'Built With',
  },
  footer: {
    description:
      'Autonomous driving motion planning research using reinforcement learning with curriculum-based progressive training.',
    bottom: 'Autonomous Driving ML Platform -- RL/IL Motion Planning Research',
  },
  hero: {
    badge: 'RL Motion Planning Research',
    titleLine1: 'Autonomous Driving',
    titleLine2: 'ML Platform',
    subtitlePre: 'End-to-end motion planning using ',
    subtitleHL1: 'PPO reinforcement learning',
    subtitleMid: ' with curriculum-based training across 7 progressive phases. Validated against ',
    subtitleHL2: 'ISO 21448 (SOTIF)',
    subtitlePost: ' safety standards.',
    viewResearch: 'View Research',
    phaseRoadmap: 'Phase Roadmap',
    peakReward: 'Peak Reward',
    phasesComplete: 'Phases Complete',
    collisionTarget: 'Collision Target',
    safetyBenchmark: 'safety benchmark',
    failedAttempts: 'failed attempts',
    overallProgress: 'Overall Progress',
    complete: 'complete',
    inProgress: 'in progress',
    failed: 'failed',
  },
  stats: {
    peakReward: 'Peak Reward',
    peakRewardSub: 'Phase I: Curves+NPC',
    phasesComplete: 'Phases Complete',
    failedAttempts: 'failed attempts',
    observationSpace: 'Observation Space',
    observationSub: 'ego + route + NPC + lane',
    parallelEnvs: 'Parallel Envs',
    parallelSub: 'simultaneous training',
    collisionRate: 'Collision Rate',
    collisionSub: 'safety target',
    policiesDiscovered: 'Policies Discovered',
    policiesSub: 'from trial-error',
  },
  research: {
    heading: 'Research',
    sub: 'Safety frameworks, competitive analysis, and policy discoveries from iterative training',
    sotifTitle: 'SOTIF Framework (ISO 21448)',
    sotifDesc:
      'ISO 21448 defines Safety of the Intended Functionality -- addressing risks from functional limitations rather than system faults. The 4-quadrant model classifies scenarios by knowledge and safety status.',
    sotifGoal:
      'Goal: shrink Quadrant 2 and 3 (unsafe areas) while expanding Quadrant 1 (known safe). Each training phase systematically moves scenarios from Unknown to Known.',
    known: 'KNOWN',
    unknown: 'UNKNOWN',
    teslaTitle: 'Tesla FSD Gap Analysis',
    teslaDesc:
      'Component-by-component comparison against Tesla FSD 12/13 architecture. This project is a research platform, not a commercial product -- the gap is intentional and informative.',
    vs: 'vs',
    thComponent: 'Component',
    thTesla: 'Tesla FSD',
    thProject: 'This Project',
    thGap: 'Gap',
    thFeasibility: 'Feasibility',
    teslaInsightLabel: 'Key Insight:',
    teslaInsight:
      "Tesla FSD is a commercial product backed by 4M+ fleet vehicles and Dojo supercomputer (1.1 ExaFLOPS). This project operates on a single RTX 4090 (82.6 TFLOPS) -- a 13-million-fold compute gap. The value lies in algorithm validation, not scale replication. Vehicle Control is the only component at parity, achieved through direct RL policy output.",
    policyTitle: 'Policy Discovery Log',
    policyDesc:
      'Design principles discovered through trial-and-error training. Each failure produced actionable policies that converge with established standards.',
    verified: 'VERIFIED',
    policyInProgress: 'IN PROGRESS',
    planned: 'PLANNED',
    fail: 'FAIL',
    fix: 'FIX',
    matchingStandard: 'Matching Standard:',
    referencesTitle: 'References & Academic Research',
    referencesDesc:
      'Standards, papers, and technical analyses that inform the training curriculum, reward design, and safety validation framework.',
    openLink: 'Open',
    convergenceInsightLabel: 'Convergence Insight:',
    convergenceInsight:
      "Our empirical policies (P-001 through P-011) discovered through trial-and-error training independently converge with established international safety standards. P-002 (Staggered Curriculum) maps to SOTIF's incremental complexity principle. P-005/P-006 (planned) directly implement UN R157/R171 numerical limits. This convergence validates that RL-based policy learning naturally rediscovers safety engineering principles.",
  },
  phases: {
    heading: 'Training Phases',
    sub: 'Progressive curriculum learning across 7 phases. Failed attempts are shown as branched paths -- each failure produced design policies reused in later phases.',
    rewardEvolution: 'Reward Evolution (all attempts)',
    success: 'Success',
    failed: 'Failed',
    inProgressLabel: 'In Progress',
    active: 'ACTIVE',
    phaseDetails: 'Phase Details',
    failureCause: 'Failure Cause',
    keyInsight: 'Key Insight',
    obs: 'Obs',
    steps: 'Steps',
    training: 'Training...',
  },
  architecture: {
    heading: 'Architecture',
    sub: 'System design, training pipeline, and hardware specifications',
    trainingPipeline: 'Training Pipeline',
    thisProject: 'This Project',
    teslaFsd: 'Tesla FSD 12/13',
    hardwareEnv: 'Hardware Environment',
    // Tech items
    simulation: 'Simulation',
    framework: 'Framework',
    mlEngine: 'ML Engine',
    rlAlgorithm: 'RL Algorithm',
    middleware: 'Middleware',
    languages: 'Languages',
    compute: 'Compute',
    monitoring: 'Monitoring',
    // Architecture items
    observation: 'Observation',
    observationValue: '280D vector',
    observationDesc: 'ego state, route, NPC, lane, intersection, signal, pedestrian',
    actionSpace: 'Action Space',
    actionValue: 'Continuous 2D',
    actionDesc: 'acceleration [-4, 2] m/s2, steering [-0.5, 0.5] rad',
    reward: 'Reward',
    rewardValue: '7 components',
    rewardDesc: 'progress, speed, lane, overtaking, violation, jerk, time',
    trainingLabel: 'Training',
    trainingValue: '16 parallel',
    trainingDesc: 'simultaneous environments with curriculum learning',
    // Pipeline steps
    unitySimulation: 'Unity Simulation',
    parallelEnvs: '16 parallel envs',
    mlAgentsBridge: 'ML-Agents Bridge',
    grpcProtocol: 'gRPC protocol',
    pytorchPpo: 'PyTorch PPO',
    gpuTraining: 'GPU training',
    onnxExport: 'ONNX Export',
    modelPackaging: 'model packaging',
    unityInference: 'Unity Inference',
    realtimeControl: 'real-time control',
    // System layers
    input: 'Input',
    perception: 'Perception',
    prediction: 'Prediction',
    planning: 'Planning',
    control: 'Control',
  },
  insights: {
    heading: 'Insights',
    sub: 'Lessons learned from 7 phases of RL training -- what worked, what failed, and how failures converge with safety standards',
    whatWorked: 'What Worked',
    whatFailed: 'What Failed',
    // What worked items
    denseReward: 'Dense Reward Shaping',
    denseRewardDetail:
      '7 reward components (progress, speed, lane keeping, overtaking, violation, jerk, time) provided clear learning signals from the start.',
    curriculum: 'Curriculum Learning',
    curriculumDetail:
      'Progressive difficulty increase (NPCs, speed zones, goal distance) enabled stable training. Staggered thresholds prevented simultaneous transitions.',
    checkpoint: 'Checkpoint Transfer',
    checkpointDetail:
      'Warm-starting from previous phases provided critical bootstrap. Phase B v2 succeeded with Phase A checkpoint where v1 (fresh start) failed.',
    parallel: '16 Parallel Environments',
    parallelDetail:
      'Simultaneous training across 16 areas provided diverse experience and faster convergence. Essential for PPO on-policy learning.',
    // What failed items
    curriculumShock: 'Simultaneous Curriculum Shock',
    curriculumShockDetail:
      'Phase D v1: 3 parameters advanced at the same threshold (reward ~400), causing reward crash from +406 to -4,825. Fixed with staggered thresholds in v2.',
    freshStart: 'Fresh Start with Complex Obs',
    freshStartDetail:
      'Phase B v1: Training from scratch with NPC interaction collapsed at 1.8M steps. Complex observations need warm-start from simpler policy.',
    excessivePenalty: 'Excessive Penalty Design',
    excessivePenaltyDetail:
      'Phase B v1: speedUnderPenalty -0.1/step taught agent to stop entirely. Reduced to -0.02 (80% decrease) in v2 to restore learning.',
    obsDimChange: 'Observation Dimension Change',
    obsDimChangeDetail:
      '242D to 254D transition breaks checkpoint compatibility. Fresh training required for lane observation integration, losing accumulated knowledge.',
    // Convergence
    convergenceTitle: 'Experiential Discovery to Standard Convergence',
    convergenceDesc:
      'Policies discovered through trial-and-error independently converge with established safety standards and best practices.',
    // Standards
    standardsTitle: 'Safety Standards Integration',
    sotifStandard: 'SOTIF (ISO 21448)',
    sotifStandardDesc:
      '4-quadrant safety model. Functional Insufficiency analysis for each observation dimension. Systematic Known/Unknown Safe/Unsafe classification.',
    unR171: 'UN R171 (DCAS)',
    unR171Desc:
      'Cut-in/cut-out test parameters: TTC 1.5-5.0s, max deceleration -7.0 m/s2, jerk limit 3.0 m/s3. Level 2 driving assistance validation.',
    unR157: 'UN R157 (ALKS)',
    unR157Desc:
      'Lateral acceleration limit 0.3g, curvature rate dk/ds < 0.1/m2, crosstrack error < 0.3m. Level 3 automated lane keeping standards.',
  },
};

export type Translations = typeof en;
