using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using ADPlatform.Environment;

namespace ADPlatform.Agents
{
    /// <summary>
    /// End-to-End Driving Agent for RL training.
    ///
    /// Observation Space (Phase-dependent):
    ///   Phase A/B/C (242D):
    ///   - Ego state (8D): x, y, vx, vy, cos_h, sin_h, ax, ay
    ///   - Ego history (40D): 5 past steps x 8D
    ///   - Surrounding agents (160D): 20 agents x 8 features
    ///   - Route info (30D): 10 waypoints x 3 (x, y, dist)
    ///   - Speed info (4D): current_speed_norm, speed_limit_norm, speed_ratio, next_limit_norm
    ///
    ///   Phase D/E/F (254D = 242D + 12D):
    ///   - + Lane info (12D): left_type(4), right_type(4), left_dist, right_dist, left_crossing, right_crossing
    ///
    ///   Phase G (260D = 254D + 6D):
    ///   - + Intersection info (6D): type(4), distance(1), turn_direction(1)
    ///
    ///   Phase J (268D = 260D + 8D):
    ///   - + Traffic signal (8D): state(4), dist_to_stop(1), time_remaining(1), should_stop(1), stopped_at_line(1)
    ///
    ///   Phase L (280D = 268D + 12D):
    ///   - + Pedestrian info (10D): 2 nearest pedestrians x 5 (rel_x, rel_z, vel_x, vel_z, dist)
    ///   - + Crosswalk info (2D): has_crosswalk_ahead(1), distance_to_crosswalk(1)
    ///
    /// Action Space (2D continuous):
    ///   - Steering: [-0.5, 0.5] rad
    ///   - Acceleration: [-4.0, 2.0] m/s²
    ///
    /// Reward Design:
    ///   - Progress toward goal
    ///   - Speed compliance (+0.3 for 80-100% of limit)
    ///   - Speed over limit (-0.5 ~ -3.0 progressive)
    ///   - Speed under limit (-0.1 for &lt;50% of limit)
    ///   - Collision penalty
    ///   - Off-road penalty
    ///   - Comfort (jerk) penalty
    ///   - Goal reached bonus
    /// </summary>
    public class E2EDrivingAgent : Agent
    {
        [Header("Vehicle Physics")]
        public float maxSpeed = 30f;            // m/s (~108 km/h)
        public float maxAcceleration = 4f;      // m/s²
        public float maxBraking = 8f;           // m/s²
        public float maxSteeringAngle = 30f;    // degrees
        public float vehicleMass = 1500f;       // kg
        public float wheelBase = 2.7f;          // meters

        [Header("Observation Settings")]
        public int historySteps = 5;            // Past timesteps to record
        public int maxAgents = 20;              // Max surrounding vehicles
        public float agentDetectionRange = 50f; // meters
        public int numWaypoints = 10;           // Route waypoints
        public float waypointSpacing = 10f;     // meters between waypoints

        [Header("Observation Space Control")]
        [Tooltip("Phase A/B/C: false (242D), Phase D/E/F: true (254D)")]
        public bool enableLaneObservation = false;      // +12D when enabled
        [Tooltip("Phase G+: true (260D)")]
        public bool enableIntersectionObservation = false;  // +6D when enabled
        [Tooltip("Phase J+: true (268D)")]
        public bool enableTrafficSignalObservation = false;  // +8D when enabled
        [Tooltip("Phase L+: true (280D)")]
        public bool enablePedestrianObservation = false;     // +12D when enabled

        [Header("Reward Weights")]
        public float progressWeight = 1.0f;
        public float collisionPenalty = -5f;
        public float nearCollisionPenalty = -1.5f;
        public float offRoadPenalty = -5f;
        public float jerkPenalty = -0.1f;
        public float goalBonus = 10f;
        public float timePenalty = -0.05f;

        [Header("Speed Policy Rewards")]
        public float speedComplianceReward = 0.3f;    // 80-100% of limit
        public float speedOverPenaltyScale = -0.5f;   // Progressive: -0.5 per 10% over
        public float speedOverMaxPenalty = -3.0f;      // Cap at -3.0
        public float speedUnderPenalty = -0.1f;        // Below 50% of limit
        public float speedZoneTransitionReward = 0.2f; // Smooth transition bonus

        [Header("Lane Keeping")]
        public float headingAlignmentReward = 0.02f;   // Reward for heading aligned with road
        public float lateralDeviationPenalty = -0.02f;  // Penalty for off-center driving
        public float maxLateralDeviation = 2.5f;        // meters from center before penalty

        [Header("Center Line Rules (Phase F)")]
        public float wrongWayPenalty = -5f;             // Penalty for crossing center line (same as off-road)
        public float centerLineCrossingPenalty = -0.5f; // Per-step penalty when on wrong side

        [Header("Traffic Signal Rules (Phase J)")]
        public float redLightViolationPenalty = -5f;    // Crossing stop line on red
        public float properRedStopReward = 0.3f;        // Per-step: stopped correctly at red
        public float yellowCautionPenalty = -2f;         // Entering intersection on yellow from far
        public float unnecessaryStopPenalty = -0.1f;     // Per-step: stopped at green for too long
        public float unnecessaryStopTimeout = 2f;        // seconds at green before penalty
        public bool terminateOnRedViolation = true;      // End episode on red light violation

        [Header("Pedestrian References (Phase L)")]
        public PedestrianController[] pedestrians;
        public float pedestrianDetectionRange = 50f;

        [Header("Pedestrian Rewards (Phase L)")]
        public float pedestrianCollisionPenalty = -10f;
        public float crosswalkYieldReward = 0.2f;
        public float crosswalkSpeedPenaltyScale = -0.3f;

        public enum TrainingVersion { v10g, v11, v12 }

        [Header("Training Version")]
        [Tooltip("v10g: Lane keeping + following, v11: Sparse overtake, v12: Dense overtake")]
        public TrainingVersion trainingVersion = TrainingVersion.v12;

        [Header("Following Behavior")]
        public float leadVehicleDetectRange = 40f;    // meters ahead to check
        public float safeFollowingDistance = 15f;      // meters (safe gap)

        [Header("Following Reward (v10g/v11 only)")]
        [Tooltip("v10g/v11: Reward for maintaining safe following distance")]
        public float followingBonus = 0.3f;           // v10g/v11: per-step when safely following
        [Tooltip("v11: Only apply followingBonus when NPC > this ratio of speedLimit")]
        public float followingBonusSpeedThreshold = 0.7f;  // v11: gated following bonus

        [Header("Overtaking (v11 - Sparse Reward)")]
        [Tooltip("v11: One-time bonus when overtake fully completed")]
        public float overtakePassBonus = 3.0f;        // v11: sparse, one-time
        [Tooltip("v11: Per-step bonus when beside NPC during overtake")]
        public float overtakeSpeedBonus = 0.15f;      // v11: per-step beside NPC

        [Header("Overtaking (v12 - Dense Reward Strategy)")]
        public float overtakeInitiateBonus = 0.5f;    // One-time: lane change started
        public float overtakeBesideBonus = 0.2f;      // Per-step: maintaining speed beside NPC
        public float overtakeAheadBonus = 1.0f;       // One-time: passed NPC (ego ahead)
        public float overtakeCompleteBonus = 2.0f;    // One-time: returned to lane after pass
        public float stuckBehindPenalty = -0.1f;      // Per-step: stuck behind slow NPC
        public float stuckBehindTimeout = 3.0f;       // seconds before penalty kicks in
        public float overtakeSlowLeadThreshold = 0.7f; // Below this ratio of speedLimit, NPC is "slow"
        public float overtakeDetectWidth = 3.0f;       // SphereCast radius for lead detection

        [Header("Debug")]
        public bool debugRewards = true;          // Enable detailed reward logging
        public int debugLogInterval = 100;        // Log every N steps

        [Header("Environment")]
        public Transform goalTarget;
        public Transform[] routeWaypoints;
        public float roadWidth = 7f;           // meters (lane width)
        public float maxEpisodeDistance = 500f;
        public LayerMask vehicleLayer;
        public LayerMask roadLayer;

        [Header("Scene")]
        public DrivingSceneManager sceneManager;
        public WaypointManager waypointManager;
        public TrafficLightController trafficLight;

        // Internal state
        private Rigidbody rb;
        private Vector3 startPosition;
        private Quaternion startRotation;
        private float previousDistanceToGoal;
        private int currentWaypointIndex;

        /// <summary>
        /// P-028: Allow external update of waypoint index for grid mode.
        /// TestFieldManager tracks per-agent waypoint progress and synchronizes here.
        /// </summary>
        public int CurrentWaypointIndex
        {
            get => currentWaypointIndex;
            set => currentWaypointIndex = value;
        }

        // History buffer (ring buffer)
        private float[][] egoHistory;
        private int historyIndex;
        private bool historyFull;

        // Previous frame state (for jerk calculation)
        private float prevAcceleration;
        private float prevSteering;
        private Vector3 prevVelocity;

        // Episode stats
        private float episodeReward;
        private int episodeSteps;
        private float totalDistance;
        private Vector3 prevPosition;

        // Collision tracking
        private float lastCollisionTime;
        private int episodeCollisions;
        private const float COLLISION_COOLDOWN = 1.0f;  // 1 second between penalties
        private const int MAX_COLLISIONS_PER_EPISODE = 3;  // End episode after 3 hits

        // Internal speed tracking (bypasses physics friction)
        private float currentSpeed = 0f;

        // Speed policy state
        private float currentSpeedLimit;
        private float prevSpeedLimit;

        // Overtaking state (v12: 5-phase dense reward)
        private enum OvertakingPhase { None, Approaching, Beside, Ahead, LaneReturn }
        private OvertakingPhase overtakingPhase = OvertakingPhase.None;
        private Transform overtakeTarget = null;
        private int overtakeCount = 0;         // NPCs passed this episode
        private bool isOvertaking = false;     // Flag for lane keeping suspension
        private float stuckBehindTimer = 0f;   // Time spent stuck behind slow NPC
        private float laneReturnStartX = 0f;   // X position when lane return started

        // Lane detection state (v13 - Phase C)
        [Header("Lane Marking Detection")]
        public LayerMask laneMarkingLayer;           // Layer for lane marking colliders
        public float laneDetectDistance = 5f;        // Raycast distance to left/right
        public float laneDetectHeight = 0.5f;        // Raycast origin height

        [Header("Lane Violation Penalties (v13)")]
        public float whiteSolidCrossPenalty = -2.0f;    // 백색 실선 위반
        public float yellowDashedCrossPenalty = -3.0f;  // 황색 점선 위반
        public float yellowSolidCrossPenalty = -5.0f;   // 황색 실선 위반
        public float doubleYellowCrossPenalty = -10.0f; // 이중 황색 위반 (episode 종료)
        public bool terminateOnDoubleYellow = true;     // 이중 황색 위반 시 종료

        // Lane state (detected via raycast)
        private LaneMarkingType leftLaneType = LaneMarkingType.None;
        private LaneMarkingType rightLaneType = LaneMarkingType.None;
        private float leftLaneDist = 1f;   // Normalized: 0=touching, 1=far
        private float rightLaneDist = 1f;
        private bool leftLaneCrossing = false;   // Currently crossing left lane
        private bool rightLaneCrossing = false;  // Currently crossing right lane
        private float lastLaneViolationTime = -1f;
        private const float LANE_VIOLATION_COOLDOWN = 0.5f;

        // Traffic signal state (Phase J)
        private bool hasPassedStopLine = false;          // Track if agent already past stop line
        private bool wasPastStopLineAtRedStart = false;  // Was agent already past stop line when Red began?
        private TrafficLightController.LightState prevSignalState = TrafficLightController.LightState.None;
        private float stoppedAtGreenTimer = 0f;          // Time spent unnecessarily stopped at green
        private float dbgTrafficSignalReward = 0f;       // Debug: traffic signal reward component

        // Pedestrian state (Phase L)
        private float dbgPedestrianReward = 0f;
        private float crosswalkZ = 75f;  // Default crosswalk Z position (before intersection at 93m)
        private float yieldRewardAccumulated = 0f;     // Cumulative yield reward this episode
        private float maxYieldRewardPerEpisode = 2f;   // Cap: max yield reward per episode (prevents farming)
        private float yieldDuration = 0f;              // How long agent has been yielding this encounter
        private float maxYieldDuration = 8f;           // Max yield time before penalty (seconds)
        private bool hasYieldedThisEncounter = false;   // Track if yield was already rewarded for current crosswalk
        private float crosswalkStopTimer = 0f;         // Total time spent slow (<2 m/s) near crosswalk this episode
        private float maxCrosswalkStopDuration = 15f;  // Episode terminates after this many seconds near crosswalk

        // Speed reward cap (P-027: anti-speed-farming)
        private float speedRewardAccumulated = 0f;
        private float maxSpeedRewardPerEpisode = 200f;

        // P-028: Inference mode flag (skip training-only termination checks)
        private bool isInferenceMode = false;

        // Debug: reward component tracking
        private float dbgProgressReward = 0f;
        private float dbgSpeedReward = 0f;
        private float dbgLaneKeepReward = 0f;
        private float dbgOvertakeReward = 0f;
        private float dbgLaneViolationPenalty = 0f;
        private float dbgStuckPenalty = 0f;
        private float dbgCollisionCount = 0f;
        private string dbgEpisodeEndReason = "";

        public override void Initialize()
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }

            rb.mass = vehicleMass;
            rb.linearDamping = 0f;  // We manage speed internally
            rb.angularDamping = 2f;
            rb.useGravity = false;  // Vehicle sits on ground, no need for gravity
            rb.constraints = RigidbodyConstraints.FreezeRotationX |
                             RigidbodyConstraints.FreezeRotationZ |
                             RigidbodyConstraints.FreezePositionY;  // Lock Y position

            // Remove friction from collider (prevents PhysX from zeroing velocity)
            var col = GetComponent<Collider>();
            if (col != null)
            {
                var frictionless = new PhysicsMaterial("Frictionless");
                frictionless.staticFriction = 0f;
                frictionless.dynamicFriction = 0f;
                frictionless.frictionCombine = PhysicsMaterialCombine.Minimum;
                col.material = frictionless;
            }

            // Disable SimpleVehicleController (interferes with RL training)
            var simpleController = GetComponent<SimpleVehicleController>();
            if (simpleController != null)
                simpleController.enabled = false;

            // Force values (override Inspector serialization) - v12
            headingAlignmentReward = 0.02f;
            lateralDeviationPenalty = -0.02f;
            overtakeInitiateBonus = 0.5f;
            overtakeBesideBonus = 0.2f;
            overtakeAheadBonus = 1.0f;
            overtakeCompleteBonus = 2.0f;
            stuckBehindPenalty = -0.1f;
            stuckBehindTimeout = 3.0f;
            overtakeSlowLeadThreshold = 0.7f;
            overtakeDetectWidth = 3.0f;
            timePenalty = -0.05f;  // P-027: force value (prevent Inspector override)

            // P-027: Limit per-episode steps to prevent infinite farming
            // But allow unlimited steps for inference-only mode (Phase M test field)
            var bp = GetComponent<Unity.MLAgents.Policies.BehaviorParameters>();
            isInferenceMode = bp != null && (bp.BehaviorType == Unity.MLAgents.Policies.BehaviorType.InferenceOnly
                                          || bp.BehaviorType == Unity.MLAgents.Policies.BehaviorType.HeuristicOnly);
            if (isInferenceMode)
                MaxStep = 0;  // Unlimited for inference
            else
                MaxStep = 3000;

            startPosition = transform.position;
            startRotation = transform.rotation;

            // Auto-find scene manager within same Training Area
            if (sceneManager == null)
                sceneManager = transform.parent.GetComponentInChildren<DrivingSceneManager>();

            // Auto-find waypoint manager within same Training Area
            if (waypointManager == null)
                waypointManager = transform.parent.GetComponentInChildren<WaypointManager>();

            // Initialize history buffer
            egoHistory = new float[historySteps][];
            for (int i = 0; i < historySteps; i++)
            {
                egoHistory[i] = new float[8];
            }
        }

        /// <summary>
        /// Set the start position/rotation for this agent (used by TestFieldManager for respawning).
        /// </summary>
        public void SetStartPose(Vector3 position, Quaternion rotation)
        {
            startPosition = position;
            startRotation = rotation;
        }

        public override void OnEpisodeBegin()
        {
            // Reset vehicle
            transform.position = startPosition;
            transform.rotation = startRotation;
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            currentSpeed = 0f;

            // Reset history
            historyIndex = 0;
            historyFull = false;
            for (int i = 0; i < historySteps; i++)
            {
                System.Array.Clear(egoHistory[i], 0, 8);
            }

            // Reset state
            prevAcceleration = 0f;
            prevSteering = 0f;
            prevVelocity = Vector3.zero;
            prevPosition = startPosition;
            currentWaypointIndex = 0;
            episodeReward = 0f;
            episodeSteps = 0;
            totalDistance = 0f;

            // Reset collision tracking
            lastCollisionTime = -COLLISION_COOLDOWN;
            episodeCollisions = 0;

            // Reset overtaking state
            overtakingPhase = OvertakingPhase.None;
            overtakeTarget = null;
            overtakeCount = 0;
            isOvertaking = false;
            stuckBehindTimer = 0f;
            laneReturnStartX = 0f;

            // Reset lane detection state
            leftLaneType = LaneMarkingType.None;
            rightLaneType = LaneMarkingType.None;
            leftLaneDist = 1f;
            rightLaneDist = 1f;
            leftLaneCrossing = false;
            rightLaneCrossing = false;
            lastLaneViolationTime = -LANE_VIOLATION_COOLDOWN;

            // Reset traffic signal state
            hasPassedStopLine = false;
            wasPastStopLineAtRedStart = false;
            prevSignalState = TrafficLightController.LightState.None;
            stoppedAtGreenTimer = 0f;
            dbgTrafficSignalReward = 0f;

            // Reset pedestrian state
            dbgPedestrianReward = 0f;
            yieldRewardAccumulated = 0f;
            yieldDuration = 0f;
            hasYieldedThisEncounter = false;
            crosswalkStopTimer = 0f;
            speedRewardAccumulated = 0f;  // P-027

            // Reset debug stats
            dbgProgressReward = 0f;
            dbgSpeedReward = 0f;
            dbgLaneKeepReward = 0f;
            dbgOvertakeReward = 0f;
            dbgStuckPenalty = 0f;
            dbgCollisionCount = 0f;
            dbgLaneViolationPenalty = 0f;
            dbgEpisodeEndReason = "";

            // Reset environment (curriculum, NPCs)
            if (sceneManager != null)
                sceneManager.ResetEpisode();

            // Reset speed policy state
            if (waypointManager != null)
            {
                currentSpeedLimit = waypointManager.GetCurrentSpeedLimit(transform.position);
            }
            else
            {
                currentSpeedLimit = maxSpeed;
            }
            prevSpeedLimit = currentSpeedLimit;

            if (goalTarget != null)
            {
                previousDistanceToGoal = Vector3.Distance(transform.position, goalTarget.position);
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // === EGO STATE (8D) ===
            float[] currentEgo = GetEgoState();
            for (int i = 0; i < 8; i++)
            {
                sensor.AddObservation(currentEgo[i]);
            }


            // Store in history
            System.Array.Copy(currentEgo, egoHistory[historyIndex], 8);
            historyIndex = (historyIndex + 1) % historySteps;
            if (historyIndex == 0) historyFull = true;

            // === EGO HISTORY (40D = 5 steps x 8D) ===
            for (int step = 0; step < historySteps; step++)
            {
                int idx = historyFull
                    ? (historyIndex + step) % historySteps
                    : step;
                for (int d = 0; d < 8; d++)
                {
                    sensor.AddObservation(egoHistory[idx][d]);
                }
            }

            // === SURROUNDING AGENTS (160D = 20 agents x 8 features) ===
            CollectAgentObservations(sensor);

            // === ROUTE INFO (30D = 10 waypoints x 3) ===
            CollectRouteObservations(sensor);

            // === SPEED INFO (4D) ===
            CollectSpeedObservations(sensor);

            // === LANE INFO (12D) === [Phase D/E/F: enabled]
            if (enableLaneObservation)
            {
                CollectLaneObservations(sensor);
            }

            // === INTERSECTION INFO (6D) === [Phase G+: enabled]
            // Note: Called independently of enableLaneObservation
            if (enableIntersectionObservation)
            {
                AddIntersectionObservations(sensor);
            }

            // === TRAFFIC SIGNAL INFO (8D) === [Phase J+: enabled]
            if (enableTrafficSignalObservation)
            {
                CollectTrafficSignalObservations(sensor);
            }

            // === PEDESTRIAN INFO (12D) === [Phase L+: enabled]
            if (enablePedestrianObservation)
            {
                CollectPedestrianObservations(sensor);
            }
        }

        private void CollectSpeedObservations(VectorSensor sensor)
        {
            float speed = Mathf.Abs(currentSpeed);
            float speedLimit = currentSpeedLimit;
            float nextSpeedLimit = currentSpeedLimit;

            if (waypointManager != null)
            {
                speedLimit = waypointManager.GetCurrentSpeedLimit(transform.position);
                nextSpeedLimit = waypointManager.GetNextSpeedLimit(transform.position);
                currentSpeedLimit = speedLimit;
            }

            // 1. Current speed normalized by max_speed [0, 1]
            sensor.AddObservation(speed / maxSpeed);

            // 2. Current zone speed limit normalized by max_speed [0, 1]
            sensor.AddObservation(speedLimit / maxSpeed);

            // 3. Speed ratio: current / limit (1.0 = optimal, >1.0 = over limit)
            float speedRatio = speedLimit > 0.1f ? speed / speedLimit : 0f;
            sensor.AddObservation(Mathf.Clamp(speedRatio, 0f, 2f) / 2f);  // normalize to [0, 1]

            // 4. Next zone speed limit (for pre-deceleration planning)
            sensor.AddObservation(nextSpeedLimit / maxSpeed);
        }

        /// <summary>
        /// Lane observation collection (v13 - Phase C).
        /// Detects left and right lane markings via raycast.
        /// Observation: 12D = left_type(4) + right_type(4) + left_dist(1) + right_dist(1) + crossings(2)
        /// </summary>
        private void CollectLaneObservations(VectorSensor sensor)
        {
            // Detect lane markings via raycast
            DetectLaneMarkings();

            // Left lane type one-hot (4D)
            float[] leftOneHot = GetLaneTypeOneHot(leftLaneType);
            for (int i = 0; i < 4; i++)
                sensor.AddObservation(leftOneHot[i]);

            // Right lane type one-hot (4D)
            float[] rightOneHot = GetLaneTypeOneHot(rightLaneType);
            for (int i = 0; i < 4; i++)
                sensor.AddObservation(rightOneHot[i]);

            // Normalized distances (1D each)
            sensor.AddObservation(leftLaneDist);   // 0=touching, 1=far
            sensor.AddObservation(rightLaneDist);

            // Crossing flags (1D each)
            sensor.AddObservation(leftLaneCrossing ? 1f : 0f);
            sensor.AddObservation(rightLaneCrossing ? 1f : 0f);
        }

        /// <summary>
        /// Add intersection-related observations (Phase G)
        /// Total: 6D (type one-hot 4D + turn direction one-hot 3D - shared = 6D)
        /// </summary>
        private void AddIntersectionObservations(VectorSensor sensor)
        {
            if (waypointManager == null)
            {
                // No intersection info - add zeros
                for (int i = 0; i < 6; i++)
                    sensor.AddObservation(0f);
                return;
            }

            // Intersection type one-hot: [None, T-junction, Cross, Y-junction] (4D)
            int intersectionType = waypointManager.intersectionType;
            sensor.AddObservation(intersectionType == 0 ? 1f : 0f);  // None
            sensor.AddObservation(intersectionType == 1 ? 1f : 0f);  // T-junction
            sensor.AddObservation(intersectionType == 2 ? 1f : 0f);  // Cross
            sensor.AddObservation(intersectionType == 3 ? 1f : 0f);  // Y-junction

            // Distance to intersection (normalized)
            float distToIntersection = waypointManager.intersectionDistance -
                transform.InverseTransformPoint(waypointManager.transform.position).z;
            sensor.AddObservation(Mathf.Clamp01(distToIntersection / 200f));

            // Turn direction: 0=straight, 1=left, 2=right (encoded as normalized value)
            sensor.AddObservation(waypointManager.turnDirection / 2f);
        }

        /// <summary>
        /// Add traffic signal observations (Phase J)
        /// Total: 8D = state one-hot(4) + dist_to_stop(1) + time_remaining(1) + should_stop(1) + stopped_at_line(1)
        /// </summary>
        private void CollectTrafficSignalObservations(VectorSensor sensor)
        {
            if (trafficLight == null)
            {
                for (int i = 0; i < 8; i++)
                    sensor.AddObservation(0f);
                return;
            }

            var state = trafficLight.GetCurrentState();

            // Traffic light state one-hot (4D): [Red, Yellow, Green, None]
            sensor.AddObservation(state == TrafficLightController.LightState.Red ? 1f : 0f);
            sensor.AddObservation(state == TrafficLightController.LightState.Yellow ? 1f : 0f);
            sensor.AddObservation(state == TrafficLightController.LightState.Green ? 1f : 0f);
            sensor.AddObservation(state == TrafficLightController.LightState.None ? 1f : 0f);

            // Distance to stop line (1D) - normalized by 200m
            sensor.AddObservation(trafficLight.GetDistanceToStopLineNormalized(transform.position));

            // Time remaining in current signal phase (1D) - normalized [0, 1]
            sensor.AddObservation(trafficLight.GetTimeRemainingNormalized());

            // Should stop indicator (1D) - 1 if red/yellow AND behind stop line
            sensor.AddObservation(trafficLight.ShouldStop(transform.position) ? 1f : 0f);

            // Stopped at line indicator (1D) - 1 if stopped within tolerance of stop line
            sensor.AddObservation(trafficLight.IsStoppedAtLine(transform.position, Mathf.Abs(currentSpeed)) ? 1f : 0f);
        }

        /// <summary>
        /// Pedestrian observation collection (Phase L).
        /// Finds 2 nearest active pedestrians and adds relative position, velocity, distance.
        /// Also adds crosswalk info: has_crosswalk_ahead(1D), distance_to_crosswalk(1D).
        /// Total: 12D = 2 pedestrians x 5D + 2D crosswalk
        /// </summary>
        private void CollectPedestrianObservations(VectorSensor sensor)
        {
            // Find nearest active pedestrians sorted by distance
            float nearestDist1 = float.MaxValue;
            float nearestDist2 = float.MaxValue;
            PedestrianController nearest1 = null;
            PedestrianController nearest2 = null;

            if (pedestrians != null)
            {
                foreach (var ped in pedestrians)
                {
                    if (ped == null || !ped.IsActive()) continue;

                    float dist = Vector3.Distance(transform.position, ped.GetPosition());
                    if (dist > pedestrianDetectionRange) continue;

                    if (dist < nearestDist1)
                    {
                        // Shift 1st to 2nd
                        nearestDist2 = nearestDist1;
                        nearest2 = nearest1;
                        nearestDist1 = dist;
                        nearest1 = ped;
                    }
                    else if (dist < nearestDist2)
                    {
                        nearestDist2 = dist;
                        nearest2 = ped;
                    }
                }
            }

            // Pedestrian 1 (5D)
            AddPedestrianObs(sensor, nearest1, nearestDist1);
            // Pedestrian 2 (5D)
            AddPedestrianObs(sensor, nearest2, nearestDist2);

            // Crosswalk info (2D)
            // Distance from ego to crosswalk (in local Z space)
            float crosswalkWorldZ = transform.parent != null
                ? transform.parent.position.z + crosswalkZ
                : crosswalkZ;
            float distToCrosswalk = crosswalkWorldZ - transform.position.z;
            bool hasCrosswalkAhead = distToCrosswalk > 0f && distToCrosswalk < 100f;

            sensor.AddObservation(hasCrosswalkAhead ? 1f : 0f);
            sensor.AddObservation(Mathf.Clamp01(distToCrosswalk / 200f));
        }

        private void AddPedestrianObs(VectorSensor sensor, PedestrianController ped, float dist)
        {
            if (ped != null && ped.IsActive())
            {
                Vector3 relPos = transform.InverseTransformPoint(ped.GetPosition());
                Vector3 pedVel = ped.GetVelocity();
                Vector3 relVel = transform.InverseTransformDirection(pedVel);

                sensor.AddObservation(relPos.x / pedestrianDetectionRange);   // rel_x
                sensor.AddObservation(relPos.z / pedestrianDetectionRange);   // rel_z
                sensor.AddObservation(relVel.x / 2f);                         // vel_x (norm by 2 m/s)
                sensor.AddObservation(relVel.z / 2f);                         // vel_z
                sensor.AddObservation(dist / pedestrianDetectionRange);        // distance
            }
            else
            {
                // Pad with zeros
                for (int i = 0; i < 5; i++)
                    sensor.AddObservation(0f);
            }
        }

        /// <summary>
        /// Detect lane markings on left and right sides via raycast.
        /// Updates leftLaneType, rightLaneType, leftLaneDist, rightLaneDist.
        /// </summary>
        private void DetectLaneMarkings()
        {
            Vector3 origin = transform.position + Vector3.up * laneDetectHeight;

            // Left raycast
            Vector3 leftDir = -transform.right;
            if (Physics.Raycast(origin, leftDir, out RaycastHit leftHit, laneDetectDistance, laneMarkingLayer))
            {
                leftLaneDist = leftHit.distance / laneDetectDistance;
                LaneMarking leftMarking = leftHit.collider.GetComponent<LaneMarking>();
                if (leftMarking != null)
                {
                    leftLaneType = leftMarking.markingType;
                    leftLaneCrossing = leftHit.distance < 0.5f;  // Very close = crossing
                }
                else
                {
                    leftLaneType = LaneMarkingType.None;
                    leftLaneCrossing = false;
                }
            }
            else
            {
                leftLaneType = LaneMarkingType.None;
                leftLaneDist = 1f;
                leftLaneCrossing = false;
            }

            // Right raycast
            Vector3 rightDir = transform.right;
            if (Physics.Raycast(origin, rightDir, out RaycastHit rightHit, laneDetectDistance, laneMarkingLayer))
            {
                rightLaneDist = rightHit.distance / laneDetectDistance;
                LaneMarking rightMarking = rightHit.collider.GetComponent<LaneMarking>();
                if (rightMarking != null)
                {
                    rightLaneType = rightMarking.markingType;
                    rightLaneCrossing = rightHit.distance < 0.5f;
                }
                else
                {
                    rightLaneType = LaneMarkingType.None;
                    rightLaneCrossing = false;
                }
            }
            else
            {
                rightLaneType = LaneMarkingType.None;
                rightLaneDist = 1f;
                rightLaneCrossing = false;
            }
        }

        /// <summary>
        /// Convert lane type to 4D one-hot encoding.
        /// [WhiteDashed, WhiteSolid, YellowDashed/Solid, DoubleYellow]
        /// </summary>
        private float[] GetLaneTypeOneHot(LaneMarkingType type)
        {
            float[] encoding = new float[4];
            switch (type)
            {
                case LaneMarkingType.WhiteDashed:
                    encoding[0] = 1f;
                    break;
                case LaneMarkingType.WhiteSolid:
                    encoding[1] = 1f;
                    break;
                case LaneMarkingType.YellowDashed:
                case LaneMarkingType.YellowSolid:
                    encoding[2] = 1f;
                    break;
                case LaneMarkingType.DoubleYellow:
                    encoding[3] = 1f;
                    break;
            }
            return encoding;
        }

        private float[] GetEgoState()
        {
            float[] ego = new float[8];

            Vector3 posOffset = transform.position - startPosition;
            Vector3 velocity = rb.linearVelocity;
            float heading = transform.eulerAngles.y * Mathf.Deg2Rad;
            Vector3 accel = (velocity - prevVelocity) / Mathf.Max(Time.fixedDeltaTime, 0.001f);

            if (isInferenceMode)
            {
                // P-029: Northify - rotate world-space vectors by -heading
                // Model trained exclusively northbound (heading=0), so we rotate
                // all vectors into heading-aligned frame for correct inference
                float cosH = Mathf.Cos(heading);
                float sinH = Mathf.Sin(heading);

                // Position offset (rotated to heading-aligned frame)
                ego[0] = (posOffset.x * cosH - posOffset.z * sinH) / 100f;
                ego[1] = (posOffset.x * sinH + posOffset.z * cosH) / 100f;

                // Velocity (rotated: forward -> ego[3], lateral -> ego[2])
                ego[2] = (velocity.x * cosH - velocity.z * sinH) / maxSpeed;
                ego[3] = (velocity.x * sinH + velocity.z * cosH) / maxSpeed;

                // Heading: always "north" in northified frame
                ego[4] = 1f;
                ego[5] = 0f;

                // Acceleration (rotated)
                ego[6] = Mathf.Clamp((accel.x * cosH - accel.z * sinH) / maxAcceleration, -1f, 1f);
                ego[7] = Mathf.Clamp((accel.x * sinH + accel.z * cosH) / maxAcceleration, -1f, 1f);
            }
            else
            {
                // Training: unchanged world-space (original behavior)
                ego[0] = posOffset.x / 100f;
                ego[1] = posOffset.z / 100f;
                ego[2] = velocity.x / maxSpeed;
                ego[3] = velocity.z / maxSpeed;
                ego[4] = Mathf.Cos(heading);
                ego[5] = Mathf.Sin(heading);
                ego[6] = Mathf.Clamp(accel.x / maxAcceleration, -1f, 1f);
                ego[7] = Mathf.Clamp(accel.z / maxAcceleration, -1f, 1f);
            }

            prevVelocity = velocity;
            return ego;
        }

        private void CollectAgentObservations(VectorSensor sensor)
        {
            // Find nearby vehicles
            Collider[] nearbyColliders = Physics.OverlapSphere(
                transform.position, agentDetectionRange, vehicleLayer);

            int agentCount = 0;
            List<(float dist, Collider col)> sortedAgents = new List<(float, Collider)>();

            foreach (var col in nearbyColliders)
            {
                if (col.gameObject == gameObject) continue;
                float dist = Vector3.Distance(transform.position, col.transform.position);
                sortedAgents.Add((dist, col));
            }

            // Sort by distance
            sortedAgents.Sort((a, b) => a.dist.CompareTo(b.dist));

            // Collect features for up to maxAgents
            for (int i = 0; i < maxAgents; i++)
            {
                if (i < sortedAgents.Count)
                {
                    var agent = sortedAgents[i];
                    Transform agentTransform = agent.col.transform;
                    Rigidbody agentRb = agent.col.GetComponent<Rigidbody>();

                    // Relative position (2D)
                    Vector3 relPos = transform.InverseTransformPoint(agentTransform.position);
                    sensor.AddObservation(relPos.x / agentDetectionRange);
                    sensor.AddObservation(relPos.z / agentDetectionRange);

                    // Relative velocity
                    Vector3 agentVel = agentRb != null ? agentRb.linearVelocity : Vector3.zero;
                    Vector3 relVel = transform.InverseTransformDirection(agentVel - rb.linearVelocity);
                    sensor.AddObservation(relVel.x / maxSpeed);
                    sensor.AddObservation(relVel.z / maxSpeed);

                    // Agent heading relative to ego
                    float relHeading = (agentTransform.eulerAngles.y - transform.eulerAngles.y) * Mathf.Deg2Rad;
                    sensor.AddObservation(Mathf.Cos(relHeading));
                    sensor.AddObservation(Mathf.Sin(relHeading));

                    // Agent speed and distance
                    float agentSpeed = agentRb != null ? agentRb.linearVelocity.magnitude : 0f;
                    sensor.AddObservation(agentSpeed / maxSpeed);
                    sensor.AddObservation(agent.dist / agentDetectionRange);
                }
                else
                {
                    // Pad with zeros for missing agents
                    for (int j = 0; j < 8; j++)
                    {
                        sensor.AddObservation(0f);
                    }
                }
            }
        }

        private void CollectRouteObservations(VectorSensor sensor)
        {
            if (routeWaypoints != null && routeWaypoints.Length > 0)
            {
                // Use predefined waypoints
                for (int i = 0; i < numWaypoints; i++)
                {
                    int wpIdx = (currentWaypointIndex + i) % routeWaypoints.Length;
                    if (routeWaypoints[wpIdx] == null)
                    {
                        sensor.AddObservation(0f);
                        sensor.AddObservation((i + 1) * waypointSpacing / 100f);
                        sensor.AddObservation((i + 1) * waypointSpacing / maxEpisodeDistance);
                        continue;
                    }
                    Vector3 relPos = transform.InverseTransformPoint(routeWaypoints[wpIdx].position);
                    float dist = Vector3.Distance(transform.position, routeWaypoints[wpIdx].position);

                    sensor.AddObservation(relPos.x / 100f);
                    sensor.AddObservation(relPos.z / 100f);
                    sensor.AddObservation(dist / maxEpisodeDistance);
                }
            }
            else if (goalTarget != null)
            {
                // Generate waypoints toward goal
                Vector3 dirToGoal = (goalTarget.position - transform.position).normalized;
                float totalDist = Vector3.Distance(transform.position, goalTarget.position);

                for (int i = 0; i < numWaypoints; i++)
                {
                    float t = (float)(i + 1) / numWaypoints;
                    Vector3 wpWorld = transform.position + dirToGoal * (totalDist * t);
                    Vector3 relPos = transform.InverseTransformPoint(wpWorld);
                    float dist = totalDist * t;

                    sensor.AddObservation(relPos.x / 100f);
                    sensor.AddObservation(relPos.z / 100f);
                    sensor.AddObservation(dist / maxEpisodeDistance);
                }
            }
            else
            {
                // No goal - emit forward waypoints
                for (int i = 0; i < numWaypoints; i++)
                {
                    sensor.AddObservation(0f);
                    sensor.AddObservation((i + 1) * waypointSpacing / 100f);
                    sensor.AddObservation((i + 1) * waypointSpacing / maxEpisodeDistance);
                }
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            episodeSteps++;

            // Actions: [steering, acceleration] in [-1, 1] range from ML-Agents
            float steeringInput = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float accelInput = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);



            // Scale to physical units (symmetric for better RL exploration)
            float steering = steeringInput * 0.5f;  // [-0.5, 0.5] rad
            float acceleration = accelInput * maxAcceleration;  // [-4, 4] m/s² symmetric

            // Apply physics
            ApplyVehiclePhysics(steering, acceleration);

            // Debug: log after physics every 50 steps
            if (episodeSteps % 50 == 1)
            {
                Debug.Log($"[Agent] Step={episodeSteps} Accel={acceleration:F2} Steer={steering:F3} Speed={currentSpeed:F2} RbV={rb.linearVelocity.magnitude:F3} Pos=({transform.position.x:F1},{transform.position.z:F1})");
            }

            // Calculate rewards
            CalculateRewards(steering, acceleration);

            // Track distance
            totalDistance += Vector3.Distance(transform.position, prevPosition);
            prevPosition = transform.position;

            // Update previous state
            prevSteering = steering;
            prevAcceleration = acceleration;

            // Episode termination conditions
            CheckTermination();
        }

        private void ApplyVehiclePhysics(float steering, float acceleration)
        {
            // Update speed internally (completely bypasses PhysX friction)
            currentSpeed += acceleration * Time.fixedDeltaTime;
            currentSpeed = Mathf.Clamp(currentSpeed, 0f, maxSpeed);

            // Light air drag (realistic rolling resistance when no input)
            // 0.001/frame → equilibrium ~80 m/s (clamped by maxSpeed=30)
            currentSpeed *= (1f - 0.001f);

            // Set velocity directly - overrides any physics solver interference
            rb.linearVelocity = transform.forward * currentSpeed;

            // Steering (bicycle model) - only when moving
            if (Mathf.Abs(currentSpeed) > 0.5f)
            {
                float turnRadius = wheelBase / Mathf.Tan(Mathf.Abs(steering) + 0.001f);
                float angularVelocity = currentSpeed / turnRadius * Mathf.Sign(steering);
                float turnAngle = angularVelocity * Time.fixedDeltaTime * Mathf.Rad2Deg;
                transform.Rotate(0f, turnAngle, 0f);
            }
        }

        private void CalculateRewards(float steering, float acceleration)
        {
            float reward = 0f;
            float progressR = 0f, speedR = 0f, laneR = 0f, overtakeR = 0f, laneViolR = 0f;

            // 1. Progress reward
            if (goalTarget != null)
            {
                float currentDist = Vector3.Distance(transform.position, goalTarget.position);
                float progress = previousDistanceToGoal - currentDist;
                progressR = progress * progressWeight;
                reward += progressR;
                previousDistanceToGoal = currentDist;

                // Goal reached
                if (currentDist < 5f)
                {
                    reward += goalBonus;
                    dbgEpisodeEndReason = "GOAL_REACHED";
                    LogEpisodeSummary();
                    AddReward(reward);
                    EndEpisode();
                    return;
                }
            }

            // 2. Speed policy rewards (P-027: cap per-episode to prevent farming)
            speedR = CalculateSpeedPolicyReward();
            if (speedR > 0f && speedRewardAccumulated + speedR > maxSpeedRewardPerEpisode)
                speedR = Mathf.Max(0f, maxSpeedRewardPerEpisode - speedRewardAccumulated);
            if (speedR > 0f)
                speedRewardAccumulated += speedR;
            reward += speedR;

            // 2.5. Lane keeping
            laneR = CalculateLaneKeepingReward();
            reward += laneR;

            // 2.6. Overtaking reward
            overtakeR = CalculateOvertakingReward();
            reward += overtakeR;

            // 2.7. Lane violation penalty (v13)
            laneViolR = CalculateLaneViolationReward();
            reward += laneViolR;
            if (laneViolR < -9f)  // DoubleYellow violation triggers episode end
            {
                dbgEpisodeEndReason = "DOUBLE_YELLOW_VIOLATION";
                LogEpisodeSummary();
                AddReward(reward);
                EndEpisode();
                return;
            }

            // 3. Jerk penalty (smooth driving)
            float accelJerk = Mathf.Abs(acceleration - prevAcceleration);
            float steerJerk = Mathf.Abs(steering - prevSteering);
            reward += (accelJerk + steerJerk) * jerkPenalty;

            // 4. Time penalty
            reward += timePenalty;

            // 5. Off-road: end episode immediately
            if (IsOffRoad())
            {
                dbgEpisodeEndReason = "OFF_ROAD";
                LogEpisodeSummary();
                AddReward(offRoadPenalty);
                EndEpisode();
                return;
            }

            // 5.5. Wrong-way driving check (Phase F: Center Line Rules)
            if (IsWrongWayDriving())
            {
                dbgEpisodeEndReason = "WRONG_WAY";
                LogEpisodeSummary();
                AddReward(wrongWayPenalty);
                EndEpisode();
                return;
            }

            // 5.7. Traffic signal compliance (Phase J)
            if (enableTrafficSignalObservation && trafficLight != null)
            {
                float signalR = CalculateTrafficSignalReward();
                reward += signalR;
                dbgTrafficSignalReward += signalR;

                // Log signal approach/interaction every 10 steps when within 50m of stop line
                if (debugRewards && episodeSteps % 10 == 0)
                {
                    float distToStop = trafficLight.GetStopLineWorldZ() - transform.position.z;
                    var sigState = trafficLight.GetCurrentState();
                    if (sigState != TrafficLightController.LightState.None && distToStop < 50f && distToStop > -20f)
                    {
                        Debug.Log($"[Signal] Step={episodeSteps} State={sigState} Dist={distToStop:F1}m Spd={currentSpeed:F1} SigR={signalR:F3} ShouldStop={trafficLight.ShouldStop(transform.position)} AtLine={trafficLight.IsStoppedAtLine(transform.position, Mathf.Abs(currentSpeed))}");
                    }
                }

                if (signalR <= redLightViolationPenalty + 0.1f && terminateOnRedViolation)
                {
                    dbgEpisodeEndReason = "RED_LIGHT_VIOLATION";
                    LogEpisodeSummary();
                    AddReward(reward);
                    EndEpisode();
                    return;
                }
            }

            // 5.8. Pedestrian reward (Phase L)
            if (enablePedestrianObservation)
            {
                float pedR = CalculatePedestrianReward();
                reward += pedR;
                dbgPedestrianReward += pedR;

                if (pedR <= pedestrianCollisionPenalty + 0.1f)
                {
                    dbgEpisodeEndReason = "PEDESTRIAN_COLLISION";
                    LogEpisodeSummary();
                    AddReward(reward);
                    EndEpisode();
                    return;
                }
            }

            // 6. Near-collision penalty (TTC < 2s)
            if (IsNearCollision())
            {
                reward += nearCollisionPenalty * Time.fixedDeltaTime;
            }

            // Accumulate debug stats
            dbgProgressReward += progressR;
            dbgSpeedReward += speedR;
            dbgLaneKeepReward += laneR;
            dbgOvertakeReward += overtakeR;
            dbgStuckPenalty += (stuckBehindTimer > stuckBehindTimeout) ? stuckBehindPenalty : 0f;
            dbgLaneViolationPenalty += laneViolR;

            // Debug logging
            if (debugRewards && episodeSteps % debugLogInterval == 0)
            {
                // Signal info for debug
                string sigInfo = "";
                if (enableTrafficSignalObservation && trafficLight != null)
                {
                    var sigState = trafficLight.GetCurrentState();
                    float stopDist = trafficLight.GetStopLineWorldZ() - transform.position.z;
                    sigInfo = $" Sig={sigState} StopDist={stopDist:F1}m SigR={dbgTrafficSignalReward:F2}";
                }
                Debug.Log($"[DBG] Step={episodeSteps} Spd={currentSpeed:F1}/{currentSpeedLimit:F1} Phase={overtakingPhase} " +
                          $"R={reward:F2} (prog={progressR:F2} spd={speedR:F2} lane={laneR:F2} ovt={overtakeR:F2} lnViol={laneViolR:F2}) " +
                          $"Stuck={stuckBehindTimer:F1}s L={leftLaneType} R={rightLaneType} Total={episodeReward:F1}{sigInfo}");
            }

            episodeReward += reward;
            AddReward(reward);

            // === v11: TensorBoard StatsRecorder Logging ===
            // Log every debugLogInterval steps (100 by default)
            if (episodeSteps % debugLogInterval == 0)
            {
                var stats = Academy.Instance.StatsRecorder;

                // Reward components (accumulated since last log)
                stats.Add("Reward/Progress", dbgProgressReward);
                stats.Add("Reward/Speed", dbgSpeedReward);
                stats.Add("Reward/LaneKeeping", dbgLaneKeepReward);
                stats.Add("Reward/Overtaking", dbgOvertakeReward);
                stats.Add("Reward/LaneViolation", dbgLaneViolationPenalty);
                stats.Add("Reward/TrafficSignal", dbgTrafficSignalReward);
                stats.Add("Reward/Pedestrian", dbgPedestrianReward);
                stats.Add("Reward/Jerk", (Mathf.Abs(steering - prevSteering) + Mathf.Abs(acceleration - prevAcceleration)) * jerkPenalty);
                stats.Add("Reward/Time", timePenalty * debugLogInterval);

                // Behavior statistics (instantaneous)
                stats.Add("Stats/Speed", currentSpeed);
                stats.Add("Stats/SpeedLimit", currentSpeedLimit);
                stats.Add("Stats/SpeedRatio", currentSpeedLimit > 0.1f ? currentSpeed / currentSpeedLimit : 0f);
                stats.Add("Stats/Acceleration", acceleration);
                stats.Add("Stats/Steering", Mathf.Abs(steering));
                stats.Add("Stats/DistanceTraveled", totalDistance);
                stats.Add("Stats/StuckTimer", stuckBehindTimer);

                // Overtaking state (one-hot encoded for visualization)
                stats.Add("Stats/OvertakePhase_None", overtakingPhase == OvertakingPhase.None ? 1f : 0f);
                stats.Add("Stats/OvertakePhase_Active", overtakingPhase != OvertakingPhase.None ? 1f : 0f);
            }
        }

        private void LogEpisodeSummary()
        {
            if (!debugRewards) return;
            Debug.Log($"[EPISODE END] Reason={dbgEpisodeEndReason} Steps={episodeSteps} TotalReward={episodeReward:F1} " +
                      $"Overtakes={overtakeCount} Collisions={episodeCollisions} " +
                      $"Components: Progress={dbgProgressReward:F1} Speed={dbgSpeedReward:F1} " +
                      $"LaneKeep={dbgLaneKeepReward:F1} Overtake={dbgOvertakeReward:F1} StuckPen={dbgStuckPenalty:F1} " +
                      $"LaneViol={dbgLaneViolationPenalty:F1} Signal={dbgTrafficSignalReward:F2}");

            // === v11: TensorBoard Episode Summary ===
            var stats = Academy.Instance.StatsRecorder;

            // Episode-level statistics
            stats.Add("Episode/Length", episodeSteps);
            stats.Add("Episode/TotalReward", episodeReward);
            stats.Add("Episode/OvertakeCount", overtakeCount);
            stats.Add("Episode/CollisionCount", episodeCollisions);
            stats.Add("Episode/DistanceTraveled", totalDistance);

            // End reason breakdown (one-hot for ratio tracking)
            stats.Add("Episode/EndReason_Goal", dbgEpisodeEndReason == "GOAL_REACHED" ? 1f : 0f);
            stats.Add("Episode/EndReason_Collision", dbgEpisodeEndReason == "MAX_COLLISIONS" ? 1f : 0f);
            stats.Add("Episode/EndReason_OffRoad", dbgEpisodeEndReason == "OFF_ROAD" ? 1f : 0f);
            stats.Add("Episode/EndReason_WrongWay", dbgEpisodeEndReason == "WRONG_WAY" ? 1f : 0f);
            stats.Add("Episode/EndReason_MaxDistance", dbgEpisodeEndReason == "MAX_DISTANCE" ? 1f : 0f);
            stats.Add("Episode/EndReason_Stuck", dbgEpisodeEndReason == "STUCK_LOW_SPEED" ? 1f : 0f);
            stats.Add("Episode/EndReason_LaneViolation", dbgEpisodeEndReason == "DOUBLE_YELLOW_VIOLATION" ? 1f : 0f);
            stats.Add("Episode/EndReason_Pedestrian", dbgEpisodeEndReason == "PEDESTRIAN_COLLISION" ? 1f : 0f);
            stats.Add("Episode/EndReason_GoalBypass", dbgEpisodeEndReason == "GOAL_BYPASS" ? 1f : 0f);
            stats.Add("Episode/SpeedRewardCapped", speedRewardAccumulated >= maxSpeedRewardPerEpisode ? 1f : 0f);

            // Cumulative reward components (episode totals)
            stats.Add("Episode/TotalProgress", dbgProgressReward);
            stats.Add("Episode/TotalSpeedReward", dbgSpeedReward);
            stats.Add("Episode/TotalLaneKeep", dbgLaneKeepReward);
            stats.Add("Episode/TotalOvertake", dbgOvertakeReward);
            stats.Add("Episode/TotalStuckPenalty", dbgStuckPenalty);
            stats.Add("Episode/TotalLaneViolation", dbgLaneViolationPenalty);
        }

        private bool IsOffRoad()
        {
            // Simple road check: raycast down to check road surface
            if (roadLayer != 0)
            {
                if (!Physics.Raycast(transform.position + Vector3.up, Vector3.down, 5f, roadLayer))
                {
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Check if vehicle is driving on wrong side of road (Phase F).
        /// Phase G (P-013): Passes Z position for intersection zone awareness.
        /// </summary>
        private bool IsWrongWayDriving()
        {
            if (waypointManager == null)
                return false;

            Vector3 localPos = waypointManager.transform.InverseTransformPoint(transform.position);

            return waypointManager.IsWrongWayDriving(localPos.x, localPos.z);
        }

        private bool IsNearCollision()
        {
            // Check TTC (Time To Collision) with forward raycast
            float speed = rb.linearVelocity.magnitude;
            if (speed < 1f) return false;

            if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hit,
                                speed * 2f, vehicleLayer))
            {
                float ttc = hit.distance / speed;
                return ttc < 2f;
            }
            return false;
        }

        /// <summary>
        /// Speed policy reward (v12): targetSpeed = speedLimit ALWAYS.
        /// Never reduces target for lead vehicles. Being stuck behind slow NPC
        /// is inherently penalized because agent can't reach speedLimit.
        /// Stuck-behind timer adds explicit penalty after timeout.
        /// </summary>
        private float CalculateSpeedPolicyReward()
        {
            float reward = 0f;
            float speed = Mathf.Abs(currentSpeed);
            float speedLimit = currentSpeedLimit;

            // If no speed limit info available, use a minimal forward reward
            if (waypointManager == null || speedLimit < 0.1f)
            {
                return Mathf.Clamp01(speed / maxSpeed) * 0.1f;
            }

            // Detect lead vehicle (for stuck-behind tracking only, NOT for target adjustment)
            float leadDist;
            float leadSpeed;
            bool isBlocked = GetLeadVehicleInfo(out leadDist, out leadSpeed);

            // v12: targetSpeed is ALWAYS speedLimit - never reduce for lead vehicle
            float targetSpeed = Mathf.Max(speedLimit, 1f);
            float speedRatio = speed / targetSpeed;

            // === Stuck Behind Slow NPC Timer (v12 only) ===
            // v10g/v11: No stuck penalty (following is rewarded instead)
            // v12: Stuck penalty encourages overtaking
            if (trainingVersion == TrainingVersion.v12)
            {
                bool stuckBehindSlowNPC = isBlocked &&
                    leadSpeed < speedLimit * overtakeSlowLeadThreshold &&
                    leadDist < safeFollowingDistance * 1.5f &&
                    speed < leadSpeed * 1.2f &&  // Agent is matching NPC speed
                    !isOvertaking;

                if (stuckBehindSlowNPC)
                {
                    stuckBehindTimer += Time.fixedDeltaTime;
                    if (stuckBehindTimer > stuckBehindTimeout)
                    {
                        reward += stuckBehindPenalty;  // -0.1/step after 3 seconds
                    }
                }
                else
                {
                    stuckBehindTimer = 0f;
                }
            }

            // === Speed Compliance: +0.3 for 80-100% of limit ===
            if (speedRatio >= 0.8f && speedRatio <= 1.05f)
            {
                reward += speedComplianceReward;
            }
            // === Speed Over Limit: progressive penalty ===
            else if (speedRatio > 1.05f)
            {
                float overRatio = speedRatio - 1.0f;
                float penalty = speedOverPenaltyScale * Mathf.Min(overRatio * 10f, 6f);
                reward += Mathf.Max(penalty, speedOverMaxPenalty);
            }
            // === Speed Under Limit: progressive penalty ===
            // Slower = more penalty, encourages acceleration
            else if (speedRatio < 0.5f)
            {
                // speedRatio 0.0 -> penalty = -0.2
                // speedRatio 0.25 -> penalty = -0.15
                // speedRatio 0.5 -> penalty = -0.1 (boundary)
                float progressivePenalty = speedUnderPenalty * (2f - speedRatio * 2f);
                reward += progressivePenalty;
            }
            // === Between 50-80%: encourage acceleration toward compliance ===
            else if (speedRatio >= 0.5f && speedRatio < 0.8f)
            {
                // Small reward that increases as approaching 80%
                reward += 0.1f * speedRatio;
            }

            // === Speed Zone Transition: smooth change bonus ===
            if (Mathf.Abs(currentSpeedLimit - prevSpeedLimit) > 0.1f)
            {
                float targetDiff = Mathf.Abs(speed - currentSpeedLimit) / currentSpeedLimit;
                if (targetDiff < 0.2f)
                {
                    reward += speedZoneTransitionReward * (1f - targetDiff / 0.2f);
                }
                prevSpeedLimit = currentSpeedLimit;
            }

            return reward;
        }

        /// <summary>
        /// Lane keeping reward: heading alignment with road + lateral deviation penalty.
        /// Encourages straight driving along waypoints while allowing temporary deviation for overtaking.
        /// </summary>
        private float CalculateLaneKeepingReward()
        {
            float reward = 0f;

            if (routeWaypoints == null || routeWaypoints.Length < 2)
                return 0f;

            // Find the closest waypoint AHEAD of the agent
            Transform aheadWP = null;
            float minAheadDist = float.MaxValue;

            for (int i = 0; i < routeWaypoints.Length; i++)
            {
                if (routeWaypoints[i] == null) continue;
                // Check if waypoint is ahead (Z > agent Z in world space)
                float dz = routeWaypoints[i].position.z - transform.position.z;
                if (dz > 2f)  // At least 2m ahead
                {
                    float dist = Vector3.Distance(transform.position, routeWaypoints[i].position);
                    if (dist < minAheadDist)
                    {
                        minAheadDist = dist;
                        aheadWP = routeWaypoints[i];
                    }
                }
            }

            if (aheadWP == null) return 0f;

            // === Heading Alignment ===
            Vector3 toWaypoint = aheadWP.position - transform.position;
            toWaypoint.y = 0f;
            toWaypoint.Normalize();

            Vector3 forward = transform.forward;
            forward.y = 0f;
            forward.Normalize();

            float headingDot = Vector3.Dot(forward, toWaypoint);
            // Reward when well-aligned (dot > 0.95 = within ~18 degrees)
            if (headingDot > 0.95f)
            {
                reward += headingAlignmentReward * ((headingDot - 0.95f) / 0.05f);
            }
            // Penalty when badly misaligned (dot < 0.7 = > 45 degrees)
            else if (headingDot < 0.7f)
            {
                reward += -0.05f;
            }

            // === Lateral Deviation (suspended during overtaking) ===
            if (!isOvertaking)
            {
                float lateralOffset = Mathf.Abs(transform.position.x - aheadWP.position.x);
                if (lateralOffset > maxLateralDeviation)
                {
                    float excessDeviation = (lateralOffset - maxLateralDeviation) / maxLateralDeviation;
                    reward += lateralDeviationPenalty * Mathf.Min(excessDeviation, 2f);
                }
            }

            return reward;
        }

        /// <summary>
        /// Overtaking reward (v12): Dense reward at every sub-phase.
        /// State machine: None -> Approaching -> Beside -> Ahead -> LaneReturn -> None
        /// Each transition gives immediate reward to guide the agent.
        /// </summary>
        private float CalculateOvertakingReward()
        {
            float reward = 0f;

            // Find closest slow NPC using OverlapSphere
            Transform closestNPC = null;
            float closestNPCSpeed = maxSpeed;
            float closestNPCAhead = float.MaxValue;

            Collider[] nearby = Physics.OverlapSphere(transform.position, leadVehicleDetectRange, vehicleLayer);
            foreach (var col in nearby)
            {
                if (col.gameObject == gameObject) continue;
                Vector3 rel = transform.InverseTransformPoint(col.transform.position);
                if (rel.z > -5f && rel.z < closestNPCAhead)
                {
                    closestNPCAhead = rel.z;
                    closestNPC = col.transform;

                    Rigidbody npcRb = col.GetComponent<Rigidbody>();
                    if (npcRb != null)
                        closestNPCSpeed = npcRb.linearVelocity.magnitude;
                    else
                    {
                        NPCVehicleController npc = col.GetComponent<NPCVehicleController>();
                        if (npc != null) closestNPCSpeed = npc.GetCurrentSpeed();
                    }
                }
            }

            // ============================================================
            // v10g MODE: Following reward only, no overtaking
            // ============================================================
            if (trainingVersion == TrainingVersion.v10g)
            {
                return CalculateFollowingReward_v10g(closestNPC, closestNPCSpeed, closestNPCAhead);
            }

            // ============================================================
            // v11 MODE: Following reward (gated) + Sparse overtaking
            // ============================================================
            if (trainingVersion == TrainingVersion.v11)
            {
                return CalculateOvertakingReward_v11(closestNPC, closestNPCSpeed, closestNPCAhead);
            }

            // ============================================================
            // v12 MODE: Dense 5-phase overtaking (original logic)
            // ============================================================

            // Use tracked target during active overtaking, otherwise use closest NPC
            Transform activeTarget = (overtakingPhase != OvertakingPhase.None && overtakeTarget != null)
                ? overtakeTarget : closestNPC;

            if (activeTarget == null)
            {
                if (overtakingPhase != OvertakingPhase.None)
                {
                    overtakingPhase = OvertakingPhase.None;
                    isOvertaking = false;
                }
                return 0f;
            }

            // Get relative position of active target
            Vector3 relNPC = transform.InverseTransformPoint(activeTarget.position);
            float npcAhead = relNPC.z;   // positive = NPC ahead
            float npcLateral = relNPC.x; // positive = NPC to the right

            // Get NPC speed
            float npcSpeed = maxSpeed;
            Rigidbody targetRb = activeTarget.GetComponent<Rigidbody>();
            if (targetRb != null)
            {
                npcSpeed = targetRb.linearVelocity.magnitude;
            }
            else
            {
                NPCVehicleController npcCtrl = activeTarget.GetComponent<NPCVehicleController>();
                if (npcCtrl != null) npcSpeed = npcCtrl.GetCurrentSpeed();
            }

            bool isSlowNPC = npcSpeed < currentSpeedLimit * overtakeSlowLeadThreshold;

            // Get lane center X from route waypoints (for lane return detection)
            float laneCenterX = transform.position.x;
            if (routeWaypoints != null)
            {
                for (int i = 0; i < routeWaypoints.Length; i++)
                {
                    if (routeWaypoints[i] == null) continue;
                    float dz = routeWaypoints[i].position.z - transform.position.z;
                    if (dz > 2f)
                    {
                        laneCenterX = routeWaypoints[i].position.x;
                        break;
                    }
                }
            }

            switch (overtakingPhase)
            {
                case OvertakingPhase.None:
                    // Trigger: slow NPC detected ahead
                    if (isSlowNPC && npcAhead > 5f && npcAhead < leadVehicleDetectRange)
                    {
                        overtakingPhase = OvertakingPhase.Approaching;
                        overtakeTarget = activeTarget;
                        isOvertaking = true;
                        // No reward yet - just tracking
                    }
                    break;

                case OvertakingPhase.Approaching:
                    if (!isSlowNPC || activeTarget == null)
                    {
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                        break;
                    }
                    // Transition to Beside: laterally offset from NPC and close
                    if (Mathf.Abs(npcLateral) > 1.5f && npcAhead < 12f && npcAhead > -3f)
                    {
                        overtakingPhase = OvertakingPhase.Beside;
                        reward += overtakeInitiateBonus;  // +0.5: lane change initiated
                        Debug.Log($"[Overtake] BESIDE phase entered. Reward +{overtakeInitiateBonus}");
                    }
                    // Timeout
                    if (npcAhead > leadVehicleDetectRange * 1.5f)
                    {
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                    }
                    break;

                case OvertakingPhase.Beside:
                    // Per-step reward for maintaining speed while beside NPC
                    if (currentSpeed > npcSpeed * 0.9f)
                    {
                        reward += overtakeBesideBonus;  // +0.2/step
                    }
                    // Transition to Ahead: ego has passed NPC
                    if (npcAhead < -3f)
                    {
                        overtakingPhase = OvertakingPhase.Ahead;
                        reward += overtakeAheadBonus;  // +1.0: passed NPC
                        overtakeCount++;
                        Debug.Log($"[Overtake] AHEAD phase. Pass #{overtakeCount}. Reward +{overtakeAheadBonus}");
                    }
                    // Timeout: lost NPC or too far
                    if (Vector3.Distance(transform.position, activeTarget.position) > 35f)
                    {
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                    }
                    break;

                case OvertakingPhase.Ahead:
                    // Transition to LaneReturn: start moving back toward lane center
                    laneReturnStartX = transform.position.x;
                    overtakingPhase = OvertakingPhase.LaneReturn;
                    break;

                case OvertakingPhase.LaneReturn:
                    // Reward for returning to lane center
                    float lateralOffset = Mathf.Abs(transform.position.x - laneCenterX);
                    if (lateralOffset < maxLateralDeviation)
                    {
                        // Successfully returned to lane
                        reward += overtakeCompleteBonus;  // +2.0: full overtake complete
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                        Debug.Log($"[Overtake] COMPLETE! Lane return done. Reward +{overtakeCompleteBonus}");
                    }
                    // Timeout: if taking too long to return (30m ahead of NPC)
                    if (npcAhead < -30f)
                    {
                        // Still give partial credit - passed but didn't return to lane
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                    }
                    break;
            }

            return reward;
        }

        /// <summary>
        /// v10g: Following reward only, no overtaking.
        /// Rewards maintaining safe following distance behind NPCs.
        /// </summary>
        private float CalculateFollowingReward_v10g(Transform closestNPC, float npcSpeed, float npcAhead)
        {
            if (closestNPC == null)
                return 0f;

            // Check if NPC is ahead and within safe following distance
            if (npcAhead > 0 && npcAhead < safeFollowingDistance * 2f)
            {
                // Maintaining safe distance? Reward!
                if (npcAhead >= safeFollowingDistance * 0.5f)
                {
                    dbgOvertakeReward += followingBonus;
                    return followingBonus;  // +0.3 per step for safe following
                }
            }
            return 0f;
        }

        /// <summary>
        /// v11: Following reward (gated) + Sparse overtaking.
        /// Following bonus only when NPC > 70% speedLimit.
        /// Sparse overtaking: overtakePassBonus when complete, overtakeSpeedBonus per-step beside NPC.
        /// </summary>
        private float CalculateOvertakingReward_v11(Transform closestNPC, float npcSpeed, float npcAhead)
        {
            float reward = 0f;

            if (closestNPC == null)
            {
                if (overtakingPhase != OvertakingPhase.None)
                {
                    overtakingPhase = OvertakingPhase.None;
                    isOvertaking = false;
                }
                return 0f;
            }

            Vector3 relNPC = transform.InverseTransformPoint(closestNPC.position);
            float npcLateral = relNPC.x;
            bool isSlowNPC = npcSpeed < currentSpeedLimit * overtakeSlowLeadThreshold;
            bool isFastNPC = npcSpeed >= currentSpeedLimit * followingBonusSpeedThreshold;

            // GATED Following bonus: only when NPC is fast (>70% speedLimit)
            if (isFastNPC && npcAhead > 0 && npcAhead < safeFollowingDistance * 2f)
            {
                if (npcAhead >= safeFollowingDistance * 0.5f)
                {
                    reward += followingBonus;
                }
            }

            // SPARSE Overtaking rewards
            switch (overtakingPhase)
            {
                case OvertakingPhase.None:
                    if (isSlowNPC && npcAhead > 5f && npcAhead < leadVehicleDetectRange)
                    {
                        overtakingPhase = OvertakingPhase.Approaching;
                        overtakeTarget = closestNPC;
                        isOvertaking = true;
                    }
                    break;

                case OvertakingPhase.Approaching:
                    if (!isSlowNPC)
                    {
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                        break;
                    }
                    // Transition to Beside
                    if (Mathf.Abs(npcLateral) > 1.5f && npcAhead < 12f && npcAhead > -3f)
                    {
                        overtakingPhase = OvertakingPhase.Beside;
                        // v11: No initiate bonus (sparse)
                    }
                    break;

                case OvertakingPhase.Beside:
                    // v11: Per-step bonus for speed beside NPC
                    if (currentSpeed > npcSpeed * 0.9f)
                    {
                        reward += overtakeSpeedBonus;  // +0.15/step (v11 sparse)
                    }
                    // Transition to Completed
                    if (npcAhead < -3f)
                    {
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                        reward += overtakePassBonus;  // +3.0 (v11 sparse, one-time)
                        overtakeCount++;
                        Debug.Log($"[v11 Overtake] COMPLETE! Pass #{overtakeCount}. Sparse bonus +{overtakePassBonus}");
                    }
                    break;
            }

            dbgOvertakeReward += reward;
            return reward;
        }

        /// <summary>
        /// Lane violation penalty (v13 - Phase C).
        /// Penalizes crossing prohibited lane markings.
        /// WhiteDashed = allowed, WhiteSolid = penalty, YellowSolid = high penalty, DoubleYellow = fatal.
        /// </summary>
        private float CalculateLaneViolationReward()
        {
            float penalty = 0f;

            // Cooldown to prevent rapid-fire penalties
            if (Time.time - lastLaneViolationTime < LANE_VIOLATION_COOLDOWN)
                return 0f;

            // Check left lane crossing
            if (leftLaneCrossing && !isOvertaking)  // Overtaking suspends some penalties
            {
                penalty += GetLaneCrossingPenalty(leftLaneType);
                if (penalty < 0f)
                {
                    lastLaneViolationTime = Time.time;
                    if (debugRewards)
                    {
                        Debug.Log($"[Lane Violation] Crossed LEFT {leftLaneType}. Penalty={penalty:F1}");
                    }
                }
            }

            // Check right lane crossing
            if (rightLaneCrossing && !isOvertaking)
            {
                float rightPenalty = GetLaneCrossingPenalty(rightLaneType);
                if (rightPenalty < 0f)
                {
                    lastLaneViolationTime = Time.time;
                    penalty += rightPenalty;
                    if (debugRewards)
                    {
                        Debug.Log($"[Lane Violation] Crossed RIGHT {rightLaneType}. Penalty={rightPenalty:F1}");
                    }
                }
            }

            // During overtaking: only center line violations apply (no white line penalties)
            if (isOvertaking)
            {
                // Check if crossing center line (yellow markings)
                bool leftCenterLine = leftLaneCrossing &&
                    (leftLaneType == LaneMarkingType.YellowDashed ||
                     leftLaneType == LaneMarkingType.YellowSolid ||
                     leftLaneType == LaneMarkingType.DoubleYellow);
                bool rightCenterLine = rightLaneCrossing &&
                    (rightLaneType == LaneMarkingType.YellowDashed ||
                     rightLaneType == LaneMarkingType.YellowSolid ||
                     rightLaneType == LaneMarkingType.DoubleYellow);

                if (leftCenterLine)
                {
                    penalty += GetLaneCrossingPenalty(leftLaneType);
                    lastLaneViolationTime = Time.time;
                    if (debugRewards)
                        Debug.Log($"[Lane Violation] CENTER LINE crossed during overtake: {leftLaneType}");
                }
                if (rightCenterLine)
                {
                    penalty += GetLaneCrossingPenalty(rightLaneType);
                    lastLaneViolationTime = Time.time;
                    if (debugRewards)
                        Debug.Log($"[Lane Violation] CENTER LINE crossed during overtake: {rightLaneType}");
                }
            }

            return penalty;
        }

        /// <summary>
        /// Get penalty for crossing specific lane marking type.
        /// </summary>
        private float GetLaneCrossingPenalty(LaneMarkingType type)
        {
            switch (type)
            {
                case LaneMarkingType.WhiteDashed:
                    return 0f;  // Allowed
                case LaneMarkingType.WhiteSolid:
                    return whiteSolidCrossPenalty;  // -2.0
                case LaneMarkingType.YellowDashed:
                    return yellowDashedCrossPenalty;  // -3.0
                case LaneMarkingType.YellowSolid:
                    return yellowSolidCrossPenalty;  // -5.0
                case LaneMarkingType.DoubleYellow:
                    return doubleYellowCrossPenalty;  // -10.0 (fatal)
                default:
                    return 0f;
            }
        }

        /// <summary>
        /// Traffic signal reward (Phase J).
        /// Rewards stopping at red lights, penalizes violations.
        /// </summary>
        private float CalculateTrafficSignalReward()
        {
            if (trafficLight == null) return 0f;

            var state = trafficLight.GetCurrentState();
            if (state == TrafficLightController.LightState.None) return 0f;

            float reward = 0f;
            float speed = Mathf.Abs(currentSpeed);
            float stopLineWorldZ = trafficLight.GetStopLineWorldZ();
            bool behindStopLine = transform.position.z < stopLineWorldZ;
            float distToStop = stopLineWorldZ - transform.position.z;

            // Detect signal state transitions
            if (state != prevSignalState)
            {
                if (state == TrafficLightController.LightState.Red)
                {
                    // Red phase just started: record whether agent was already past stop line
                    wasPastStopLineAtRedStart = !behindStopLine;
                    hasPassedStopLine = !behindStopLine;
                }
                prevSignalState = state;
            }

            // Track stop line crossing during current phase
            if (!behindStopLine && !hasPassedStopLine)
            {
                hasPassedStopLine = true;
            }

            switch (state)
            {
                case TrafficLightController.LightState.Red:
                    if (hasPassedStopLine && !wasPastStopLineAtRedStart
                        && trafficLight.HasViolatedRedLight(transform.position))
                    {
                        // Crossed stop line DURING red phase — true violation
                        reward = redLightViolationPenalty;
                        Debug.Log($"[TrafficSignal] RED LIGHT VIOLATION at step {episodeSteps}");
                    }
                    else if (trafficLight.IsStoppedAtLine(transform.position, speed))
                    {
                        // Properly stopped at red — reward
                        reward = properRedStopReward * Time.fixedDeltaTime;
                    }
                    else if (behindStopLine && distToStop < 50f && distToStop > 0f)
                    {
                        // Approaching red light within 50m — reward deceleration
                        // Target: slow down proportionally to distance (closer = slower)
                        float maxApproachSpeed = currentSpeedLimit;
                        float desiredSpeed = Mathf.Lerp(0f, maxApproachSpeed, distToStop / 50f);
                        if (speed <= desiredSpeed + 1f)
                        {
                            // Good: speed is at or below the appropriate approach speed
                            reward = 0.1f * Time.fixedDeltaTime;
                        }
                        else
                        {
                            // Bad: approaching too fast for the distance
                            float excessRatio = Mathf.Clamp01((speed - desiredSpeed) / maxApproachSpeed);
                            reward = -0.2f * excessRatio * Time.fixedDeltaTime;
                        }
                    }
                    // Reset green timer when red
                    stoppedAtGreenTimer = 0f;
                    break;

                case TrafficLightController.LightState.Yellow:
                    if (behindStopLine && distToStop > 15f && speed > 2f)
                    {
                        // Far from intersection + moving = should start slowing
                        float excessRatio = Mathf.Clamp01((speed - 5f) / currentSpeedLimit);
                        reward = -0.1f * excessRatio * Time.fixedDeltaTime;
                    }
                    else if (!behindStopLine && !hasPassedStopLine)
                    {
                        // Entering intersection on yellow from far away
                        reward = yellowCautionPenalty;
                        Debug.Log($"[TrafficSignal] Yellow caution at step {episodeSteps}");
                    }
                    stoppedAtGreenTimer = 0f;
                    break;

                case TrafficLightController.LightState.Green:
                    // Track unnecessary stopping at green
                    if (behindStopLine && speed < 0.5f)
                    {
                        stoppedAtGreenTimer += Time.fixedDeltaTime;
                        if (stoppedAtGreenTimer > unnecessaryStopTimeout)
                        {
                            reward = unnecessaryStopPenalty * Time.fixedDeltaTime;
                        }
                    }
                    else
                    {
                        stoppedAtGreenTimer = 0f;
                    }

                    // Reset stop line tracking on green (agent can proceed)
                    hasPassedStopLine = !behindStopLine;
                    wasPastStopLineAtRedStart = false;
                    break;
            }

            return reward;
        }

        /// <summary>
        /// Pedestrian reward (Phase L).
        /// - Collision with pedestrian: terminate with large penalty
        /// - Approaching crosswalk with active pedestrian while fast: speed penalty
        /// - Yielding (slowing/stopping) at crosswalk with crossing pedestrian: positive reward
        /// </summary>
        private float CalculatePedestrianReward()
        {
            if (pedestrians == null) return 0f;

            float reward = 0f;
            float speed = Mathf.Abs(currentSpeed);
            float crosswalkWorldZ = transform.parent != null
                ? transform.parent.position.z + crosswalkZ
                : crosswalkZ;
            float distToCrosswalk = crosswalkWorldZ - transform.position.z;

            // Check collision via OverlapSphere (small radius around ego)
            bool anyPedestrianActive = false;
            bool pedestrianAtCrosswalk = false;

            foreach (var ped in pedestrians)
            {
                if (ped == null || !ped.IsActive()) continue;
                anyPedestrianActive = true;

                float distToPed = Vector3.Distance(transform.position, ped.GetPosition());

                // Collision detection: very close to pedestrian
                if (distToPed < 3f)
                {
                    Debug.Log($"[Pedestrian] COLLISION at step {episodeSteps}, dist={distToPed:F1}m");
                    return pedestrianCollisionPenalty;  // -10f, triggers episode end
                }

                // Check if pedestrian is near crosswalk (within 5m Z)
                float pedDistFromCrosswalk = Mathf.Abs(ped.GetPosition().z - crosswalkWorldZ);
                if (pedDistFromCrosswalk < 10f)
                {
                    pedestrianAtCrosswalk = true;
                }
            }

            // Crosswalk yielding reward (v3: capped + episode termination for overstay)
            // Track time spent slow near crosswalk (anti-farming, regardless of pedestrian presence)
            if (distToCrosswalk > 0f && distToCrosswalk < 20f && speed < 2f)
            {
                crosswalkStopTimer += Time.fixedDeltaTime;
            }
            else if (distToCrosswalk > 30f || speed >= 3f)
            {
                // Reset timer when clearly past crosswalk or driving normally
                crosswalkStopTimer = Mathf.Max(0f, crosswalkStopTimer - Time.fixedDeltaTime * 2f);
            }

            if (pedestrianAtCrosswalk && distToCrosswalk > 0f && distToCrosswalk < 50f)
            {
                // Agent is approaching crosswalk with active pedestrian
                if (speed < 1f && distToCrosswalk < 15f)
                {
                    yieldDuration += Time.fixedDeltaTime;

                    if (yieldDuration <= maxYieldDuration && yieldRewardAccumulated < maxYieldRewardPerEpisode)
                    {
                        // Yielding: capped positive reward
                        float yieldR = crosswalkYieldReward * Time.fixedDeltaTime;
                        yieldR = Mathf.Min(yieldR, maxYieldRewardPerEpisode - yieldRewardAccumulated);
                        reward += yieldR;
                        yieldRewardAccumulated += yieldR;
                        hasYieldedThisEncounter = true;
                    }
                    else if (yieldDuration > maxYieldDuration)
                    {
                        // Overstaying penalty: strong enough to deter farming
                        reward += -1.0f * Time.fixedDeltaTime;
                    }
                }
                else if (speed > currentSpeedLimit * 0.5f && distToCrosswalk < 30f)
                {
                    // Going too fast near crosswalk with pedestrian
                    float excessRatio = speed / Mathf.Max(currentSpeedLimit, 1f);
                    reward += crosswalkSpeedPenaltyScale * excessRatio * Time.fixedDeltaTime;
                }
            }
            else
            {
                // Not near crosswalk with pedestrian: reset yield tracking for next encounter
                if (hasYieldedThisEncounter)
                {
                    yieldDuration = 0f;
                    hasYieldedThisEncounter = false;
                }
            }

            return reward;
        }

        /// <summary>
        /// Detect lead vehicle ahead using SphereCast (catches laterally offset NPCs).
        /// Returns true if a vehicle is within leadVehicleDetectRange ahead.
        /// Also updates overtakeTarget for overtaking reward tracking.
        /// </summary>
        private bool GetLeadVehicleInfo(out float distance, out float leadSpeed)
        {
            distance = leadVehicleDetectRange;
            leadSpeed = maxSpeed;
            overtakeTarget = null;

            if (vehicleLayer == 0) return false;

            // SphereCast with overtakeDetectWidth radius to catch offset NPCs
            if (Physics.SphereCast(transform.position, overtakeDetectWidth, transform.forward,
                                   out RaycastHit hit, leadVehicleDetectRange, vehicleLayer))
            {
                distance = hit.distance;
                overtakeTarget = hit.collider.transform;

                // Get lead vehicle speed
                Rigidbody leadRb = hit.collider.GetComponent<Rigidbody>();
                if (leadRb != null)
                {
                    leadSpeed = leadRb.linearVelocity.magnitude;
                }
                else
                {
                    NPCVehicleController npc = hit.collider.GetComponent<NPCVehicleController>();
                    if (npc != null)
                    {
                        leadSpeed = npc.GetCurrentSpeed();
                    }
                }
                return true;
            }
            return false;
        }

        private void CheckTermination()
        {
            // P-028: Skip training-only termination checks in inference mode
            // In inference mode (Phase M grid), TestFieldManager handles goal/timeout
            if (!isInferenceMode)
            {
                // Max distance exceeded
                float distFromStart = Vector3.Distance(transform.position, startPosition);
                if (distFromStart > maxEpisodeDistance)
                {
                    dbgEpisodeEndReason = "MAX_DISTANCE";
                    LogEpisodeSummary();
                    EndEpisode();
                    return;
                }

                // P-027: Goal bypass detection - agent drove past goal without reaching it
                // Uses world Z axis - only valid for straight roads (training)
                if (goalTarget != null)
                {
                    float agentZ = transform.position.z;
                    float goalZ = goalTarget.position.z;
                    if (agentZ > goalZ + 20f)
                    {
                        dbgEpisodeEndReason = "GOAL_BYPASS";
                        LogEpisodeSummary();
                        AddReward(-2f);
                        EndEpisode();
                        return;
                    }
                }
            }

            // Max steps (handled by ML-Agents MaxStep setting)
            // Stuck detection (low speed for too long)
            if (episodeSteps > 500 && currentSpeed < 0.1f)
            {
                dbgEpisodeEndReason = "STUCK_LOW_SPEED";
                LogEpisodeSummary();
                AddReward(-1f);
                EndEpisode();
                return;
            }

            // Crosswalk anti-farming: terminate if loitering near crosswalk (P-026 v3)
            if (enablePedestrianObservation && crosswalkStopTimer > maxCrosswalkStopDuration)
            {
                dbgEpisodeEndReason = "CROSSWALK_OVERSTAY";
                LogEpisodeSummary();
                AddReward(-2f);
                EndEpisode();
            }
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.gameObject.CompareTag("Obstacle") ||
                collision.gameObject.CompareTag("Vehicle"))
            {
                // Cooldown: prevent rapid-fire penalties from same collision
                if (Time.time - lastCollisionTime < COLLISION_COOLDOWN) return;
                lastCollisionTime = Time.time;
                episodeCollisions++;
                dbgCollisionCount++;

                if (debugRewards)
                {
                    Debug.Log($"[COLLISION] #{episodeCollisions} with {collision.gameObject.name} at step {episodeSteps}");
                }

                AddReward(collisionPenalty);

                // Slow down on collision (lose momentum)
                currentSpeed *= 0.3f;

                // End episode only after repeated collisions (unable to learn)
                if (episodeCollisions >= MAX_COLLISIONS_PER_EPISODE)
                {
                    dbgEpisodeEndReason = "MAX_COLLISIONS";
                    LogEpisodeSummary();
                    EndEpisode();
                }
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Use expert controller if available (for recording demonstrations)
            var expert = GetComponent<ExpertDriverController>();
            if (expert != null && expert.enabled)
            {
                expert.GetExpertAction(actionsOut);
                return;
            }

            // Fallback: keyboard control (zero actions if legacy Input unavailable)
            var continuousActions = actionsOut.ContinuousActions;
            try
            {
                continuousActions[0] = Input.GetAxis("Horizontal");  // Steering
                continuousActions[1] = Input.GetAxis("Vertical");    // Acceleration
            }
            catch (System.InvalidOperationException)
            {
                continuousActions[0] = 0f;
                continuousActions[1] = 0f;
            }
        }

        // Public API for external monitoring
        public float GetEpisodeReward() => episodeReward;
        public int GetEpisodeSteps() => episodeSteps;
        public float GetTotalDistance() => totalDistance;
        public float GetCurrentSpeed() => rb != null ? rb.linearVelocity.magnitude : 0f;
    }
}
