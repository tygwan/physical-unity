using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using ADPlatform.Environment;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Phase B v2: Decision Learning Agent
    ///
    /// Copy of E2EDrivingAgent with:
    ///   1. Enum renamed: v10g/v11/v12 → Phase0/Phase0Sparse/PhaseA
    ///   2. Boosted overtaking rewards (4x initiate, 2.5x beside, 2x ahead, 1.5x complete)
    ///   3. Reduced penalties (80% speedUnder reduction, 50% stuckBehind reduction)
    ///   4. Blocked detection: suspend speed penalty when NPC ahead
    ///
    /// Design Doc: experiments/phase-B-decision-v2/DESIGN.md
    /// v1 Failure Ref: experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md
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
    /// Action Space (2D continuous):
    ///   - Steering: [-0.5, 0.5] rad
    ///   - Acceleration: [-4.0, 2.0] m/s²
    ///
    /// Reward Design (Phase B v2 - boosted overtaking):
    ///   - Progress toward goal
    ///   - Speed compliance (+0.3 for 80-100% of limit)
    ///   - Speed over limit (-0.5 ~ -3.0 progressive)
    ///   - Speed under limit (-0.02 for &lt;50% of limit, suspended when blocked)
    ///   - Collision penalty
    ///   - Off-road penalty
    ///   - Comfort (jerk) penalty
    ///   - Goal reached bonus
    ///   - Overtake initiate: +2.0 (4x from Phase A)
    ///   - Overtake beside: +0.5/step (2.5x from Phase A)
    ///   - Overtake ahead: +2.0 (2x from Phase A)
    ///   - Overtake complete: +3.0 (1.5x from Phase A)
    /// </summary>
    public class E2EDrivingAgentBv2 : Agent
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

        [Header("Reward Weights")]
        public float progressWeight = 1.0f;
        public float collisionPenalty = -5f;
        public float nearCollisionPenalty = -1.5f;
        public float offRoadPenalty = -5f;
        public float jerkPenalty = -0.1f;
        public float goalBonus = 10f;
        public float timePenalty = -0.001f;

        [Header("Speed Policy Rewards")]
        public float speedComplianceReward = 0.3f;    // 80-100% of limit
        public float speedOverPenaltyScale = -0.5f;   // Progressive: -0.5 per 10% over
        public float speedOverMaxPenalty = -3.0f;      // Cap at -3.0
        public float speedUnderPenalty = -0.02f;       // Phase B v2: -0.02 (was -0.1, 80% reduction)
        public float speedZoneTransitionReward = 0.2f; // Smooth transition bonus

        [Header("Lane Keeping")]
        public float headingAlignmentReward = 0.02f;   // Reward for heading aligned with road
        public float lateralDeviationPenalty = -0.02f;  // Penalty for off-center driving
        public float maxLateralDeviation = 2.5f;        // meters from center before penalty

        [Header("Center Line Rules (Phase F)")]
        public float wrongWayPenalty = -5f;             // Penalty for crossing center line (same as off-road)
        public float centerLineCrossingPenalty = -0.5f; // Per-step penalty when on wrong side

        public enum TrainingVersion { Phase0, Phase0Sparse, PhaseA }

        [Header("Training Version")]
        [Tooltip("Phase0: Lane keeping + following, Phase0Sparse: Sparse overtake (deprecated), PhaseA: Dense overtake")]
        public TrainingVersion trainingVersion = TrainingVersion.PhaseA;

        [Header("Following Behavior")]
        public float leadVehicleDetectRange = 40f;    // meters ahead to check
        public float safeFollowingDistance = 15f;      // meters (safe gap)

        [Header("Following Reward (Phase0/Phase0Sparse only)")]
        [Tooltip("Phase0/Phase0Sparse: Reward for maintaining safe following distance")]
        public float followingBonus = 0.3f;           // Phase0/Phase0Sparse: per-step when safely following
        [Tooltip("Phase0Sparse: Only apply followingBonus when NPC > this ratio of speedLimit")]
        public float followingBonusSpeedThreshold = 0.7f;  // Phase0Sparse: gated following bonus

        [Header("Overtaking (Phase0Sparse - Sparse Reward)")]
        [Tooltip("Phase0Sparse: One-time bonus when overtake fully completed")]
        public float overtakePassBonus = 3.0f;        // Phase0Sparse: sparse, one-time
        [Tooltip("Phase0Sparse: Per-step bonus when beside NPC during overtake")]
        public float overtakeSpeedBonus = 0.15f;      // Phase0Sparse: per-step beside NPC

        [Header("Overtaking (PhaseA - Dense Reward Strategy)")]
        public float overtakeInitiateBonus = 2.0f;    // Phase B v2: 2.0 (was 0.5, 4x increase)
        public float overtakeBesideBonus = 0.5f;      // Phase B v2: 0.5 (was 0.2, 2.5x increase)
        public float overtakeAheadBonus = 2.0f;       // Phase B v2: 2.0 (was 1.0, 2x increase)
        public float overtakeCompleteBonus = 3.0f;    // Phase B v2: 3.0 (was 2.0, 1.5x increase)
        public float stuckBehindPenalty = -0.05f;     // Phase B v2: -0.05 (was -0.1, 50% reduction)
        public float stuckBehindTimeout = 5.0f;       // Phase B v2: 5.0 (was 3.0, more patience)
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

        // Internal state
        private Rigidbody rb;
        private Vector3 startPosition;
        private Quaternion startRotation;
        private float previousDistanceToGoal;
        private int currentWaypointIndex;

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

        // Overtaking state (PhaseA: 5-phase dense reward)
        private enum OvertakingPhase { None, Approaching, Beside, Ahead, LaneReturn }
        private OvertakingPhase overtakingPhase = OvertakingPhase.None;
        private Transform overtakeTarget = null;
        private int overtakeCount = 0;         // NPCs passed this episode
        private bool isOvertaking = false;     // Flag for lane keeping suspension
        private float stuckBehindTimer = 0f;   // Time spent stuck behind slow NPC
        private float laneReturnStartX = 0f;   // X position when lane return started

        // Lane detection state (Phase C)
        [Header("Lane Marking Detection")]
        public LayerMask laneMarkingLayer;           // Layer for lane marking colliders
        public float laneDetectDistance = 5f;        // Raycast distance to left/right
        public float laneDetectHeight = 0.5f;        // Raycast origin height

        [Header("Lane Violation Penalties")]
        public float whiteSolidCrossPenalty = -2.0f;
        public float yellowDashedCrossPenalty = -3.0f;
        public float yellowSolidCrossPenalty = -5.0f;
        public float doubleYellowCrossPenalty = -10.0f;
        public bool terminateOnDoubleYellow = true;

        // Lane state (detected via raycast)
        private LaneMarkingType leftLaneType = LaneMarkingType.None;
        private LaneMarkingType rightLaneType = LaneMarkingType.None;
        private float leftLaneDist = 1f;   // Normalized: 0=touching, 1=far
        private float rightLaneDist = 1f;
        private bool leftLaneCrossing = false;   // Currently crossing left lane
        private bool rightLaneCrossing = false;  // Currently crossing right lane
        private float lastLaneViolationTime = -1f;
        private const float LANE_VIOLATION_COOLDOWN = 0.5f;

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

            // Phase B v2: Boosted overtaking + reduced penalties
            // See: experiments/phase-B-decision-v2/DESIGN.md
            headingAlignmentReward = 0.02f;
            lateralDeviationPenalty = -0.02f;
            overtakeInitiateBonus = 2.0f;       // 0.5 -> 2.0 (4x increase)
            overtakeBesideBonus = 0.5f;          // 0.2 -> 0.5 (2.5x increase)
            overtakeAheadBonus = 2.0f;           // 1.0 -> 2.0 (2x increase)
            overtakeCompleteBonus = 3.0f;        // 2.0 -> 3.0 (1.5x increase)
            stuckBehindPenalty = -0.05f;         // -0.1 -> -0.05 (50% reduction)
            stuckBehindTimeout = 5.0f;           // 3.0 -> 5.0 (more patience)
            speedUnderPenalty = -0.02f;          // -0.1 -> -0.02 (80% reduction)
            overtakeSlowLeadThreshold = 0.7f;
            overtakeDetectWidth = 3.0f;

            Debug.Log("[Phase B v2] Reward values loaded: overtakeInitiate=2.0, beside=0.5, ahead=2.0, complete=3.0, speedUnder=-0.02, stuckBehind=-0.05");

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
            if (enableIntersectionObservation)
            {
                AddIntersectionObservations(sensor);
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
        /// Lane observation collection (Phase C).
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

            // Position (local, normalized)
            Vector3 localPos = transform.position - startPosition;
            ego[0] = localPos.x / 100f;
            ego[1] = localPos.z / 100f;

            // Velocity (normalized)
            Vector3 velocity = rb.linearVelocity;
            ego[2] = velocity.x / maxSpeed;
            ego[3] = velocity.z / maxSpeed;

            // Heading (cos, sin)
            float heading = transform.eulerAngles.y * Mathf.Deg2Rad;
            ego[4] = Mathf.Cos(heading);
            ego[5] = Mathf.Sin(heading);

            // Acceleration (estimated from velocity change)
            Vector3 accel = (velocity - prevVelocity) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            ego[6] = Mathf.Clamp(accel.x / maxAcceleration, -1f, 1f);
            ego[7] = Mathf.Clamp(accel.z / maxAcceleration, -1f, 1f);

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
                    int wpIdx = Mathf.Min(currentWaypointIndex + i, routeWaypoints.Length - 1);
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
                Debug.Log($"[AgentBv2] Step={episodeSteps} Accel={acceleration:F2} Steer={steering:F3} Speed={currentSpeed:F2} RbV={rb.linearVelocity.magnitude:F3} Pos=({transform.position.x:F1},{transform.position.z:F1})");
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
            // 0.001/frame -> equilibrium ~80 m/s (clamped by maxSpeed=30)
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

            // 2. Speed policy rewards
            speedR = CalculateSpeedPolicyReward();
            reward += speedR;

            // 2.5. Lane keeping
            laneR = CalculateLaneKeepingReward();
            reward += laneR;

            // 2.6. Overtaking reward
            overtakeR = CalculateOvertakingReward();
            reward += overtakeR;

            // 2.7. Lane violation penalty
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
                Debug.Log($"[DBG-Bv2] Step={episodeSteps} Spd={currentSpeed:F1}/{currentSpeedLimit:F1} Phase={overtakingPhase} " +
                          $"R={reward:F2} (prog={progressR:F2} spd={speedR:F2} lane={laneR:F2} ovt={overtakeR:F2} lnViol={laneViolR:F2}) " +
                          $"Stuck={stuckBehindTimer:F1}s L={leftLaneType} R={rightLaneType} Total={episodeReward:F1}");
            }

            episodeReward += reward;
            AddReward(reward);

            // TensorBoard StatsRecorder Logging
            if (episodeSteps % debugLogInterval == 0)
            {
                var stats = Academy.Instance.StatsRecorder;

                // Reward components (accumulated since last log)
                stats.Add("Reward/Progress", dbgProgressReward);
                stats.Add("Reward/Speed", dbgSpeedReward);
                stats.Add("Reward/LaneKeeping", dbgLaneKeepReward);
                stats.Add("Reward/Overtaking", dbgOvertakeReward);
                stats.Add("Reward/LaneViolation", dbgLaneViolationPenalty);
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
            Debug.Log($"[EPISODE END Bv2] Reason={dbgEpisodeEndReason} Steps={episodeSteps} TotalReward={episodeReward:F1} " +
                      $"Overtakes={overtakeCount} Collisions={episodeCollisions} " +
                      $"Components: Progress={dbgProgressReward:F1} Speed={dbgSpeedReward:F1} " +
                      $"LaneKeep={dbgLaneKeepReward:F1} Overtake={dbgOvertakeReward:F1} StuckPen={dbgStuckPenalty:F1} " +
                      $"LaneViol={dbgLaneViolationPenalty:F1}");

            // TensorBoard Episode Summary
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
        /// Check if vehicle is driving on wrong side of road (Phase F)
        /// Uses WaypointManager to check center line rules
        /// </summary>
        private bool IsWrongWayDriving()
        {
            if (waypointManager == null)
                return false;

            // Get vehicle's local X position relative to road
            Vector3 localPos = waypointManager.transform.InverseTransformPoint(transform.position);

            // Check with WaypointManager if this violates center line
            return waypointManager.IsWrongWayDriving(localPos.x);
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
        /// Speed policy reward (Phase B v2): targetSpeed = speedLimit ALWAYS.
        /// Never reduces target for lead vehicles. Being stuck behind slow NPC
        /// is inherently penalized because agent can't reach speedLimit.
        /// Stuck-behind timer adds explicit penalty after timeout.
        ///
        /// Phase B v2 change: Speed under penalty suspended when blocked by NPC ahead.
        /// This prevents the agent from learning to STOP as optimal policy.
        /// See: experiments/phase-B-decision/ROOT-CAUSE-ANALYSIS.md
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

            // targetSpeed is ALWAYS speedLimit - never reduce for lead vehicle
            float targetSpeed = Mathf.Max(speedLimit, 1f);
            float speedRatio = speed / targetSpeed;

            // === Stuck Behind Slow NPC Timer (PhaseA only) ===
            // Phase0/Phase0Sparse: No stuck penalty (following is rewarded instead)
            // PhaseA: Stuck penalty encourages overtaking
            if (trainingVersion == TrainingVersion.PhaseA)
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
                        reward += stuckBehindPenalty;  // -0.05/step after 5 seconds (v2: reduced)
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
            // Phase B v2: Suspend penalty when blocked by NPC ahead
            else if (speedRatio < 0.5f)
            {
                // Phase B v2: Check if agent is blocked by NPC ahead
                bool isBlockedByNPC = isBlocked &&
                    leadDist < safeFollowingDistance * 1.5f;

                if (!isBlockedByNPC)
                {
                    // speedRatio 0.0 -> penalty = -0.04 (v2: was -0.2)
                    // speedRatio 0.25 -> penalty = -0.03 (v2: was -0.15)
                    // speedRatio 0.5 -> penalty = -0.02 (v2: was -0.1)
                    float progressivePenalty = speedUnderPenalty * (2f - speedRatio * 2f);
                    reward += progressivePenalty;
                }
                // When blocked by NPC: no speed penalty
                // Agent shouldn't be punished for physics impossibility
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
        /// Overtaking reward (PhaseA): Dense reward at every sub-phase.
        /// State machine: None -> Approaching -> Beside -> Ahead -> LaneReturn -> None
        /// Each transition gives immediate reward to guide the agent.
        ///
        /// Phase B v2: Boosted rewards (4x initiate, 2.5x beside, 2x ahead, 1.5x complete)
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
            // Phase0 MODE: Following reward only, no overtaking
            // ============================================================
            if (trainingVersion == TrainingVersion.Phase0)
            {
                return CalculateFollowingReward_Phase0(closestNPC, closestNPCSpeed, closestNPCAhead);
            }

            // ============================================================
            // Phase0Sparse MODE: Following reward (gated) + Sparse overtaking
            // ============================================================
            if (trainingVersion == TrainingVersion.Phase0Sparse)
            {
                return CalculateOvertakingReward_Phase0Sparse(closestNPC, closestNPCSpeed, closestNPCAhead);
            }

            // ============================================================
            // PhaseA MODE: Dense 5-phase overtaking
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
                        reward += overtakeInitiateBonus;  // +2.0: lane change initiated (v2: 4x)
                        Debug.Log($"[Overtake Bv2] BESIDE phase entered. Reward +{overtakeInitiateBonus}");
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
                        reward += overtakeBesideBonus;  // +0.5/step (v2: 2.5x)
                    }
                    // Transition to Ahead: ego has passed NPC
                    if (npcAhead < -3f)
                    {
                        overtakingPhase = OvertakingPhase.Ahead;
                        reward += overtakeAheadBonus;  // +2.0: passed NPC (v2: 2x)
                        overtakeCount++;
                        Debug.Log($"[Overtake Bv2] AHEAD phase. Pass #{overtakeCount}. Reward +{overtakeAheadBonus}");
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
                        reward += overtakeCompleteBonus;  // +3.0: full overtake complete (v2: 1.5x)
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                        Debug.Log($"[Overtake Bv2] COMPLETE! Lane return done. Reward +{overtakeCompleteBonus}");
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
        /// Phase0: Following reward only, no overtaking.
        /// Rewards maintaining safe following distance behind NPCs.
        /// </summary>
        private float CalculateFollowingReward_Phase0(Transform closestNPC, float npcSpeed, float npcAhead)
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
        /// Phase0Sparse: Following reward (gated) + Sparse overtaking.
        /// Following bonus only when NPC > 70% speedLimit.
        /// Sparse overtaking: overtakePassBonus when complete, overtakeSpeedBonus per-step beside NPC.
        /// </summary>
        private float CalculateOvertakingReward_Phase0Sparse(Transform closestNPC, float npcSpeed, float npcAhead)
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
                        // Phase0Sparse: No initiate bonus (sparse)
                    }
                    break;

                case OvertakingPhase.Beside:
                    // Phase0Sparse: Per-step bonus for speed beside NPC
                    if (currentSpeed > npcSpeed * 0.9f)
                    {
                        reward += overtakeSpeedBonus;  // +0.15/step (Phase0Sparse sparse)
                    }
                    // Transition to Completed
                    if (npcAhead < -3f)
                    {
                        overtakingPhase = OvertakingPhase.None;
                        isOvertaking = false;
                        reward += overtakePassBonus;  // +3.0 (Phase0Sparse sparse, one-time)
                        overtakeCount++;
                        Debug.Log($"[Phase0Sparse Overtake] COMPLETE! Pass #{overtakeCount}. Sparse bonus +{overtakePassBonus}");
                    }
                    break;
            }

            dbgOvertakeReward += reward;
            return reward;
        }

        /// <summary>
        /// Lane violation penalty (Phase C).
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
            // Max distance exceeded
            float distFromStart = Vector3.Distance(transform.position, startPosition);
            if (distFromStart > maxEpisodeDistance)
            {
                dbgEpisodeEndReason = "MAX_DISTANCE";
                LogEpisodeSummary();
                EndEpisode();
                return;
            }

            // Max steps (handled by ML-Agents MaxStep setting)
            // Stuck detection (low speed for too long)
            if (episodeSteps > 500 && currentSpeed < 0.1f)
            {
                dbgEpisodeEndReason = "STUCK_LOW_SPEED";
                LogEpisodeSummary();
                AddReward(-1f);
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
                    Debug.Log($"[COLLISION Bv2] #{episodeCollisions} with {collision.gameObject.name} at step {episodeSteps}");
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

            // Fallback: keyboard control
            var continuousActions = actionsOut.ContinuousActions;
            continuousActions[0] = Input.GetAxis("Horizontal");  // Steering
            continuousActions[1] = Input.GetAxis("Vertical");    // Acceleration
        }

        // Public API for external monitoring
        public float GetEpisodeReward() => episodeReward;
        public int GetEpisodeSteps() => episodeSteps;
        public float GetTotalDistance() => totalDistance;
        public float GetCurrentSpeed() => rb != null ? rb.linearVelocity.magnitude : 0f;
    }
}