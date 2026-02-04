using UnityEngine;
using Unity.MLAgents.Policies;
using ADPlatform.Agents;
using ADPlatform.Environment;
using ADPlatform.DebugTools;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Central orchestrator for Phase M multi-agent inference test field.
    /// Manages 12 RL agents, 25 NPCs, 8 pedestrians, and traffic signals.
    /// Supports both linear road (v1) and grid network (v2) modes.
    /// </summary>
    [DefaultExecutionOrder(-100)]
    public class TestFieldManager : MonoBehaviour
    {
        [Header("Agents")]
        public E2EDrivingAgent[] agents;
        public Transform[] goalTargets;

        [Header("NPCs")]
        public NPCVehicleController[] npcVehicles;

        [Header("Pedestrians")]
        public PedestrianController[] pedestrians;

        [Header("Environment (Legacy Linear)")]
        public WaypointManager waypointManager;
        public TrafficLightController trafficLight;

        [Header("Environment (Grid v2)")]
        public GridRoadNetwork gridNetwork;
        public GridTrafficLightManager gridTrafficManager;

        [Header("Camera")]
        public FollowCamera followCamera;

        [Header("Config")]
        public float goalDistance = 230f;
        public float roadLength = 2000f;
        public float npcSpeedRatio = 0.85f;
        public float respawnTimeout = 120f;
        public int goalWaypointLookahead = 12;

        [Header("Heuristic Mode")]
        [Tooltip("Enable Pure Pursuit heuristic control instead of ONNX inference")]
        public bool enableHeuristicMode = false;

        // Runtime
        private float[] agentTimers;
        private int currentAgentIndex = 0;
        private float halfRoad;

        // Grid mode runtime
        private bool isGridMode;
        private Transform[][] agentRouteWaypoints;
        private GridWaypointProxy[] agentProxies;
        private int[] agentCurrentWpIndex;

        void Awake()
        {
            // P-031: Set BehaviorType BEFORE ML-Agents Agent.LazyInitialize()
            // Must run in Awake() because ML-Agents reads BehaviorType in first FixedUpdate
            if (enableHeuristicMode && agents != null)
            {
                for (int i = 0; i < agents.Length; i++)
                {
                    if (agents[i] != null)
                        SetupHeuristicAgent(agents[i]);
                }
            }
        }

        void Start()
        {
            halfRoad = roadLength / 2f;
            agentTimers = new float[agents != null ? agents.Length : 0];

            isGridMode = gridNetwork != null;

            if (isGridMode)
            {
                InitializeGridRoutes();
                InitializeGridTrafficLights();
            }
            else
            {
                InitializeTrafficLight();
            }

            InitializeNPCs();
            InitializePedestrians();
            InitializeAgentGoals();

            string mode = isGridMode ? "Grid 4x4" : "Linear";
            string control = enableHeuristicMode ? "Heuristic (Pure Pursuit)" : "ONNX Inference";
            Debug.Log($"[TestFieldManager] Initialized ({mode}, {control}): {agents?.Length ?? 0} agents, " +
                      $"{npcVehicles?.Length ?? 0} NPCs, {pedestrians?.Length ?? 0} pedestrians");
        }

        void Update()
        {
            if (agents == null) return;

            for (int i = 0; i < agents.Length; i++)
            {
                if (agents[i] == null) continue;

                agentTimers[i] += Time.deltaTime;

                // Grid mode: update dynamic traffic light and proxy
                if (isGridMode)
                {
                    UpdateGridAgent(i);
                }

                // Check goal reached
                if (goalTargets != null && i < goalTargets.Length && goalTargets[i] != null)
                {
                    float distToGoal = Vector3.Distance(agents[i].transform.position, goalTargets[i].position);
                    if (distToGoal < 10f)
                    {
                        Debug.Log($"[TestFieldManager] Agent_{i} reached goal in {agentTimers[i]:F1}s");
                        RespawnAgent(i);
                    }
                }

                // Grid mode: check if agent went off the grid boundaries
                if (isGridMode)
                {
                    float boundary = gridNetwork.blockSize * (gridNetwork.gridSize - 1) / 2f + 30f;
                    Vector3 pos = agents[i].transform.position;
                    if (Mathf.Abs(pos.x) > boundary || Mathf.Abs(pos.z) > boundary)
                    {
                        Debug.Log($"[TestFieldManager] Agent_{i} out of bounds at ({pos.x:F0},{pos.z:F0}), respawning");
                        RespawnAgent(i);
                        continue;
                    }
                }

                // Check timeout
                if (agentTimers[i] > respawnTimeout)
                {
                    Debug.Log($"[TestFieldManager] Agent_{i} timed out after {respawnTimeout}s");
                    RespawnAgent(i);
                }
            }
        }

        // ============================================================
        // Grid Mode Initialization
        // ============================================================

        private void InitializeGridRoutes()
        {
            if (agents == null || gridNetwork == null) return;

            var routes = GridRoutes.GetAgentRoutes();
            agentRouteWaypoints = new Transform[agents.Length][];
            agentProxies = new GridWaypointProxy[agents.Length];
            agentCurrentWpIndex = new int[agents.Length];

            for (int i = 0; i < agents.Length; i++)
            {
                if (agents[i] == null) continue;

                int routeIdx = i % routes.Length;
                var route = routes[routeIdx];

                // Create waypoint container per agent
                var wpParent = new GameObject($"Route_{i}_Waypoints");
                wpParent.transform.SetParent(transform);

                // Generate waypoints
                agentRouteWaypoints[i] = gridNetwork.GenerateRouteWaypoints(route, wpParent.transform);

                // Assign to agent
                agents[i].routeWaypoints = agentRouteWaypoints[i];

                // Setup proxy
                agentProxies[i] = agents[i].GetComponent<GridWaypointProxy>();
                if (agentProxies[i] == null)
                    agentProxies[i] = agents[i].gameObject.AddComponent<GridWaypointProxy>();

                agentProxies[i].gridNetwork = gridNetwork;
                agentProxies[i].InitializeForRoute(route, routeIdx);

                // Assign proxy as the waypointManager
                agents[i].waypointManager = agentProxies[i];

                agentCurrentWpIndex[i] = 0;

                // P-030: Ensure correct 280D observation vector for ONNX model
                agents[i].enableLaneObservation = true;
                agents[i].enableIntersectionObservation = true;
                agents[i].enableTrafficSignalObservation = true;
                agents[i].enablePedestrianObservation = true;

                Debug.Log($"[TestFieldManager] Agent_{i}: route {routeIdx}, {agentRouteWaypoints[i].Length} waypoints");
            }
        }

        private void InitializeGridTrafficLights()
        {
            if (gridTrafficManager == null || gridNetwork == null) return;
            gridTrafficManager.InitializeCoordination(gridNetwork);
        }

        /// <summary>
        /// P-031: Configure agent for heuristic Pure Pursuit mode.
        /// Attaches ExpertDriverController and sets BehaviorType to HeuristicOnly.
        /// </summary>
        private void SetupHeuristicAgent(E2EDrivingAgent agent)
        {
            // Add ExpertDriverController if not present
            var expert = agent.GetComponent<ExpertDriverController>();
            if (expert == null)
                expert = agent.gameObject.AddComponent<ExpertDriverController>();

            expert.enabled = true;
            expert.autoRecord = false;

            // Set BehaviorType to HeuristicOnly
            var bp = agent.GetComponent<BehaviorParameters>();
            if (bp != null)
            {
                bp.BehaviorType = BehaviorType.HeuristicOnly;
            }
        }

        // ============================================================
        // Grid Mode Update
        // ============================================================

        private void UpdateGridAgent(int i)
        {
            if (agents[i] == null) return;

            // Update current waypoint index and sync to agent
            if (agentRouteWaypoints != null && agentRouteWaypoints[i] != null)
            {
                UpdateAgentWaypointIndex(i);
                // P-028: Sync waypoint index so agent observes correct upcoming waypoints
                agents[i].CurrentWaypointIndex = agentCurrentWpIndex[i];
                UpdateGridGoalPosition(i);
            }

            // Dynamic traffic light assignment
            if (gridTrafficManager != null)
            {
                var light = gridTrafficManager.GetRelevantLight(
                    agents[i].transform.position, agents[i].transform.forward);
                agents[i].trafficLight = light;
            }

            // Update proxy
            if (agentProxies != null && agentProxies[i] != null)
            {
                agentProxies[i].UpdateForAgent(
                    agents[i].transform.position,
                    agents[i].transform.forward,
                    agentCurrentWpIndex[i]);
            }
        }

        private void UpdateAgentWaypointIndex(int i)
        {
            var wps = agentRouteWaypoints[i];
            if (wps == null || wps.Length == 0) return;

            int current = agentCurrentWpIndex[i];
            float reachDist = 5f;

            // Advance through reached waypoints
            int checks = 0;
            while (checks < 5)
            {
                int idx = current % wps.Length;
                if (wps[idx] == null) break;

                float dist = Vector3.Distance(agents[i].transform.position, wps[idx].position);
                if (dist < reachDist)
                {
                    current++;
                    checks++;
                }
                else
                {
                    break;
                }
            }

            // P-030: Snap-to-nearest recovery when agent drifts off-route.
            // If current waypoint is too far, find the nearest ahead waypoint
            // to prevent stuck observations pointing at wrong angles.
            int curIdx = current % wps.Length;
            if (wps[curIdx] != null)
            {
                float distToCurrent = Vector3.Distance(agents[i].transform.position, wps[curIdx].position);
                if (distToCurrent > 15f)
                {
                    Vector3 agentFwd = agents[i].transform.forward;
                    float bestScore = float.MaxValue;
                    int bestIdx = current;

                    for (int w = 0; w < wps.Length; w++)
                    {
                        if (wps[w] == null) continue;
                        Vector3 toWp = wps[w].position - agents[i].transform.position;
                        float d = toWp.magnitude;
                        float dot = d > 0.1f ? Vector3.Dot(toWp / d, agentFwd) : 0f;

                        // Prefer waypoints ahead (dot > 0) and nearby
                        // Score: distance penalized if behind agent
                        float score = d + (dot < 0f ? 50f : 0f);
                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestIdx = w;
                        }
                    }
                    current = bestIdx;
                }
            }

            agentCurrentWpIndex[i] = current;
        }

        private void UpdateGridGoalPosition(int i)
        {
            if (goalTargets == null || i >= goalTargets.Length || goalTargets[i] == null) return;

            var wps = agentRouteWaypoints[i];
            if (wps == null || wps.Length == 0) return;

            // Place goal at N waypoints ahead
            int goalIdx = (agentCurrentWpIndex[i] + goalWaypointLookahead) % wps.Length;
            if (wps[goalIdx] != null)
            {
                goalTargets[i].position = wps[goalIdx].position + Vector3.up * 0.5f;
            }
        }

        // ============================================================
        // NPC / Pedestrian / Legacy Init
        // ============================================================

        private void InitializeNPCs()
        {
            if (npcVehicles == null) return;

            float defaultSpeed = waypointManager != null ? waypointManager.defaultSpeedLimit : 16.67f;
            if (gridNetwork != null)
                defaultSpeed = gridNetwork.defaultSpeedLimit;

            for (int i = 0; i < npcVehicles.Length; i++)
            {
                if (npcVehicles[i] == null) continue;

                npcVehicles[i].gameObject.SetActive(true);

                float speedVariation = Random.Range(0.7f, 1.0f);
                float npcSpeed = defaultSpeed * npcSpeedRatio * speedVariation;

                var speedField = npcVehicles[i].GetType().GetField("cruiseSpeed");
                if (speedField != null)
                    speedField.SetValue(npcVehicles[i], npcSpeed);

                // Assign waypoints (legacy linear mode)
                if (waypointManager != null && gridNetwork == null)
                {
                    var wpField = npcVehicles[i].GetType().GetField("waypointManager");
                    if (wpField != null)
                        wpField.SetValue(npcVehicles[i], waypointManager);
                }
            }
        }

        private void InitializePedestrians()
        {
            if (pedestrians == null) return;

            for (int i = 0; i < pedestrians.Length; i++)
            {
                if (pedestrians[i] == null) continue;
                pedestrians[i].gameObject.SetActive(true);
            }
        }

        private void InitializeTrafficLight()
        {
            if (trafficLight == null) return;
            trafficLight.ResetSignal(true, 0.5f);
        }

        private void InitializeAgentGoals()
        {
            if (agents == null || goalTargets == null) return;

            for (int i = 0; i < agents.Length && i < goalTargets.Length; i++)
            {
                if (agents[i] != null && goalTargets[i] != null)
                {
                    agents[i].goalTarget = goalTargets[i];

                    if (!isGridMode)
                        UpdateGoalPosition(i);
                }
            }
        }

        // ============================================================
        // Respawn
        // ============================================================

        private void RespawnAgent(int i)
        {
            if (agents[i] == null) return;

            if (isGridMode)
                RespawnGridAgent(i);
            else
                RespawnLinearAgent(i);

            agentTimers[i] = 0f;
        }

        private void RespawnGridAgent(int i)
        {
            if (agentRouteWaypoints == null || agentRouteWaypoints[i] == null) return;

            var wps = agentRouteWaypoints[i];
            if (wps.Length == 0) return;

            // P-030: Pick random waypoint that's safely inside the grid boundary.
            // Outer-edge waypoints can cause immediate out-of-bounds after spawning.
            float safeBoundary = gridNetwork.blockSize * (gridNetwork.gridSize - 1) / 2f - 20f;
            int startIdx = -1;
            for (int attempt = 0; attempt < 20; attempt++)
            {
                int candidateIdx = Random.Range(0, wps.Length);
                if (wps[candidateIdx] == null) continue;
                Vector3 cPos = wps[candidateIdx].position;
                if (Mathf.Abs(cPos.x) < safeBoundary && Mathf.Abs(cPos.z) < safeBoundary)
                {
                    startIdx = candidateIdx;
                    break;
                }
            }
            // Fallback: use a waypoint near the center of the route
            if (startIdx < 0)
                startIdx = wps.Length / 4;

            int nextIdx = (startIdx + 1) % wps.Length;

            Vector3 newPos = wps[startIdx].position;
            newPos.y = 0.75f;

            // Face toward the next waypoint
            Vector3 fwd = (wps[nextIdx].position - wps[startIdx].position).normalized;
            if (fwd.sqrMagnitude < 0.01f) fwd = Vector3.forward;
            fwd.y = 0;
            Quaternion newRot = Quaternion.LookRotation(fwd, Vector3.up);

            agents[i].SetStartPose(newPos, newRot);

            // P-030: Immediately teleport transform to prevent OOB race condition.
            // OnEpisodeBegin() only runs on next FixedUpdate, but Update() boundary
            // check can fire again before that. Direct teleport ensures safe position.
            agents[i].transform.position = newPos;
            agents[i].transform.rotation = newRot;
            var agentRb = agents[i].GetComponent<Rigidbody>();
            if (agentRb != null)
            {
                agentRb.linearVelocity = Vector3.zero;
                agentRb.angularVelocity = Vector3.zero;
            }

            agents[i].EndEpisode();

            // P-030: Sync index AFTER EndEpisode (OnEpisodeBegin resets to 0)
            agentCurrentWpIndex[i] = startIdx;
            agents[i].CurrentWaypointIndex = startIdx;
            UpdateGridGoalPosition(i);
        }

        private void RespawnLinearAgent(int i)
        {
            float newZ = Random.Range(-halfRoad + 20f, 0f);
            float[] laneXPositions = { 1.75f, 0f, -1.75f };
            float newX = laneXPositions[Random.Range(0, laneXPositions.Length)];

            Vector3 newPos = new Vector3(newX, 0.75f, newZ);
            Quaternion newRot = Quaternion.identity;

            agents[i].SetStartPose(newPos, newRot);
            agents[i].EndEpisode();

            UpdateGoalPosition(i);
        }

        private void UpdateGoalPosition(int i)
        {
            if (goalTargets == null || i >= goalTargets.Length || goalTargets[i] == null) return;
            if (agents == null || i >= agents.Length || agents[i] == null) return;

            float goalZ = Mathf.Min(agents[i].transform.position.z + goalDistance, halfRoad - 20f);
            goalTargets[i].position = new Vector3(0f, 1f, goalZ);
        }
    }
}
