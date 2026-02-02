using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.MLAgents;
using System.Collections.Generic;
using ADPlatform.Agents;
using ADPlatform.Inference;

namespace ADPlatform.Environment
{
    /// <summary>
    /// Manages the driving scene lifecycle.
    /// Connects waypoints, NPC vehicles, and the ego agent.
    /// Handles environment reset for RL training episodes.
    /// Supports curriculum learning via ML-Agents EnvironmentParameters.
    /// </summary>
    public class DrivingSceneManager : MonoBehaviour
    {
        [Header("References")]
        public E2EDrivingAgent egoAgent;
        public AutonomousDrivingController adController;
        public WaypointManager waypointManager;
        public Transform goalTarget;

        [Header("NPC Vehicles")]
        public NPCVehicleController[] npcVehicles;

        [Header("Intersection Visuals")]
        public GameObject intersectionArea;
        public GameObject leftArm;
        public GameObject rightArm;
        public GameObject leftAngledArm;
        public GameObject rightAngledArm;

        [Header("Traffic Signal (Phase J)")]
        public TrafficLightController trafficLight;

        [Header("NPC Spawning")]
        public float npcMinSpawnDistance = 20f;    // Min distance ahead of ego (meters)
        public float npcMaxSpawnDistance = 150f;   // Max distance ahead of ego (meters)
        public float npcMinSpacing = 15f;          // Min distance between NPCs (meters)
        public float npcLateralOffsetMin = 1.5f;   // Min lateral offset from center (meters)
        public float npcLateralOffsetMax = 2.5f;   // Max lateral offset from center (meters)

        [Header("Episode Settings")]
        public float maxEpisodeTime = 60f;
        public bool autoReset = true;

        [Header("Curriculum")]
        public bool useCurriculum = true;
        public float defaultGoalDistance = 230f;

        [Header("Inference Testing")]
        [Tooltip("When true, use override values below instead of ML-Agents curriculum parameters")]
        public bool inferenceMode = false;
        public int inferenceNumNPCs = 3;
        public float inferenceGoalDistance = 230f;
        public int inferenceSpeedZoneCount = 1;
        public float inferenceRoadCurvature = 0f;
        public float inferenceCurveVariation = 0f;
        public int inferenceNumLanes = 2;
        public bool inferenceCenterLine = true;
        public int inferenceIntersectionType = 3;
        public int inferenceTurnDirection = 2;
        public float inferenceNpcSpeedRatio = 0.85f;
        public float inferenceNpcSpeedVariation = 0.15f;
        public bool inferenceTrafficSignalEnabled = true;
        public float inferenceSignalGreenRatio = 0.5f;

        [Header("Stats")]
        public int episodeCount = 0;
        public float episodeTime = 0f;
        public int activeNPCCount = 0;

        void Start()
        {
            // Search within Training Area root (parent of DrivingEnvironment)
            Transform areaRoot = transform.parent;

            if (egoAgent == null)
                egoAgent = areaRoot.GetComponentInChildren<E2EDrivingAgent>();
            if (adController == null)
                adController = areaRoot.GetComponentInChildren<AutonomousDrivingController>();
            if (waypointManager == null)
                waypointManager = areaRoot.GetComponentInChildren<WaypointManager>();

            if (npcVehicles == null || npcVehicles.Length == 0)
                npcVehicles = areaRoot.GetComponentsInChildren<NPCVehicleController>();

            // Defer connection to ensure WaypointManager has initialized
            Invoke(nameof(DelayedConnect), 0.1f);
        }

        private void DelayedConnect()
        {
            FindIntersectionVisuals();
            FindTrafficLight();
            ValidateActiveScene();
            ConnectWaypoints();

            Debug.Log($"[DrivingSceneManager] Initialized - {npcVehicles.Length} NPCs, " +
                     $"Agent={egoAgent != null}, Controller={adController != null}");
        }

        /// <summary>
        /// Validate that the active Unity scene matches curriculum requirements (P-011).
        /// Logs warnings if multi-lane or intersection features are expected but the scene
        /// doesn't support them. Prevents silent training failures from scene mismatch.
        /// </summary>
        private void ValidateActiveScene()
        {
            string sceneName = SceneManager.GetActiveScene().name;

            // Check multi-lane requirement
            float maxLanes = GetCurriculumParam("num_lanes", 1f);
            if (maxLanes > 1f && !sceneName.Contains("MultiLane") && !sceneName.Contains("PhaseF") && !sceneName.Contains("PhaseK"))
            {
                Debug.LogError($"[SCENE MISMATCH] num_lanes curriculum expects multi-lane but active scene is '{sceneName}'. " +
                    $"Expected: PhaseF_MultiLane. Training will likely fail at lane transition. (P-011)");
            }

            // Check intersection requirement
            float intersectionType = GetCurriculumParam("intersection_type", 0f);
            if (intersectionType > 0f && !sceneName.Contains("Intersection") && !sceneName.Contains("PhaseG") && !sceneName.Contains("PhaseH") && !sceneName.Contains("PhaseK"))
            {
                Debug.LogError($"[SCENE MISMATCH] intersection_type curriculum expects intersections but active scene is '{sceneName}'. " +
                    $"Expected: PhaseG_Intersection or PhaseH_NPCIntersection. (P-011)");
            }

            // Check road width for multi-lane
            if (waypointManager != null && maxLanes > 1f)
            {
                float requiredWidth = maxLanes * waypointManager.laneWidth + 1f;  // lanes * 3.5m + margin
                // Log the expected road configuration for verification
                Debug.Log($"[SceneValidation] Scene='{sceneName}', MaxLanes={maxLanes}, " +
                    $"RequiredRoadWidth={requiredWidth:F1}m, LaneWidth={waypointManager.laneWidth}m");
            }

            Debug.Log($"[SceneValidation] Active scene: '{sceneName}' - validation passed");
        }

        /// <summary>
        /// Connect generated waypoints to the driving agent and controller
        /// </summary>
        private void ConnectWaypoints()
        {
            if (waypointManager == null) return;

            Transform[] waypoints = waypointManager.GetAllWaypoints();
            if (waypoints.Length == 0)
            {
                waypointManager.GenerateWaypoints();
                waypoints = waypointManager.GetAllWaypoints();
            }

            if (egoAgent != null)
            {
                egoAgent.routeWaypoints = waypoints;
                if (goalTarget != null)
                    egoAgent.goalTarget = goalTarget;
                Debug.Log($"[DrivingSceneManager] Connected {waypoints.Length} waypoints to E2EDrivingAgent");
            }

            if (adController != null)
            {
                adController.routeWaypoints = waypoints;
                if (goalTarget != null)
                    adController.goalTarget = goalTarget;
            }
        }

        void Update()
        {
            episodeTime += Time.deltaTime;

            // Auto-reset on timeout
            if (autoReset && episodeTime > maxEpisodeTime)
            {
                ResetEpisode();
            }
        }

        /// <summary>
        /// Reset the entire driving environment for a new episode
        /// </summary>
        public void ResetEpisode()
        {
            episodeCount++;
            episodeTime = 0f;

            // Reset waypoint progress
            if (waypointManager != null)
                waypointManager.ResetProgress();

            // Apply curriculum parameters
            if (useCurriculum || inferenceMode)
                ApplyCurriculumParameters();

            Debug.Log($"[DrivingSceneManager] Episode {episodeCount} started (NPCs: {activeNPCCount})");
        }

        /// <summary>
        /// Get a curriculum parameter value, using inference overrides when in inference mode.
        /// </summary>
        private float GetCurriculumParam(string name, float defaultValue)
        {
            if (inferenceMode)
            {
                switch (name)
                {
                    case "num_active_npcs": return inferenceNumNPCs;
                    case "goal_distance": return inferenceGoalDistance;
                    case "speed_zone_count": return inferenceSpeedZoneCount;
                    case "road_curvature": return inferenceRoadCurvature;
                    case "curve_direction_variation": return inferenceCurveVariation;
                    case "num_lanes": return inferenceNumLanes;
                    case "center_line_enabled": return inferenceCenterLine ? 1f : 0f;
                    case "intersection_type": return inferenceIntersectionType;
                    case "turn_direction": return inferenceTurnDirection;
                    case "npc_speed_ratio": return inferenceNpcSpeedRatio;
                    case "npc_speed_variation": return inferenceNpcSpeedVariation;
                    case "traffic_signal_enabled": return inferenceTrafficSignalEnabled ? 1f : 0f;
                    case "signal_green_ratio": return inferenceSignalGreenRatio;
                    default: return defaultValue;
                }
            }
            return Academy.Instance.EnvironmentParameters.GetWithDefault(name, defaultValue);
        }

        /// <summary>
        /// Read curriculum parameters from ML-Agents and adjust environment
        /// </summary>
        private void ApplyCurriculumParameters()
        {
            // Control number of active NPCs
            int numNPCs = Mathf.RoundToInt(GetCurriculumParam("num_active_npcs", 0f));
            activeNPCCount = Mathf.Clamp(numNPCs, 0, npcVehicles.Length);

            for (int i = 0; i < npcVehicles.Length; i++)
            {
                if (npcVehicles[i] != null)
                    npcVehicles[i].gameObject.SetActive(i < activeNPCCount);
            }

            // Control goal distance (relative to Training Area origin)
            float goalDist = GetCurriculumParam("goal_distance", defaultGoalDistance);
            if (goalTarget != null)
            {
                Vector3 areaOrigin = transform.position;
                Vector3 goalPos = goalTarget.position;
                goalPos.z = areaOrigin.z + goalDist;
                goalTarget.position = goalPos;
            }

            // Control speed zone count (Stage 4: Speed Policy)
            int speedZoneCount = Mathf.RoundToInt(GetCurriculumParam("speed_zone_count", 1f));
            if (waypointManager != null)
            {
                waypointManager.SetSpeedZoneCount(speedZoneCount);
            }

            // Road curvature (Phase E / Phase K: Dense Urban)
            // Always set curvature fields to clear stale values from previous episodes.
            // Phase K supports curved approach + intersection simultaneously.
            float roadCurvature = GetCurriculumParam("road_curvature", 0f);
            float curveDirectionVariation = GetCurriculumParam("curve_direction_variation", 0f);
            if (waypointManager != null)
            {
                waypointManager.roadCurvature = Mathf.Clamp01(roadCurvature);
                waypointManager.curveDirectionVariation = Mathf.Clamp01(curveDirectionVariation);
            }

            // Multi-lane support (Phase F)
            int numLanes = Mathf.RoundToInt(GetCurriculumParam("num_lanes", 1f));
            bool centerLineEnabled = GetCurriculumParam("center_line_enabled", 0f) > 0.5f;
            if (waypointManager != null)
            {
                waypointManager.SetLaneCount(numLanes);
                waypointManager.SetCenterLineEnabled(centerLineEnabled);
            }

            // Intersection support (Phase G / Phase K)
            // Always call SetIntersection to regenerate waypoints with current curvature + intersection.
            int intersectionType = Mathf.RoundToInt(GetCurriculumParam("intersection_type", 0f));
            int turnDirection = Mathf.RoundToInt(GetCurriculumParam("turn_direction", 0f));
            if (waypointManager != null)
            {
                waypointManager.SetIntersection(intersectionType, turnDirection);
            }

            // Toggle intersection road visuals to match current type
            UpdateIntersectionVisuals(intersectionType);

            // Traffic signal (Phase J)
            float trafficSignalEnabled = GetCurriculumParam("traffic_signal_enabled", 0f);
            float signalGreenRatio = GetCurriculumParam("signal_green_ratio", 0.5f);
            if (trafficLight != null)
            {
                trafficLight.ResetSignal(trafficSignalEnabled > 0.5f, signalGreenRatio);
            }

            // NPC speed variation range (curriculum: 0.0 -> 0.3)
            float speedVariation = GetCurriculumParam("npc_speed_variation", 0.0f);

            // NPC base speed ratio (v12: 0.3 = very slow for overtaking training)
            // Default 1.0 = NPCs drive at speed limit
            float npcSpeedRatio = GetCurriculumParam("npc_speed_ratio", 1.0f);

            // Get current zone speed limit for NPC speed calculation
            float zoneSpeedLimit = waypointManager != null
                ? waypointManager.defaultSpeedLimit
                : 16.67f;  // fallback: 60 km/h

            // Spawn NPCs at random positions with varied speeds
            SpawnNPCsRandomly(zoneSpeedLimit, speedVariation, npcSpeedRatio);
        }

        /// <summary>
        /// Randomly place active NPCs on waypoints ahead of ego vehicle.
        /// Speed = speedLimit * baseRatio * (1.0 +/- variation).
        /// v12: baseRatio controls how slow NPCs are (0.3 = 30% of limit).
        /// Phase H: Uses waypoint-index-based spawning for intersections.
        /// </summary>
        private void SpawnNPCsRandomly(float speedLimit, float speedVariation, float baseRatio = 1.0f)
        {
            if (activeNPCCount == 0) return;

            Transform[] waypoints = waypointManager != null
                ? waypointManager.GetAllWaypoints()
                : null;

            Vector3 egoPosition = egoAgent != null
                ? egoAgent.transform.position
                : transform.position;

            // Phase H: Use waypoint-index-based spawning for intersections
            // Z-distance filtering breaks when waypoints turn at intersections
            bool isIntersection = waypointManager != null && waypointManager.intersectionType > 0;
            if (isIntersection && waypoints != null && waypoints.Length > 0)
            {
                SpawnNPCsOnWaypoints(waypoints, egoPosition, speedLimit, speedVariation, baseRatio);
                return;
            }

            // ---- Original Z-distance-based logic (non-intersection) ----

            // Collect valid spawn positions (ahead of ego, on waypoints)
            List<Vector3> spawnPositions = new List<Vector3>();

            if (waypoints != null && waypoints.Length > 0)
            {
                foreach (var wp in waypoints)
                {
                    if (wp == null) continue;
                    float distAhead = wp.position.z - egoPosition.z;
                    if (distAhead >= npcMinSpawnDistance && distAhead <= npcMaxSpawnDistance)
                    {
                        spawnPositions.Add(wp.position);
                    }
                }
            }

            // Fallback: generate positions along forward direction
            if (spawnPositions.Count < activeNPCCount)
            {
                for (int i = 0; i < activeNPCCount * 3; i++)
                {
                    float dist = Random.Range(npcMinSpawnDistance, npcMaxSpawnDistance);
                    Vector3 pos = egoPosition + Vector3.forward * dist;
                    spawnPositions.Add(pos);
                }
            }

            // Shuffle spawn positions
            for (int i = spawnPositions.Count - 1; i > 0; i--)
            {
                int j = Random.Range(0, i + 1);
                Vector3 temp = spawnPositions[i];
                spawnPositions[i] = spawnPositions[j];
                spawnPositions[j] = temp;
            }

            // Assign NPCs to positions with minimum spacing check
            List<Vector3> usedPositions = new List<Vector3>();
            int spawnIdx = 0;

            for (int i = 0; i < activeNPCCount && spawnIdx < spawnPositions.Count; i++)
            {
                if (npcVehicles[i] == null) continue;

                // Find a position with enough spacing from already-placed NPCs
                Vector3 chosenPos = Vector3.zero;
                bool found = false;

                while (spawnIdx < spawnPositions.Count)
                {
                    Vector3 candidate = spawnPositions[spawnIdx];
                    spawnIdx++;

                    bool tooClose = false;
                    foreach (var used in usedPositions)
                    {
                        if (Vector3.Distance(candidate, used) < npcMinSpacing)
                        {
                            tooClose = true;
                            break;
                        }
                    }

                    if (!tooClose)
                    {
                        chosenPos = candidate;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    // Fallback: place at fixed distance
                    chosenPos = egoPosition + Vector3.forward * (npcMinSpawnDistance + i * npcMinSpacing);
                }

                usedPositions.Add(chosenPos);

                // Apply lateral offset for overtaking
                // Phase F: Use lane positions if multi-lane, otherwise use random offset
                float lateralOffset;
                if (waypointManager != null && waypointManager.numLanes > 1)
                {
                    // Multi-lane: spawn NPC in a random lane (different from ego's lane 0)
                    int npcLane = Random.Range(0, waypointManager.numLanes);
                    lateralOffset = waypointManager.GetLaneXPosition(npcLane) - chosenPos.x;
                }
                else
                {
                    // Single lane: random side offset
                    lateralOffset = Random.Range(npcLateralOffsetMin, npcLateralOffsetMax);
                    if (Random.value > 0.5f) lateralOffset = -lateralOffset;
                }
                Vector3 spawnPos = chosenPos + Vector3.right * lateralOffset;

                // NPC speed = speedLimit * baseRatio * (1.0 +/- variation)
                // v12 Phase A: baseRatio=0.3, variation=0.0 -> NPC at 30% of limit
                float speedMultiplier = baseRatio + Random.Range(-speedVariation, speedVariation);
                float npcSpeed = speedLimit * Mathf.Max(speedMultiplier, 0.2f);  // floor at 20%

                Quaternion forward = Quaternion.LookRotation(Vector3.forward);
                npcVehicles[i].SpawnAt(spawnPos, forward, npcSpeed);
            }
        }

        /// <summary>
        /// Spawn NPCs on waypoints using path-distance (waypoint index) instead of Z-distance.
        /// Correct for intersections where waypoints turn and Z-distance is misleading.
        /// NPCs are placed ahead of ego with minimum waypoint-index spacing.
        /// </summary>
        private void SpawnNPCsOnWaypoints(Transform[] waypoints, Vector3 egoPosition,
            float speedLimit, float speedVariation, float baseRatio)
        {
            // Find ego's nearest waypoint index
            int egoWpIndex = 0;
            float minDist = float.MaxValue;
            for (int i = 0; i < waypoints.Length; i++)
            {
                if (waypoints[i] == null) continue;
                float dist = Vector3.Distance(egoPosition, waypoints[i].position);
                if (dist < minDist)
                {
                    minDist = dist;
                    egoWpIndex = i;
                }
            }

            // Spawn NPCs at waypoint indices ahead of ego
            // Min 3 waypoints ahead (~60m gap at 20m spacing) for first NPC
            // Min 2 waypoints between NPCs (~40m spacing)
            int minAheadWPs = 3;
            int minSpacingWPs = 2;
            int nextWpIndex = egoWpIndex + minAheadWPs;

            for (int i = 0; i < activeNPCCount; i++)
            {
                if (npcVehicles[i] == null) continue;

                // Check bounds
                if (nextWpIndex >= waypoints.Length)
                {
                    // Not enough waypoints ahead — deactivate remaining NPCs
                    npcVehicles[i].gameObject.SetActive(false);
                    continue;
                }

                // NPC speed = speedLimit * baseRatio * (1.0 +/- variation)
                float speedMultiplier = baseRatio + Random.Range(-speedVariation, speedVariation);
                float npcSpeed = speedLimit * Mathf.Max(speedMultiplier, 0.2f);

                npcVehicles[i].SpawnAtWaypoint(nextWpIndex, npcSpeed, waypoints);
                nextWpIndex += minSpacingWPs;
            }
        }

        /// <summary>
        /// Auto-discover intersection visual GameObjects by name in the Road hierarchy.
        /// Called once in Start() as a fallback if not wired by PhaseSceneCreator.
        /// </summary>
        private void FindIntersectionVisuals()
        {
            Transform areaRoot = transform.parent;
            if (areaRoot == null) return;

            Transform road = areaRoot.Find("Road");
            if (road == null) return;

            if (intersectionArea == null)
            {
                Transform t = road.Find("IntersectionArea");
                if (t != null) intersectionArea = t.gameObject;
            }
            if (leftArm == null)
            {
                Transform t = road.Find("LeftArm");
                if (t != null) leftArm = t.gameObject;
            }
            if (rightArm == null)
            {
                Transform t = road.Find("RightArm");
                if (t != null) rightArm = t.gameObject;
            }
            if (leftAngledArm == null)
            {
                Transform t = road.Find("LeftAngledArm");
                if (t != null) leftAngledArm = t.gameObject;
            }
            if (rightAngledArm == null)
            {
                Transform t = road.Find("RightAngledArm");
                if (t != null) rightAngledArm = t.gameObject;
            }
        }

        /// <summary>
        /// Auto-discover TrafficLightController in the Training Area hierarchy.
        /// </summary>
        private void FindTrafficLight()
        {
            if (trafficLight != null) return;

            Transform areaRoot = transform.parent;
            if (areaRoot == null) return;

            trafficLight = areaRoot.GetComponentInChildren<TrafficLightController>(true);

            // Wire to agent
            if (trafficLight != null && egoAgent != null && egoAgent.trafficLight == null)
            {
                egoAgent.trafficLight = trafficLight;
            }
        }

        /// <summary>
        /// Toggle intersection arm visuals based on intersection type.
        /// 0=None, 1=T-junction(right only), 2=Cross(left+right), 3=Y-junction(angled arms).
        /// </summary>
        private void UpdateIntersectionVisuals(int intersectionType)
        {
            bool showIntersection = intersectionType > 0;
            bool showLeftArm = intersectionType == 2;           // Cross only
            bool showRightArm = intersectionType == 1 || intersectionType == 2;  // T-junction + Cross
            bool showLeftAngled = intersectionType == 3;        // Y-junction
            bool showRightAngled = intersectionType == 3;       // Y-junction

            if (intersectionArea != null) intersectionArea.SetActive(showIntersection);
            if (leftArm != null) leftArm.SetActive(showLeftArm);
            if (rightArm != null) rightArm.SetActive(showRightArm);
            if (leftAngledArm != null) leftAngledArm.SetActive(showLeftAngled);
            if (rightAngledArm != null) rightAngledArm.SetActive(showRightAngled);
        }

        /// <summary>
        /// Get current episode info for logging
        /// </summary>
        public (int episode, float time, float agentSpeed) GetEpisodeInfo()
        {
            float speed = 0f;
            if (egoAgent != null)
                speed = egoAgent.GetCurrentSpeed();
            else if (adController != null)
                speed = adController.currentSpeed;

            return (episodeCount, episodeTime, speed);
        }
    }
}
