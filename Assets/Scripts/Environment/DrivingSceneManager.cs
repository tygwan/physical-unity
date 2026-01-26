using UnityEngine;
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
            ConnectWaypoints();

            Debug.Log($"[DrivingSceneManager] Initialized - {npcVehicles.Length} NPCs, " +
                     $"Agent={egoAgent != null}, Controller={adController != null}");
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
            if (useCurriculum)
                ApplyCurriculumParameters();

            Debug.Log($"[DrivingSceneManager] Episode {episodeCount} started (NPCs: {activeNPCCount})");
        }

        /// <summary>
        /// Read curriculum parameters from ML-Agents and adjust environment
        /// </summary>
        private void ApplyCurriculumParameters()
        {
            var envParams = Academy.Instance.EnvironmentParameters;

            // Control number of active NPCs
            int numNPCs = Mathf.RoundToInt(envParams.GetWithDefault("num_active_npcs", 0f));
            activeNPCCount = Mathf.Clamp(numNPCs, 0, npcVehicles.Length);

            for (int i = 0; i < npcVehicles.Length; i++)
            {
                if (npcVehicles[i] != null)
                    npcVehicles[i].gameObject.SetActive(i < activeNPCCount);
            }

            // Control goal distance (relative to Training Area origin)
            float goalDist = envParams.GetWithDefault("goal_distance", defaultGoalDistance);
            if (goalTarget != null)
            {
                Vector3 areaOrigin = transform.position;
                Vector3 goalPos = goalTarget.position;
                goalPos.z = areaOrigin.z + goalDist;
                goalTarget.position = goalPos;
            }

            // Control speed zone count (Stage 4: Speed Policy)
            int speedZoneCount = Mathf.RoundToInt(envParams.GetWithDefault("speed_zone_count", 1f));
            if (waypointManager != null)
            {
                waypointManager.SetSpeedZoneCount(speedZoneCount);
            }

            // NPC speed variation range (curriculum: 0.0 -> 0.3)
            float speedVariation = envParams.GetWithDefault("npc_speed_variation", 0.0f);

            // NPC base speed ratio (v12: 0.3 = very slow for overtaking training)
            // Default 1.0 = NPCs drive at speed limit
            float npcSpeedRatio = envParams.GetWithDefault("npc_speed_ratio", 1.0f);

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

                // Apply lateral offset so ego can overtake
                // Random side: +X (right) or -X (left)
                float lateralOffset = Random.Range(npcLateralOffsetMin, npcLateralOffsetMax);
                if (Random.value > 0.5f) lateralOffset = -lateralOffset;
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
