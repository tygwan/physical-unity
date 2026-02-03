using UnityEngine;
using ADPlatform.Agents;
using ADPlatform.Environment;
using ADPlatform.DebugTools;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Central orchestrator for Phase M multi-agent inference test field.
    /// Manages 12 RL agents, 25 NPCs, 8 pedestrians, and traffic signals
    /// on a single shared 2000m road.
    /// </summary>
    public class TestFieldManager : MonoBehaviour
    {
        [Header("Agents")]
        public E2EDrivingAgent[] agents;
        public Transform[] goalTargets;

        [Header("NPCs")]
        public NPCVehicleController[] npcVehicles;

        [Header("Pedestrians")]
        public PedestrianController[] pedestrians;

        [Header("Environment")]
        public WaypointManager waypointManager;
        public TrafficLightController trafficLight;

        [Header("Camera")]
        public FollowCamera followCamera;

        [Header("Config")]
        public float goalDistance = 230f;
        public float roadLength = 2000f;
        public float npcSpeedRatio = 0.85f;
        public float respawnTimeout = 120f;

        // Runtime
        private float[] agentTimers;
        private int currentAgentIndex = 0;
        private float halfRoad;

        void Start()
        {
            halfRoad = roadLength / 2f;
            agentTimers = new float[agents != null ? agents.Length : 0];

            InitializeNPCs();
            InitializePedestrians();
            InitializeTrafficLight();
            InitializeAgentGoals();

            Debug.Log($"[TestFieldManager] Initialized: {agents?.Length ?? 0} agents, " +
                      $"{npcVehicles?.Length ?? 0} NPCs, {pedestrians?.Length ?? 0} pedestrians");
        }

        void Update()
        {
            if (agents == null) return;

            // Update agent timers and check for respawn conditions
            for (int i = 0; i < agents.Length; i++)
            {
                if (agents[i] == null) continue;

                agentTimers[i] += Time.deltaTime;

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

                // Check timeout
                if (agentTimers[i] > respawnTimeout)
                {
                    Debug.Log($"[TestFieldManager] Agent_{i} timed out after {respawnTimeout}s");
                    RespawnAgent(i);
                }
            }

            // Camera cycling is handled by FollowCamera component (Tab / number keys)
        }

        private void InitializeNPCs()
        {
            if (npcVehicles == null) return;

            float defaultSpeedLimit = waypointManager != null ? waypointManager.defaultSpeedLimit : 16.67f;

            for (int i = 0; i < npcVehicles.Length; i++)
            {
                if (npcVehicles[i] == null) continue;

                npcVehicles[i].gameObject.SetActive(true);

                // Set random speed based on speed limit
                float speedVariation = Random.Range(0.7f, 1.0f);
                float npcSpeed = defaultSpeedLimit * npcSpeedRatio * speedVariation;

                // Use reflection to set cruiseSpeed (same pattern as DrivingSceneManager)
                var speedField = npcVehicles[i].GetType().GetField("cruiseSpeed");
                if (speedField != null)
                    speedField.SetValue(npcVehicles[i], npcSpeed);

                // Assign waypoints
                if (waypointManager != null)
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
                    UpdateGoalPosition(i);
                }
            }
        }

        private void RespawnAgent(int i)
        {
            if (agents[i] == null) return;

            // Random Z in first half of road
            float newZ = Random.Range(-halfRoad + 20f, 0f);
            // Cycle across 3 lanes
            float[] laneXPositions = { 1.75f, 0f, -1.75f };
            float newX = laneXPositions[Random.Range(0, laneXPositions.Length)];

            Vector3 newPos = new Vector3(newX, 0.75f, newZ);
            Quaternion newRot = Quaternion.identity;

            agents[i].SetStartPose(newPos, newRot);
            agents[i].EndEpisode();

            // Update goal target
            UpdateGoalPosition(i);

            agentTimers[i] = 0f;
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
