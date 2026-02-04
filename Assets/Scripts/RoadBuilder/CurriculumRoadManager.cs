using System;
using Unity.MLAgents;
using UnityEngine;
using ADPlatform.Agents;

namespace ADPlatform.RoadBuilder
{
    /// <summary>
    /// Manages curriculum-based road generation for ML-Agents RL training.
    /// Reads environment parameters to determine current stage and configures
    /// ProceduralRoadBuilder accordingly.
    ///
    /// 4-stage curriculum:
    ///   Stage 0 (Straight): Simple straight roads, 1-2 lanes
    ///   Stage 1 (Curves): Curved roads, 1-3 lanes
    ///   Stage 2 (Intersections): Intersections with turns, 2-3 lanes
    ///   Stage 3 (FullTraffic): Everything combined, 2-4 lanes + signals
    /// </summary>
    public class CurriculumRoadManager : MonoBehaviour
    {
        public enum CurriculumStage
        {
            Straight = 0,
            Curves = 1,
            Intersections = 2,
            FullTraffic = 3
        }

        [Header("References")]
        public ProceduralRoadBuilder roadBuilder;
        public WaypointManager waypointManager;

        [Header("Current State")]
        [SerializeField] private CurriculumStage currentStage = CurriculumStage.Straight;
        public bool useCurriculum = true;

        [Header("Stage 1: Straight")]
        public float straightRoadLength = 300f;
        public int straightNumLanes = 2;

        [Header("Stage 2: Curves")]
        public float curvesRoadLength = 400f;
        [Range(0f, 1f)] public float curveIntensity = 0.5f;
        public int curvesNumLanes = 2;

        [Header("Stage 3: Intersections")]
        public float intersectionRoadLength = 500f;
        public int intersectionNumLanes = 2;
        public int defaultIntersectionType = 2; // Cross

        [Header("Stage 4: Full Traffic")]
        public float fullTrafficRoadLength = 500f;
        public int fullTrafficNumLanes = 3;

        [Header("Waypoint Container")]
        public Transform waypointParent;

        private Transform[] lastWaypoints = Array.Empty<Transform>();

        public CurriculumStage CurrentStage => currentStage;

        void Start()
        {
            EnsureReferences();
            ConfigureForEpisode();
        }

        /// <summary>
        /// Configure road for current episode based on curriculum or inspector settings.
        /// Call this from DrivingSceneManager.ResetEpisode() or E2EDrivingAgent.OnEpisodeBegin().
        /// </summary>
        public void ConfigureForEpisode()
        {
            EnsureReferences();

            CurriculumStage stage;
            int numLanes;
            float curvature;
            int intersectionType;
            int turnDir;

            if (useCurriculum)
            {
                var env = Academy.Instance.EnvironmentParameters;
                int stageIdx = Mathf.Clamp(Mathf.RoundToInt(env.GetWithDefault("road_stage", 0f)), 0, 3);
                stage = (CurriculumStage)stageIdx;
                numLanes = Mathf.RoundToInt(env.GetWithDefault("num_lanes", GetDefaultLanes(stage)));
                curvature = env.GetWithDefault("road_curvature", stage >= CurriculumStage.Curves ? curveIntensity : 0f);
                intersectionType = Mathf.RoundToInt(env.GetWithDefault("intersection_type", stage >= CurriculumStage.Intersections ? defaultIntersectionType : 0));
                turnDir = Mathf.RoundToInt(env.GetWithDefault("turn_direction", 0f));
            }
            else
            {
                stage = currentStage;
                numLanes = GetDefaultLanes(stage);
                curvature = (stage >= CurriculumStage.Curves) ? curveIntensity : 0f;
                intersectionType = (stage >= CurriculumStage.Intersections) ? defaultIntersectionType : 0;
                turnDir = 0;
            }

            ApplySettings(stage, numLanes, curvature, intersectionType, turnDir);
        }

        public Transform[] GetCurrentWaypoints()
        {
            return lastWaypoints;
        }

        private void ApplySettings(CurriculumStage stage, int numLanes, float curvature,
            int intersectionType, int turnDirection)
        {
            currentStage = stage;
            float roadLength = GetStageRoadLength(stage);

            // Clamp values per stage
            numLanes = Mathf.Clamp(numLanes, GetMinLanes(stage), GetMaxLanes(stage));
            if (stage < CurriculumStage.Curves) curvature = 0f;
            if (stage < CurriculumStage.Intersections) intersectionType = 0;

            // Configure road builder directly
            if (roadBuilder != null)
            {
                roadBuilder.roadLength = roadLength;
                roadBuilder.numLanes = numLanes;
                roadBuilder.roadCurvature = Mathf.Clamp01(curvature);
                roadBuilder.intersectionType = Mathf.Clamp(intersectionType, 0, 3);
                roadBuilder.turnDirection = Mathf.Clamp(turnDirection, 0, 2);

                roadBuilder.CleanupRoad();
                roadBuilder.GenerateRoad();
            }

            // Generate waypoints
            if (waypointManager != null)
            {
                waypointManager.roadLength = roadLength;
                waypointManager.numLanes = numLanes;
                waypointManager.roadCurvature = Mathf.Clamp01(curvature);
                waypointManager.intersectionType = Mathf.Clamp(intersectionType, 0, 3);
                waypointManager.turnDirection = Mathf.Clamp(turnDirection, 0, 2);
                waypointManager.GenerateWaypoints();
                lastWaypoints = waypointManager.GetAllWaypoints();
            }
            else if (roadBuilder != null)
            {
                // Use road builder's waypoint generation
                Transform parent = waypointParent != null ? waypointParent : transform;
                lastWaypoints = roadBuilder.GenerateWaypoints(parent);
            }

            // Update agent routes
            UpdateAgentRoutes();

            Debug.Log($"[CurriculumRoadManager] Stage={stage}, Lanes={numLanes}, " +
                $"Curvature={curvature:F2}, Intersection={intersectionType}, " +
                $"Waypoints={lastWaypoints.Length}");
        }

        private void UpdateAgentRoutes()
        {
            if (lastWaypoints == null || lastWaypoints.Length == 0) return;

            var agents = FindObjectsByType<E2EDrivingAgent>(FindObjectsSortMode.None);
            foreach (var agent in agents)
            {
                agent.routeWaypoints = lastWaypoints;
                if (agent.waypointManager == null && waypointManager != null)
                    agent.waypointManager = waypointManager;
            }
        }

        private void EnsureReferences()
        {
            if (roadBuilder == null)
                roadBuilder = FindAnyObjectByType<ProceduralRoadBuilder>();
            if (waypointManager == null)
                waypointManager = FindAnyObjectByType<WaypointManager>();
        }

        private float GetStageRoadLength(CurriculumStage stage)
        {
            switch (stage)
            {
                case CurriculumStage.Curves: return curvesRoadLength;
                case CurriculumStage.Intersections: return intersectionRoadLength;
                case CurriculumStage.FullTraffic: return fullTrafficRoadLength;
                default: return straightRoadLength;
            }
        }

        private int GetDefaultLanes(CurriculumStage stage)
        {
            switch (stage)
            {
                case CurriculumStage.Curves: return curvesNumLanes;
                case CurriculumStage.Intersections: return intersectionNumLanes;
                case CurriculumStage.FullTraffic: return fullTrafficNumLanes;
                default: return straightNumLanes;
            }
        }

        private int GetMinLanes(CurriculumStage stage)
        {
            return stage >= CurriculumStage.Intersections ? 2 : 1;
        }

        private int GetMaxLanes(CurriculumStage stage)
        {
            switch (stage)
            {
                case CurriculumStage.Curves: return 3;
                case CurriculumStage.Intersections: return 3;
                case CurriculumStage.FullTraffic: return 4;
                default: return 2;
            }
        }
    }
}
