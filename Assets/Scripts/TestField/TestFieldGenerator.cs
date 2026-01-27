using UnityEngine;
using System.Collections.Generic;
using System.Linq;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Difficulty presets for test field generation
    /// </summary>
    public enum TestFieldDifficulty
    {
        Easy,       // Mostly straight, 2 lanes, slow NPCs
        Medium,     // Mixed curves, 2-3 lanes, moderate NPCs
        Hard,       // Sharp curves, 3-4 lanes, fast NPCs
        Expert,     // Complex scenarios, lane changes, high NPC density
        Random      // Fully random within constraints
    }

    /// <summary>
    /// Configuration for procedural test field generation
    /// </summary>
    [System.Serializable]
    public class TestFieldConfig
    {
        [Header("Basic Settings")]
        public int seed = 12345;
        public float totalLength = 2000f;        // Total road length in meters
        public TestFieldDifficulty difficulty = TestFieldDifficulty.Medium;

        [Header("Segment Distribution")]
        [Range(0f, 1f)] public float straightWeight = 0.4f;
        [Range(0f, 1f)] public float gentleCurveWeight = 0.25f;
        [Range(0f, 1f)] public float sharpCurveWeight = 0.15f;
        [Range(0f, 1f)] public float sCurveWeight = 0.1f;
        [Range(0f, 1f)] public float laneTransitionWeight = 0.1f;

        [Header("Road Properties")]
        public int minLanes = 2;
        public int maxLanes = 4;
        public float minSegmentLength = 50f;
        public float maxSegmentLength = 200f;

        [Header("Curve Parameters")]
        public float minCurveRadius = 30f;       // Sharp curves
        public float maxCurveRadius = 150f;      // Gentle curves
        public float minCurveAngle = 15f;
        public float maxCurveAngle = 90f;

        [Header("Variation")]
        [Range(0f, 1f)] public float npcDensity = 0.5f;
        [Range(0f, 1f)] public float obstacleChance = 0.1f;
        [Range(0f, 1f)] public float speedZoneVariation = 0.5f;

        [Header("Route Planning")]
        public int numCheckpoints = 5;           // Intermediate goals
        public bool enableAlternateRoutes = false;

        /// <summary>
        /// Apply difficulty preset
        /// </summary>
        public void ApplyDifficulty()
        {
            switch (difficulty)
            {
                case TestFieldDifficulty.Easy:
                    straightWeight = 0.6f;
                    gentleCurveWeight = 0.3f;
                    sharpCurveWeight = 0.05f;
                    sCurveWeight = 0.05f;
                    laneTransitionWeight = 0f;
                    minLanes = 2; maxLanes = 2;
                    npcDensity = 0.2f;
                    minCurveRadius = 100f;
                    break;

                case TestFieldDifficulty.Medium:
                    straightWeight = 0.4f;
                    gentleCurveWeight = 0.3f;
                    sharpCurveWeight = 0.15f;
                    sCurveWeight = 0.1f;
                    laneTransitionWeight = 0.05f;
                    minLanes = 2; maxLanes = 3;
                    npcDensity = 0.4f;
                    minCurveRadius = 60f;
                    break;

                case TestFieldDifficulty.Hard:
                    straightWeight = 0.25f;
                    gentleCurveWeight = 0.25f;
                    sharpCurveWeight = 0.25f;
                    sCurveWeight = 0.15f;
                    laneTransitionWeight = 0.1f;
                    minLanes = 2; maxLanes = 4;
                    npcDensity = 0.6f;
                    minCurveRadius = 40f;
                    break;

                case TestFieldDifficulty.Expert:
                    straightWeight = 0.15f;
                    gentleCurveWeight = 0.2f;
                    sharpCurveWeight = 0.3f;
                    sCurveWeight = 0.2f;
                    laneTransitionWeight = 0.15f;
                    minLanes = 2; maxLanes = 4;
                    npcDensity = 0.8f;
                    minCurveRadius = 30f;
                    obstacleChance = 0.2f;
                    break;
            }
        }
    }

    /// <summary>
    /// Generated test field data
    /// </summary>
    public class TestFieldData
    {
        public TestFieldConfig config;
        public List<RoadSegmentData> segments = new List<RoadSegmentData>();
        public List<Vector3> allWaypoints = new List<Vector3>();
        public List<Vector3> checkpoints = new List<Vector3>();
        public float totalArcLength;
        public Bounds bounds;

        // Variation data
        public List<Vector3> npcSpawnPoints = new List<Vector3>();
        public List<Vector3> obstaclePositions = new List<Vector3>();
        public Dictionary<int, SpeedZoneType> speedZones = new Dictionary<int, SpeedZoneType>();
    }

    /// <summary>
    /// Procedural test field generator
    /// Combines segments, parametric curves, and variations for ultimate testbed
    /// </summary>
    public class TestFieldGenerator : MonoBehaviour
    {
        [Header("Configuration")]
        public TestFieldConfig config;

        [Header("Generated Data")]
        public TestFieldData generatedField;

        [Header("Visualization")]
        public bool showGizmos = true;
        public bool showWaypoints = true;
        public bool showBoundaries = true;
        public bool showSpeedZones = true;

        private System.Random rng;

        /// <summary>
        /// Generate a new test field with given seed
        /// </summary>
        public TestFieldData Generate(int? overrideSeed = null)
        {
            int seed = overrideSeed ?? config.seed;
            rng = new System.Random(seed);

            config.ApplyDifficulty();

            generatedField = new TestFieldData { config = config };

            // Phase 1: Generate road segments
            GenerateSegments();

            // Phase 2: Collect all waypoints
            CollectWaypoints();

            // Phase 3: Place checkpoints (for route planning)
            PlaceCheckpoints();

            // Phase 4: Generate variations (NPCs, obstacles, speed zones)
            GenerateVariations();

            // Phase 5: Calculate bounds
            CalculateBounds();

            Debug.Log($"[TestFieldGenerator] Generated test field: seed={seed}, " +
                     $"segments={generatedField.segments.Count}, " +
                     $"waypoints={generatedField.allWaypoints.Count}, " +
                     $"length={generatedField.totalArcLength:F0}m");

            return generatedField;
        }

        /// <summary>
        /// Generate road segments based on configuration
        /// </summary>
        private void GenerateSegments()
        {
            Vector3 currentPos = transform.position;
            float currentHeading = 0f;
            float totalLength = 0f;
            int segmentIndex = 0;
            int currentLanes = config.minLanes;

            while (totalLength < config.totalLength)
            {
                // Select segment type based on weights
                SegmentType segType = SelectSegmentType();

                // Generate segment config
                var segConfig = CreateSegmentConfig(segType, ref currentLanes);

                // Ensure we don't exceed total length
                float remainingLength = config.totalLength - totalLength;
                if (segConfig.length > remainingLength)
                    segConfig.length = remainingLength;

                if (segConfig.length < 20f) break;  // Minimum segment

                // Generate the segment
                var segment = RoadSegmentGenerator.Generate(segConfig, currentPos, currentHeading);
                segment.segmentIndex = segmentIndex++;

                generatedField.segments.Add(segment);

                // Update position for next segment
                currentPos = segment.endPoint;
                currentHeading = segment.endHeading;
                totalLength += segment.arcLength;
                generatedField.totalArcLength = totalLength;
            }
        }

        private SegmentType SelectSegmentType()
        {
            float total = config.straightWeight + config.gentleCurveWeight +
                         config.sharpCurveWeight + config.sCurveWeight +
                         config.laneTransitionWeight;

            float roll = (float)rng.NextDouble() * total;
            float cumulative = 0f;

            cumulative += config.straightWeight;
            if (roll < cumulative) return SegmentType.Straight;

            cumulative += config.gentleCurveWeight;
            if (roll < cumulative) return rng.NextDouble() > 0.5 ? SegmentType.ArcLeft : SegmentType.ArcRight;

            cumulative += config.sharpCurveWeight;
            if (roll < cumulative) return rng.NextDouble() > 0.5 ? SegmentType.ArcLeft : SegmentType.ArcRight;

            cumulative += config.sCurveWeight;
            if (roll < cumulative) return SegmentType.SCurve;

            cumulative += config.laneTransitionWeight;
            if (roll < cumulative) return rng.NextDouble() > 0.5 ? SegmentType.LaneMerge : SegmentType.LaneExpand;

            return SegmentType.Straight;
        }

        private RoadSegmentConfig CreateSegmentConfig(SegmentType type, ref int currentLanes)
        {
            var cfg = new RoadSegmentConfig
            {
                type = type,
                length = Mathf.Lerp(config.minSegmentLength, config.maxSegmentLength, (float)rng.NextDouble()),
                laneCount = currentLanes,
                npcDensity = config.npcDensity,
                obstacleChance = config.obstacleChance
            };

            // Speed zone based on variation setting
            cfg.speedZone = SelectSpeedZone();

            // Curve parameters
            bool isSharpCurve = (float)rng.NextDouble() > 0.5f;
            if (isSharpCurve)
            {
                cfg.curveRadius = Mathf.Lerp(config.minCurveRadius, (config.minCurveRadius + config.maxCurveRadius) / 2f, (float)rng.NextDouble());
                cfg.curveAngle = Mathf.Lerp(45f, config.maxCurveAngle, (float)rng.NextDouble());
            }
            else
            {
                cfg.curveRadius = Mathf.Lerp((config.minCurveRadius + config.maxCurveRadius) / 2f, config.maxCurveRadius, (float)rng.NextDouble());
                cfg.curveAngle = Mathf.Lerp(config.minCurveAngle, 45f, (float)rng.NextDouble());
            }

            // Handle lane transitions
            if (type == SegmentType.LaneMerge)
            {
                cfg.laneCount = Mathf.Min(currentLanes + 2, config.maxLanes);
                currentLanes = Mathf.Max(config.minLanes, currentLanes - 2);
            }
            else if (type == SegmentType.LaneExpand)
            {
                currentLanes = Mathf.Min(config.maxLanes, currentLanes + 2);
                cfg.laneCount = currentLanes;
            }

            return cfg;
        }

        private SpeedZoneType SelectSpeedZone()
        {
            if (config.speedZoneVariation < 0.1f)
                return SpeedZoneType.General;

            float roll = (float)rng.NextDouble();
            if (roll < 0.1f) return SpeedZoneType.Residential;
            if (roll < 0.3f) return SpeedZoneType.Urban;
            if (roll < 0.7f) return SpeedZoneType.General;
            if (roll < 0.9f) return SpeedZoneType.Highway;
            return SpeedZoneType.Expressway;
        }

        private void CollectWaypoints()
        {
            generatedField.allWaypoints.Clear();

            foreach (var segment in generatedField.segments)
            {
                // Avoid duplicating endpoints
                int startIdx = generatedField.allWaypoints.Count > 0 ? 1 : 0;
                for (int i = startIdx; i < segment.waypoints.Count; i++)
                {
                    generatedField.allWaypoints.Add(segment.waypoints[i]);
                }
            }
        }

        private void PlaceCheckpoints()
        {
            generatedField.checkpoints.Clear();

            if (generatedField.allWaypoints.Count == 0) return;

            // Place checkpoints evenly along the route
            int interval = generatedField.allWaypoints.Count / (config.numCheckpoints + 1);

            for (int i = 1; i <= config.numCheckpoints; i++)
            {
                int idx = Mathf.Min(i * interval, generatedField.allWaypoints.Count - 1);
                generatedField.checkpoints.Add(generatedField.allWaypoints[idx]);
            }

            // Final checkpoint is the end
            generatedField.checkpoints.Add(generatedField.allWaypoints[generatedField.allWaypoints.Count - 1]);
        }

        private void GenerateVariations()
        {
            // NPC spawn points
            generatedField.npcSpawnPoints.Clear();
            foreach (var segment in generatedField.segments)
            {
                if ((float)rng.NextDouble() > segment.config.npcDensity) continue;

                // Place 1-3 NPCs per segment
                int npcCount = rng.Next(1, 4);
                for (int i = 0; i < npcCount; i++)
                {
                    int wpIdx = rng.Next(0, segment.waypoints.Count);
                    Vector3 spawnPos = segment.waypoints[wpIdx];

                    // Random lane offset
                    float laneOffset = (float)(rng.NextDouble() - 0.5) * segment.config.laneCount * segment.config.laneWidth;
                    spawnPos.x += laneOffset;

                    generatedField.npcSpawnPoints.Add(spawnPos);
                }
            }

            // Obstacle positions
            generatedField.obstaclePositions.Clear();
            foreach (var segment in generatedField.segments)
            {
                if ((float)rng.NextDouble() > segment.config.obstacleChance) continue;

                int wpIdx = rng.Next(segment.waypoints.Count / 4, segment.waypoints.Count * 3 / 4);
                Vector3 obstaclePos = segment.waypoints[wpIdx];
                float offset = (float)(rng.NextDouble() - 0.5) * segment.config.laneWidth;
                obstaclePos.x += offset;

                generatedField.obstaclePositions.Add(obstaclePos);
            }

            // Speed zones (per segment)
            generatedField.speedZones.Clear();
            for (int i = 0; i < generatedField.segments.Count; i++)
            {
                generatedField.speedZones[i] = generatedField.segments[i].config.speedZone;
            }
        }

        private void CalculateBounds()
        {
            if (generatedField.allWaypoints.Count == 0)
            {
                generatedField.bounds = new Bounds(transform.position, Vector3.one * 100f);
                return;
            }

            Vector3 min = generatedField.allWaypoints[0];
            Vector3 max = generatedField.allWaypoints[0];

            foreach (var wp in generatedField.allWaypoints)
            {
                min = Vector3.Min(min, wp);
                max = Vector3.Max(max, wp);
            }

            // Add some padding
            float padding = 50f;
            generatedField.bounds = new Bounds(
                (min + max) / 2f,
                (max - min) + Vector3.one * padding
            );
        }

        void OnDrawGizmos()
        {
            if (!showGizmos || generatedField == null) return;

            // Draw waypoints
            if (showWaypoints)
            {
                Gizmos.color = Color.cyan;
                foreach (var wp in generatedField.allWaypoints)
                {
                    Gizmos.DrawSphere(wp, 0.5f);
                }
            }

            // Draw boundaries
            if (showBoundaries)
            {
                foreach (var segment in generatedField.segments)
                {
                    // Left boundary
                    Gizmos.color = Color.white;
                    for (int i = 0; i < segment.leftBoundary.Count - 1; i++)
                    {
                        Gizmos.DrawLine(segment.leftBoundary[i], segment.leftBoundary[i + 1]);
                    }

                    // Right boundary
                    for (int i = 0; i < segment.rightBoundary.Count - 1; i++)
                    {
                        Gizmos.DrawLine(segment.rightBoundary[i], segment.rightBoundary[i + 1]);
                    }
                }
            }

            // Draw checkpoints
            Gizmos.color = Color.green;
            foreach (var cp in generatedField.checkpoints)
            {
                Gizmos.DrawWireSphere(cp, 3f);
            }

            // Draw NPC spawn points
            Gizmos.color = Color.red;
            foreach (var npc in generatedField.npcSpawnPoints)
            {
                Gizmos.DrawWireCube(npc, Vector3.one * 2f);
            }

            // Draw obstacles
            Gizmos.color = Color.yellow;
            foreach (var obs in generatedField.obstaclePositions)
            {
                Gizmos.DrawWireCube(obs, Vector3.one * 1.5f);
            }

            // Draw speed zones
            if (showSpeedZones)
            {
                foreach (var segment in generatedField.segments)
                {
                    Color zoneColor = GetSpeedZoneColor(segment.config.speedZone);
                    Gizmos.color = zoneColor;
                    if (segment.waypoints.Count > 0)
                    {
                        Gizmos.DrawWireCube(segment.waypoints[0] + Vector3.up * 5f, new Vector3(10f, 2f, 0.5f));
                    }
                }
            }
        }

        private Color GetSpeedZoneColor(SpeedZoneType zone)
        {
            switch (zone)
            {
                case SpeedZoneType.Residential: return Color.red;
                case SpeedZoneType.Urban: return new Color(1f, 0.5f, 0f);  // Orange
                case SpeedZoneType.General: return Color.yellow;
                case SpeedZoneType.Highway: return Color.green;
                case SpeedZoneType.Expressway: return Color.cyan;
                default: return Color.white;
            }
        }
    }
}
