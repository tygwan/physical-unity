using UnityEngine;
using System.Collections.Generic;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Speed zone types based on Korean road traffic regulations (도로교통법 제17조)
    /// </summary>
    public enum SpeedZoneType
    {
        Residential,      // 주거지역/스쿨존: 30 km/h (8.33 m/s)
        UrbanNarrow,      // 시가지 이면도로: 50 km/h (13.89 m/s)
        UrbanGeneral,     // 일반도로 (도시부): 60 km/h (16.67 m/s)
        Expressway        // 자동차전용도로: 80 km/h (22.22 m/s)
    }

    /// <summary>
    /// Defines a speed zone along the road
    /// </summary>
    [System.Serializable]
    public struct SpeedZone
    {
        public SpeedZoneType type;
        public float startZ;      // Zone start position (local Z)
        public float speedLimit;   // m/s

        public SpeedZone(SpeedZoneType type, float startZ, float speedLimit)
        {
            this.type = type;
            this.startZ = startZ;
            this.speedLimit = speedLimit;
        }
    }

    /// <summary>
    /// Manages route waypoints for the ego vehicle.
    /// Generates waypoints along the road and provides them to the E2EDrivingAgent.
    /// Includes speed zone management for traffic regulation compliance.
    /// </summary>
    public class WaypointManager : MonoBehaviour
    {
        [Header("Waypoint Generation")]
        public float roadLength = 500f;
        public float waypointSpacing = 20f;
        public float laneX = 1.75f;          // X position of the lane
        public float waypointY = 0.5f;

        [Header("Speed Zones")]
        [Tooltip("Number of speed zones (controlled by curriculum 'speed_zone_count' parameter)")]
        public int numSpeedZones = 1;         // Curriculum-controlled: 1~4 zones
        public float defaultSpeedLimit = 16.67f;  // 60 km/h default

        [Header("Visualization")]
        public bool showGizmos = true;
        public Color waypointColor = Color.cyan;
        public float gizmoSize = 0.5f;

        [Header("Runtime")]
        public Transform egoVehicle;
        public int currentWaypointIndex = 0;
        public float waypointReachDistance = 5f;

        private List<Transform> waypoints = new List<Transform>();
        private List<SpeedZone> speedZones = new List<SpeedZone>();
        private float[] waypointSpeedLimits;  // Speed limit per waypoint

        void Start()
        {
            GenerateSpeedZones();
            GenerateWaypoints();
        }

        /// <summary>
        /// Generate speed zones based on numSpeedZones (curriculum-controlled)
        /// Korean road regulations: 30/50/60/80 km/h zones
        /// </summary>
        public void GenerateSpeedZones()
        {
            speedZones.Clear();
            float startZ = -roadLength / 2f;
            float zoneLength = roadLength / Mathf.Max(numSpeedZones, 1);

            if (numSpeedZones <= 1)
            {
                // Single zone: default speed limit (60 km/h)
                speedZones.Add(new SpeedZone(SpeedZoneType.UrbanGeneral, startZ, defaultSpeedLimit));
            }
            else
            {
                // Multi-zone layout: Residential → UrbanNarrow → UrbanGeneral → Expressway
                SpeedZoneType[] zoneTypes = {
                    SpeedZoneType.Residential,    // 30 km/h
                    SpeedZoneType.UrbanNarrow,    // 50 km/h
                    SpeedZoneType.UrbanGeneral,   // 60 km/h
                    SpeedZoneType.Expressway      // 80 km/h
                };
                float[] zoneLimits = { 8.33f, 13.89f, 16.67f, 22.22f };

                for (int i = 0; i < numSpeedZones; i++)
                {
                    int typeIdx = Mathf.Min(i, zoneTypes.Length - 1);
                    float zoneStart = startZ + i * zoneLength;
                    speedZones.Add(new SpeedZone(zoneTypes[typeIdx], zoneStart, zoneLimits[typeIdx]));
                }
            }
        }

        /// <summary>
        /// Generate waypoints along the road with speed limit tags
        /// </summary>
        public void GenerateWaypoints()
        {
            // Clear existing
            foreach (Transform child in transform)
            {
                Destroy(child.gameObject);
            }
            waypoints.Clear();

            // Generate waypoints from -roadLength/2 to +roadLength/2
            float startZ = -roadLength / 2f;
            float endZ = roadLength / 2f;
            int count = 0;

            for (float z = startZ; z <= endZ; z += waypointSpacing)
            {
                GameObject wp = new GameObject($"WP_{count:D3}");
                wp.transform.SetParent(transform);
                wp.transform.localPosition = new Vector3(laneX, waypointY, z);
                waypoints.Add(wp.transform);
                count++;
            }

            // Assign speed limits to waypoints
            waypointSpeedLimits = new float[count];
            for (int i = 0; i < count; i++)
            {
                float wpZ = waypoints[i].localPosition.z;
                waypointSpeedLimits[i] = GetSpeedLimitAtZ(wpZ);
            }

            UnityEngine.Debug.Log($"[WaypointManager] Generated {count} waypoints, {speedZones.Count} speed zones");
        }

        /// <summary>
        /// Get speed limit at a given local Z position
        /// </summary>
        private float GetSpeedLimitAtZ(float localZ)
        {
            float limit = defaultSpeedLimit;
            for (int i = speedZones.Count - 1; i >= 0; i--)
            {
                if (localZ >= speedZones[i].startZ)
                {
                    limit = speedZones[i].speedLimit;
                    break;
                }
            }
            return limit;
        }

        /// <summary>
        /// Get current speed limit at the ego vehicle's position
        /// </summary>
        public float GetCurrentSpeedLimit(Vector3 worldPosition)
        {
            Vector3 localPos = transform.InverseTransformPoint(worldPosition);
            return GetSpeedLimitAtZ(localPos.z);
        }

        /// <summary>
        /// Get the next speed zone's limit (for pre-deceleration planning)
        /// Returns current limit if no next zone exists
        /// </summary>
        public float GetNextSpeedLimit(Vector3 worldPosition)
        {
            Vector3 localPos = transform.InverseTransformPoint(worldPosition);
            float currentZ = localPos.z;

            for (int i = 0; i < speedZones.Count; i++)
            {
                if (speedZones[i].startZ > currentZ)
                {
                    return speedZones[i].speedLimit;
                }
            }
            // No next zone, return current
            return GetSpeedLimitAtZ(currentZ);
        }

        /// <summary>
        /// Get distance to next speed zone boundary (for transition planning)
        /// Returns float.MaxValue if no upcoming zone change
        /// </summary>
        public float GetDistanceToNextZone(Vector3 worldPosition)
        {
            Vector3 localPos = transform.InverseTransformPoint(worldPosition);
            float currentZ = localPos.z;

            for (int i = 0; i < speedZones.Count; i++)
            {
                if (speedZones[i].startZ > currentZ)
                {
                    return speedZones[i].startZ - currentZ;
                }
            }
            return float.MaxValue;
        }

        /// <summary>
        /// Update speed zones from curriculum parameter
        /// Called by DrivingSceneManager when environment resets
        /// </summary>
        public void SetSpeedZoneCount(int count)
        {
            numSpeedZones = Mathf.Clamp(count, 1, 4);
            GenerateSpeedZones();
            // Re-assign speed limits to existing waypoints
            if (waypointSpeedLimits != null && waypoints.Count > 0)
            {
                for (int i = 0; i < waypoints.Count; i++)
                {
                    waypointSpeedLimits[i] = GetSpeedLimitAtZ(waypoints[i].localPosition.z);
                }
            }
        }

        /// <summary>
        /// Get the N nearest upcoming waypoints from current position
        /// </summary>
        public Transform[] GetUpcomingWaypoints(Vector3 position, int count = 10)
        {
            // Find nearest waypoint
            UpdateCurrentWaypoint(position);

            Transform[] result = new Transform[count];
            for (int i = 0; i < count; i++)
            {
                int idx = Mathf.Min(currentWaypointIndex + i, waypoints.Count - 1);
                result[i] = waypoints[idx];
            }
            return result;
        }

        /// <summary>
        /// Update current waypoint index based on ego position
        /// </summary>
        private void UpdateCurrentWaypoint(Vector3 position)
        {
            if (waypoints.Count == 0) return;

            // Check if we've passed the current waypoint
            while (currentWaypointIndex < waypoints.Count - 1)
            {
                float dist = Vector3.Distance(position, waypoints[currentWaypointIndex].position);
                if (dist < waypointReachDistance)
                {
                    currentWaypointIndex++;
                }
                else
                {
                    break;
                }
            }
        }

        /// <summary>
        /// Get all waypoints as Transform array (for E2EDrivingAgent.routeWaypoints)
        /// </summary>
        public Transform[] GetAllWaypoints()
        {
            return waypoints.ToArray();
        }

        /// <summary>
        /// Get the final waypoint (goal)
        /// </summary>
        public Transform GetGoal()
        {
            return waypoints.Count > 0 ? waypoints[waypoints.Count - 1] : null;
        }

        /// <summary>
        /// Reset to start
        /// </summary>
        public void ResetProgress()
        {
            currentWaypointIndex = 0;
        }

        void OnDrawGizmos()
        {
            if (!showGizmos) return;

            // Draw waypoints with speed-zone-based colors
            for (int i = 0; i < waypoints.Count; i++)
            {
                if (waypoints[i] == null) continue;
                if (waypointSpeedLimits != null && i < waypointSpeedLimits.Length)
                {
                    // Color by speed zone: red=slow, yellow=medium, green=fast
                    float t = Mathf.InverseLerp(8.33f, 22.22f, waypointSpeedLimits[i]);
                    Gizmos.color = Color.Lerp(Color.red, Color.green, t);
                }
                else
                {
                    Gizmos.color = waypointColor;
                }
                Gizmos.DrawSphere(waypoints[i].position, gizmoSize);
            }

            // Draw path
            Gizmos.color = Color.green;
            for (int i = 0; i < waypoints.Count - 1; i++)
            {
                if (waypoints[i] == null || waypoints[i + 1] == null) continue;
                Gizmos.DrawLine(waypoints[i].position, waypoints[i + 1].position);
            }

            // Draw speed zone boundaries
            for (int i = 0; i < speedZones.Count; i++)
            {
                Vector3 zoneStart = transform.TransformPoint(new Vector3(0, 1f, speedZones[i].startZ));
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireCube(zoneStart, new Vector3(10f, 2f, 0.5f));
            }
        }
    }
}
