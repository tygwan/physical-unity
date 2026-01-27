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
        public float laneX = 1.75f;          // X position of the lane (single lane mode)
        public float waypointY = 0.5f;

        [Header("Multi-Lane Support (Phase F)")]
        [Tooltip("Number of lanes: 1-4 (curriculum-controlled)")]
        [Range(1, 4)]
        public int numLanes = 1;
        [Tooltip("Lane width in meters (Korean standard: 3.5m)")]
        public float laneWidth = 3.5f;
        [Tooltip("Enable center line rule enforcement (no wrong-way driving)")]
        public bool centerLineEnabled = false;
        [Tooltip("Current lane index (0 = rightmost lane in direction of travel)")]
        public int currentLane = 0;

        [Header("Road Curvature (Phase E)")]
        [Tooltip("Road curvature intensity: 0 = straight, 1.0 = max curves")]
        [Range(0f, 1f)]
        public float roadCurvature = 0f;
        [Tooltip("Curve direction variation: 0 = single direction, 1.0 = random left/right")]
        [Range(0f, 1f)]
        public float curveDirectionVariation = 0f;
        [Tooltip("Maximum curve angle in degrees (at curvature = 1.0)")]
        public float maxCurveAngle = 45f;
        [Tooltip("Minimum curve segment length in meters")]
        public float minCurveSegmentLength = 50f;
        [Tooltip("Maximum curve segment length in meters")]
        public float maxCurveSegmentLength = 150f;

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
        /// Supports curved roads when roadCurvature > 0 (Phase E)
        /// </summary>
        public void GenerateWaypoints()
        {
            // Clear existing
            foreach (Transform child in transform)
            {
                Destroy(child.gameObject);
            }
            waypoints.Clear();

            if (roadCurvature <= 0.01f)
            {
                // Straight road (Phase A-D behavior)
                GenerateStraightWaypoints();
            }
            else
            {
                // Curved road (Phase E+)
                GenerateCurvedWaypoints();
            }

            // Assign speed limits to waypoints
            waypointSpeedLimits = new float[waypoints.Count];
            for (int i = 0; i < waypoints.Count; i++)
            {
                // Use arc length approximation for curved roads
                float progressZ = (float)i / waypoints.Count * roadLength - roadLength / 2f;
                waypointSpeedLimits[i] = GetSpeedLimitAtZ(progressZ);
            }

            UnityEngine.Debug.Log($"[WaypointManager] Generated {waypoints.Count} waypoints, lanes={numLanes}, curvature={roadCurvature:F2}, centerLine={centerLineEnabled}");
        }

        /// <summary>
        /// Generate straight waypoints (Phase A-D original behavior)
        /// Updated in Phase F to support multi-lane
        /// </summary>
        private void GenerateStraightWaypoints()
        {
            float startZ = -roadLength / 2f;
            float endZ = roadLength / 2f;
            int count = 0;

            // Calculate lane X position based on numLanes and currentLane
            float activeLaneX = GetLaneXPosition(currentLane);

            for (float z = startZ; z <= endZ; z += waypointSpacing)
            {
                GameObject wp = new GameObject($"WP_{count:D3}");
                wp.transform.SetParent(transform);
                wp.transform.localPosition = new Vector3(activeLaneX, waypointY, z);
                waypoints.Add(wp.transform);
                count++;
            }
        }

        /// <summary>
        /// Get X position for a specific lane index (Phase F)
        /// Lane 0 = rightmost lane (Korean traffic: drive on right side)
        /// </summary>
        public float GetLaneXPosition(int laneIndex)
        {
            if (numLanes <= 1)
            {
                return laneX;  // Single lane mode (Phase A-E behavior)
            }

            // Multi-lane: lanes are numbered 0 (rightmost) to numLanes-1 (leftmost)
            // Road center is at X = 0, lanes spread symmetrically
            // Korean traffic: drive on right side, so lane 0 is right of center
            float totalWidth = numLanes * laneWidth;
            float rightEdge = totalWidth / 2f;
            float laneCenterOffset = laneWidth / 2f;

            // Lane 0 = rightmost = rightEdge - laneCenterOffset
            // Lane 1 = next to right = rightEdge - laneWidth - laneCenterOffset
            return rightEdge - (laneIndex * laneWidth) - laneCenterOffset;
        }

        /// <summary>
        /// Get lane index from X position (Phase F)
        /// Returns -1 if off-road, 0-3 for valid lanes
        /// </summary>
        public int GetLaneFromXPosition(float xPos)
        {
            if (numLanes <= 1)
            {
                // Single lane: check if within lane bounds
                float halfWidth = laneWidth / 2f;
                if (Mathf.Abs(xPos - laneX) <= halfWidth)
                    return 0;
                return -1;  // Off lane
            }

            // Multi-lane
            float totalWidth = numLanes * laneWidth;
            float rightEdge = totalWidth / 2f;

            for (int i = 0; i < numLanes; i++)
            {
                float laneCenter = GetLaneXPosition(i);
                float halfWidth = laneWidth / 2f;
                if (Mathf.Abs(xPos - laneCenter) <= halfWidth)
                    return i;
            }
            return -1;  // Off-road
        }

        /// <summary>
        /// Check if position violates center line rule (Phase F)
        /// In Korean traffic, driving on left side of center is wrong-way
        /// </summary>
        public bool IsWrongWayDriving(float xPos)
        {
            if (!centerLineEnabled || numLanes <= 1)
                return false;

            // Center line is at X = 0
            // Wrong-way = driving on the opposite side (X < 0 for Korean right-side traffic)
            // Actually for our lane setup, wrong-way would be negative lanes
            // Let's define: positive X = right side (correct), negative X = wrong way
            float tolerance = 0.5f;  // Small tolerance
            return xPos < -tolerance;
        }

        /// <summary>
        /// Get all available lane positions (for NPC spawning, lane change decisions)
        /// </summary>
        public float[] GetAllLanePositions()
        {
            float[] positions = new float[numLanes];
            for (int i = 0; i < numLanes; i++)
            {
                positions[i] = GetLaneXPosition(i);
            }
            return positions;
        }

        /// <summary>
        /// Set lane count from curriculum parameter (Phase F)
        /// </summary>
        public void SetLaneCount(int count)
        {
            numLanes = Mathf.Clamp(count, 1, 4);
            currentLane = 0;  // Reset to rightmost lane
            GenerateWaypoints();
        }

        /// <summary>
        /// Set center line enforcement (Phase F)
        /// </summary>
        public void SetCenterLineEnabled(bool enabled)
        {
            centerLineEnabled = enabled;
        }

        /// <summary>
        /// Generate curved waypoints (Phase E)
        /// Creates smooth curves with varying directions based on curriculum parameters
        /// Updated in Phase F to support multi-lane
        /// </summary>
        private void GenerateCurvedWaypoints()
        {
            // Calculate lane X position based on numLanes and currentLane
            float activeLaneX = GetLaneXPosition(currentLane);

            // Current position and heading
            Vector3 currentPos = new Vector3(activeLaneX, waypointY, -roadLength / 2f);
            float currentHeading = 0f;  // 0 = forward (Z+), in degrees
            float totalDistance = 0f;
            int count = 0;
            int curveDirection = 1;  // 1 = right, -1 = left

            // Determine curve segments
            float segmentLength = Random.Range(minCurveSegmentLength, maxCurveSegmentLength);
            float segmentProgress = 0f;
            float targetCurveAngle = GetRandomCurveAngle(curveDirection);

            while (totalDistance < roadLength)
            {
                // Create waypoint
                GameObject wp = new GameObject($"WP_{count:D3}");
                wp.transform.SetParent(transform);
                wp.transform.localPosition = currentPos;
                waypoints.Add(wp.transform);
                count++;

                // Update segment progress
                segmentProgress += waypointSpacing;
                if (segmentProgress >= segmentLength)
                {
                    // Start new curve segment
                    segmentProgress = 0f;
                    segmentLength = Random.Range(minCurveSegmentLength, maxCurveSegmentLength);

                    // Determine curve direction
                    if (curveDirectionVariation > 0.5f)
                    {
                        // Random direction
                        curveDirection = Random.value > 0.5f ? 1 : -1;
                    }
                    else
                    {
                        // Alternate or maintain direction
                        curveDirection = -curveDirection;
                    }
                    targetCurveAngle = GetRandomCurveAngle(curveDirection);
                }

                // Apply gradual heading change (smooth curve)
                float curveRate = (targetCurveAngle / segmentLength) * waypointSpacing;
                currentHeading += curveRate;
                currentHeading = Mathf.Clamp(currentHeading, -maxCurveAngle, maxCurveAngle);

                // Move to next position
                float headingRad = currentHeading * Mathf.Deg2Rad;
                Vector3 moveDir = new Vector3(Mathf.Sin(headingRad), 0f, Mathf.Cos(headingRad));
                currentPos += moveDir * waypointSpacing;
                totalDistance += waypointSpacing;
            }
        }

        /// <summary>
        /// Get random curve angle based on curvature intensity and direction
        /// </summary>
        private float GetRandomCurveAngle(int direction)
        {
            float baseAngle = maxCurveAngle * roadCurvature;
            float angle = Random.Range(baseAngle * 0.3f, baseAngle);
            return angle * direction;
        }

        /// <summary>
        /// Set road curvature from curriculum parameter (Phase E)
        /// Called by DrivingSceneManager when environment resets
        /// </summary>
        public void SetRoadCurvature(float curvature, float directionVariation = 0f)
        {
            roadCurvature = Mathf.Clamp01(curvature);
            curveDirectionVariation = Mathf.Clamp01(directionVariation);
            GenerateWaypoints();
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

            // Draw lane boundaries (Phase F)
            if (numLanes > 1)
            {
                float totalWidth = numLanes * laneWidth;
                float startZ = -roadLength / 2f;
                float endZ = roadLength / 2f;

                // Draw lane dividers
                for (int i = 0; i <= numLanes; i++)
                {
                    float x = totalWidth / 2f - (i * laneWidth);
                    Vector3 start = transform.TransformPoint(new Vector3(x, 0.1f, startZ));
                    Vector3 end = transform.TransformPoint(new Vector3(x, 0.1f, endZ));

                    if (i == numLanes / 2 && centerLineEnabled)
                    {
                        // Center line (yellow)
                        Gizmos.color = Color.yellow;
                        Gizmos.DrawLine(start, end);
                        Gizmos.DrawLine(start + Vector3.up * 0.1f, end + Vector3.up * 0.1f);
                    }
                    else
                    {
                        // Regular lane divider (white)
                        Gizmos.color = Color.white;
                        Gizmos.DrawLine(start, end);
                    }
                }

                // Draw lane center markers
                for (int i = 0; i < numLanes; i++)
                {
                    float laneCenter = GetLaneXPosition(i);
                    Gizmos.color = new Color(0.5f, 0.5f, 1f, 0.5f);  // Light blue
                    for (float z = startZ; z <= endZ; z += 50f)
                    {
                        Vector3 markerPos = transform.TransformPoint(new Vector3(laneCenter, 0.05f, z));
                        Gizmos.DrawSphere(markerPos, 0.3f);
                    }
                }
            }
        }
    }
}
