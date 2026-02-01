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

        [Header("Intersection (Phase G)")]
        [Tooltip("Intersection type: 0=None, 1=T-junction, 2=Cross, 3=Y-junction")]
        [Range(0, 3)]
        public int intersectionType = 0;
        [Tooltip("Turn direction at intersection: 0=Straight, 1=Left, 2=Right")]
        [Range(0, 2)]
        public int turnDirection = 0;
        [Tooltip("Distance to intersection from start (meters)")]
        public float intersectionDistance = 100f;
        [Tooltip("Intersection road width (meters)")]
        public float intersectionWidth = 14f;  // 4-lane intersection
        [Tooltip("Turn radius for smooth turns (meters)")]
        public float turnRadius = 10f;

        [Header("Visualization")]
        public bool showGizmos = true;
        public Color waypointColor = Color.cyan;
        public float gizmoSize = 0.5f;

        [Header("Runtime")]
        public Transform egoVehicle;
        public int currentWaypointIndex = 0;
        public float waypointReachDistance = 5f;

        [Header("Real-time Update (Inspector에서 조절)")]
        [Tooltip("실시간 경로 갱신 활성화")]
        public bool enableRealtimeUpdate = false;
        [Tooltip("갱신 주기 (초). 0.05 = 20Hz, 0.1 = 10Hz")]
        [Range(0.02f, 1f)]
        public float updateInterval = 0.1f;
        [Tooltip("차량 전방 몇 개의 waypoint만 갱신할지")]
        [Range(3, 20)]
        public int updateAheadCount = 10;

        private List<Transform> waypoints = new List<Transform>();
        private List<SpeedZone> speedZones = new List<SpeedZone>();
        private float[] waypointSpeedLimits;  // Speed limit per waypoint
        private float lastUpdateTime = 0f;

        void Start()
        {
            GenerateSpeedZones();
            GenerateWaypoints();
        }

        void Update()
        {
            if (enableRealtimeUpdate && Time.time - lastUpdateTime >= updateInterval)
            {
                UpdateWaypointsAhead();
                lastUpdateTime = Time.time;
            }
        }

        /// <summary>
        /// 차량 전방의 waypoint만 부분 갱신 (성능 최적화)
        /// </summary>
        private void UpdateWaypointsAhead()
        {
            if (egoVehicle == null || waypoints.Count == 0) return;

            Vector3 egoPos = egoVehicle.position;
            int startIdx = Mathf.Max(0, currentWaypointIndex);
            int endIdx = Mathf.Min(waypoints.Count, startIdx + updateAheadCount);

            for (int i = startIdx; i < endIdx; i++)
            {
                if (waypoints[i] != null)
                {
                    // Waypoint 위치를 현재 도로 상태에 맞게 업데이트
                    // (곡선 도로의 경우 동적 조정 가능)
                    Vector3 wp = waypoints[i].position;
                    // 시각화를 위해 Gizmo가 자동 갱신됨
                }
            }
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
                // Descending speed layout: familiar default speed first, slower zones later
                // Prevents curriculum shock: 1-zone (60 km/h) → 2-zones starts at (60, 50)
                // count=2: UrbanGeneral(60) → UrbanNarrow(50)       [gentle 10 km/h drop]
                // count=3: UrbanGeneral(60) → UrbanNarrow(50) → Residential(30)  [progressive]
                // count=4: UrbanGeneral(60) → UrbanNarrow(50) → Residential(30) → Expressway(80)
                SpeedZoneType[] zoneTypes = {
                    SpeedZoneType.UrbanGeneral,   // 60 km/h (same as single-zone default)
                    SpeedZoneType.UrbanNarrow,    // 50 km/h
                    SpeedZoneType.Residential,    // 30 km/h
                    SpeedZoneType.Expressway      // 80 km/h
                };
                float[] zoneLimits = { 16.67f, 13.89f, 8.33f, 22.22f };

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
        /// Phase F v3 fix: reuses existing GameObjects to maintain Transform reference
        /// continuity during curriculum transitions (prevents observation mismatch)
        /// </summary>
        public void GenerateWaypoints()
        {
            // Phase F v3: reuse existing child GameObjects instead of destroying them
            // Destroying Transforms during curriculum transitions breaks agent observation
            // references, causing policy-observation mismatch and entropy collapse
            int existingChildCount = transform.childCount;
            waypoints.Clear();

            // Collect new positions without creating GameObjects
            List<Vector3> positions = new List<Vector3>();
            if (intersectionType > 0)
            {
                CollectIntersectionPositions(positions);
            }
            else if (roadCurvature <= 0.01f)
            {
                CollectStraightPositions(positions);
            }
            else
            {
                CollectCurvedPositions(positions);
            }

            // Apply positions: reuse existing GameObjects, create new only if needed
            for (int i = 0; i < positions.Count; i++)
            {
                Transform wp;
                if (i < existingChildCount)
                {
                    wp = transform.GetChild(i);
                    wp.gameObject.SetActive(true);
                }
                else
                {
                    GameObject go = new GameObject($"WP_{i:D3}");
                    go.transform.SetParent(transform);
                    wp = go.transform;
                }
                wp.localPosition = positions[i];
                wp.name = $"WP_{i:D3}";
                waypoints.Add(wp);
            }

            // Deactivate excess waypoints (don't destroy - may be reused later)
            for (int i = positions.Count; i < existingChildCount; i++)
            {
                transform.GetChild(i).gameObject.SetActive(false);
            }

            // Assign speed limits to waypoints
            waypointSpeedLimits = new float[waypoints.Count];
            for (int i = 0; i < waypoints.Count; i++)
            {
                float progressZ = (float)i / waypoints.Count * roadLength - roadLength / 2f;
                waypointSpeedLimits[i] = GetSpeedLimitAtZ(progressZ);
            }

            int reused = Mathf.Min(positions.Count, existingChildCount);
            int created = Mathf.Max(0, positions.Count - existingChildCount);
            UnityEngine.Debug.Log($"[WaypointManager] Generated {waypoints.Count} waypoints (reused={reused}, new={created}), lanes={numLanes}, curvature={roadCurvature:F2}, intersection={intersectionType}, turn={turnDirection}");
        }

        /// <summary>
        /// Collect straight waypoint positions (Phase A-D original behavior)
        /// Updated in Phase F to support multi-lane
        /// Phase F v3: outputs positions instead of creating GameObjects
        /// </summary>
        private void CollectStraightPositions(List<Vector3> positions)
        {
            float startZ = -roadLength / 2f;
            float endZ = roadLength / 2f;
            float activeLaneX = GetLaneXPosition(currentLane);

            for (float z = startZ; z <= endZ; z += waypointSpacing)
            {
                positions.Add(new Vector3(activeLaneX, waypointY, z));
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
        /// In Korean traffic, driving on left side of center is wrong-way.
        /// Phase G: Disabled in/after intersection zone where turns require negative X positions.
        /// </summary>
        public bool IsWrongWayDriving(float xPos, float zPos)
        {
            if (!centerLineEnabled || numLanes <= 1)
                return false;

            // Phase G (P-013): Disable WrongWay check in intersection zone and beyond.
            // Left turns exit at negative X, which is valid post-intersection.
            if (intersectionType > 0 && zPos >= intersectionDistance - intersectionWidth)
                return false;

            float tolerance = 0.5f;
            return xPos < -tolerance;
        }

        /// <summary>
        /// Legacy overload for backward compatibility (no Z position).
        /// </summary>
        public bool IsWrongWayDriving(float xPos)
        {
            return IsWrongWayDriving(xPos, float.MinValue);
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
        /// Phase F v3 fix: shifts existing waypoint positions instead of regenerating
        /// Maintains Transform reference continuity for agent observations
        /// </summary>
        public void SetLaneCount(int count)
        {
            float oldLaneX = GetLaneXPosition(currentLane);

            numLanes = Mathf.Clamp(count, 1, 4);
            currentLane = 0;  // Reset to rightmost lane

            float newLaneX = GetLaneXPosition(currentLane);

            if (waypoints.Count > 0)
            {
                // Shift existing waypoint X positions without destroying GameObjects
                float deltaX = newLaneX - oldLaneX;
                if (Mathf.Abs(deltaX) > 0.001f)
                {
                    foreach (var wp in waypoints)
                    {
                        if (wp != null)
                        {
                            Vector3 pos = wp.localPosition;
                            pos.x += deltaX;
                            wp.localPosition = pos;
                        }
                    }
                }
                // Re-assign speed limits for shifted positions
                if (waypointSpeedLimits != null)
                {
                    for (int i = 0; i < waypoints.Count; i++)
                    {
                        waypointSpeedLimits[i] = GetSpeedLimitAtZ(waypoints[i].localPosition.z);
                    }
                }
                UnityEngine.Debug.Log($"[WaypointManager] SetLaneCount({count}): {waypoints.Count} waypoints repositioned (deltaX={deltaX:F2})");
            }
            else
            {
                GenerateWaypoints();
            }
        }

        /// <summary>
        /// Set center line enforcement (Phase F)
        /// </summary>
        public void SetCenterLineEnabled(bool enabled)
        {
            centerLineEnabled = enabled;
        }

        /// <summary>
        /// Collect curved waypoint positions (Phase E)
        /// Creates smooth curves with varying directions based on curriculum parameters
        /// Updated in Phase F to support multi-lane
        /// Phase F v3: outputs positions instead of creating GameObjects
        /// </summary>
        private void CollectCurvedPositions(List<Vector3> positions)
        {
            float activeLaneX = GetLaneXPosition(currentLane);

            Vector3 currentPos = new Vector3(activeLaneX, waypointY, -roadLength / 2f);
            float currentHeading = 0f;
            float totalDistance = 0f;
            int curveDirection = 1;

            float segmentLength = Random.Range(minCurveSegmentLength, maxCurveSegmentLength);
            float segmentProgress = 0f;
            float targetCurveAngle = GetRandomCurveAngle(curveDirection);

            while (totalDistance < roadLength)
            {
                positions.Add(currentPos);

                segmentProgress += waypointSpacing;
                if (segmentProgress >= segmentLength)
                {
                    segmentProgress = 0f;
                    segmentLength = Random.Range(minCurveSegmentLength, maxCurveSegmentLength);

                    if (curveDirectionVariation > 0.5f)
                        curveDirection = Random.value > 0.5f ? 1 : -1;
                    else
                        curveDirection = -curveDirection;
                    targetCurveAngle = GetRandomCurveAngle(curveDirection);
                }

                float curveRate = (targetCurveAngle / segmentLength) * waypointSpacing;
                currentHeading += curveRate;
                currentHeading = Mathf.Clamp(currentHeading, -maxCurveAngle, maxCurveAngle);

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
        /// Collect intersection waypoint positions (Phase G)
        /// Supports T-junction, Cross, Y-junction with turn directions
        /// Phase F v3: outputs positions instead of creating GameObjects
        /// </summary>
        private void CollectIntersectionPositions(List<Vector3> positions)
        {
            float activeLaneX = GetLaneXPosition(currentLane);

            // Phase 1: Approach to intersection (straight section)
            float approachStart = -roadLength / 2f;
            float approachEnd = intersectionDistance - intersectionWidth / 2f;

            for (float z = approachStart; z < approachEnd; z += waypointSpacing)
            {
                positions.Add(new Vector3(activeLaneX, waypointY, z));
            }

            // Phase 2: Intersection maneuver based on turn direction
            switch (turnDirection)
            {
                case 0:
                    CollectStraightThroughPositions(activeLaneX, positions);
                    break;
                case 1:
                    CollectLeftTurnPositions(activeLaneX, positions);
                    break;
                case 2:
                    CollectRightTurnPositions(activeLaneX, positions);
                    break;
            }

            // Phase 3: Exit section
            CollectExitSectionPositions(positions);
        }

        private void CollectStraightThroughPositions(float laneX, List<Vector3> positions)
        {
            float startZ = intersectionDistance - intersectionWidth / 2f;
            float endZ = intersectionDistance + intersectionWidth / 2f;

            for (float z = startZ; z <= endZ; z += waypointSpacing * 0.5f)
            {
                positions.Add(new Vector3(laneX, waypointY, z));
            }
        }

        private void CollectLeftTurnPositions(float startLaneX, List<Vector3> positions)
        {
            float turnStartZ = intersectionDistance - turnRadius;

            int arcSegments = 8;
            for (int i = 0; i <= arcSegments; i++)
            {
                float t = (float)i / arcSegments;
                float angle = t * Mathf.PI / 2f;

                float x = startLaneX - turnRadius * (1f - Mathf.Cos(angle));
                float z = turnStartZ + turnRadius * Mathf.Sin(angle);

                positions.Add(new Vector3(x, waypointY, z));
            }
        }

        private void CollectRightTurnPositions(float startLaneX, List<Vector3> positions)
        {
            float turnStartZ = intersectionDistance - turnRadius * 0.5f;

            int arcSegments = 6;
            for (int i = 0; i <= arcSegments; i++)
            {
                float t = (float)i / arcSegments;
                float angle = t * Mathf.PI / 2f;

                float x = startLaneX + turnRadius * (1f - Mathf.Cos(angle));
                float z = turnStartZ + turnRadius * 0.7f * Mathf.Sin(angle);

                positions.Add(new Vector3(x, waypointY, z));
            }
        }

        private void CollectExitSectionPositions(List<Vector3> positions)
        {
            Vector3 lastPos = positions[positions.Count - 1];
            float exitLength = roadLength / 2f - intersectionDistance - intersectionWidth / 2f;

            switch (turnDirection)
            {
                case 0:
                    for (float d = waypointSpacing; d <= exitLength; d += waypointSpacing)
                        positions.Add(new Vector3(lastPos.x, waypointY, lastPos.z + d));
                    break;
                case 1:
                    for (float d = waypointSpacing; d <= exitLength; d += waypointSpacing)
                        positions.Add(new Vector3(lastPos.x - d, waypointY, lastPos.z));
                    break;
                case 2:
                    for (float d = waypointSpacing; d <= exitLength; d += waypointSpacing)
                        positions.Add(new Vector3(lastPos.x + d, waypointY, lastPos.z));
                    break;
            }
        }

        /// <summary>
        /// Set intersection parameters from curriculum (Phase G)
        /// </summary>
        public void SetIntersection(int type, int direction)
        {
            intersectionType = Mathf.Clamp(type, 0, 3);
            turnDirection = Mathf.Clamp(direction, 0, 2);
            GenerateWaypoints();
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
