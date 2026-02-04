using UnityEngine;
using System.Collections.Generic;
using ADPlatform.Environment;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Generates 4x4 city block grid road geometry and route waypoints.
    /// 5x5 intersections, 100m block size, 3-lane roads.
    /// </summary>
    public class GridRoadNetwork : MonoBehaviour
    {
        [Header("Grid Config")]
        public int gridSize = 5;
        public float blockSize = 100f;
        public int numLanes = 3;
        public float laneWidth = 3.5f;
        public float intersectionWidth = 14f;
        public float turnRadius = 10f;
        public float defaultSpeedLimit = 16.67f;

        [Header("Waypoint Config")]
        public float waypointSpacing = 10f;
        public float waypointY = 0.5f;

        // Grid data
        private Vector3[,] intersectionPositions;

        void Awake()
        {
            ComputeIntersectionPositions();
        }

        private void ComputeIntersectionPositions()
        {
            intersectionPositions = new Vector3[gridSize, gridSize];
            float halfGrid = (gridSize - 1) * blockSize / 2f;

            for (int col = 0; col < gridSize; col++)
            {
                for (int row = 0; row < gridSize; row++)
                {
                    float x = col * blockSize - halfGrid;
                    float z = row * blockSize - halfGrid;
                    intersectionPositions[col, row] = new Vector3(x, 0f, z);
                }
            }
        }

        /// <summary>
        /// Get the world position of an intersection by (col, row) index.
        /// </summary>
        public Vector3 GetIntersectionPosition(int col, int row)
        {
            if (intersectionPositions == null)
                ComputeIntersectionPositions();
            return intersectionPositions[
                Mathf.Clamp(col, 0, gridSize - 1),
                Mathf.Clamp(row, 0, gridSize - 1)];
        }

        /// <summary>
        /// Generate all road geometry (surfaces, markings) under the given parent.
        /// </summary>
        public void GenerateRoadGeometry(Transform parent)
        {
            if (intersectionPositions == null)
                ComputeIntersectionPositions();

            float roadWidth = numLanes * laneWidth;
            float halfGrid = (gridSize - 1) * blockSize / 2f;

            // Horizontal roads (along X axis, fixed Z per row)
            for (int row = 0; row < gridSize; row++)
            {
                float z = row * blockSize - halfGrid;
                CreateRoadSegment(parent, $"HRoad_{row}",
                    new Vector3(0f, 0.01f, z),
                    new Vector3((gridSize - 1) * blockSize + roadWidth, roadWidth, 1f),
                    isHorizontal: true);
            }

            // Vertical roads (along Z axis, fixed X per column)
            for (int col = 0; col < gridSize; col++)
            {
                float x = col * blockSize - halfGrid;
                CreateRoadSegment(parent, $"VRoad_{col}",
                    new Vector3(x, 0.01f, 0f),
                    new Vector3(roadWidth, (gridSize - 1) * blockSize + roadWidth, 1f),
                    isHorizontal: false);
            }

            // Intersection surfaces (slightly raised for visual distinction)
            for (int col = 0; col < gridSize; col++)
            {
                for (int row = 0; row < gridSize; row++)
                {
                    Vector3 pos = intersectionPositions[col, row];
                    CreateIntersectionSurface(parent, $"Intersection_{col}{row}",
                        new Vector3(pos.x, 0.015f, pos.z), intersectionWidth);
                }
            }

            // Lane markings
            CreateLaneMarkings(parent);
        }

        private void CreateRoadSegment(Transform parent, string name, Vector3 center,
            Vector3 dimensions, bool isHorizontal)
        {
            var road = GameObject.CreatePrimitive(PrimitiveType.Plane);
            road.name = name;
            road.transform.SetParent(parent);
            road.transform.localPosition = center;

            if (isHorizontal)
                road.transform.localScale = new Vector3(dimensions.x / 10f, 1f, dimensions.y / 10f);
            else
                road.transform.localScale = new Vector3(dimensions.x / 10f, 1f, dimensions.y / 10f);

            road.GetComponent<Renderer>().sharedMaterial = GetAsphaltMaterial();
            Object.DestroyImmediate(road.GetComponent<Collider>());
        }

        private void CreateIntersectionSurface(Transform parent, string name, Vector3 center, float size)
        {
            var surface = GameObject.CreatePrimitive(PrimitiveType.Plane);
            surface.name = name;
            surface.transform.SetParent(parent);
            surface.transform.localPosition = center;
            surface.transform.localScale = new Vector3(size / 10f, 1f, size / 10f);
            surface.GetComponent<Renderer>().sharedMaterial = GetLightAsphaltMaterial();
            Object.DestroyImmediate(surface.GetComponent<Collider>());
        }

        private void CreateLaneMarkings(Transform parent)
        {
            var markingsParent = new GameObject("LaneMarkings");
            markingsParent.transform.SetParent(parent);
            markingsParent.transform.localPosition = Vector3.zero;

            float roadWidth = numLanes * laneWidth;
            float halfWidth = roadWidth / 2f;
            float halfGrid = (gridSize - 1) * blockSize / 2f;
            float dashLength = 3f;
            float gapLength = 5f;

            // Horizontal road markings
            for (int row = 0; row < gridSize; row++)
            {
                float z = row * blockSize - halfGrid;

                for (int col = 0; col < gridSize - 1; col++)
                {
                    float startX = col * blockSize - halfGrid + intersectionWidth / 2f;
                    float endX = (col + 1) * blockSize - halfGrid - intersectionWidth / 2f;

                    // Edge lines (white solid)
                    CreateLineSegment(markingsParent.transform, $"HEdge_{row}_{col}_top",
                        new Vector3((startX + endX) / 2f, 0.02f, z + halfWidth),
                        new Vector3(endX - startX, 0.02f, 0.15f), false);
                    CreateLineSegment(markingsParent.transform, $"HEdge_{row}_{col}_bot",
                        new Vector3((startX + endX) / 2f, 0.02f, z - halfWidth),
                        new Vector3(endX - startX, 0.02f, 0.15f), false);

                    // Center dashed line (yellow)
                    for (float x = startX; x < endX; x += dashLength + gapLength)
                    {
                        float len = Mathf.Min(dashLength, endX - x);
                        CreateDashMark(markingsParent.transform,
                            new Vector3(x + len / 2f, 0.02f, z),
                            new Vector3(len, 0.02f, 0.12f), true);
                    }
                }
            }

            // Vertical road markings
            for (int col = 0; col < gridSize; col++)
            {
                float x = col * blockSize - halfGrid;

                for (int row = 0; row < gridSize - 1; row++)
                {
                    float startZ = row * blockSize - halfGrid + intersectionWidth / 2f;
                    float endZ = (row + 1) * blockSize - halfGrid - intersectionWidth / 2f;

                    // Edge lines (white solid)
                    CreateLineSegment(markingsParent.transform, $"VEdge_{col}_{row}_left",
                        new Vector3(x - halfWidth, 0.02f, (startZ + endZ) / 2f),
                        new Vector3(0.15f, 0.02f, endZ - startZ), false);
                    CreateLineSegment(markingsParent.transform, $"VEdge_{col}_{row}_right",
                        new Vector3(x + halfWidth, 0.02f, (startZ + endZ) / 2f),
                        new Vector3(0.15f, 0.02f, endZ - startZ), false);

                    // Center dashed line (yellow)
                    for (float z = startZ; z < endZ; z += dashLength + gapLength)
                    {
                        float len = Mathf.Min(dashLength, endZ - z);
                        CreateDashMark(markingsParent.transform,
                            new Vector3(x, 0.02f, z + len / 2f),
                            new Vector3(0.12f, 0.02f, len), true);
                    }
                }
            }
        }

        private void CreateLineSegment(Transform parent, string name, Vector3 pos, Vector3 scale, bool isYellow)
        {
            var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
            line.name = name;
            line.transform.SetParent(parent);
            line.transform.localPosition = pos;
            line.transform.localScale = scale;
            line.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
            Object.DestroyImmediate(line.GetComponent<Collider>());
        }

        private void CreateDashMark(Transform parent, Vector3 pos, Vector3 scale, bool isYellow)
        {
            var dash = GameObject.CreatePrimitive(PrimitiveType.Cube);
            dash.name = "CenterDash";
            dash.transform.SetParent(parent);
            dash.transform.localPosition = pos;
            dash.transform.localScale = scale;
            dash.GetComponent<Renderer>().sharedMaterial = GetYellowLineMaterial();
            Object.DestroyImmediate(dash.GetComponent<Collider>());
        }

        /// <summary>
        /// Generate waypoints for a route and return them as Transform array.
        /// Creates waypoint GameObjects under the given parent.
        /// </summary>
        public Transform[] GenerateRouteWaypoints(GridRouteDefinition route, Transform parent)
        {
            if (intersectionPositions == null)
                ComputeIntersectionPositions();

            var positions = new List<Vector3>();

            for (int i = 0; i < route.waypoints.Length; i++)
            {
                int nextIdx = (i + 1) % route.waypoints.Length;
                if (!route.isCyclic && i == route.waypoints.Length - 1)
                    break;

                Vector2Int from = route.waypoints[i];
                Vector2Int to = route.waypoints[nextIdx];
                int turnType = i < route.turnTypes.Length ? route.turnTypes[i] : 0;

                Vector3 fromPos = GetIntersectionPosition(from.x, from.y);
                Vector3 toPos = GetIntersectionPosition(to.x, to.y);

                // Determine travel direction
                Vector3 dir = (toPos - fromPos).normalized;
                float segmentLength = Vector3.Distance(fromPos, toPos);

                if (turnType == 0)
                {
                    // Straight through: waypoints along the segment
                    // Use the right-side lane offset (Korean traffic: drive on right)
                    Vector3 laneOffset = GetLaneOffset(dir);

                    for (float d = 0; d < segmentLength; d += waypointSpacing)
                    {
                        positions.Add(fromPos + dir * d + laneOffset + Vector3.up * waypointY);
                    }
                }
                else
                {
                    // Approach to intersection center
                    Vector3 laneOffset = GetLaneOffset(dir);
                    float approachDist = segmentLength - turnRadius;

                    for (float d = 0; d < approachDist; d += waypointSpacing)
                    {
                        positions.Add(fromPos + dir * d + laneOffset + Vector3.up * waypointY);
                    }

                    // Turn arc
                    if (turnType == 1)
                        CollectLeftTurnArc(positions, fromPos + dir * approachDist + laneOffset, dir, to, from);
                    else
                        CollectRightTurnArc(positions, fromPos + dir * approachDist + laneOffset, dir, to, from);
                }
            }

            // Create Transform GameObjects
            var waypoints = new Transform[positions.Count];
            for (int i = 0; i < positions.Count; i++)
            {
                var go = new GameObject($"WP_{i:D3}");
                go.transform.SetParent(parent);
                go.transform.position = positions[i];
                waypoints[i] = go.transform;
            }

            return waypoints;
        }

        /// <summary>
        /// Get lane offset perpendicular to travel direction (right side of road).
        /// </summary>
        private Vector3 GetLaneOffset(Vector3 direction)
        {
            // Right-hand side lane: perpendicular to direction, offset by half lane width
            Vector3 right = Vector3.Cross(Vector3.up, direction).normalized;
            return right * laneWidth * 0.5f;
        }

        private void CollectLeftTurnArc(List<Vector3> positions, Vector3 startPos,
            Vector3 approachDir, Vector2Int toIntersection, Vector2Int fromIntersection)
        {
            // Left turn: 90-degree arc turning left
            int arcSegments = 8;
            Vector3 left = -Vector3.Cross(Vector3.up, approachDir).normalized;

            // Arc center is to the left of the start position
            Vector3 arcCenter = startPos + left * turnRadius;

            for (int i = 1; i <= arcSegments; i++)
            {
                float t = (float)i / arcSegments;
                float angle = t * Mathf.PI / 2f;

                // Start from approach direction, turn 90 degrees left
                Vector3 offset = -left * Mathf.Cos(angle) * turnRadius
                               + approachDir * Mathf.Sin(angle) * turnRadius;
                positions.Add(arcCenter + offset + Vector3.up * waypointY);
            }
        }

        private void CollectRightTurnArc(List<Vector3> positions, Vector3 startPos,
            Vector3 approachDir, Vector2Int toIntersection, Vector2Int fromIntersection)
        {
            // Right turn: 90-degree arc turning right
            int arcSegments = 6;
            Vector3 right = Vector3.Cross(Vector3.up, approachDir).normalized;

            Vector3 arcCenter = startPos + right * turnRadius * 0.7f;

            for (int i = 1; i <= arcSegments; i++)
            {
                float t = (float)i / arcSegments;
                float angle = t * Mathf.PI / 2f;

                Vector3 offset = -right * Mathf.Cos(angle) * turnRadius * 0.7f
                               + approachDir * Mathf.Sin(angle) * turnRadius * 0.7f;
                positions.Add(arcCenter + offset + Vector3.up * waypointY);
            }
        }

        /// <summary>
        /// Get the nearest traffic light for a position + forward direction.
        /// Searches grid intersections within range.
        /// </summary>
        public TrafficLightController GetNearestTrafficLight(Vector3 pos, Vector3 forward,
            GridTrafficLightManager lightManager)
        {
            if (lightManager == null) return null;
            return lightManager.GetRelevantLight(pos, forward);
        }

        // Material cache
        private static Material s_asphalt;
        private static Material s_lightAsphalt;
        private static Material s_whiteLine;
        private static Material s_yellowLine;

        private Material GetAsphaltMaterial()
        {
            if (s_asphalt == null)
            {
                s_asphalt = new Material(Shader.Find("Standard"));
                s_asphalt.color = new Color(0.25f, 0.25f, 0.25f);
            }
            return s_asphalt;
        }

        private Material GetLightAsphaltMaterial()
        {
            if (s_lightAsphalt == null)
            {
                s_lightAsphalt = new Material(Shader.Find("Standard"));
                s_lightAsphalt.color = new Color(0.28f, 0.28f, 0.28f);
            }
            return s_lightAsphalt;
        }

        private Material GetWhiteLineMaterial()
        {
            if (s_whiteLine == null)
            {
                s_whiteLine = new Material(Shader.Find("Standard"));
                s_whiteLine.color = Color.white;
                s_whiteLine.EnableKeyword("_EMISSION");
                s_whiteLine.SetColor("_EmissionColor", Color.white * 0.3f);
            }
            return s_whiteLine;
        }

        private Material GetYellowLineMaterial()
        {
            if (s_yellowLine == null)
            {
                s_yellowLine = new Material(Shader.Find("Standard"));
                s_yellowLine.color = Color.yellow;
                s_yellowLine.EnableKeyword("_EMISSION");
                s_yellowLine.SetColor("_EmissionColor", Color.yellow * 0.3f);
            }
            return s_yellowLine;
        }
    }
}
