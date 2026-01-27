using UnityEngine;
using System.Collections.Generic;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Road segment types for procedural generation
    /// </summary>
    public enum SegmentType
    {
        Straight,           // 직선 구간
        ArcLeft,            // 왼쪽 원호 곡선
        ArcRight,           // 오른쪽 원호 곡선
        ClothoidEntry,      // 완화곡선 진입
        ClothoidExit,       // 완화곡선 탈출
        SCurve,             // S자 곡선
        LaneMerge,          // 차선 감소 (4→2)
        LaneExpand,         // 차선 증가 (2→4)
        SpeedZoneChange,    // 속도 구간 전환
        Intersection        // 교차로 (Phase G+)
    }

    /// <summary>
    /// Speed zone types (Korean road regulations)
    /// </summary>
    public enum SpeedZoneType
    {
        Residential = 30,    // 주거지역: 30 km/h
        Urban = 50,          // 도시부: 50 km/h
        General = 60,        // 일반도로: 60 km/h
        Highway = 80,        // 자동차전용도로: 80 km/h
        Expressway = 100     // 고속도로: 100 km/h
    }

    /// <summary>
    /// Configuration for a single road segment
    /// </summary>
    [System.Serializable]
    public class RoadSegmentConfig
    {
        public SegmentType type = SegmentType.Straight;
        public float length = 100f;              // 세그먼트 길이 (m)
        public int laneCount = 2;                // 차선 수 (1-4)
        public float laneWidth = 3.5f;           // 차선 폭 (m)
        public SpeedZoneType speedZone = SpeedZoneType.General;

        // Curve parameters
        public float curveRadius = 100f;         // 곡선 반경 (m)
        public float curveAngle = 30f;           // 곡선 각도 (degrees)

        // Variation parameters
        public float npcDensity = 0.5f;          // NPC 밀도 (0-1)
        public float obstacleChance = 0f;        // 장애물 확률 (0-1)
        public bool hasCenterLine = true;        // 중앙선 유무

        public RoadSegmentConfig() { }

        public RoadSegmentConfig(SegmentType type, float length, int lanes = 2)
        {
            this.type = type;
            this.length = length;
            this.laneCount = lanes;
        }
    }

    /// <summary>
    /// Runtime data for a generated road segment
    /// </summary>
    public class RoadSegmentData
    {
        public RoadSegmentConfig config;
        public Vector3 startPoint;
        public Vector3 endPoint;
        public float startHeading;      // degrees
        public float endHeading;        // degrees
        public List<Vector3> waypoints = new List<Vector3>();
        public List<Vector3> leftBoundary = new List<Vector3>();
        public List<Vector3> rightBoundary = new List<Vector3>();
        public float arcLength;         // 실제 도로 길이
        public int segmentIndex;

        public float SpeedLimitMps => (float)config.speedZone * 1000f / 3600f;
    }

    /// <summary>
    /// Generates individual road segment geometry
    /// </summary>
    public static class RoadSegmentGenerator
    {
        private const float WAYPOINT_SPACING = 5f;  // 웨이포인트 간격 (m)

        /// <summary>
        /// Generate a road segment from configuration
        /// </summary>
        public static RoadSegmentData Generate(RoadSegmentConfig config, Vector3 startPoint, float startHeading)
        {
            var data = new RoadSegmentData
            {
                config = config,
                startPoint = startPoint,
                startHeading = startHeading
            };

            switch (config.type)
            {
                case SegmentType.Straight:
                    GenerateStraight(data);
                    break;
                case SegmentType.ArcLeft:
                    GenerateArc(data, -1);  // Left = negative angle
                    break;
                case SegmentType.ArcRight:
                    GenerateArc(data, 1);   // Right = positive angle
                    break;
                case SegmentType.SCurve:
                    GenerateSCurve(data);
                    break;
                case SegmentType.ClothoidEntry:
                case SegmentType.ClothoidExit:
                    GenerateClothoid(data);
                    break;
                case SegmentType.LaneMerge:
                case SegmentType.LaneExpand:
                    GenerateLaneTransition(data);
                    break;
                default:
                    GenerateStraight(data);
                    break;
            }

            // Generate boundary lines
            GenerateBoundaries(data);

            return data;
        }

        private static void GenerateStraight(RoadSegmentData data)
        {
            float headingRad = data.startHeading * Mathf.Deg2Rad;
            Vector3 direction = new Vector3(Mathf.Sin(headingRad), 0, Mathf.Cos(headingRad));

            int numWaypoints = Mathf.CeilToInt(data.config.length / WAYPOINT_SPACING);

            for (int i = 0; i <= numWaypoints; i++)
            {
                float t = (float)i / numWaypoints;
                Vector3 point = data.startPoint + direction * (t * data.config.length);
                data.waypoints.Add(point);
            }

            data.endPoint = data.startPoint + direction * data.config.length;
            data.endHeading = data.startHeading;
            data.arcLength = data.config.length;
        }

        private static void GenerateArc(RoadSegmentData data, int direction)
        {
            float radius = data.config.curveRadius;
            float angle = data.config.curveAngle * direction;
            float angleRad = angle * Mathf.Deg2Rad;
            float startHeadingRad = data.startHeading * Mathf.Deg2Rad;

            // Arc length
            data.arcLength = Mathf.Abs(angleRad) * radius;
            int numWaypoints = Mathf.CeilToInt(data.arcLength / WAYPOINT_SPACING);

            // Find center of the arc
            Vector3 toCenter = new Vector3(
                Mathf.Cos(startHeadingRad) * direction,
                0,
                -Mathf.Sin(startHeadingRad) * direction
            );
            Vector3 center = data.startPoint + toCenter * radius;

            // Generate waypoints along arc
            for (int i = 0; i <= numWaypoints; i++)
            {
                float t = (float)i / numWaypoints;
                float currentAngle = startHeadingRad - (Mathf.PI / 2 * direction) + (angleRad * t);

                Vector3 point = center + new Vector3(
                    Mathf.Cos(currentAngle) * radius,
                    0,
                    Mathf.Sin(currentAngle) * radius
                );
                data.waypoints.Add(point);
            }

            data.endPoint = data.waypoints[data.waypoints.Count - 1];
            data.endHeading = data.startHeading + angle;
        }

        private static void GenerateSCurve(RoadSegmentData data)
        {
            // S-curve = two arcs in opposite directions
            float halfLength = data.config.length / 2f;
            float radius = data.config.curveRadius;
            float angle = data.config.curveAngle;

            // First arc (right)
            var firstArcConfig = new RoadSegmentConfig
            {
                type = SegmentType.ArcRight,
                curveRadius = radius,
                curveAngle = angle,
                laneCount = data.config.laneCount
            };
            var firstArc = Generate(firstArcConfig, data.startPoint, data.startHeading);

            // Second arc (left) - from end of first
            var secondArcConfig = new RoadSegmentConfig
            {
                type = SegmentType.ArcLeft,
                curveRadius = radius,
                curveAngle = angle,
                laneCount = data.config.laneCount
            };
            var secondArc = Generate(secondArcConfig, firstArc.endPoint, firstArc.endHeading);

            // Combine waypoints
            data.waypoints.AddRange(firstArc.waypoints);
            data.waypoints.AddRange(secondArc.waypoints.GetRange(1, secondArc.waypoints.Count - 1));

            data.endPoint = secondArc.endPoint;
            data.endHeading = secondArc.endHeading;
            data.arcLength = firstArc.arcLength + secondArc.arcLength;
        }

        private static void GenerateClothoid(RoadSegmentData data)
        {
            // Simplified clothoid using cubic bezier approximation
            // Real clothoid: curvature increases linearly with arc length
            float headingRad = data.startHeading * Mathf.Deg2Rad;
            Vector3 forward = new Vector3(Mathf.Sin(headingRad), 0, Mathf.Cos(headingRad));

            float length = data.config.length;
            float finalCurvature = 1f / data.config.curveRadius;
            bool isEntry = data.config.type == SegmentType.ClothoidEntry;

            int numWaypoints = Mathf.CeilToInt(length / WAYPOINT_SPACING);
            float currentHeading = data.startHeading;
            Vector3 currentPos = data.startPoint;

            for (int i = 0; i <= numWaypoints; i++)
            {
                float t = (float)i / numWaypoints;
                data.waypoints.Add(currentPos);

                // Curvature increases/decreases linearly
                float curvature = isEntry ? (finalCurvature * t) : (finalCurvature * (1 - t));
                float turnRate = curvature * WAYPOINT_SPACING * Mathf.Rad2Deg;

                currentHeading += turnRate;
                float headRad = currentHeading * Mathf.Deg2Rad;
                Vector3 dir = new Vector3(Mathf.Sin(headRad), 0, Mathf.Cos(headRad));
                currentPos += dir * WAYPOINT_SPACING;
            }

            data.endPoint = currentPos;
            data.endHeading = currentHeading;
            data.arcLength = length;
        }

        private static void GenerateLaneTransition(RoadSegmentData data)
        {
            // Lane merge/expand is essentially a straight section with changing width
            // The actual lane change happens in the boundary generation
            GenerateStraight(data);
        }

        private static void GenerateBoundaries(RoadSegmentData data)
        {
            float halfWidth = data.config.laneCount * data.config.laneWidth / 2f;

            for (int i = 0; i < data.waypoints.Count; i++)
            {
                Vector3 point = data.waypoints[i];
                Vector3 forward;

                if (i < data.waypoints.Count - 1)
                    forward = (data.waypoints[i + 1] - point).normalized;
                else if (i > 0)
                    forward = (point - data.waypoints[i - 1]).normalized;
                else
                    forward = Vector3.forward;

                Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;

                // Handle lane transitions
                float widthMultiplier = 1f;
                if (data.config.type == SegmentType.LaneMerge)
                {
                    float t = (float)i / (data.waypoints.Count - 1);
                    widthMultiplier = Mathf.Lerp(1f, 0.5f, t);  // 4→2 lanes
                }
                else if (data.config.type == SegmentType.LaneExpand)
                {
                    float t = (float)i / (data.waypoints.Count - 1);
                    widthMultiplier = Mathf.Lerp(0.5f, 1f, t);  // 2→4 lanes
                }

                data.leftBoundary.Add(point - right * halfWidth * widthMultiplier);
                data.rightBoundary.Add(point + right * halfWidth * widthMultiplier);
            }
        }
    }
}
