using System.Collections.Generic;
using UnityEngine;

namespace ADPlatform.RoadBuilder
{
    public class ProceduralRoadBuilder : MonoBehaviour
    {
        [Header("Road Dimensions")]
        public float roadLength = 500f;
        [Range(1, 4)] public int numLanes = 2;
        [Tooltip("Lane width in meters (Korean standard: 3.5m)")]
        public float laneWidth = 3.5f;
        public float shoulderWidth = 1.0f;

        [Header("Road Shape")]
        [Range(0f, 1f)] public float roadCurvature = 0f;
        public float maxCurveAngle = 30f;
        public float waypointSpacing = 10f;

        [Header("Intersection")]
        [Range(0, 3)] public int intersectionType = 0; // None/T/Cross/Y
        [Range(0, 2)] public int turnDirection = 0; // Straight/Left/Right
        public float intersectionDistance = 150f;

        private const float MeshSpacing = 2f;
        private const float MinCurveSegmentLength = 50f;
        private const float MaxCurveSegmentLength = 150f;
        private const float IntersectionWidth = 14f;
        private const float TurnRadius = 10f;

        private static Material asphaltMaterial;
        private static Material whiteLineMaterial;
        private static Material yellowLineMaterial;
        private static Material sidewalkMaterial;

        private readonly List<GameObject> generatedObjects = new List<GameObject>();

        public void GenerateRoad()
        {
            CleanupRoad();

            List<Vector3> path = CollectRoadPositions(MeshSpacing);
            if (path.Count < 2)
                return;

            BuildRoadSurface(path);
            BuildLaneMarkings(path);
            BuildSidewalks(path);
            BuildRumbleStrips(path);
        }

        public Transform[] GenerateWaypoints(Transform parent)
        {
            if (parent == null)
                parent = transform;

            List<Vector3> positions = CollectRoadPositions(Mathf.Max(1f, waypointSpacing));
            Transform[] waypoints = new Transform[positions.Count];

            for (int i = 0; i < positions.Count; i++)
            {
                GameObject go = new GameObject($"WP_{i:D3}");
                go.transform.SetParent(parent, false);
                go.transform.localPosition = positions[i];
                waypoints[i] = go.transform;
            }

            return waypoints;
        }

        public void CleanupRoad()
        {
            for (int i = generatedObjects.Count - 1; i >= 0; i--)
            {
                GameObject obj = generatedObjects[i];
                if (obj == null)
                    continue;
                DestroyObject(obj);
            }
            generatedObjects.Clear();
        }

        private void BuildRoadSurface(List<Vector3> path)
        {
            float width = numLanes * laneWidth + 2f * shoulderWidth;
            float halfWidth = width * 0.5f;

            Mesh mesh = BuildStripMesh(path, 0f, width, 0f, true, 0f, 0f);
            if (mesh == null)
                return;

            GameObject road = CreateChild("RoadSurface");
            MeshFilter mf = road.AddComponent<MeshFilter>();
            MeshRenderer mr = road.AddComponent<MeshRenderer>();
            MeshCollider mc = road.AddComponent<MeshCollider>();

            mf.sharedMesh = mesh;
            mr.sharedMaterial = GetAsphaltMaterial();
            mc.sharedMesh = mesh;
            road.tag = "Road";
            road.layer = gameObject.layer;

            // Shoulder edge offsets for UV sanity on very narrow roads
            if (halfWidth < 0.1f)
                halfWidth = 0.1f;
        }

        private void BuildLaneMarkings(List<Vector3> path)
        {
            GameObject markingsRoot = CreateChild("LaneMarkings");

            float roadWidth = numLanes * laneWidth + 2f * shoulderWidth;
            float halfWidth = roadWidth * 0.5f;
            float lineWidth = 0.15f;
            float lineHeight = 0.02f;

            // Center line (dashed yellow)
            GameObject centerLine = BuildLineMesh(path, 0f, lineWidth, lineHeight, true, 3f, 5f, GetYellowLineMaterial(), "CenterLine");
            if (centerLine != null)
            {
                centerLine.transform.SetParent(markingsRoot.transform, false);
                centerLine.tag = "LaneMarking";
            }

            // Edge lines (solid white)
            float edgeOffset = halfWidth - (lineWidth * 0.5f);
            GameObject leftEdge = BuildLineMesh(path, -edgeOffset, lineWidth, lineHeight, false, 0f, 0f, GetWhiteLineMaterial(), "LeftEdgeLine");
            GameObject rightEdge = BuildLineMesh(path, edgeOffset, lineWidth, lineHeight, false, 0f, 0f, GetWhiteLineMaterial(), "RightEdgeLine");
            if (leftEdge != null)
            {
                leftEdge.transform.SetParent(markingsRoot.transform, false);
                leftEdge.tag = "LaneMarking";
            }
            if (rightEdge != null)
            {
                rightEdge.transform.SetParent(markingsRoot.transform, false);
                rightEdge.tag = "LaneMarking";
            }

            // Lane dividers for 3+ lanes (white dashed)
            if (numLanes > 2)
            {
                for (int i = 1; i < numLanes; i++)
                {
                    float dividerOffset = -halfWidth + shoulderWidth + (i * laneWidth);
                    if (Mathf.Abs(dividerOffset) < 0.01f)
                        continue;

                    GameObject divider = BuildLineMesh(path, dividerOffset, lineWidth, lineHeight, true, 3f, 5f, GetWhiteLineMaterial(), $"LaneDivider_{i}");
                    if (divider != null)
                    {
                        divider.transform.SetParent(markingsRoot.transform, false);
                        divider.tag = "LaneMarking";
                    }
                }
            }
        }

        private void BuildSidewalks(List<Vector3> path)
        {
            GameObject sidewalksRoot = CreateChild("Sidewalks");

            float roadWidth = numLanes * laneWidth + 2f * shoulderWidth;
            float halfWidth = roadWidth * 0.5f;
            float sidewalkWidth = 2f;
            float sidewalkHeight = 0.15f;
            float leftOffset = -(halfWidth + sidewalkWidth * 0.5f);
            float rightOffset = (halfWidth + sidewalkWidth * 0.5f);

            Mesh leftMesh = BuildStripMesh(path, leftOffset, sidewalkWidth, sidewalkHeight, false, 0f, 0f);
            Mesh rightMesh = BuildStripMesh(path, rightOffset, sidewalkWidth, sidewalkHeight, false, 0f, 0f);

            if (leftMesh != null)
            {
                GameObject leftSidewalk = CreateChild("LeftSidewalk");
                leftSidewalk.transform.SetParent(sidewalksRoot.transform, false);
                MeshFilter mf = leftSidewalk.AddComponent<MeshFilter>();
                MeshRenderer mr = leftSidewalk.AddComponent<MeshRenderer>();
                MeshCollider mc = leftSidewalk.AddComponent<MeshCollider>();
                mf.sharedMesh = leftMesh;
                mr.sharedMaterial = GetSidewalkMaterial();
                mc.sharedMesh = leftMesh;
            }

            if (rightMesh != null)
            {
                GameObject rightSidewalk = CreateChild("RightSidewalk");
                rightSidewalk.transform.SetParent(sidewalksRoot.transform, false);
                MeshFilter mf = rightSidewalk.AddComponent<MeshFilter>();
                MeshRenderer mr = rightSidewalk.AddComponent<MeshRenderer>();
                MeshCollider mc = rightSidewalk.AddComponent<MeshCollider>();
                mf.sharedMesh = rightMesh;
                mr.sharedMaterial = GetSidewalkMaterial();
                mc.sharedMesh = rightMesh;
            }
        }

        private void BuildRumbleStrips(List<Vector3> path)
        {
            GameObject edgesRoot = CreateChild("RoadEdges");

            float roadWidth = numLanes * laneWidth + 2f * shoulderWidth;
            float halfWidth = roadWidth * 0.5f;
            float stripOffset = Mathf.Max(0.1f, shoulderWidth * 0.5f);
            float offsetFromCenter = halfWidth - stripOffset;

            for (int i = 0; i < path.Count - 1; i++)
            {
                Vector3 a = path[i];
                Vector3 b = path[i + 1];
                Vector3 forward = (b - a).normalized;
                Vector3 right = Vector3.Cross(Vector3.up, forward);
                float segLen = Vector3.Distance(a, b);
                Vector3 center = (a + b) * 0.5f;

                CreateRumbleStripSegment(edgesRoot.transform, center + right * offsetFromCenter, forward, segLen, "RumbleRight");
                CreateRumbleStripSegment(edgesRoot.transform, center - right * offsetFromCenter, forward, segLen, "RumbleLeft");
            }
        }

        private void CreateRumbleStripSegment(Transform parent, Vector3 position, Vector3 forward, float length, string name)
        {
            GameObject go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = position;
            go.transform.localRotation = Quaternion.LookRotation(forward, Vector3.up);
            go.tag = "RoadEdge";
            generatedObjects.Add(go);

            BoxCollider bc = go.AddComponent<BoxCollider>();
            bc.size = new Vector3(0.3f, 0.05f, Mathf.Max(0.5f, length));
            bc.center = new Vector3(0f, 0.025f, 0f);
        }

        private Mesh BuildStripMesh(List<Vector3> path, float offset, float width, float height, bool generateUV, float dashLength, float gapLength)
        {
            if (path == null || path.Count < 2)
                return null;

            int count = path.Count;
            List<Vector3> vertices = new List<Vector3>(count * 2);
            List<Vector3> normals = new List<Vector3>(count * 2);
            List<Vector2> uvs = new List<Vector2>(count * 2);
            List<int> triangles = new List<int>((count - 1) * 6);

            float totalLength = 0f;
            float[] cumulative = new float[count];
            for (int i = 1; i < count; i++)
            {
                totalLength += Vector3.Distance(path[i - 1], path[i]);
                cumulative[i] = totalLength;
            }
            if (totalLength < 0.001f)
                totalLength = 1f;

            for (int i = 0; i < count; i++)
            {
                Vector3 forward;
                if (i == 0)
                    forward = (path[i + 1] - path[i]).normalized;
                else if (i == count - 1)
                    forward = (path[i] - path[i - 1]).normalized;
                else
                    forward = (path[i + 1] - path[i - 1]).normalized;

                Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;
                Vector3 center = path[i] + right * offset + Vector3.up * height;

                Vector3 left = center - right * (width * 0.5f);
                Vector3 rightPos = center + right * (width * 0.5f);

                vertices.Add(left);
                vertices.Add(rightPos);
                normals.Add(Vector3.up);
                normals.Add(Vector3.up);

                float v = cumulative[i] / totalLength;
                uvs.Add(new Vector2(0f, v));
                uvs.Add(new Vector2(1f, v));
            }

            for (int i = 0; i < count - 1; i++)
            {
                int idx = i * 2;
                triangles.Add(idx);
                triangles.Add(idx + 2);
                triangles.Add(idx + 1);
                triangles.Add(idx + 1);
                triangles.Add(idx + 2);
                triangles.Add(idx + 3);
            }

            Mesh mesh = new Mesh();
            mesh.name = "StripMesh";
            mesh.SetVertices(vertices);
            mesh.SetNormals(normals);
            mesh.SetTriangles(triangles, 0);
            if (generateUV)
                mesh.SetUVs(0, uvs);
            mesh.RecalculateBounds();
            return mesh;
        }

        private GameObject BuildLineMesh(List<Vector3> path, float offset, float width, float height, bool dashed, float dashLength, float gapLength, Material material, string name)
        {
            if (path == null || path.Count < 2)
                return null;

            List<Vector3> vertices = new List<Vector3>();
            List<Vector3> normals = new List<Vector3>();
            List<Vector2> uvs = new List<Vector2>();
            List<int> triangles = new List<int>();

            float distance = 0f;
            float pattern = Mathf.Max(0.01f, dashLength + gapLength);

            for (int i = 0; i < path.Count - 1; i++)
            {
                Vector3 a = path[i];
                Vector3 b = path[i + 1];
                float segLen = Vector3.Distance(a, b);
                float segStart = distance;
                distance += segLen;

                bool draw = true;
                if (dashed)
                {
                    float mod = segStart % pattern;
                    draw = mod < dashLength;
                }

                if (!draw)
                    continue;

                Vector3 forward = (b - a).normalized;
                Vector3 right = Vector3.Cross(Vector3.up, forward).normalized;

                Vector3 centerA = a + right * offset + Vector3.up * height;
                Vector3 centerB = b + right * offset + Vector3.up * height;

                Vector3 leftA = centerA - right * (width * 0.5f);
                Vector3 rightA = centerA + right * (width * 0.5f);
                Vector3 leftB = centerB - right * (width * 0.5f);
                Vector3 rightB = centerB + right * (width * 0.5f);

                int baseIndex = vertices.Count;
                vertices.Add(leftA);
                vertices.Add(rightA);
                vertices.Add(leftB);
                vertices.Add(rightB);

                normals.Add(Vector3.up);
                normals.Add(Vector3.up);
                normals.Add(Vector3.up);
                normals.Add(Vector3.up);

                uvs.Add(new Vector2(0f, 0f));
                uvs.Add(new Vector2(1f, 0f));
                uvs.Add(new Vector2(0f, 1f));
                uvs.Add(new Vector2(1f, 1f));

                triangles.Add(baseIndex + 0);
                triangles.Add(baseIndex + 2);
                triangles.Add(baseIndex + 1);
                triangles.Add(baseIndex + 1);
                triangles.Add(baseIndex + 2);
                triangles.Add(baseIndex + 3);
            }

            if (vertices.Count == 0)
                return null;

            Mesh mesh = new Mesh();
            mesh.name = name;
            mesh.SetVertices(vertices);
            mesh.SetNormals(normals);
            mesh.SetTriangles(triangles, 0);
            mesh.SetUVs(0, uvs);
            mesh.RecalculateBounds();

            GameObject go = CreateChild(name);
            MeshFilter mf = go.AddComponent<MeshFilter>();
            MeshRenderer mr = go.AddComponent<MeshRenderer>();
            MeshCollider mc = go.AddComponent<MeshCollider>();
            mf.sharedMesh = mesh;
            mr.sharedMaterial = material;
            mc.sharedMesh = mesh;
            return go;
        }

        private List<Vector3> CollectRoadPositions(float spacing)
        {
            List<Vector3> positions = new List<Vector3>();
            if (intersectionType > 0 && roadCurvature > 0.01f)
                CollectCurvedIntersectionPositions(positions, spacing);
            else if (intersectionType > 0)
                CollectIntersectionPositions(positions, spacing);
            else if (roadCurvature <= 0.01f)
                CollectStraightPositions(positions, spacing);
            else
                CollectCurvedPositions(positions, spacing);
            return positions;
        }

        private void CollectStraightPositions(List<Vector3> positions, float spacing)
        {
            float startZ = -roadLength / 2f;
            float endZ = roadLength / 2f;
            for (float z = startZ; z <= endZ; z += spacing)
                positions.Add(new Vector3(0f, 0f, z));
        }

        private void CollectCurvedPositions(List<Vector3> positions, float spacing)
        {
            Vector3 currentPos = new Vector3(0f, 0f, -roadLength / 2f);
            float currentHeading = 0f;
            float totalDistance = 0f;
            int curveDirection = 1;

            float segmentLength = Random.Range(MinCurveSegmentLength, MaxCurveSegmentLength);
            float segmentProgress = 0f;
            float targetCurveAngle = GetRandomCurveAngle(curveDirection);

            while (totalDistance < roadLength)
            {
                positions.Add(currentPos);

                segmentProgress += spacing;
                if (segmentProgress >= segmentLength)
                {
                    segmentProgress = 0f;
                    segmentLength = Random.Range(MinCurveSegmentLength, MaxCurveSegmentLength);
                    curveDirection = -curveDirection;
                    targetCurveAngle = GetRandomCurveAngle(curveDirection);
                }

                float curveRate = (targetCurveAngle / segmentLength) * spacing;
                currentHeading += curveRate;
                currentHeading = Mathf.Clamp(currentHeading, -maxCurveAngle, maxCurveAngle);

                float headingRad = currentHeading * Mathf.Deg2Rad;
                Vector3 moveDir = new Vector3(Mathf.Sin(headingRad), 0f, Mathf.Cos(headingRad));
                currentPos += moveDir * spacing;
                totalDistance += spacing;
            }
        }

        private float GetRandomCurveAngle(int direction)
        {
            float baseAngle = maxCurveAngle * roadCurvature;
            float angle = Random.Range(baseAngle * 0.3f, baseAngle);
            return angle * direction;
        }

        private void CollectCurvedIntersectionPositions(List<Vector3> positions, float spacing)
        {
            float approachEnd = intersectionDistance - IntersectionWidth / 2f;
            float straightenStart = approachEnd - 30f;

            Vector3 currentPos = new Vector3(0f, 0f, -roadLength / 2f);
            float currentHeading = 0f;
            int curveDirection = 1;

            float segmentLength = Random.Range(MinCurveSegmentLength, MaxCurveSegmentLength);
            float segmentProgress = 0f;
            float targetCurveAngle = GetRandomCurveAngle(curveDirection);

            while (currentPos.z < straightenStart)
            {
                positions.Add(currentPos);

                segmentProgress += spacing;
                if (segmentProgress >= segmentLength)
                {
                    segmentProgress = 0f;
                    segmentLength = Random.Range(MinCurveSegmentLength, MaxCurveSegmentLength);
                    curveDirection = -curveDirection;
                    targetCurveAngle = GetRandomCurveAngle(curveDirection);
                }

                float curveRate = (targetCurveAngle / segmentLength) * spacing;
                currentHeading += curveRate;
                currentHeading = Mathf.Clamp(currentHeading, -maxCurveAngle, maxCurveAngle);

                float headingRad = currentHeading * Mathf.Deg2Rad;
                Vector3 moveDir = new Vector3(Mathf.Sin(headingRad), 0f, Mathf.Cos(headingRad));
                currentPos += moveDir * spacing;
            }

            float straightenDistance = approachEnd - currentPos.z;
            int straightenSteps = Mathf.Max(1, Mathf.FloorToInt(straightenDistance / (spacing * 0.5f)));
            float startX = currentPos.x;
            float startZ = currentPos.z;

            for (int i = 1; i <= straightenSteps; i++)
            {
                float t = (float)i / straightenSteps;
                float smooth = t * t * (3f - 2f * t);
                float x = Mathf.Lerp(startX, 0f, smooth);
                float z = Mathf.Lerp(startZ, approachEnd, t);
                positions.Add(new Vector3(x, 0f, z));
            }

            switch (turnDirection)
            {
                case 0:
                    CollectStraightThroughPositions(positions, spacing);
                    break;
                case 1:
                    CollectLeftTurnPositions(positions);
                    break;
                case 2:
                    CollectRightTurnPositions(positions);
                    break;
            }

            CollectExitSectionPositions(positions, spacing);
        }

        private void CollectIntersectionPositions(List<Vector3> positions, float spacing)
        {
            float approachStart = -roadLength / 2f;
            float approachEnd = intersectionDistance - IntersectionWidth / 2f;

            for (float z = approachStart; z < approachEnd; z += spacing)
                positions.Add(new Vector3(0f, 0f, z));

            switch (turnDirection)
            {
                case 0:
                    CollectStraightThroughPositions(positions, spacing);
                    break;
                case 1:
                    CollectLeftTurnPositions(positions);
                    break;
                case 2:
                    CollectRightTurnPositions(positions);
                    break;
            }

            CollectExitSectionPositions(positions, spacing);
        }

        private void CollectStraightThroughPositions(List<Vector3> positions, float spacing)
        {
            float startZ = intersectionDistance - IntersectionWidth / 2f;
            float endZ = intersectionDistance + IntersectionWidth / 2f;

            for (float z = startZ; z <= endZ; z += spacing * 0.5f)
                positions.Add(new Vector3(0f, 0f, z));
        }

        private void CollectLeftTurnPositions(List<Vector3> positions)
        {
            float turnStartZ = intersectionDistance - TurnRadius;
            int arcSegments = 8;
            for (int i = 0; i <= arcSegments; i++)
            {
                float t = (float)i / arcSegments;
                float angle = t * Mathf.PI / 2f;
                float x = -TurnRadius * (1f - Mathf.Cos(angle));
                float z = turnStartZ + TurnRadius * Mathf.Sin(angle);
                positions.Add(new Vector3(x, 0f, z));
            }
        }

        private void CollectRightTurnPositions(List<Vector3> positions)
        {
            float turnStartZ = intersectionDistance - TurnRadius * 0.5f;
            int arcSegments = 6;
            for (int i = 0; i <= arcSegments; i++)
            {
                float t = (float)i / arcSegments;
                float angle = t * Mathf.PI / 2f;
                float x = TurnRadius * (1f - Mathf.Cos(angle));
                float z = turnStartZ + TurnRadius * 0.7f * Mathf.Sin(angle);
                positions.Add(new Vector3(x, 0f, z));
            }
        }

        private void CollectExitSectionPositions(List<Vector3> positions, float spacing)
        {
            if (positions.Count == 0)
                return;

            Vector3 lastPos = positions[positions.Count - 1];
            float exitLength = roadLength / 2f - intersectionDistance - IntersectionWidth / 2f;

            switch (turnDirection)
            {
                case 0:
                    for (float d = spacing; d <= exitLength; d += spacing)
                        positions.Add(new Vector3(lastPos.x, 0f, lastPos.z + d));
                    break;
                case 1:
                    for (float d = spacing; d <= exitLength; d += spacing)
                        positions.Add(new Vector3(lastPos.x - d, 0f, lastPos.z));
                    break;
                case 2:
                    for (float d = spacing; d <= exitLength; d += spacing)
                        positions.Add(new Vector3(lastPos.x + d, 0f, lastPos.z));
                    break;
            }
        }

        public Material GetAsphaltMaterial()
        {
            if (asphaltMaterial == null)
            {
                asphaltMaterial = new Material(Shader.Find("Standard"));
                asphaltMaterial.color = new Color(0.18f, 0.18f, 0.18f, 1f);
            }
            return asphaltMaterial;
        }

        public Material GetWhiteLineMaterial()
        {
            if (whiteLineMaterial == null)
            {
                whiteLineMaterial = new Material(Shader.Find("Standard"));
                whiteLineMaterial.color = Color.white;
                whiteLineMaterial.EnableKeyword("_EMISSION");
                whiteLineMaterial.SetColor("_EmissionColor", Color.white * 0.6f);
            }
            return whiteLineMaterial;
        }

        public Material GetYellowLineMaterial()
        {
            if (yellowLineMaterial == null)
            {
                yellowLineMaterial = new Material(Shader.Find("Standard"));
                yellowLineMaterial.color = new Color(1f, 0.85f, 0.1f, 1f);
                yellowLineMaterial.EnableKeyword("_EMISSION");
                yellowLineMaterial.SetColor("_EmissionColor", new Color(1f, 0.85f, 0.1f, 1f) * 0.6f);
            }
            return yellowLineMaterial;
        }

        public Material GetSidewalkMaterial()
        {
            if (sidewalkMaterial == null)
            {
                sidewalkMaterial = new Material(Shader.Find("Standard"));
                sidewalkMaterial.color = new Color(0.75f, 0.75f, 0.75f, 1f);
            }
            return sidewalkMaterial;
        }

        private GameObject CreateChild(string name)
        {
            GameObject go = new GameObject(name);
            go.transform.SetParent(transform, false);
            generatedObjects.Add(go);
            return go;
        }

        private void DestroyObject(GameObject obj)
        {
#if UNITY_EDITOR
            if (!Application.isPlaying)
                DestroyImmediate(obj);
            else
                Destroy(obj);
#else
            Destroy(obj);
#endif
        }
    }
}
