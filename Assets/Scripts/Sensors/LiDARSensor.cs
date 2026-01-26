using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;
using System.Collections.Generic;

namespace ADPlatform.Sensors
{
    /// <summary>
    /// LiDAR sensor using raycasting to simulate point cloud.
    /// Publishes to ROS2 as sensor_msgs/PointCloud2.
    /// </summary>
    public class LiDARSensor : MonoBehaviour
    {
        [Header("LiDAR Settings")]
        [Tooltip("Number of vertical layers (channels)")]
        public int verticalRays = 16;

        [Tooltip("Number of horizontal rays per layer")]
        public int horizontalRays = 360;

        [Tooltip("Vertical FOV range (degrees)")]
        public float verticalFOVMin = -15f;
        public float verticalFOVMax = 15f;

        [Tooltip("Horizontal FOV (360 for full rotation)")]
        public float horizontalFOV = 360f;

        [Tooltip("Maximum detection range (meters)")]
        public float maxRange = 100f;

        [Tooltip("Minimum detection range (meters)")]
        public float minRange = 0.5f;

        [Header("ROS Settings")]
        public string pointCloudTopic = "/vehicle/lidar/points";
        public float publishRate = 10f;  // Hz
        public string frameId = "lidar_link";

        [Header("Debug")]
        public bool showDebugRays = false;
        public Color hitColor = Color.red;
        public Color missColor = Color.green;

        private ROSConnection ros;
        private float publishInterval;
        private float lastPublishTime;
        private List<Vector3> pointBuffer;
        private List<float> intensityBuffer;

        // PointCloud2 field offsets
        private const int POINT_STEP = 16;  // x(4) + y(4) + z(4) + intensity(4)

        void Start()
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<PointCloud2Msg>(pointCloudTopic);

            publishInterval = 1f / publishRate;
            lastPublishTime = Time.time;

            pointBuffer = new List<Vector3>(verticalRays * horizontalRays);
            intensityBuffer = new List<float>(verticalRays * horizontalRays);

            Debug.Log($"[LiDARSensor] Initialized - {verticalRays}x{horizontalRays} rays, Range: {minRange}-{maxRange}m, Topic: {pointCloudTopic}");
        }

        void FixedUpdate()
        {
            if (Time.time - lastPublishTime >= publishInterval)
            {
                ScanAndPublish();
                lastPublishTime = Time.time;
            }
        }

        void ScanAndPublish()
        {
            pointBuffer.Clear();
            intensityBuffer.Clear();

            float verticalStep = (verticalFOVMax - verticalFOVMin) / Mathf.Max(1, verticalRays - 1);
            float horizontalStep = horizontalFOV / horizontalRays;

            // Perform raycasting
            for (int v = 0; v < verticalRays; v++)
            {
                float verticalAngle = verticalFOVMin + v * verticalStep;

                for (int h = 0; h < horizontalRays; h++)
                {
                    float horizontalAngle = -horizontalFOV / 2 + h * horizontalStep;

                    // Calculate ray direction
                    Quaternion rotation = Quaternion.Euler(verticalAngle, horizontalAngle, 0);
                    Vector3 direction = transform.rotation * rotation * Vector3.forward;

                    RaycastHit hit;
                    if (Physics.Raycast(transform.position, direction, out hit, maxRange))
                    {
                        if (hit.distance >= minRange)
                        {
                            // Convert hit point to local coordinates
                            Vector3 localPoint = transform.InverseTransformPoint(hit.point);
                            pointBuffer.Add(localPoint);

                            // Intensity based on distance and surface angle
                            float intensity = CalculateIntensity(hit);
                            intensityBuffer.Add(intensity);

                            if (showDebugRays)
                            {
                                Debug.DrawLine(transform.position, hit.point, hitColor, publishInterval);
                            }
                        }
                    }
                    else if (showDebugRays)
                    {
                        Debug.DrawRay(transform.position, direction * maxRange, missColor, publishInterval);
                    }
                }
            }

            // Publish point cloud
            if (pointBuffer.Count > 0)
            {
                PublishPointCloud();
            }
        }

        float CalculateIntensity(RaycastHit hit)
        {
            // Simple intensity model based on distance and angle
            float distanceFactor = 1f - (hit.distance / maxRange);
            float angleFactor = Mathf.Abs(Vector3.Dot(hit.normal, -transform.forward));
            return Mathf.Clamp01(distanceFactor * angleFactor) * 255f;
        }

        void PublishPointCloud()
        {
            int pointCount = pointBuffer.Count;
            byte[] data = new byte[pointCount * POINT_STEP];

            for (int i = 0; i < pointCount; i++)
            {
                Vector3 point = pointBuffer[i];
                float intensity = intensityBuffer[i];

                int offset = i * POINT_STEP;

                // Convert Unity coordinates to ROS (x=forward, y=left, z=up)
                // Unity: x=right, y=up, z=forward
                float rosX = point.z;   // Unity z -> ROS x
                float rosY = -point.x;  // Unity x -> ROS -y
                float rosZ = point.y;   // Unity y -> ROS z

                // Write point data (little-endian float)
                WriteFloat(data, offset + 0, rosX);
                WriteFloat(data, offset + 4, rosY);
                WriteFloat(data, offset + 8, rosZ);
                WriteFloat(data, offset + 12, intensity);
            }

            // Create PointCloud2 message
            var timestamp = GetROSTimestamp();

            PointCloud2Msg msg = new PointCloud2Msg
            {
                header = new HeaderMsg
                {
                    stamp = timestamp,
                    frame_id = frameId
                },
                height = 1,
                width = (uint)pointCount,
                fields = new PointFieldMsg[]
                {
                    new PointFieldMsg { name = "x", offset = 0, datatype = 7, count = 1 },      // FLOAT32
                    new PointFieldMsg { name = "y", offset = 4, datatype = 7, count = 1 },
                    new PointFieldMsg { name = "z", offset = 8, datatype = 7, count = 1 },
                    new PointFieldMsg { name = "intensity", offset = 12, datatype = 7, count = 1 }
                },
                is_bigendian = false,
                point_step = POINT_STEP,
                row_step = (uint)(pointCount * POINT_STEP),
                data = data,
                is_dense = true
            };

            ros.Publish(pointCloudTopic, msg);
        }

        void WriteFloat(byte[] buffer, int offset, float value)
        {
            byte[] bytes = System.BitConverter.GetBytes(value);
            buffer[offset] = bytes[0];
            buffer[offset + 1] = bytes[1];
            buffer[offset + 2] = bytes[2];
            buffer[offset + 3] = bytes[3];
        }

        TimeMsg GetROSTimestamp()
        {
            double totalSeconds = Time.timeAsDouble;
            int sec = (int)totalSeconds;
            uint nanosec = (uint)((totalSeconds - sec) * 1e9);
            return new TimeMsg { sec = sec, nanosec = nanosec };
        }
    }
}
