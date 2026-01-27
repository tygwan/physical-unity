// ROS2 기능은 Unity Robotics 패키지가 설치된 경우에만 활성화됩니다.

#if UNITY_ROBOTICS_ROS_TCP_CONNECTOR
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
        public int verticalRays = 16;
        public int horizontalRays = 360;
        public float verticalFOVMin = -15f;
        public float verticalFOVMax = 15f;
        public float horizontalFOV = 360f;
        public float maxRange = 100f;
        public float minRange = 0.5f;

        [Header("ROS Settings")]
        public string pointCloudTopic = "/vehicle/lidar/points";
        public float publishRate = 10f;
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

        private const int POINT_STEP = 16;

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

            for (int v = 0; v < verticalRays; v++)
            {
                float verticalAngle = verticalFOVMin + v * verticalStep;

                for (int h = 0; h < horizontalRays; h++)
                {
                    float horizontalAngle = -horizontalFOV / 2 + h * horizontalStep;

                    Quaternion rotation = Quaternion.Euler(verticalAngle, horizontalAngle, 0);
                    Vector3 direction = transform.rotation * rotation * Vector3.forward;

                    RaycastHit hit;
                    if (Physics.Raycast(transform.position, direction, out hit, maxRange))
                    {
                        if (hit.distance >= minRange)
                        {
                            Vector3 localPoint = transform.InverseTransformPoint(hit.point);
                            pointBuffer.Add(localPoint);

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

            if (pointBuffer.Count > 0)
            {
                PublishPointCloud();
            }
        }

        float CalculateIntensity(RaycastHit hit)
        {
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

                float rosX = point.z;
                float rosY = -point.x;
                float rosZ = point.y;

                WriteFloat(data, offset + 0, rosX);
                WriteFloat(data, offset + 4, rosY);
                WriteFloat(data, offset + 8, rosZ);
                WriteFloat(data, offset + 12, intensity);
            }

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
                    new PointFieldMsg { name = "x", offset = 0, datatype = 7, count = 1 },
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
#else
// Stub implementation when ROS TCP Connector is not installed
namespace ADPlatform.Sensors
{
    public class LiDARSensor : UnityEngine.MonoBehaviour
    {
        public int verticalRays = 16;
        public int horizontalRays = 360;
        public string pointCloudTopic = "/vehicle/lidar/points";
        public bool showDebugRays = false;

        void Start()
        {
            UnityEngine.Debug.LogWarning("[LiDARSensor] ROS TCP Connector not installed. ROS features disabled.");
            enabled = false;
        }
    }
}
#endif
