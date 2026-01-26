using UnityEngine;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Geometry;
using RosMessageTypes.Std;

namespace ADPlatform.ROS2
{
    /// <summary>
    /// Bridge between Unity Vehicle and ROS2.
    /// Publishes vehicle state (Odometry) and subscribes to control commands (Twist).
    /// </summary>
    public class VehicleROSBridge : MonoBehaviour
    {
        [Header("ROS Topics")]
        public string odomTopic = "/vehicle/odom";
        public string cmdVelTopic = "/vehicle/cmd_vel";
        public string poseTopic = "/vehicle/pose";

        [Header("Publish Settings")]
        public float publishRate = 20f;  // Hz

        [Header("References")]
        public Rigidbody vehicleRigidbody;

        private ROSConnection ros;
        private float publishInterval;
        private float lastPublishTime;

        // Received command from ROS
        private Vector3 cmdLinearVelocity;
        private Vector3 cmdAngularVelocity;
        private bool hasNewCommand = false;

        void Start()
        {
            ros = ROSConnection.GetOrCreateInstance();

            // Register publishers
            ros.RegisterPublisher<TwistMsg>(odomTopic);
            ros.RegisterPublisher<PoseMsg>(poseTopic);

            // Subscribe to command velocity
            ros.Subscribe<TwistMsg>(cmdVelTopic, OnCmdVelReceived);

            publishInterval = 1f / publishRate;
            lastPublishTime = Time.time;

            if (vehicleRigidbody == null)
            {
                vehicleRigidbody = GetComponent<Rigidbody>();
            }

            Debug.Log($"[VehicleROSBridge] Initialized - Publishing: {odomTopic}, {poseTopic} | Subscribing: {cmdVelTopic}");
        }

        void FixedUpdate()
        {
            // Publish vehicle state at specified rate
            if (Time.time - lastPublishTime >= publishInterval)
            {
                PublishVehicleState();
                lastPublishTime = Time.time;
            }
        }

        void OnCmdVelReceived(TwistMsg msg)
        {
            // Convert ROS coordinate (x=forward, y=left, z=up) to Unity (x=right, y=up, z=forward)
            cmdLinearVelocity = new Vector3(
                (float)-msg.linear.y,  // ROS y (left) -> Unity -x (right)
                (float)msg.linear.z,   // ROS z (up) -> Unity y (up)
                (float)msg.linear.x    // ROS x (forward) -> Unity z (forward)
            );

            cmdAngularVelocity = new Vector3(
                (float)-msg.angular.y,
                (float)msg.angular.z,
                (float)msg.angular.x
            );

            hasNewCommand = true;

            Debug.Log($"[VehicleROSBridge] Received cmd_vel: linear=({msg.linear.x:F2}, {msg.linear.y:F2}, {msg.linear.z:F2}), angular=({msg.angular.x:F2}, {msg.angular.y:F2}, {msg.angular.z:F2})");
        }

        void PublishVehicleState()
        {
            if (vehicleRigidbody == null) return;

            // Get vehicle state in Unity coordinates
            Vector3 position = transform.position;
            Vector3 velocity = vehicleRigidbody.linearVelocity;
            Vector3 angularVelocity = vehicleRigidbody.angularVelocity;
            Quaternion rotation = transform.rotation;

            // Publish Twist (velocity)
            TwistMsg twistMsg = new TwistMsg
            {
                linear = new Vector3Msg
                {
                    x = velocity.z,    // Unity z (forward) -> ROS x
                    y = -velocity.x,   // Unity x (right) -> ROS -y (left)
                    z = velocity.y     // Unity y (up) -> ROS z
                },
                angular = new Vector3Msg
                {
                    x = angularVelocity.z,
                    y = -angularVelocity.x,
                    z = angularVelocity.y
                }
            };
            ros.Publish(odomTopic, twistMsg);

            // Publish Pose (position + orientation)
            PoseMsg poseMsg = new PoseMsg
            {
                position = new PointMsg
                {
                    x = position.z,
                    y = -position.x,
                    z = position.y
                },
                orientation = new QuaternionMsg
                {
                    x = rotation.z,
                    y = -rotation.x,
                    z = rotation.y,
                    w = rotation.w
                }
            };
            ros.Publish(poseTopic, poseMsg);
        }

        /// <summary>
        /// Get the latest command velocity from ROS (for use by controllers).
        /// </summary>
        public bool GetCommandVelocity(out Vector3 linear, out Vector3 angular)
        {
            linear = cmdLinearVelocity;
            angular = cmdAngularVelocity;

            bool hadCommand = hasNewCommand;
            hasNewCommand = false;
            return hadCommand;
        }

        /// <summary>
        /// Check if connected to ROS.
        /// </summary>
        public bool IsConnected()
        {
            return ros != null && !ros.HasConnectionError;
        }
    }
}
