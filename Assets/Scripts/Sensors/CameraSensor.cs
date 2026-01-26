using UnityEngine;
using UnityEngine.Rendering;
using Unity.Robotics.ROSTCPConnector;
using RosMessageTypes.Sensor;
using RosMessageTypes.Std;
using RosMessageTypes.BuiltinInterfaces;
using System;

namespace ADPlatform.Sensors
{
    /// <summary>
    /// Camera sensor for capturing RGB images.
    /// Publishes to ROS2 as sensor_msgs/Image for VLA and perception tasks.
    /// </summary>
    public class CameraSensor : MonoBehaviour
    {
        [Header("Camera Settings")]
        public int imageWidth = 640;
        public int imageHeight = 480;
        public float fieldOfView = 60f;
        public float nearClip = 0.1f;
        public float farClip = 100f;

        [Header("ROS Settings")]
        public string imageTopic = "/vehicle/camera/image_raw";
        public string cameraInfoTopic = "/vehicle/camera/camera_info";
        public float publishRate = 10f;  // Hz
        public string frameId = "camera_link";

        [Header("Debug")]
        public bool showPreview = false;  // Disabled - use SensorDebugHUD instead
        public RenderTexture previewTexture;

        private Camera sensorCamera;
        private RenderTexture renderTexture;
        private Texture2D texture2D;
        private ROSConnection ros;
        private float publishInterval;
        private float lastPublishTime;
        private byte[] imageData;
        private bool isInitialized = false;

        void Start()
        {
            InitializeCamera();
            InitializeROS();

            publishInterval = 1f / publishRate;
            lastPublishTime = Time.time;

            Debug.Log($"[CameraSensor] Initialized - Resolution: {imageWidth}x{imageHeight}, Topic: {imageTopic}");
        }

        void InitializeCamera()
        {
            // Create or get camera component
            sensorCamera = GetComponent<Camera>();
            if (sensorCamera == null)
            {
                sensorCamera = gameObject.AddComponent<Camera>();
            }

            // Configure camera
            sensorCamera.fieldOfView = fieldOfView;
            sensorCamera.nearClipPlane = nearClip;
            sensorCamera.farClipPlane = farClip;
            sensorCamera.enabled = false;  // We render manually

            // Create render texture
            renderTexture = new RenderTexture(imageWidth, imageHeight, 24, RenderTextureFormat.ARGB32);
            renderTexture.Create();
            sensorCamera.targetTexture = renderTexture;

            // Create texture for reading pixels
            texture2D = new Texture2D(imageWidth, imageHeight, TextureFormat.RGB24, false);

            // Allocate image data buffer
            imageData = new byte[imageWidth * imageHeight * 3];

            // Preview texture (always set for external access like SensorDebugHUD)
            previewTexture = renderTexture;

            isInitialized = true;
        }

        void InitializeROS()
        {
            ros = ROSConnection.GetOrCreateInstance();
            ros.RegisterPublisher<ImageMsg>(imageTopic);
            ros.RegisterPublisher<CameraInfoMsg>(cameraInfoTopic);
        }

        void LateUpdate()
        {
            if (!isInitialized) return;

            if (Time.time - lastPublishTime >= publishInterval)
            {
                CaptureAndPublish();
                lastPublishTime = Time.time;
            }
        }

        void CaptureAndPublish()
        {
            // Render the camera
            sensorCamera.Render();

            // Read pixels from render texture
            RenderTexture.active = renderTexture;
            texture2D.ReadPixels(new Rect(0, 0, imageWidth, imageHeight), 0, 0);
            texture2D.Apply();
            RenderTexture.active = null;

            // Convert to byte array (RGB format, flip vertically)
            Color32[] pixels = texture2D.GetPixels32();
            int idx = 0;
            for (int y = imageHeight - 1; y >= 0; y--)  // Flip Y
            {
                for (int x = 0; x < imageWidth; x++)
                {
                    Color32 pixel = pixels[y * imageWidth + x];
                    imageData[idx++] = pixel.r;
                    imageData[idx++] = pixel.g;
                    imageData[idx++] = pixel.b;
                }
            }

            // Create and publish Image message
            var timestamp = GetROSTimestamp();

            ImageMsg imageMsg = new ImageMsg
            {
                header = new HeaderMsg
                {
                    stamp = timestamp,
                    frame_id = frameId
                },
                height = (uint)imageHeight,
                width = (uint)imageWidth,
                encoding = "rgb8",
                is_bigendian = 0,
                step = (uint)(imageWidth * 3),
                data = imageData
            };

            ros.Publish(imageTopic, imageMsg);

            // Publish camera info
            PublishCameraInfo(timestamp);
        }

        void PublishCameraInfo(TimeMsg timestamp)
        {
            // Simple pinhole camera model
            double fx = imageWidth / (2.0 * Math.Tan(fieldOfView * Math.PI / 360.0));
            double fy = fx;
            double cx = imageWidth / 2.0;
            double cy = imageHeight / 2.0;

            CameraInfoMsg cameraInfoMsg = new CameraInfoMsg
            {
                header = new HeaderMsg
                {
                    stamp = timestamp,
                    frame_id = frameId
                },
                height = (uint)imageHeight,
                width = (uint)imageWidth,
                distortion_model = "plumb_bob",
                d = new double[] { 0, 0, 0, 0, 0 },  // No distortion
                k = new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 },
                r = new double[] { 1, 0, 0, 0, 1, 0, 0, 0, 1 },
                p = new double[] { fx, 0, cx, 0, 0, fy, cy, 0, 0, 0, 1, 0 }
            };

            ros.Publish(cameraInfoTopic, cameraInfoMsg);
        }

        TimeMsg GetROSTimestamp()
        {
            double totalSeconds = Time.timeAsDouble;
            int sec = (int)totalSeconds;
            uint nanosec = (uint)((totalSeconds - sec) * 1e9);
            return new TimeMsg { sec = sec, nanosec = nanosec };
        }

        void OnDestroy()
        {
            if (renderTexture != null)
            {
                renderTexture.Release();
                Destroy(renderTexture);
            }
            if (texture2D != null)
            {
                Destroy(texture2D);
            }
        }

        // For preview in editor
        void OnGUI()
        {
            if (showPreview && previewTexture != null)
            {
                GUI.DrawTexture(new Rect(10, 10, 160, 120), previewTexture, ScaleMode.ScaleToFit);
            }
        }
    }
}
