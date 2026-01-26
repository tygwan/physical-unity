using UnityEngine;
using ADPlatform.Sensors;
using ADPlatform.ROS2;

namespace ADPlatform.DebugTools
{
    /// <summary>
    /// Debug HUD for visualizing sensor data and vehicle state in real-time.
    /// Press F1 to toggle HUD visibility.
    /// </summary>
    public class SensorDebugHUD : MonoBehaviour
    {
        [Header("References")]
        public Rigidbody vehicleRigidbody;
        public CameraSensor cameraSensor;
        public LiDARSensor lidarSensor;
        public VehicleROSBridge rosBridge;

        [Header("HUD Settings")]
        public bool showHUD = true;
        public bool showCameraPreview = true;
        public bool showLiDARVisualization = true;
        public KeyCode toggleKey = KeyCode.F1;

        [Header("Camera Preview")]
        public int previewWidth = 320;
        public int previewHeight = 240;
        public Vector2 previewPosition = new Vector2(10, 10);

        [Header("LiDAR Visualization")]
        public bool showLiDARRays = false;
        public float pointSize = 2f;
        public Gradient distanceGradient;

        // Internal state
        private GUIStyle headerStyle;
        private GUIStyle valueStyle;
        private GUIStyle boxStyle;
        private bool stylesInitialized = false;
        private float fps;
        private float fpsUpdateInterval = 0.5f;
        private float fpsAccumulator;
        private int fpsFrames;
        private float fpsTimeLeft;

        // LiDAR visualization
        private Material lineMaterial;

        void Start()
        {
            // Auto-find references if not set
            if (vehicleRigidbody == null)
                vehicleRigidbody = FindObjectOfType<Rigidbody>();
            if (cameraSensor == null)
                cameraSensor = FindObjectOfType<CameraSensor>();
            if (lidarSensor == null)
                lidarSensor = FindObjectOfType<LiDARSensor>();
            if (rosBridge == null)
                rosBridge = FindObjectOfType<VehicleROSBridge>();

            // Initialize gradient if not set
            if (distanceGradient == null)
            {
                distanceGradient = new Gradient();
                distanceGradient.SetKeys(
                    new GradientColorKey[] {
                        new GradientColorKey(Color.red, 0f),
                        new GradientColorKey(Color.yellow, 0.3f),
                        new GradientColorKey(Color.green, 0.6f),
                        new GradientColorKey(Color.cyan, 1f)
                    },
                    new GradientAlphaKey[] {
                        new GradientAlphaKey(1f, 0f),
                        new GradientAlphaKey(1f, 1f)
                    }
                );
            }

            // Create line material for LiDAR visualization
            CreateLineMaterial();

            fpsTimeLeft = fpsUpdateInterval;

            UnityEngine.Debug.Log("[SensorDebugHUD] Initialized - Press F1 to toggle HUD");
        }

        void Update()
        {
            // Toggle HUD (F1)
            if (Input.GetKeyDown(toggleKey))
            {
                showHUD = !showHUD;
            }

            // Toggle LiDAR rays (F2)
            if (Input.GetKeyDown(KeyCode.F2))
            {
                showLiDARRays = !showLiDARRays;
                UnityEngine.Debug.Log($"[SensorDebugHUD] LiDAR Rays: {(showLiDARRays ? "ON" : "OFF")}");
            }

            // Toggle Camera Preview (F3)
            if (Input.GetKeyDown(KeyCode.F3))
            {
                showCameraPreview = !showCameraPreview;
                UnityEngine.Debug.Log($"[SensorDebugHUD] Camera Preview: {(showCameraPreview ? "ON" : "OFF")}");
            }

            // FPS calculation
            fpsTimeLeft -= Time.deltaTime;
            fpsAccumulator += Time.timeScale / Time.deltaTime;
            fpsFrames++;

            if (fpsTimeLeft <= 0f)
            {
                fps = fpsAccumulator / fpsFrames;
                fpsTimeLeft = fpsUpdateInterval;
                fpsAccumulator = 0f;
                fpsFrames = 0;
            }

            // Update LiDAR debug rays
            if (lidarSensor != null)
            {
                lidarSensor.showDebugRays = showLiDARRays && showLiDARVisualization;
            }
        }

        void InitStyles()
        {
            if (stylesInitialized) return;

            headerStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                normal = { textColor = Color.white }
            };

            valueStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                normal = { textColor = Color.green }
            };

            boxStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = MakeTexture(2, 2, new Color(0f, 0f, 0f, 0.7f)) }
            };

            stylesInitialized = true;
        }

        void OnGUI()
        {
            if (!showHUD) return;

            InitStyles();

            float yOffset = 10f;
            float xOffset = 10f;
            float panelWidth = 280f;

            // Camera Preview (top-left)
            if (showCameraPreview && cameraSensor != null && cameraSensor.previewTexture != null)
            {
                GUI.DrawTexture(
                    new Rect(previewPosition.x, previewPosition.y, previewWidth, previewHeight),
                    cameraSensor.previewTexture,
                    ScaleMode.ScaleToFit
                );
                GUI.Label(
                    new Rect(previewPosition.x, previewPosition.y + previewHeight, previewWidth, 20),
                    $"Camera: {cameraSensor.imageWidth}x{cameraSensor.imageHeight} @ {cameraSensor.publishRate}Hz",
                    valueStyle
                );
                yOffset = previewPosition.y + previewHeight + 30;
            }

            // Vehicle State Panel
            DrawVehicleStatePanel(xOffset, yOffset, panelWidth);

            // Sensor Status Panel (right side)
            DrawSensorStatusPanel(Screen.width - panelWidth - 10, 10, panelWidth);

            // Controls Help (bottom)
            DrawControlsHelp();
        }

        void DrawVehicleStatePanel(float x, float y, float width)
        {
            if (vehicleRigidbody == null) return;

            float height = 200f;
            GUI.Box(new Rect(x, y, width, height), "", boxStyle);

            float innerX = x + 10;
            float innerY = y + 5;
            float lineHeight = 18f;

            GUI.Label(new Rect(innerX, innerY, width, 20), "VEHICLE STATE", headerStyle);
            innerY += 25;

            // Position
            Vector3 pos = vehicleRigidbody.position;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Position: ({pos.x:F2}, {pos.y:F2}, {pos.z:F2})", valueStyle);
            innerY += lineHeight;

            // Rotation
            Vector3 rot = vehicleRigidbody.rotation.eulerAngles;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Rotation: ({rot.x:F1}°, {rot.y:F1}°, {rot.z:F1}°)", valueStyle);
            innerY += lineHeight;

            // Velocity
            Vector3 vel = vehicleRigidbody.linearVelocity;
            float speed = vel.magnitude;
            float speedKmh = speed * 3.6f;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Speed: {speed:F2} m/s ({speedKmh:F1} km/h)", valueStyle);
            innerY += lineHeight;

            // Velocity components
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Velocity: ({vel.x:F2}, {vel.y:F2}, {vel.z:F2})", valueStyle);
            innerY += lineHeight;

            // Angular velocity
            Vector3 angVel = vehicleRigidbody.angularVelocity;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Angular: ({angVel.x:F2}, {angVel.y:F2}, {angVel.z:F2})", valueStyle);
            innerY += lineHeight;

            // Heading
            float heading = vehicleRigidbody.rotation.eulerAngles.y;
            string headingDir = GetHeadingDirection(heading);
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Heading: {heading:F1}° ({headingDir})", valueStyle);
            innerY += lineHeight;

            // FPS
            GUI.Label(new Rect(innerX, innerY, width, 20), $"FPS: {fps:F1}", valueStyle);
        }

        void DrawSensorStatusPanel(float x, float y, float width)
        {
            float height = 180f;
            GUI.Box(new Rect(x, y, width, height), "", boxStyle);

            float innerX = x + 10;
            float innerY = y + 5;
            float lineHeight = 18f;

            GUI.Label(new Rect(innerX, innerY, width, 20), "SENSOR STATUS", headerStyle);
            innerY += 25;

            // Camera status
            string camStatus = cameraSensor != null ? "✓ Active" : "✗ Not Found";
            Color camColor = cameraSensor != null ? Color.green : Color.red;
            GUI.contentColor = camColor;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"Camera: {camStatus}", valueStyle);
            innerY += lineHeight;

            if (cameraSensor != null)
            {
                GUI.contentColor = Color.white;
                GUI.Label(new Rect(innerX + 20, innerY, width, 20), $"Topic: {cameraSensor.imageTopic}", valueStyle);
                innerY += lineHeight;
            }

            // LiDAR status
            string lidarStatus = lidarSensor != null ? "✓ Active" : "✗ Not Found";
            Color lidarColor = lidarSensor != null ? Color.green : Color.red;
            GUI.contentColor = lidarColor;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"LiDAR: {lidarStatus}", valueStyle);
            innerY += lineHeight;

            if (lidarSensor != null)
            {
                GUI.contentColor = Color.white;
                GUI.Label(new Rect(innerX + 20, innerY, width, 20), $"Topic: {lidarSensor.pointCloudTopic}", valueStyle);
                innerY += lineHeight;
                GUI.Label(new Rect(innerX + 20, innerY, width, 20), $"Rays: {lidarSensor.verticalRays}x{lidarSensor.horizontalRays}", valueStyle);
                innerY += lineHeight;
            }

            // ROS Bridge status
            string rosStatus = rosBridge != null ? (rosBridge.IsConnected() ? "✓ Connected" : "⚠ Disconnected") : "✗ Not Found";
            Color rosColor = rosBridge != null && rosBridge.IsConnected() ? Color.green : Color.yellow;
            GUI.contentColor = rosColor;
            GUI.Label(new Rect(innerX, innerY, width, 20), $"ROS Bridge: {rosStatus}", valueStyle);
            GUI.contentColor = Color.white;
        }

        void DrawControlsHelp()
        {
            float width = 300f;
            float height = 80f;
            float x = (Screen.width - width) / 2;
            float y = Screen.height - height - 10;

            GUI.Box(new Rect(x, y, width, height), "", boxStyle);

            float innerX = x + 10;
            float innerY = y + 5;

            GUI.Label(new Rect(innerX, innerY, width, 20), "CONTROLS", headerStyle);
            innerY += 22;
            GUI.Label(new Rect(innerX, innerY, width, 20), "W/S: Forward/Backward | A/D: Steer", valueStyle);
            innerY += 16;
            GUI.Label(new Rect(innerX, innerY, width, 20), "F1: Toggle HUD | F2: Toggle LiDAR Rays", valueStyle);
            innerY += 16;
            GUI.Label(new Rect(innerX, innerY, width, 20), "F3: Toggle Camera Preview", valueStyle);
        }

        string GetHeadingDirection(float heading)
        {
            if (heading >= 337.5f || heading < 22.5f) return "N";
            if (heading >= 22.5f && heading < 67.5f) return "NE";
            if (heading >= 67.5f && heading < 112.5f) return "E";
            if (heading >= 112.5f && heading < 157.5f) return "SE";
            if (heading >= 157.5f && heading < 202.5f) return "S";
            if (heading >= 202.5f && heading < 247.5f) return "SW";
            if (heading >= 247.5f && heading < 292.5f) return "W";
            if (heading >= 292.5f && heading < 337.5f) return "NW";
            return "?";
        }

        Texture2D MakeTexture(int width, int height, Color color)
        {
            Color[] pixels = new Color[width * height];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = color;

            Texture2D texture = new Texture2D(width, height);
            texture.SetPixels(pixels);
            texture.Apply();
            return texture;
        }

        void CreateLineMaterial()
        {
            if (!lineMaterial)
            {
                Shader shader = Shader.Find("Hidden/Internal-Colored");
                lineMaterial = new Material(shader);
                lineMaterial.hideFlags = HideFlags.HideAndDontSave;
                lineMaterial.SetInt("_SrcBlend", (int)UnityEngine.Rendering.BlendMode.SrcAlpha);
                lineMaterial.SetInt("_DstBlend", (int)UnityEngine.Rendering.BlendMode.OneMinusSrcAlpha);
                lineMaterial.SetInt("_Cull", (int)UnityEngine.Rendering.CullMode.Off);
                lineMaterial.SetInt("_ZWrite", 0);
            }
        }
    }
}
