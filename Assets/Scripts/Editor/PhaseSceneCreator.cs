using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using System.IO;

/// <summary>
/// Creates training scenes for each Phase of the curriculum.
/// Use: Tools > Create Phase Scenes > [Phase Name]
/// </summary>
public class PhaseSceneCreator
{
    private const string SCENES_PATH = "Assets/Scenes";
    private const int NUM_TRAINING_AREAS = 16;  // 1x16 linear layout
    private const float AREA_SPACING = 100f;    // 100m between areas (X axis)

    // Intersection geometry constants
    private const float INTERSECTION_Z = 100f;           // Intersection center Z (from WaypointManager)
    private const float INTERSECTION_WIDTH = 14f;         // Intersection zone width
    private const float INTERSECTION_ZONE_START = 93f;    // Z start of intersection zone
    private const float INTERSECTION_ZONE_END = 107f;     // Z end of intersection zone
    private const float ROAD_WIDTH = 8f;                  // 2 lanes * 3.5 + 1m margin
    private const float ARM_LENGTH = 25f;                 // Length of intersection arms
    private const float CURB_WIDTH = 0.3f;
    private const float CURB_HEIGHT = 0.15f;

    // Shared material cache (reused across training areas)
    private static Material s_asphaltMaterial;
    private static Material s_lightAsphaltMaterial;
    private static Material s_curbMaterial;
    private static Material s_whiteLineMaterial;
    private static Material s_yellowLineMaterial;

    #region Menu Items

    [MenuItem("Tools/Create Phase Scenes/Create All Phase Scenes")]
    public static void CreateAllPhaseScenes()
    {
        CreatePhaseAScene();
        CreatePhaseBScene();
        CreatePhaseCScene();
        CreatePhaseEScene();
        CreatePhaseFScene();
        CreatePhaseGScene();
        CreatePhaseHScene();
        CreatePhaseJScene();
        CreatePhaseKScene();
        CreatePhaseLScene();
        CreatePhaseMScene();

        EditorUtility.DisplayDialog("Complete",
            "All Phase scenes created in Assets/Scenes/", "OK");
    }

    [MenuItem("Tools/Create Phase Scenes/Phase M - Test Field")]
    public static void CreatePhaseMScene() { CreatePhaseM_TestField(); }

    [MenuItem("Tools/Create Phase Scenes/Phase A - Dense Overtaking")]
    public static void CreatePhaseAScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseA_DenseOvertaking",
            description = "Dense traffic overtaking training",
            numNPCs = 5,
            numLanes = 1,
            roadCurvature = 0f,
            intersectionType = 0,
            npcSpeedRatio = 0.3f,
            roadLength = 300f,
            observationSize = 242,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase B - Decision Learning")]
    public static void CreatePhaseBScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseB_DecisionLearning",
            description = "Decision making in traffic",
            numNPCs = 3,
            numLanes = 1,
            roadCurvature = 0f,
            intersectionType = 0,
            npcSpeedRatio = 0.5f,
            roadLength = 300f,
            observationSize = 242,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase C - Multi NPC")]
    public static void CreatePhaseCScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseC_MultiNPC",
            description = "Complex multi-vehicle interaction",
            numNPCs = 5,
            numLanes = 1,
            roadCurvature = 0f,
            intersectionType = 0,
            npcSpeedRatio = 0.6f,
            numSpeedZones = 2,
            roadLength = 400f,
            observationSize = 242,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase E - Curved Roads")]
    public static void CreatePhaseEScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseE_CurvedRoads",
            description = "Curved road navigation",
            numNPCs = 3,
            numLanes = 1,
            roadCurvature = 0.5f,
            curveDirectionVariation = 0.5f,
            intersectionType = 0,
            npcSpeedRatio = 0.7f,
            roadLength = 500f,
            observationSize = 254,
            enableLaneObservation = true,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase F - Multi Lane")]
    public static void CreatePhaseFScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseF_MultiLane",
            description = "Multi-lane highway driving",
            numNPCs = 4,
            numLanes = 3,
            centerLineEnabled = true,
            roadCurvature = 0.2f,
            intersectionType = 0,
            npcSpeedRatio = 0.8f,
            roadLength = 500f,
            observationSize = 254,
            enableLaneObservation = true,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase G - Intersection")]
    public static void CreatePhaseGScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseG_Intersection",
            description = "Intersection navigation",
            numNPCs = 2,
            numLanes = 2,
            roadCurvature = 0f,
            intersectionType = 2, // Cross intersection
            turnDirection = 0,    // Configurable at runtime
            npcSpeedRatio = 0.6f,
            roadLength = 300f,
            observationSize = 260,
            enableLaneObservation = true,
            enableIntersectionObservation = true,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase H - NPC Intersection")]
    public static void CreatePhaseHScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseH_NPCIntersection",
            description = "NPC interaction in intersections",
            numNPCs = 3,
            numLanes = 2,
            roadCurvature = 0f,
            intersectionType = 2, // Cross intersection default
            turnDirection = 0,    // Configurable at runtime
            npcSpeedRatio = 0.6f,
            roadLength = 300f,
            observationSize = 260,
            enableLaneObservation = true,
            enableIntersectionObservation = true,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase J - Traffic Signals")]
    public static void CreatePhaseJScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseJ_TrafficSignals",
            description = "Traffic signals + stop lines at intersections",
            numNPCs = 3,
            numLanes = 2,
            roadCurvature = 0f,
            intersectionType = 2, // Cross intersection default
            turnDirection = 0,    // Configurable at runtime
            npcSpeedRatio = 0.6f,
            roadLength = 300f,
            observationSize = 268,
            enableLaneObservation = true,
            enableIntersectionObservation = true,
            enableTrafficSignalObservation = true,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase K - Dense Urban")]
    public static void CreatePhaseKScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseK_DenseUrban",
            description = "Dense urban: curved roads + intersections + traffic signals + NPCs",
            numNPCs = 5,
            numLanes = 2,
            centerLineEnabled = true,
            roadCurvature = 0.3f,
            curveDirectionVariation = 0.5f,
            intersectionType = 2, // Cross intersection default
            turnDirection = 0,    // Configurable at runtime
            npcSpeedRatio = 0.85f,
            roadLength = 500f,
            observationSize = 268,
            enableLaneObservation = true,
            enableIntersectionObservation = true,
            enableTrafficSignalObservation = true,
        });
    }

    [MenuItem("Tools/Create Phase Scenes/Phase L - Crosswalks")]
    public static void CreatePhaseLScene()
    {
        CreatePhaseScene(new PhaseConfig
        {
            name = "PhaseL_Crosswalks",
            description = "Crosswalk + pedestrian yielding in urban environment",
            numNPCs = 5,
            numLanes = 2,
            centerLineEnabled = true,
            roadCurvature = 0.3f,
            curveDirectionVariation = 0.5f,
            intersectionType = 2,
            turnDirection = 0,
            npcSpeedRatio = 0.85f,
            roadLength = 500f,
            observationSize = 280,
            enableLaneObservation = true,
            enableIntersectionObservation = true,
            enableTrafficSignalObservation = true,
            enablePedestrianObservation = true,
            numPedestrians = 3,
        });
    }

    #endregion

    #region Scene Creation

    private struct PhaseConfig
    {
        public string name;
        public string description;
        public int numNPCs;
        public int numLanes;
        public bool centerLineEnabled;
        public float roadCurvature;
        public float curveDirectionVariation;
        public int intersectionType;
        public int turnDirection;
        public float npcSpeedRatio;
        public int numSpeedZones;
        public float roadLength;
        // Observation configuration
        public int observationSize;            // 242, 254, 260, 268, or 280
        public bool enableLaneObservation;     // true for Phase D+
        public bool enableIntersectionObservation; // true for Phase G+
        public bool enableTrafficSignalObservation; // true for Phase J+
        public bool enablePedestrianObservation;   // true for Phase L+
        public int numPedestrians;             // max pedestrian pool (Phase L+)
    }

    private static void CreatePhaseScene(PhaseConfig config)
    {
        // Ensure scenes folder exists
        if (!Directory.Exists(SCENES_PATH))
        {
            Directory.CreateDirectory(SCENES_PATH);
            AssetDatabase.Refresh();
        }

        // Reset material cache for fresh scene
        InitMaterialCache();

        // Create new scene
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // 1. Main Camera with FollowCamera (positioned based on road length)
        var camera = CreateMainCamera(config.roadLength);

        // 2. Directional Light
        CreateDirectionalLight();

        // 3. Training Areas parent
        var trainingAreasParent = new GameObject("TrainingAreas");
        trainingAreasParent.transform.position = Vector3.zero;

        // 4. Create 16 parallel training areas (linear layout along X axis)
        GameObject firstAgent = null;
        for (int i = 0; i < NUM_TRAINING_AREAS; i++)
        {
            Vector3 areaOffset = new Vector3(i * AREA_SPACING, 0, 0);

            var trainingArea = CreateTrainingArea(config, i, areaOffset, trainingAreasParent.transform);

            // Store first agent for camera follow
            if (i == 0)
            {
                firstAgent = trainingArea.transform.Find("E2EDrivingAgent")?.gameObject;
            }
        }

        // Wire camera to first agent
        if (firstAgent != null)
        {
            var followCamera = camera.GetComponent("FollowCamera");
            if (followCamera != null)
            {
                SetProperty(followCamera, "target", firstAgent.transform);
            }
        }

        // Save scene
        string scenePath = $"{SCENES_PATH}/{config.name}.unity";
        EditorSceneManager.SaveScene(scene, scenePath);

        // Add to build settings
        AddSceneToBuildSettings(scenePath);

        Debug.Log($"[PhaseSceneCreator] Created {config.name} with {NUM_TRAINING_AREAS} parallel training areas: {config.description}");
    }

    private static GameObject CreateTrainingArea(PhaseConfig config, int areaIndex, Vector3 offset, Transform parent)
    {
        var trainingArea = new GameObject($"TrainingArea_{areaIndex}");
        trainingArea.transform.SetParent(parent);
        trainingArea.transform.position = offset;

        // Ground for this area (wider for intersection phases with arms)
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(trainingArea.transform);
        ground.transform.localPosition = Vector3.zero;

        float width = config.intersectionType > 0
            ? 80f  // Wide ground for intersection arms
            : Mathf.Max(50f, config.numLanes * 5f + 30f);
        float depth = config.roadLength + 20f;
        ground.transform.localScale = new Vector3(width / 10f, 1, depth / 10f);

        var groundRenderer = ground.GetComponent<Renderer>();
        var groundMaterial = new Material(Shader.Find("Standard"));
        groundMaterial.color = new Color(0.15f, 0.4f, 0.15f);
        groundRenderer.material = groundMaterial;

        // Road with WaypointManager
        var road = CreateRoadForArea(config, trainingArea.transform);

        // DrivingSceneManager for this area
        var sceneManager = CreateDrivingSceneManagerForArea(config, trainingArea.transform);

        // Agent Vehicle
        var agent = CreateAgentVehicleForArea(config.roadLength, areaIndex, trainingArea.transform, config);

        // NPC Vehicles
        var npcs = CreateNPCVehiclesForArea(config.numNPCs, config.roadLength, trainingArea.transform);

        // Goal Target
        var goal = CreateGoalTargetForArea(config.roadLength, trainingArea.transform);

        // Traffic Light (Phase J)
        GameObject trafficLightObj = null;
        if (config.enableTrafficSignalObservation)
        {
            trafficLightObj = CreateTrafficLightForArea(trainingArea.transform);
        }

        // Pedestrians + Crosswalk (Phase L)
        GameObject[] pedestrianObjs = null;
        if (config.enablePedestrianObservation)
        {
            pedestrianObjs = CreatePedestriansForArea(config.numPedestrians, trainingArea.transform);
            CreateCrosswalkForArea(trainingArea.transform);
        }

        // Wire up references within this training area
        WireAreaReferences(sceneManager, agent, road, goal, npcs, trafficLightObj, pedestrianObjs);

        return trainingArea;
    }

    private static GameObject CreateMainCamera(float roadLength)
    {
        var camera = new GameObject("Main Camera");
        camera.tag = "MainCamera";
        var cam = camera.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.Skybox;
        cam.fieldOfView = 60f;

        // Position camera behind the first agent
        // Agent starts at z = -roadLength/2 + 10, camera 25m behind
        float agentStartZ = -roadLength / 2f + 10f;
        camera.transform.position = new Vector3(1.75f, 15, agentStartZ - 25f);
        camera.transform.rotation = Quaternion.Euler(30, 0, 0);
        camera.AddComponent<AudioListener>();

        // Add FollowCamera component
        var followCameraType = System.Type.GetType("FollowCamera, Assembly-CSharp");
        if (followCameraType != null)
        {
            camera.AddComponent(followCameraType);
        }

        return camera;
    }

    private static void CreateDirectionalLight()
    {
        var light = new GameObject("Directional Light");
        var lightComp = light.AddComponent<Light>();
        lightComp.type = LightType.Directional;
        lightComp.intensity = 1.2f;
        lightComp.shadows = LightShadows.Soft;
        lightComp.color = new Color(1f, 0.95f, 0.85f);
        light.transform.rotation = Quaternion.Euler(50, -30, 0);
    }

    private static void CreateGround(float roadLength, int numLanes)
    {
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = Vector3.zero;

        float width = Mathf.Max(50f, numLanes * 5f + 30f);
        ground.transform.localScale = new Vector3(width / 10f, 1, roadLength / 10f + 10f);

        var renderer = ground.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.15f, 0.4f, 0.15f); // Green grass
        renderer.material = material;
    }

    private static GameObject CreateRoad(PhaseConfig config)
    {
        var road = new GameObject("Road");

        // Road surface
        var roadSurface = GameObject.CreatePrimitive(PrimitiveType.Plane);
        roadSurface.name = "RoadSurface";
        roadSurface.transform.SetParent(road.transform);
        roadSurface.transform.localPosition = new Vector3(0, 0.01f, 0);

        float roadWidth = config.numLanes * 3.5f + 1f; // Lane width + margins
        roadSurface.transform.localScale = new Vector3(roadWidth / 10f, 1, config.roadLength / 10f);

        var renderer = roadSurface.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.25f, 0.25f, 0.25f); // Asphalt gray
        renderer.material = material;

        // Add WaypointManager with Phase-specific settings
        var wpManagerType = System.Type.GetType("ADPlatform.Agents.WaypointManager, Assembly-CSharp");
        if (wpManagerType != null)
        {
            var wpManager = road.AddComponent(wpManagerType);

            // Set properties via reflection
            SetProperty(wpManager, "roadLength", config.roadLength);
            SetProperty(wpManager, "numLanes", config.numLanes);
            SetProperty(wpManager, "centerLineEnabled", config.centerLineEnabled);
            SetProperty(wpManager, "roadCurvature", config.roadCurvature);
            SetProperty(wpManager, "curveDirectionVariation", config.curveDirectionVariation);
            SetProperty(wpManager, "intersectionType", config.intersectionType);
            SetProperty(wpManager, "turnDirection", config.turnDirection);
            if (config.numSpeedZones > 0)
                SetProperty(wpManager, "numSpeedZones", config.numSpeedZones);
        }

        // Create lane markings
        CreateLaneMarkings(road.transform, config);

        return road;
    }

    private static void CreateLaneMarkings(Transform roadParent, PhaseConfig config)
    {
        float roadLength = config.roadLength;
        float markingSpacing = 10f;
        float dashLength = 3f;

        // Center line (if applicable)
        if (config.numLanes > 1)
        {
            var centerLine = new GameObject("CenterLine");
            centerLine.transform.SetParent(roadParent);

            for (float z = -roadLength / 2f; z < roadLength / 2f; z += markingSpacing)
            {
                var dash = GameObject.CreatePrimitive(PrimitiveType.Cube);
                dash.name = "Dash";
                dash.transform.SetParent(centerLine.transform);
                dash.transform.localPosition = new Vector3(0, 0.02f, z);
                dash.transform.localScale = new Vector3(0.15f, 0.01f, dashLength);

                var renderer = dash.GetComponent<Renderer>();
                var material = new Material(Shader.Find("Standard"));
                material.color = config.centerLineEnabled ? Color.yellow : Color.white;
                renderer.material = material;

                // Remove collider from marking
                Object.DestroyImmediate(dash.GetComponent<Collider>());
            }
        }
    }

    private static GameObject CreateDrivingSceneManager(PhaseConfig config)
    {
        var manager = new GameObject("DrivingSceneManager");

        var managerType = System.Type.GetType("ADPlatform.Environment.DrivingSceneManager, Assembly-CSharp");
        if (managerType != null)
        {
            var component = manager.AddComponent(managerType);

            // Set default values
            SetProperty(component, "maxEpisodeTime", 90f);
            SetProperty(component, "useCurriculum", true);
            SetProperty(component, "defaultGoalDistance", config.roadLength * 0.9f);
            SetProperty(component, "npcMinSpawnDistance", 30f);
            SetProperty(component, "npcMaxSpawnDistance", config.roadLength * 0.6f);
        }

        return manager;
    }

    private static GameObject CreateAgentVehicle(float roadLength)
    {
        // Create vehicle body
        var agent = GameObject.CreatePrimitive(PrimitiveType.Cube);
        agent.name = "E2EDrivingAgent";
        agent.transform.position = new Vector3(1.75f, 0.75f, -roadLength / 2f + 10f);
        agent.transform.localScale = new Vector3(2f, 1.5f, 4.5f);

        var renderer = agent.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.2f, 0.4f, 0.8f); // Blue agent
        renderer.material = material;

        // Add Rigidbody
        var rb = agent.AddComponent<Rigidbody>();
        rb.mass = 1500f;
        rb.linearDamping = 1f;
        rb.angularDamping = 2f;
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        // Add BoxCollider (replace default)
        Object.DestroyImmediate(agent.GetComponent<Collider>());
        var collider = agent.AddComponent<BoxCollider>();
        collider.size = Vector3.one;

        // Add E2EDrivingAgent component
        var agentType = System.Type.GetType("ADPlatform.Agents.E2EDrivingAgent, Assembly-CSharp");
        if (agentType != null)
        {
            agent.AddComponent(agentType);
        }

        return agent;
    }

    private static GameObject[] CreateNPCVehicles(int count, float roadLength)
    {
        var npcs = new GameObject[count];
        var npcParent = new GameObject("NPCVehicles");

        for (int i = 0; i < count; i++)
        {
            var npc = GameObject.CreatePrimitive(PrimitiveType.Cube);
            npc.name = $"NPC_{i}";
            npc.transform.SetParent(npcParent.transform);

            // Position NPCs ahead of agent
            float zOffset = 50f + i * 30f;
            float xOffset = (i % 2 == 0) ? 1.75f : -1.75f;
            npc.transform.position = new Vector3(xOffset, 0.75f, -roadLength / 2f + zOffset);
            npc.transform.localScale = new Vector3(2f, 1.5f, 4.5f);

            var renderer = npc.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            // Varied NPC colors
            material.color = GetNPCColor(i);
            renderer.material = material;

            // Add Rigidbody
            var rb = npc.AddComponent<Rigidbody>();
            rb.mass = 1500f;
            rb.isKinematic = true;

            // Add NPCVehicleController
            var npcType = System.Type.GetType("ADPlatform.Agents.NPCVehicleController, Assembly-CSharp");
            if (npcType != null)
            {
                npc.AddComponent(npcType);
            }

            npc.SetActive(false); // Start inactive, curriculum controls activation
            npcs[i] = npc;
        }

        return npcs;
    }

    private static Color GetNPCColor(int index)
    {
        Color[] colors = {
            new Color(0.8f, 0.2f, 0.2f), // Red
            new Color(0.9f, 0.9f, 0.9f), // White
            new Color(0.2f, 0.2f, 0.2f), // Black
            new Color(0.7f, 0.7f, 0.2f), // Yellow
            new Color(0.5f, 0.5f, 0.5f), // Gray
        };
        return colors[index % colors.Length];
    }

    private static GameObject CreateGoalTarget(float roadLength)
    {
        var goal = new GameObject("GoalTarget");
        goal.transform.position = new Vector3(0, 1f, roadLength / 2f - 20f);

        // Visual indicator
        var visual = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        visual.name = "GoalVisual";
        visual.transform.SetParent(goal.transform);
        visual.transform.localPosition = Vector3.zero;
        visual.transform.localScale = new Vector3(5f, 0.1f, 5f);

        var renderer = visual.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.2f, 0.8f, 0.2f, 0.5f);
        renderer.material = material;

        // Remove collider from visual
        Object.DestroyImmediate(visual.GetComponent<Collider>());

        return goal;
    }

    private static GameObject CreateRoadForArea(PhaseConfig config, Transform parent)
    {
        var road = new GameObject("Road");
        road.transform.SetParent(parent);
        road.transform.localPosition = Vector3.zero;

        // Main road surface
        var roadSurface = GameObject.CreatePrimitive(PrimitiveType.Plane);
        roadSurface.name = "MainRoadSurface";
        roadSurface.transform.SetParent(road.transform);
        roadSurface.transform.localPosition = new Vector3(0, 0.01f, 0);

        float roadWidth = config.numLanes * 3.5f + 1f;
        roadSurface.transform.localScale = new Vector3(roadWidth / 10f, 1, config.roadLength / 10f);
        roadSurface.GetComponent<Renderer>().sharedMaterial = GetAsphaltMaterial();

        // Add WaypointManager
        var wpManagerType = System.Type.GetType("ADPlatform.Agents.WaypointManager, Assembly-CSharp");
        if (wpManagerType != null)
        {
            var wpManager = road.AddComponent(wpManagerType);
            SetProperty(wpManager, "roadLength", config.roadLength);
            SetProperty(wpManager, "numLanes", config.numLanes);
            SetProperty(wpManager, "centerLineEnabled", config.centerLineEnabled);
            SetProperty(wpManager, "roadCurvature", config.roadCurvature);
            SetProperty(wpManager, "curveDirectionVariation", config.curveDirectionVariation);
            SetProperty(wpManager, "intersectionType", config.intersectionType);
            SetProperty(wpManager, "turnDirection", config.turnDirection);
            if (config.numSpeedZones > 0)
                SetProperty(wpManager, "numSpeedZones", config.numSpeedZones);
        }

        // Add intersection road visuals for intersection phases
        if (config.intersectionType > 0)
        {
            CreateIntersectionRoadVisuals(road.transform, config);
        }

        return road;
    }

    private static GameObject CreateDrivingSceneManagerForArea(PhaseConfig config, Transform parent)
    {
        var manager = new GameObject("DrivingSceneManager");
        manager.transform.SetParent(parent);
        manager.transform.localPosition = Vector3.zero;

        var managerType = System.Type.GetType("ADPlatform.Environment.DrivingSceneManager, Assembly-CSharp");
        if (managerType != null)
        {
            var component = manager.AddComponent(managerType);
            SetProperty(component, "maxEpisodeTime", 90f);
            SetProperty(component, "useCurriculum", true);
            SetProperty(component, "defaultGoalDistance", config.roadLength * 0.9f);
            SetProperty(component, "npcMinSpawnDistance", 30f);
            SetProperty(component, "npcMaxSpawnDistance", config.roadLength * 0.6f);
        }

        return manager;
    }

    private static GameObject CreateAgentVehicleForArea(float roadLength, int areaIndex, Transform parent, PhaseConfig config)
    {
        var agent = GameObject.CreatePrimitive(PrimitiveType.Cube);
        agent.name = "E2EDrivingAgent";
        agent.transform.SetParent(parent);
        agent.transform.localPosition = new Vector3(1.75f, 0.75f, -roadLength / 2f + 10f);
        agent.transform.localScale = new Vector3(2f, 1.5f, 4.5f);

        var renderer = agent.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.2f, 0.4f, 0.8f);
        renderer.material = material;

        var rb = agent.AddComponent<Rigidbody>();
        rb.mass = 1500f;
        rb.linearDamping = 1f;
        rb.angularDamping = 2f;
        rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

        Object.DestroyImmediate(agent.GetComponent<Collider>());
        var collider = agent.AddComponent<BoxCollider>();
        collider.size = Vector3.one;

        var agentType = System.Type.GetType("ADPlatform.Agents.E2EDrivingAgent, Assembly-CSharp");
        if (agentType != null)
        {
            var agentComponent = agent.AddComponent(agentType);
            // Set observation flags
            SetProperty(agentComponent, "enableLaneObservation", config.enableLaneObservation);
            SetProperty(agentComponent, "enableIntersectionObservation", config.enableIntersectionObservation);
            SetProperty(agentComponent, "enableTrafficSignalObservation", config.enableTrafficSignalObservation);
            SetProperty(agentComponent, "enablePedestrianObservation", config.enablePedestrianObservation);
        }

        // Configure BehaviorParameters for ML-Agents training via SerializedObject
        ConfigureBehaviorParameters(agent, config);

        // Add DecisionRequester (required for agents to send observations to Python trainer)
        var drType = System.Type.GetType("Unity.MLAgents.DecisionRequester, Unity.ML-Agents");
        if (drType != null)
        {
            var dr = agent.AddComponent(drType);
            SetProperty(dr, "DecisionPeriod", 5);
            SetProperty(dr, "TakeActionsBetweenDecisions", true);
        }

        return agent;
    }

    private static void ConfigureBehaviorParameters(GameObject agent, PhaseConfig config)
    {
        var bp = agent.GetComponent("BehaviorParameters") as UnityEngine.Component;
        if (bp == null) return;

        var so = new SerializedObject(bp);

        var nameProp = so.FindProperty("m_BehaviorName");
        if (nameProp != null) nameProp.stringValue = "E2EDrivingAgent";

        int obsSize = config.observationSize > 0 ? config.observationSize : 242;

        var obsSizeProp = so.FindProperty("m_BrainParameters.VectorObservationSize");
        if (obsSizeProp != null) obsSizeProp.intValue = obsSize;

        var stackProp = so.FindProperty("m_BrainParameters.NumStackedVectorObservations");
        if (stackProp != null) stackProp.intValue = 1;

        var continuousProp = so.FindProperty("m_BrainParameters.m_ActionSpec.m_NumContinuousActions");
        if (continuousProp != null) continuousProp.intValue = 2;

        // Clear discrete branches (set size to 0)
        var branchProp = so.FindProperty("m_BrainParameters.m_ActionSpec.BranchSizes");
        if (branchProp != null)
            branchProp.ClearArray();

        so.ApplyModifiedPropertiesWithoutUndo();
    }

    private static GameObject[] CreateNPCVehiclesForArea(int count, float roadLength, Transform parent)
    {
        var npcs = new GameObject[count];
        var npcParent = new GameObject("NPCVehicles");
        npcParent.transform.SetParent(parent);
        npcParent.transform.localPosition = Vector3.zero;

        for (int i = 0; i < count; i++)
        {
            var npc = GameObject.CreatePrimitive(PrimitiveType.Cube);
            npc.name = $"NPC_{i}";
            npc.transform.SetParent(npcParent.transform);

            float zOffset = 50f + i * 30f;
            float xOffset = (i % 2 == 0) ? 1.75f : -1.75f;
            npc.transform.localPosition = new Vector3(xOffset, 0.75f, -roadLength / 2f + zOffset);
            npc.transform.localScale = new Vector3(2f, 1.5f, 4.5f);

            var renderer = npc.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            material.color = GetNPCColor(i);
            renderer.material = material;

            var rb = npc.AddComponent<Rigidbody>();
            rb.mass = 1500f;
            rb.isKinematic = true;

            var npcType = System.Type.GetType("ADPlatform.Agents.NPCVehicleController, Assembly-CSharp");
            if (npcType != null)
            {
                npc.AddComponent(npcType);
            }

            npc.SetActive(false);
            npcs[i] = npc;
        }

        return npcs;
    }

    private static GameObject CreateGoalTargetForArea(float roadLength, Transform parent)
    {
        var goal = new GameObject("GoalTarget");
        goal.transform.SetParent(parent);
        goal.transform.localPosition = new Vector3(0, 1f, roadLength / 2f - 20f);

        var visual = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        visual.name = "GoalVisual";
        visual.transform.SetParent(goal.transform);
        visual.transform.localPosition = Vector3.zero;
        visual.transform.localScale = new Vector3(5f, 0.1f, 5f);

        var renderer = visual.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.2f, 0.8f, 0.2f, 0.5f);
        renderer.material = material;

        Object.DestroyImmediate(visual.GetComponent<Collider>());

        return goal;
    }

    /// <summary>
    /// Create a traffic light with pole and 3 signal spheres for Phase J.
    /// Positioned at the stop line (Z=93) on the right side of the road.
    /// </summary>
    private static GameObject CreateTrafficLightForArea(Transform parent)
    {
        var trafficLight = new GameObject("TrafficLight");
        trafficLight.transform.SetParent(parent);
        // Position: right side of road at stop line Z
        trafficLight.transform.localPosition = new Vector3(ROAD_WIDTH / 2f + 1.5f, 0f, 0f);

        // Pole (tall cylinder)
        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "Pole";
        pole.transform.SetParent(trafficLight.transform);
        pole.transform.localPosition = new Vector3(0f, 3f, INTERSECTION_ZONE_START);
        pole.transform.localScale = new Vector3(0.15f, 3f, 0.15f);
        Object.DestroyImmediate(pole.GetComponent<Collider>());
        var poleMat = new Material(Shader.Find("Standard"));
        poleMat.color = new Color(0.3f, 0.3f, 0.3f);
        pole.GetComponent<Renderer>().material = poleMat;

        // Signal housing (box)
        var housing = GameObject.CreatePrimitive(PrimitiveType.Cube);
        housing.name = "SignalHousing";
        housing.transform.SetParent(trafficLight.transform);
        housing.transform.localPosition = new Vector3(0f, 6.5f, INTERSECTION_ZONE_START);
        housing.transform.localScale = new Vector3(0.6f, 1.8f, 0.4f);
        Object.DestroyImmediate(housing.GetComponent<Collider>());
        var housingMat = new Material(Shader.Find("Standard"));
        housingMat.color = new Color(0.2f, 0.2f, 0.2f);
        housing.GetComponent<Renderer>().material = housingMat;

        // Red light (top)
        var redLight = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        redLight.name = "RedLight";
        redLight.transform.SetParent(trafficLight.transform);
        redLight.transform.localPosition = new Vector3(0f, 7.1f, INTERSECTION_ZONE_START - 0.15f);
        redLight.transform.localScale = new Vector3(0.35f, 0.35f, 0.35f);
        Object.DestroyImmediate(redLight.GetComponent<Collider>());
        var redMat = new Material(Shader.Find("Standard"));
        redMat.color = new Color(0.15f, 0.15f, 0.15f);
        redMat.EnableKeyword("_EMISSION");
        redLight.GetComponent<Renderer>().material = redMat;

        // Yellow light (middle)
        var yellowLight = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        yellowLight.name = "YellowLight";
        yellowLight.transform.SetParent(trafficLight.transform);
        yellowLight.transform.localPosition = new Vector3(0f, 6.5f, INTERSECTION_ZONE_START - 0.15f);
        yellowLight.transform.localScale = new Vector3(0.35f, 0.35f, 0.35f);
        Object.DestroyImmediate(yellowLight.GetComponent<Collider>());
        var yellowMat = new Material(Shader.Find("Standard"));
        yellowMat.color = new Color(0.15f, 0.15f, 0.15f);
        yellowMat.EnableKeyword("_EMISSION");
        yellowLight.GetComponent<Renderer>().material = yellowMat;

        // Green light (bottom)
        var greenLight = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        greenLight.name = "GreenLight";
        greenLight.transform.SetParent(trafficLight.transform);
        greenLight.transform.localPosition = new Vector3(0f, 5.9f, INTERSECTION_ZONE_START - 0.15f);
        greenLight.transform.localScale = new Vector3(0.35f, 0.35f, 0.35f);
        Object.DestroyImmediate(greenLight.GetComponent<Collider>());
        var greenMat = new Material(Shader.Find("Standard"));
        greenMat.color = new Color(0.15f, 0.15f, 0.15f);
        greenMat.EnableKeyword("_EMISSION");
        greenLight.GetComponent<Renderer>().material = greenMat;

        // Add TrafficLightController component
        var tlType = System.Type.GetType("ADPlatform.Environment.TrafficLightController, Assembly-CSharp");
        if (tlType != null)
        {
            var tlComponent = trafficLight.AddComponent(tlType);
            SetProperty(tlComponent, "redLight", redLight.GetComponent<Renderer>());
            SetProperty(tlComponent, "yellowLight", yellowLight.GetComponent<Renderer>());
            SetProperty(tlComponent, "greenLight", greenLight.GetComponent<Renderer>());
            SetProperty(tlComponent, "stopLineZ", INTERSECTION_ZONE_START);
        }

        return trafficLight;
    }

    private static void WireAreaReferences(GameObject manager, GameObject agent,
        GameObject road, GameObject goal, GameObject[] npcs, GameObject trafficLightObj = null,
        GameObject[] pedestrianObjs = null)
    {
        var managerComponent = manager.GetComponent("DrivingSceneManager");
        if (managerComponent == null) return;

        var agentComponent = agent.GetComponent("E2EDrivingAgent");
        SetProperty(managerComponent, "egoAgent", agentComponent);

        var wpManager = road.GetComponent("WaypointManager");
        SetProperty(managerComponent, "waypointManager", wpManager);

        SetProperty(managerComponent, "goalTarget", goal.transform);

        var npcType = System.Type.GetType("ADPlatform.Agents.NPCVehicleController, Assembly-CSharp");
        if (npcType != null)
        {
            var array = System.Array.CreateInstance(npcType, npcs.Length);
            for (int i = 0; i < npcs.Length; i++)
            {
                array.SetValue(npcs[i].GetComponent(npcType), i);
            }
            SetProperty(managerComponent, "npcVehicles", array);
        }

        // Wire intersection visual references
        Transform roadTransform = road.transform;
        Transform intersectionArea = roadTransform.Find("IntersectionArea");
        Transform leftArm = roadTransform.Find("LeftArm");
        Transform rightArm = roadTransform.Find("RightArm");
        Transform leftAngledArm = roadTransform.Find("LeftAngledArm");
        Transform rightAngledArm = roadTransform.Find("RightAngledArm");

        if (intersectionArea != null)
            SetProperty(managerComponent, "intersectionArea", intersectionArea.gameObject);
        if (leftArm != null)
            SetProperty(managerComponent, "leftArm", leftArm.gameObject);
        if (rightArm != null)
            SetProperty(managerComponent, "rightArm", rightArm.gameObject);
        if (leftAngledArm != null)
            SetProperty(managerComponent, "leftAngledArm", leftAngledArm.gameObject);
        if (rightAngledArm != null)
            SetProperty(managerComponent, "rightAngledArm", rightAngledArm.gameObject);

        // Wire traffic light (Phase J)
        if (trafficLightObj != null)
        {
            var tlComponent = trafficLightObj.GetComponent("TrafficLightController");
            if (tlComponent != null)
            {
                SetProperty(managerComponent, "trafficLight", tlComponent);
                // Also wire to agent
                if (agentComponent != null)
                    SetProperty(agentComponent, "trafficLight", tlComponent);
            }
        }

        // Wire pedestrians (Phase L)
        if (pedestrianObjs != null && pedestrianObjs.Length > 0)
        {
            var pedType = System.Type.GetType("ADPlatform.Environment.PedestrianController, Assembly-CSharp");
            if (pedType != null)
            {
                var pedArray = System.Array.CreateInstance(pedType, pedestrianObjs.Length);
                for (int i = 0; i < pedestrianObjs.Length; i++)
                {
                    pedArray.SetValue(pedestrianObjs[i].GetComponent(pedType), i);
                }
                SetProperty(managerComponent, "pedestrians", pedArray);
                if (agentComponent != null)
                    SetProperty(agentComponent, "pedestrians", pedArray);
            }
        }
    }

    private static void WireReferences(GameObject manager, GameObject agent,
        GameObject road, GameObject goal, GameObject[] npcs, GameObject camera)
    {
        var managerComponent = manager.GetComponent("DrivingSceneManager");
        if (managerComponent == null) return;

        // Wire agent
        var agentComponent = agent.GetComponent("E2EDrivingAgent");
        SetProperty(managerComponent, "egoAgent", agentComponent);

        // Wire waypoint manager
        var wpManager = road.GetComponent("WaypointManager");
        SetProperty(managerComponent, "waypointManager", wpManager);

        // Wire goal
        SetProperty(managerComponent, "goalTarget", goal.transform);

        // Wire NPCs
        var npcControllers = new Component[npcs.Length];
        for (int i = 0; i < npcs.Length; i++)
        {
            npcControllers[i] = npcs[i].GetComponent("NPCVehicleController");
        }

        // Get the array type and set
        var npcType = System.Type.GetType("ADPlatform.Agents.NPCVehicleController, Assembly-CSharp");
        if (npcType != null)
        {
            var array = System.Array.CreateInstance(npcType, npcs.Length);
            for (int i = 0; i < npcs.Length; i++)
            {
                array.SetValue(npcs[i].GetComponent(npcType), i);
            }
            SetProperty(managerComponent, "npcVehicles", array);
        }

        // Wire camera follow target
        var followCamera = camera.GetComponent("FollowCamera");
        if (followCamera != null)
        {
            SetProperty(followCamera, "target", agent.transform);
        }
    }

    #region Phase M - Multi-Agent Test Field (v2: 4x4 City Block Grid)

    private const int PHASE_M_NUM_AGENTS = 12;
    private const int PHASE_M_NUM_NPCS = 25;
    private const int PHASE_M_NUM_PEDESTRIANS = 8;
    private const float PHASE_M_GRID_SIZE = 520f;
    private const float PHASE_M_GOAL_DISTANCE = 230f;
    private const float PHASE_M_BLOCK_SIZE = 100f;

    private static void CreatePhaseM_TestField()
    {
        if (!Directory.Exists(SCENES_PATH))
        {
            Directory.CreateDirectory(SCENES_PATH);
            AssetDatabase.Refresh();
        }

        InitMaterialCache();

        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // 1. Main Camera (positioned for 500m grid overview)
        var camera = CreatePhaseMCamera();

        // 2. Directional Light
        CreateDirectionalLight();

        // 3. TestFieldManager (root orchestrator)
        var testFieldManagerObj = new GameObject("TestFieldManager");
        testFieldManagerObj.transform.position = Vector3.zero;

        // 4. TestField parent
        var testField = new GameObject("TestField");
        testField.transform.position = Vector3.zero;

        // 5. Ground (520m x 520m)
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(testField.transform);
        ground.transform.localPosition = Vector3.zero;
        ground.transform.localScale = new Vector3(PHASE_M_GRID_SIZE / 10f, 1f, PHASE_M_GRID_SIZE / 10f);
        var groundMat = new Material(Shader.Find("Standard"));
        groundMat.color = new Color(0.15f, 0.4f, 0.15f);
        ground.GetComponent<Renderer>().material = groundMat;

        // 6. GridRoadNetwork
        var gridNetworkObj = new GameObject("GridRoadNetwork");
        gridNetworkObj.transform.SetParent(testField.transform);
        gridNetworkObj.transform.localPosition = Vector3.zero;
        var gridNetworkType = System.Type.GetType("ADPlatform.TestField.GridRoadNetwork, Assembly-CSharp");
        UnityEngine.Component gridNetworkComp = null;
        if (gridNetworkType != null)
        {
            gridNetworkComp = gridNetworkObj.AddComponent(gridNetworkType);
            SetProperty(gridNetworkComp, "gridSize", 5);
            SetProperty(gridNetworkComp, "blockSize", PHASE_M_BLOCK_SIZE);
            SetProperty(gridNetworkComp, "numLanes", 3);
            SetProperty(gridNetworkComp, "laneWidth", 3.5f);
            SetProperty(gridNetworkComp, "intersectionWidth", 14f);
            SetProperty(gridNetworkComp, "turnRadius", 10f);
            SetProperty(gridNetworkComp, "defaultSpeedLimit", 16.67f);
        }

        // 7. Generate road geometry
        var roadGeometryParent = new GameObject("RoadGeometry");
        roadGeometryParent.transform.SetParent(testField.transform);
        roadGeometryParent.transform.localPosition = Vector3.zero;
        if (gridNetworkComp != null)
        {
            var genMethod = gridNetworkType.GetMethod("GenerateRoadGeometry");
            if (genMethod != null)
                genMethod.Invoke(gridNetworkComp, new object[] { roadGeometryParent.transform });
        }

        // 8. GridTrafficLightManager + 12 signalized intersections
        var gridTLManagerObj = new GameObject("GridTrafficLightManager");
        gridTLManagerObj.transform.SetParent(testField.transform);
        gridTLManagerObj.transform.localPosition = Vector3.zero;
        var gridTLManagerType = System.Type.GetType("ADPlatform.TestField.GridTrafficLightManager, Assembly-CSharp");
        UnityEngine.Component gridTLComp = null;
        if (gridTLManagerType != null)
        {
            gridTLComp = gridTLManagerObj.AddComponent(gridTLManagerType);
            CreateGridTrafficLights(gridTLManagerObj.transform, gridTLComp, gridNetworkComp, gridNetworkType);
        }

        // 9. Create 12 agents at grid start positions
        var agents = CreatePhaseMGridAgents(testField.transform, gridNetworkComp, gridNetworkType);

        // 10. Create 12 goal targets
        var goals = CreatePhaseMGoalTargets(agents, testField.transform);

        // 11. Create 25 NPCs distributed across grid roads
        var npcs = CreatePhaseMGridNPCs(testField.transform, gridNetworkComp, gridNetworkType);

        // 12. Create 8 pedestrians at signalized intersections
        var pedestrianObjs = CreatePhaseMGridPedestrians(testField.transform, gridNetworkComp, gridNetworkType);

        // 13. Wire TestFieldManager references
        WirePhaseMGridReferences(testFieldManagerObj, agents, goals, npcs,
            gridNetworkObj, gridTLManagerObj, pedestrianObjs, camera);

        // 14. Wire agent-level references
        WirePhaseMGridAgentReferences(agents, pedestrianObjs, goals);

        // Save scene
        string scenePath = $"{SCENES_PATH}/PhaseM_TestField.unity";
        EditorSceneManager.SaveScene(scene, scenePath);
        AddSceneToBuildSettings(scenePath);

        Debug.Log($"[PhaseSceneCreator] Created PhaseM_TestField v2 (4x4 Grid): {PHASE_M_NUM_AGENTS} agents, " +
                  $"{PHASE_M_NUM_NPCS} NPCs, {PHASE_M_NUM_PEDESTRIANS} pedestrians, 25 intersections (12 signalized)");
    }

    private static GameObject CreatePhaseMCamera()
    {
        var camera = new GameObject("Main Camera");
        camera.tag = "MainCamera";
        var cam = camera.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.Skybox;
        cam.fieldOfView = 60f;
        // Elevated position overlooking the grid center
        camera.transform.position = new Vector3(0f, 40f, -280f);
        camera.transform.rotation = Quaternion.Euler(30, 0, 0);
        camera.AddComponent<AudioListener>();

        var followCameraType = System.Type.GetType("ADPlatform.DebugTools.FollowCamera, Assembly-CSharp");
        if (followCameraType != null)
            camera.AddComponent(followCameraType);

        var freeFlyType = System.Type.GetType("ADPlatform.DebugTools.FreeFlyCamera, Assembly-CSharp");
        if (freeFlyType != null)
            camera.AddComponent(freeFlyType);

        return camera;
    }

    /// <summary>
    /// Create traffic lights at the 12 signalized intersections.
    /// Each intersection gets an NS and EW light pair.
    /// </summary>
    private static void CreateGridTrafficLights(Transform parent, UnityEngine.Component gridTLComp,
        UnityEngine.Component gridNetworkComp, System.Type gridNetworkType)
    {
        var signalizedType = System.Type.GetType("ADPlatform.TestField.GridTrafficLightManager, Assembly-CSharp");
        if (signalizedType == null) return;

        // Get signalized intersection positions
        var signalizedField = signalizedType.GetField("SignalizedIntersections",
            System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        if (signalizedField == null) return;

        var signalized = (Vector2Int[])signalizedField.GetValue(null);

        // Create IntersectionLights array
        var ilType = signalizedType.GetNestedType("IntersectionLights");
        if (ilType == null) return;

        var intersectionsArray = System.Array.CreateInstance(ilType, signalized.Length);

        var getIntersectionPos = gridNetworkType?.GetMethod("GetIntersectionPosition");

        for (int i = 0; i < signalized.Length; i++)
        {
            int col = signalized[i].x;
            int row = signalized[i].y;

            Vector3 pos = Vector3.zero;
            if (getIntersectionPos != null && gridNetworkComp != null)
                pos = (Vector3)getIntersectionPos.Invoke(gridNetworkComp, new object[] { col, row });

            // NS traffic light (faces Z+ direction, controls Z-axis traffic)
            var nsLight = CreateGridTrafficLight(parent, $"TL_{col}{row}_NS",
                new Vector3(pos.x + 8f, 0f, pos.z), Quaternion.identity);

            // EW traffic light (faces X+ direction, rotated 90 degrees, controls X-axis traffic)
            var ewLight = CreateGridTrafficLight(parent, $"TL_{col}{row}_EW",
                new Vector3(pos.x, 0f, pos.z + 8f), Quaternion.Euler(0f, 90f, 0f));

            // Build struct
            var ilInstance = System.Activator.CreateInstance(ilType);
            ilType.GetField("col").SetValue(ilInstance, col);
            ilType.GetField("row").SetValue(ilInstance, row);
            var tlType = System.Type.GetType("ADPlatform.Environment.TrafficLightController, Assembly-CSharp");
            if (tlType != null)
            {
                ilType.GetField("nsLight").SetValue(ilInstance, nsLight.GetComponent(tlType));
                ilType.GetField("ewLight").SetValue(ilInstance, ewLight.GetComponent(tlType));
            }
            intersectionsArray.SetValue(ilInstance, i);
        }

        SetProperty(gridTLComp, "intersections", intersectionsArray);
    }

    private static GameObject CreateGridTrafficLight(Transform parent, string name,
        Vector3 position, Quaternion rotation)
    {
        var trafficLight = new GameObject(name);
        trafficLight.transform.SetParent(parent);
        trafficLight.transform.localPosition = position;
        trafficLight.transform.localRotation = rotation;

        // Pole
        var pole = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
        pole.name = "Pole";
        pole.transform.SetParent(trafficLight.transform);
        pole.transform.localPosition = new Vector3(0f, 3f, 0f);
        pole.transform.localScale = new Vector3(0.15f, 3f, 0.15f);
        Object.DestroyImmediate(pole.GetComponent<Collider>());
        var poleMat = new Material(Shader.Find("Standard"));
        poleMat.color = new Color(0.3f, 0.3f, 0.3f);
        pole.GetComponent<Renderer>().material = poleMat;

        // Housing
        var housing = GameObject.CreatePrimitive(PrimitiveType.Cube);
        housing.name = "SignalHousing";
        housing.transform.SetParent(trafficLight.transform);
        housing.transform.localPosition = new Vector3(0f, 6.5f, 0f);
        housing.transform.localScale = new Vector3(0.6f, 1.8f, 0.4f);
        Object.DestroyImmediate(housing.GetComponent<Collider>());
        var housingMat = new Material(Shader.Find("Standard"));
        housingMat.color = new Color(0.2f, 0.2f, 0.2f);
        housing.GetComponent<Renderer>().material = housingMat;

        // Lights
        var redLight = CreateLightSphere(trafficLight.transform, "RedLight", new Vector3(0f, 7.1f, -0.15f));
        var yellowLight = CreateLightSphere(trafficLight.transform, "YellowLight", new Vector3(0f, 6.5f, -0.15f));
        var greenLight = CreateLightSphere(trafficLight.transform, "GreenLight", new Vector3(0f, 5.9f, -0.15f));

        // TrafficLightController
        var tlType = System.Type.GetType("ADPlatform.Environment.TrafficLightController, Assembly-CSharp");
        if (tlType != null)
        {
            var tlComp = trafficLight.AddComponent(tlType);
            SetProperty(tlComp, "redLight", redLight.GetComponent<Renderer>());
            SetProperty(tlComp, "yellowLight", yellowLight.GetComponent<Renderer>());
            SetProperty(tlComp, "greenLight", greenLight.GetComponent<Renderer>());
            SetProperty(tlComp, "stopLineZ", 0f);  // Stop line at local origin
        }

        return trafficLight;
    }

    private static GameObject CreateLightSphere(Transform parent, string name, Vector3 localPos)
    {
        var sphere = GameObject.CreatePrimitive(PrimitiveType.Sphere);
        sphere.name = name;
        sphere.transform.SetParent(parent);
        sphere.transform.localPosition = localPos;
        sphere.transform.localScale = new Vector3(0.35f, 0.35f, 0.35f);
        Object.DestroyImmediate(sphere.GetComponent<Collider>());
        var mat = new Material(Shader.Find("Standard"));
        mat.color = new Color(0.15f, 0.15f, 0.15f);
        mat.EnableKeyword("_EMISSION");
        sphere.GetComponent<Renderer>().material = mat;
        return sphere;
    }

    private static GameObject[] CreatePhaseMGridAgents(Transform parent,
        UnityEngine.Component gridNetworkComp, System.Type gridNetworkType)
    {
        var agentsParent = new GameObject("Agents");
        agentsParent.transform.SetParent(parent);
        agentsParent.transform.localPosition = Vector3.zero;

        var agents = new GameObject[PHASE_M_NUM_AGENTS];

        // Load ONNX model
        string onnxPath = "Assets/ML-Agents/Models/E2EDrivingAgent_PhaseL.onnx";
        var onnxAsset = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(onnxPath);
        if (onnxAsset == null)
        {
            onnxPath = "Assets/ML-Agents/Models/E2EDrivingAgent-5000029.onnx";
            onnxAsset = AssetDatabase.LoadAssetAtPath<UnityEngine.Object>(onnxPath);
        }

        // Get route definitions for start positions
        var routeType = System.Type.GetType("ADPlatform.TestField.GridRoutes, Assembly-CSharp");
        Vector2Int[][] routeStarts = null;
        if (routeType != null)
        {
            var getRoutes = routeType.GetMethod("GetAgentRoutes",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            if (getRoutes != null)
            {
                var routes = getRoutes.Invoke(null, null) as System.Array;
                if (routes != null)
                {
                    routeStarts = new Vector2Int[routes.Length][];
                    for (int i = 0; i < routes.Length; i++)
                    {
                        var route = routes.GetValue(i);
                        var wpField = route.GetType().GetField("waypoints");
                        if (wpField != null)
                            routeStarts[i] = (Vector2Int[])wpField.GetValue(route);
                    }
                }
            }
        }

        var getIntersectionPos = gridNetworkType?.GetMethod("GetIntersectionPosition");

        for (int i = 0; i < PHASE_M_NUM_AGENTS; i++)
        {
            var agent = GameObject.CreatePrimitive(PrimitiveType.Cube);
            agent.name = $"Agent_{i}";
            agent.transform.SetParent(agentsParent.transform);

            // Start position: first intersection of agent's route
            Vector3 startPos = new Vector3(i * 30f - 150f, 0.75f, -200f);
            if (routeStarts != null && i < routeStarts.Length && routeStarts[i] != null
                && routeStarts[i].Length > 0 && getIntersectionPos != null && gridNetworkComp != null)
            {
                var firstIntersection = routeStarts[i][0];
                startPos = (Vector3)getIntersectionPos.Invoke(gridNetworkComp,
                    new object[] { firstIntersection.x, firstIntersection.y });
                startPos.y = 0.75f;
                // Offset slightly along the route direction
                if (routeStarts[i].Length > 1)
                {
                    var secondIntersection = routeStarts[i][1];
                    var secondPos = (Vector3)getIntersectionPos.Invoke(gridNetworkComp,
                        new object[] { secondIntersection.x, secondIntersection.y });
                    Vector3 dir = (secondPos - startPos).normalized;
                    startPos += dir * 10f;  // 10m offset from intersection center
                    startPos.y = 0.75f;
                    agent.transform.rotation = Quaternion.LookRotation(new Vector3(dir.x, 0, dir.z));
                }
            }

            agent.transform.localPosition = startPos;
            agent.transform.localScale = new Vector3(2f, 1.5f, 4.5f);

            // Blue-ish color per agent
            var renderer = agent.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            float hue = 0.58f + (i * 0.03f);
            material.color = Color.HSVToRGB(hue % 1f, 0.7f, 0.85f);
            renderer.material = material;

            agent.tag = "Vehicle";

            var rb = agent.AddComponent<Rigidbody>();
            rb.mass = 1500f;
            rb.linearDamping = 1f;
            rb.angularDamping = 2f;
            rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

            Object.DestroyImmediate(agent.GetComponent<Collider>());
            var collider = agent.AddComponent<BoxCollider>();
            collider.size = Vector3.one;

            // E2EDrivingAgent
            var agentType = System.Type.GetType("ADPlatform.Agents.E2EDrivingAgent, Assembly-CSharp");
            if (agentType != null)
            {
                var agentComponent = agent.AddComponent(agentType);
                SetProperty(agentComponent, "enableLaneObservation", true);
                SetProperty(agentComponent, "enableIntersectionObservation", true);
                SetProperty(agentComponent, "enableTrafficSignalObservation", true);
                SetProperty(agentComponent, "enablePedestrianObservation", true);
            }

            ConfigurePhaseMBehaviorParameters(agent, onnxAsset);

            var drType = System.Type.GetType("Unity.MLAgents.DecisionRequester, Unity.ML-Agents");
            if (drType != null)
            {
                var dr = agent.AddComponent(drType);
                SetProperty(dr, "DecisionPeriod", 5);
                SetProperty(dr, "TakeActionsBetweenDecisions", true);
            }

            agents[i] = agent;
        }

        return agents;
    }

    private static void ConfigurePhaseMBehaviorParameters(GameObject agent, UnityEngine.Object onnxModel)
    {
        var bp = agent.GetComponent("BehaviorParameters") as UnityEngine.Component;
        if (bp == null) return;

        var so = new SerializedObject(bp);

        var nameProp = so.FindProperty("m_BehaviorName");
        if (nameProp != null) nameProp.stringValue = "E2EDrivingAgent";

        var obsSizeProp = so.FindProperty("m_BrainParameters.VectorObservationSize");
        if (obsSizeProp != null) obsSizeProp.intValue = 280;

        var stackProp = so.FindProperty("m_BrainParameters.NumStackedVectorObservations");
        if (stackProp != null) stackProp.intValue = 1;

        var continuousProp = so.FindProperty("m_BrainParameters.m_ActionSpec.m_NumContinuousActions");
        if (continuousProp != null) continuousProp.intValue = 2;

        var branchProp = so.FindProperty("m_BrainParameters.m_ActionSpec.BranchSizes");
        if (branchProp != null) branchProp.ClearArray();

        // BehaviorType = 2 (InferenceOnly) -- P-025: NOT 1 which is HeuristicOnly
        var behaviorTypeProp = so.FindProperty("m_BehaviorType");
        if (behaviorTypeProp != null) behaviorTypeProp.enumValueIndex = 2;

        if (onnxModel != null)
        {
            var modelProp = so.FindProperty("m_Model");
            if (modelProp != null) modelProp.objectReferenceValue = onnxModel;
        }

        so.ApplyModifiedPropertiesWithoutUndo();
    }

    private static GameObject[] CreatePhaseMGoalTargets(GameObject[] agents, Transform parent)
    {
        var goalsParent = new GameObject("GoalTargets");
        goalsParent.transform.SetParent(parent);
        goalsParent.transform.localPosition = Vector3.zero;

        var goals = new GameObject[agents.Length];

        for (int i = 0; i < agents.Length; i++)
        {
            var goal = new GameObject($"GoalTarget_{i}");
            goal.transform.SetParent(goalsParent.transform);
            // Initial position near agent (will be updated dynamically by TestFieldManager)
            goal.transform.position = agents[i].transform.position + agents[i].transform.forward * 50f;
            goal.transform.position = new Vector3(goal.transform.position.x, 1f, goal.transform.position.z);

            var visual = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            visual.name = "GoalVisual";
            visual.transform.SetParent(goal.transform);
            visual.transform.localPosition = Vector3.zero;
            visual.transform.localScale = new Vector3(3f, 0.05f, 3f);

            var renderer = visual.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            material.color = new Color(0.2f, 0.8f, 0.2f, 0.3f);
            renderer.material = material;
            Object.DestroyImmediate(visual.GetComponent<Collider>());

            goals[i] = goal;
        }

        return goals;
    }

    /// <summary>
    /// Create 25 NPCs distributed across the grid road network.
    /// </summary>
    private static GameObject[] CreatePhaseMGridNPCs(Transform parent,
        UnityEngine.Component gridNetworkComp, System.Type gridNetworkType)
    {
        var npcParent = new GameObject("NPCVehicles");
        npcParent.transform.SetParent(parent);
        npcParent.transform.localPosition = Vector3.zero;

        var npcs = new GameObject[PHASE_M_NUM_NPCS];
        var getIntersectionPos = gridNetworkType?.GetMethod("GetIntersectionPosition");

        for (int i = 0; i < PHASE_M_NUM_NPCS; i++)
        {
            var npc = GameObject.CreatePrimitive(PrimitiveType.Cube);
            npc.name = $"NPC_{i}";
            npc.transform.SetParent(npcParent.transform);

            // Distribute NPCs across grid: alternate horizontal/vertical roads
            Vector3 npcPos;
            Quaternion npcRot;
            GetNPCGridPosition(i, out npcPos, out npcRot, getIntersectionPos, gridNetworkComp);

            npc.transform.localPosition = npcPos;
            npc.transform.localRotation = npcRot;
            npc.transform.localScale = new Vector3(2f, 1.5f, 4.5f);

            var renderer = npc.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            material.color = GetNPCColor(i);
            renderer.material = material;

            npc.tag = "Vehicle";

            var rb = npc.AddComponent<Rigidbody>();
            rb.mass = 1500f;
            rb.isKinematic = true;

            var npcType = System.Type.GetType("ADPlatform.Agents.NPCVehicleController, Assembly-CSharp");
            if (npcType != null)
                npc.AddComponent(npcType);

            npc.SetActive(true);
            npcs[i] = npc;
        }

        return npcs;
    }

    private static void GetNPCGridPosition(int index, out Vector3 pos, out Quaternion rot,
        System.Reflection.MethodInfo getIntersectionPos, UnityEngine.Component gridNetworkComp)
    {
        float halfGrid = 2f * PHASE_M_BLOCK_SIZE;  // 200m

        if (index < 13)
        {
            // Horizontal road NPCs (13 on H roads)
            int roadRow = index % 5;
            float z = roadRow * PHASE_M_BLOCK_SIZE - halfGrid;
            float x = -halfGrid + 30f + index * 28f;
            x = Mathf.Repeat(x + halfGrid, halfGrid * 2f) - halfGrid;
            pos = new Vector3(x, 0.75f, z + 1.75f);
            rot = Quaternion.Euler(0, 90, 0);  // Facing +X
        }
        else
        {
            // Vertical road NPCs (12 on V roads)
            int idx = index - 13;
            int roadCol = idx % 5;
            float x = roadCol * PHASE_M_BLOCK_SIZE - halfGrid;
            float z = -halfGrid + 30f + idx * 28f;
            z = Mathf.Repeat(z + halfGrid, halfGrid * 2f) - halfGrid;
            pos = new Vector3(x + 1.75f, 0.75f, z);
            rot = Quaternion.identity;  // Facing +Z
        }
    }

    /// <summary>
    /// Create 8 pedestrians at signalized intersection crosswalks.
    /// </summary>
    private static GameObject[] CreatePhaseMGridPedestrians(Transform parent,
        UnityEngine.Component gridNetworkComp, System.Type gridNetworkType)
    {
        var pedestrians = new GameObject[PHASE_M_NUM_PEDESTRIANS];
        var pedParent = new GameObject("Pedestrians");
        pedParent.transform.SetParent(parent);
        pedParent.transform.localPosition = Vector3.zero;

        var getIntersectionPos = gridNetworkType?.GetMethod("GetIntersectionPosition");

        // Place pedestrians near the first 8 signalized intersections
        var signalizedType = System.Type.GetType("ADPlatform.TestField.GridTrafficLightManager, Assembly-CSharp");
        Vector2Int[] signalized = null;
        if (signalizedType != null)
        {
            var signalizedField = signalizedType.GetField("SignalizedIntersections",
                System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
            if (signalizedField != null)
                signalized = (Vector2Int[])signalizedField.GetValue(null);
        }

        for (int i = 0; i < PHASE_M_NUM_PEDESTRIANS; i++)
        {
            var ped = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            ped.name = $"Pedestrian_{i}";
            ped.transform.SetParent(pedParent.transform);

            // Position near signalized intersection crosswalk
            Vector3 pedPos = new Vector3(-10f - i * 2f, 0.9f, 0f);
            if (signalized != null && i < signalized.Length && getIntersectionPos != null && gridNetworkComp != null)
            {
                Vector3 intPos = (Vector3)getIntersectionPos.Invoke(gridNetworkComp,
                    new object[] { signalized[i].x, signalized[i].y });
                // Offset to crosswalk position (just outside intersection)
                pedPos = new Vector3(intPos.x - 10f, 0.9f, intPos.z + 8f);
            }

            ped.transform.localPosition = pedPos;
            ped.transform.localScale = new Vector3(0.5f, 0.9f, 0.5f);

            var renderer = ped.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            material.color = GetPedestrianColor(i);
            renderer.material = material;

            var rb = ped.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;

            var pedType = System.Type.GetType("ADPlatform.Environment.PedestrianController, Assembly-CSharp");
            if (pedType != null)
                ped.AddComponent(pedType);

            ped.SetActive(true);
            pedestrians[i] = ped;
        }

        // Create crosswalks at signalized intersections
        if (signalized != null && getIntersectionPos != null && gridNetworkComp != null)
        {
            var crosswalksParent = new GameObject("Crosswalks");
            crosswalksParent.transform.SetParent(parent);
            crosswalksParent.transform.localPosition = Vector3.zero;

            for (int i = 0; i < signalized.Length; i++)
            {
                Vector3 intPos = (Vector3)getIntersectionPos.Invoke(gridNetworkComp,
                    new object[] { signalized[i].x, signalized[i].y });
                CreateGridCrosswalk(crosswalksParent.transform, intPos, i);
            }
        }

        return pedestrians;
    }

    private static void CreateGridCrosswalk(Transform parent, Vector3 intersectionPos, int index)
    {
        float stripeWidth = 0.4f;
        float stripeSpacing = 0.6f;
        float crosswalkWidth = 4f;
        float stripeLength = 10.5f;  // 3 lanes
        int numStripes = Mathf.FloorToInt(crosswalkWidth / stripeSpacing);

        // North side crosswalk
        for (int i = 0; i < numStripes; i++)
        {
            var stripe = GameObject.CreatePrimitive(PrimitiveType.Cube);
            stripe.name = $"Crosswalk_{index}_N_{i}";
            stripe.transform.SetParent(parent);
            float zOff = 8f + i * stripeSpacing;
            stripe.transform.localPosition = new Vector3(intersectionPos.x, 0.025f, intersectionPos.z + zOff);
            stripe.transform.localScale = new Vector3(stripeLength, 0.02f, stripeWidth);
            stripe.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
            Object.DestroyImmediate(stripe.GetComponent<Collider>());
        }
    }

    private static void WirePhaseMGridReferences(GameObject managerObj, GameObject[] agents,
        GameObject[] goals, GameObject[] npcs, GameObject gridNetworkObj,
        GameObject gridTLManagerObj, GameObject[] pedestrianObjs, GameObject camera)
    {
        var tfmType = System.Type.GetType("ADPlatform.TestField.TestFieldManager, Assembly-CSharp");
        if (tfmType == null)
        {
            Debug.LogError("[PhaseSceneCreator] TestFieldManager type not found");
            return;
        }

        var tfm = managerObj.AddComponent(tfmType);

        // Wire agents array
        var agentType = System.Type.GetType("ADPlatform.Agents.E2EDrivingAgent, Assembly-CSharp");
        if (agentType != null)
        {
            var agentArray = System.Array.CreateInstance(agentType, agents.Length);
            for (int i = 0; i < agents.Length; i++)
                agentArray.SetValue(agents[i].GetComponent(agentType), i);
            SetProperty(tfm, "agents", agentArray);
        }

        // Wire goal targets
        var goalTransforms = new Transform[goals.Length];
        for (int i = 0; i < goals.Length; i++)
            goalTransforms[i] = goals[i].transform;
        SetProperty(tfm, "goalTargets", goalTransforms);

        // Wire NPCs
        var npcType = System.Type.GetType("ADPlatform.Agents.NPCVehicleController, Assembly-CSharp");
        if (npcType != null)
        {
            var npcArray = System.Array.CreateInstance(npcType, npcs.Length);
            for (int i = 0; i < npcs.Length; i++)
                npcArray.SetValue(npcs[i].GetComponent(npcType), i);
            SetProperty(tfm, "npcVehicles", npcArray);
        }

        // Wire pedestrians
        var pedType = System.Type.GetType("ADPlatform.Environment.PedestrianController, Assembly-CSharp");
        if (pedType != null)
        {
            var pedArray = System.Array.CreateInstance(pedType, pedestrianObjs.Length);
            for (int i = 0; i < pedestrianObjs.Length; i++)
                pedArray.SetValue(pedestrianObjs[i].GetComponent(pedType), i);
            SetProperty(tfm, "pedestrians", pedArray);
        }

        // Wire GridRoadNetwork
        var gridNetworkType = System.Type.GetType("ADPlatform.TestField.GridRoadNetwork, Assembly-CSharp");
        if (gridNetworkType != null)
        {
            var gridComp = gridNetworkObj.GetComponent(gridNetworkType);
            if (gridComp != null)
                SetProperty(tfm, "gridNetwork", gridComp);
        }

        // Wire GridTrafficLightManager
        var gridTLType = System.Type.GetType("ADPlatform.TestField.GridTrafficLightManager, Assembly-CSharp");
        if (gridTLType != null)
        {
            var gridTLComp = gridTLManagerObj.GetComponent(gridTLType);
            if (gridTLComp != null)
                SetProperty(tfm, "gridTrafficManager", gridTLComp);
        }

        // Wire FollowCamera
        var followCamera = camera.GetComponent("FollowCamera");
        if (followCamera != null)
        {
            SetProperty(tfm, "followCamera", followCamera);

            var agentTransforms = new Transform[agents.Length];
            for (int i = 0; i < agents.Length; i++)
                agentTransforms[i] = agents[i].transform;
            SetProperty(followCamera, "targets", agentTransforms);

            if (agents.Length > 0)
                SetProperty(followCamera, "target", agents[0].transform);
        }

        // Config
        SetProperty(tfm, "goalDistance", PHASE_M_GOAL_DISTANCE);
    }

    private static void WirePhaseMGridAgentReferences(GameObject[] agents,
        GameObject[] pedestrianObjs, GameObject[] goals)
    {
        var agentType = System.Type.GetType("ADPlatform.Agents.E2EDrivingAgent, Assembly-CSharp");

        // Build pedestrian array
        var pedType = System.Type.GetType("ADPlatform.Environment.PedestrianController, Assembly-CSharp");
        System.Array pedArray = null;
        if (pedType != null && pedestrianObjs != null)
        {
            pedArray = System.Array.CreateInstance(pedType, pedestrianObjs.Length);
            for (int i = 0; i < pedestrianObjs.Length; i++)
                pedArray.SetValue(pedestrianObjs[i].GetComponent(pedType), i);
        }

        for (int i = 0; i < agents.Length; i++)
        {
            if (agentType == null) continue;
            var agentComponent = agents[i].GetComponent(agentType);
            if (agentComponent == null) continue;

            // waypointManager and trafficLight assigned dynamically by TestFieldManager at runtime

            // Wire pedestrians
            if (pedArray != null)
                SetProperty(agentComponent, "pedestrians", pedArray);

            // Wire goal target
            if (i < goals.Length)
                SetProperty(agentComponent, "goalTarget", goals[i].transform);
        }
    }

    #endregion

    #region Pedestrian & Crosswalk Helpers

    private const float CROSSWALK_Z = 75f;  // Before intersection (93m)
    private const float CROSSWALK_WIDTH = 4f;

    /// <summary>
    /// Create pedestrian capsule primitives with PedestrianController.
    /// Start inactive; DrivingSceneManager activates via curriculum.
    /// </summary>
    private static GameObject[] CreatePedestriansForArea(int count, Transform parent)
    {
        var pedestrians = new GameObject[count];
        var pedParent = new GameObject("Pedestrians");
        pedParent.transform.SetParent(parent);
        pedParent.transform.localPosition = Vector3.zero;

        for (int i = 0; i < count; i++)
        {
            var ped = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            ped.name = $"Pedestrian_{i}";
            ped.transform.SetParent(pedParent.transform);
            // Start position off-road (will be set by SpawnAtCrosswalk)
            ped.transform.localPosition = new Vector3(-10f - i * 2f, 0.9f, CROSSWALK_Z);
            ped.transform.localScale = new Vector3(0.5f, 0.9f, 0.5f);

            var renderer = ped.GetComponent<Renderer>();
            var material = new Material(Shader.Find("Standard"));
            material.color = GetPedestrianColor(i);
            renderer.material = material;

            // Kinematic rigidbody
            var rb = ped.AddComponent<Rigidbody>();
            rb.isKinematic = true;
            rb.useGravity = false;

            // Add PedestrianController
            var pedType = System.Type.GetType("ADPlatform.Environment.PedestrianController, Assembly-CSharp");
            if (pedType != null)
            {
                ped.AddComponent(pedType);
            }

            ped.SetActive(false);
            pedestrians[i] = ped;
        }

        return pedestrians;
    }

    private static Color GetPedestrianColor(int index)
    {
        Color[] colors = {
            new Color(0.9f, 0.6f, 0.2f), // Orange
            new Color(0.3f, 0.7f, 0.9f), // Light blue
            new Color(0.8f, 0.3f, 0.6f), // Pink
        };
        return colors[index % colors.Length];
    }

    /// <summary>
    /// Create crosswalk visual (white stripes) at Z=75m.
    /// </summary>
    private static void CreateCrosswalkForArea(Transform parent)
    {
        var crosswalk = new GameObject("Crosswalk");
        crosswalk.transform.SetParent(parent);
        crosswalk.transform.localPosition = Vector3.zero;

        float stripeWidth = 0.4f;
        float stripeLength = ROAD_WIDTH;
        float stripeSpacing = 0.6f;
        int numStripes = Mathf.FloorToInt(CROSSWALK_WIDTH / stripeSpacing);

        for (int i = 0; i < numStripes; i++)
        {
            var stripe = GameObject.CreatePrimitive(PrimitiveType.Cube);
            stripe.name = $"CrosswalkStripe_{i}";
            stripe.transform.SetParent(crosswalk.transform);

            float zOffset = CROSSWALK_Z - CROSSWALK_WIDTH / 2f + i * stripeSpacing;
            stripe.transform.localPosition = new Vector3(0f, 0.025f, zOffset);
            stripe.transform.localScale = new Vector3(stripeLength, 0.02f, stripeWidth);

            stripe.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
            Object.DestroyImmediate(stripe.GetComponent<Collider>());
        }
    }

    #endregion

    #region Material Factory

    private static void InitMaterialCache()
    {
        s_asphaltMaterial = null;
        s_lightAsphaltMaterial = null;
        s_curbMaterial = null;
        s_whiteLineMaterial = null;
        s_yellowLineMaterial = null;
    }

    private static Material GetAsphaltMaterial()
    {
        if (s_asphaltMaterial == null)
        {
            s_asphaltMaterial = new Material(Shader.Find("Standard"));
            s_asphaltMaterial.color = new Color(0.25f, 0.25f, 0.25f);
        }
        return s_asphaltMaterial;
    }

    private static Material GetLightAsphaltMaterial()
    {
        if (s_lightAsphaltMaterial == null)
        {
            s_lightAsphaltMaterial = new Material(Shader.Find("Standard"));
            s_lightAsphaltMaterial.color = new Color(0.28f, 0.28f, 0.28f);
        }
        return s_lightAsphaltMaterial;
    }

    private static Material GetCurbMaterial()
    {
        if (s_curbMaterial == null)
        {
            s_curbMaterial = new Material(Shader.Find("Standard"));
            s_curbMaterial.color = new Color(0.5f, 0.5f, 0.5f);
        }
        return s_curbMaterial;
    }

    private static Material GetWhiteLineMaterial()
    {
        if (s_whiteLineMaterial == null)
        {
            s_whiteLineMaterial = new Material(Shader.Find("Standard"));
            s_whiteLineMaterial.color = Color.white;
            s_whiteLineMaterial.EnableKeyword("_EMISSION");
            s_whiteLineMaterial.SetColor("_EmissionColor", Color.white * 0.3f);
        }
        return s_whiteLineMaterial;
    }

    private static Material GetYellowLineMaterial()
    {
        if (s_yellowLineMaterial == null)
        {
            s_yellowLineMaterial = new Material(Shader.Find("Standard"));
            s_yellowLineMaterial.color = Color.yellow;
            s_yellowLineMaterial.EnableKeyword("_EMISSION");
            s_yellowLineMaterial.SetColor("_EmissionColor", Color.yellow * 0.3f);
        }
        return s_yellowLineMaterial;
    }

    #endregion

    #region Geometry Helpers

    private static GameObject CreateCurb(Transform parent, string name, Vector3 pos, Vector3 scale)
    {
        var curb = GameObject.CreatePrimitive(PrimitiveType.Cube);
        curb.name = name;
        curb.transform.SetParent(parent);
        curb.transform.localPosition = pos;
        curb.transform.localScale = scale;
        curb.GetComponent<Renderer>().sharedMaterial = GetCurbMaterial();
        Object.DestroyImmediate(curb.GetComponent<Collider>());
        return curb;
    }

    private static GameObject CreateDash(Transform parent, Vector3 pos, Vector3 scale, Material mat)
    {
        var dash = GameObject.CreatePrimitive(PrimitiveType.Cube);
        dash.name = "Dash";
        dash.transform.SetParent(parent);
        dash.transform.localPosition = pos;
        dash.transform.localScale = scale;
        dash.GetComponent<Renderer>().sharedMaterial = mat;
        Object.DestroyImmediate(dash.GetComponent<Collider>());
        return dash;
    }

    private static GameObject CreateStopLine(Transform parent, string name, Vector3 pos, Vector3 scale)
    {
        var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
        line.name = name;
        line.transform.SetParent(parent);
        line.transform.localPosition = pos;
        line.transform.localScale = scale;
        line.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
        Object.DestroyImmediate(line.GetComponent<Collider>());
        return line;
    }

    private static GameObject CreateSolidEdgeLine(Transform parent, string name, float xPos, float zStart, float zEnd)
    {
        float zCenter = (zStart + zEnd) / 2f;
        float zLength = Mathf.Abs(zEnd - zStart);

        var line = GameObject.CreatePrimitive(PrimitiveType.Cube);
        line.name = name;
        line.transform.SetParent(parent);
        line.transform.localPosition = new Vector3(xPos, 0.02f, zCenter);
        line.transform.localScale = new Vector3(0.12f, 0.01f, zLength);
        line.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
        Object.DestroyImmediate(line.GetComponent<Collider>());
        return line;
    }

    #endregion

    #region Intersection Road Visuals

    private static void CreateIntersectionRoadVisuals(Transform roadParent, PhaseConfig config)
    {
        float halfRoad = config.roadLength / 2f;
        float halfWidth = ROAD_WIDTH / 2f;

        // Main road curbs (split at intersection zone)
        CreateMainRoadCurbs(roadParent, halfRoad, halfWidth);

        // Main road markings (center line + edge lines)
        CreateMainRoadMarkings(roadParent, halfRoad, halfWidth, config.roadLength);

        // Intersection area (inactive by default)
        CreateIntersectionArea(roadParent, halfWidth);

        // Left arm (Cross intersection)
        CreateLeftArm(roadParent, halfWidth);

        // Right arm (T-junction + Cross)
        CreateRightArm(roadParent, halfWidth);

        // Angled arms (Y-junction)
        CreateLeftAngledArm(roadParent, halfWidth);
        CreateRightAngledArm(roadParent, halfWidth);
    }

    private static void CreateMainRoadCurbs(Transform roadParent, float halfRoad, float halfWidth)
    {
        var curbParent = new GameObject("MainRoadCurbs");
        curbParent.transform.SetParent(roadParent);
        curbParent.transform.localPosition = Vector3.zero;

        float curbOffset = halfWidth + CURB_WIDTH / 2f;

        // Left curbs (negative X side)
        float preLength = INTERSECTION_ZONE_START + halfRoad;
        float preCenter = (-halfRoad + INTERSECTION_ZONE_START) / 2f;
        CreateCurb(curbParent.transform, "LeftCurb_Pre",
            new Vector3(-curbOffset, CURB_HEIGHT / 2f, preCenter),
            new Vector3(CURB_WIDTH, CURB_HEIGHT, preLength));

        float postLength = halfRoad - INTERSECTION_ZONE_END;
        float postCenter = (INTERSECTION_ZONE_END + halfRoad) / 2f;
        CreateCurb(curbParent.transform, "LeftCurb_Post",
            new Vector3(-curbOffset, CURB_HEIGHT / 2f, postCenter),
            new Vector3(CURB_WIDTH, CURB_HEIGHT, postLength));

        // Right curbs (positive X side)
        CreateCurb(curbParent.transform, "RightCurb_Pre",
            new Vector3(curbOffset, CURB_HEIGHT / 2f, preCenter),
            new Vector3(CURB_WIDTH, CURB_HEIGHT, preLength));

        CreateCurb(curbParent.transform, "RightCurb_Post",
            new Vector3(curbOffset, CURB_HEIGHT / 2f, postCenter),
            new Vector3(CURB_WIDTH, CURB_HEIGHT, postLength));
    }

    private static void CreateMainRoadMarkings(Transform roadParent, float halfRoad, float halfWidth, float roadLength)
    {
        var markingsParent = new GameObject("MainRoadMarkings");
        markingsParent.transform.SetParent(roadParent);
        markingsParent.transform.localPosition = Vector3.zero;

        // Yellow dashed center line (skip intersection zone)
        var centerLine = new GameObject("CenterLine");
        centerLine.transform.SetParent(markingsParent.transform);
        centerLine.transform.localPosition = Vector3.zero;

        float dashLength = 3f;
        float dashSpacing = 10f;

        for (float z = -halfRoad; z < halfRoad; z += dashSpacing)
        {
            // Skip intersection zone
            if (z + dashLength > INTERSECTION_ZONE_START && z < INTERSECTION_ZONE_END)
                continue;

            CreateDash(centerLine.transform,
                new Vector3(0f, 0.02f, z),
                new Vector3(0.15f, 0.01f, dashLength),
                GetYellowLineMaterial());
        }

        // White solid edge lines (split at intersection zone)
        float edgeX = halfWidth - 0.1f;

        CreateSolidEdgeLine(markingsParent.transform, "LeftEdge_Pre",
            -edgeX, -halfRoad, INTERSECTION_ZONE_START);
        CreateSolidEdgeLine(markingsParent.transform, "LeftEdge_Post",
            -edgeX, INTERSECTION_ZONE_END, halfRoad);
        CreateSolidEdgeLine(markingsParent.transform, "RightEdge_Pre",
            edgeX, -halfRoad, INTERSECTION_ZONE_START);
        CreateSolidEdgeLine(markingsParent.transform, "RightEdge_Post",
            edgeX, INTERSECTION_ZONE_END, halfRoad);
    }

    private static void CreateIntersectionArea(Transform roadParent, float halfWidth)
    {
        var intersectionArea = new GameObject("IntersectionArea");
        intersectionArea.transform.SetParent(roadParent);
        intersectionArea.transform.localPosition = Vector3.zero;

        // Intersection surface (14x14m, slightly above main road)
        var surface = GameObject.CreatePrimitive(PrimitiveType.Plane);
        surface.name = "IntersectionSurface";
        surface.transform.SetParent(intersectionArea.transform);
        surface.transform.localPosition = new Vector3(0f, 0.015f, INTERSECTION_Z);
        surface.transform.localScale = new Vector3(INTERSECTION_WIDTH / 10f, 1f, INTERSECTION_WIDTH / 10f);
        surface.GetComponent<Renderer>().sharedMaterial = GetLightAsphaltMaterial();
        Object.DestroyImmediate(surface.GetComponent<Collider>());

        // Stop line at approach (south side of intersection, Z=93)
        CreateStopLine(intersectionArea.transform, "StopLine_Approach",
            new Vector3(halfWidth / 2f, 0.02f, INTERSECTION_ZONE_START),
            new Vector3(halfWidth, 0.02f, 0.5f));

        // Stop line for left arm (east side, X = -halfWidth at intersection)
        CreateStopLine(intersectionArea.transform, "StopLine_LeftArm",
            new Vector3(-INTERSECTION_WIDTH / 2f, 0.02f, INTERSECTION_Z - halfWidth / 4f),
            new Vector3(0.5f, 0.02f, halfWidth));

        // Stop line for right arm (west side)
        CreateStopLine(intersectionArea.transform, "StopLine_RightArm",
            new Vector3(INTERSECTION_WIDTH / 2f, 0.02f, INTERSECTION_Z + halfWidth / 4f),
            new Vector3(0.5f, 0.02f, halfWidth));

        intersectionArea.SetActive(false);
    }

    private static void CreateArmRoad(Transform parent, float halfWidth, float armLength)
    {
        // Arm road surface
        var surface = GameObject.CreatePrimitive(PrimitiveType.Plane);
        surface.name = "ArmSurface";
        surface.transform.SetParent(parent);
        surface.transform.localPosition = Vector3.zero;
        surface.transform.localScale = new Vector3(armLength / 10f, 1f, ROAD_WIDTH / 10f);
        surface.GetComponent<Renderer>().sharedMaterial = GetAsphaltMaterial();
        Object.DestroyImmediate(surface.GetComponent<Collider>());

        // Top curb (positive Z relative to arm)
        float curbOffset = halfWidth + CURB_WIDTH / 2f;
        CreateCurb(parent, "TopCurb",
            new Vector3(0f, CURB_HEIGHT / 2f, curbOffset),
            new Vector3(armLength, CURB_HEIGHT, CURB_WIDTH));

        // Bottom curb (negative Z relative to arm)
        CreateCurb(parent, "BottomCurb",
            new Vector3(0f, CURB_HEIGHT / 2f, -curbOffset),
            new Vector3(armLength, CURB_HEIGHT, CURB_WIDTH));

        // Center line dashes (along arm X-axis, mapped to local Z)
        var centerLine = new GameObject("CenterLine");
        centerLine.transform.SetParent(parent);
        centerLine.transform.localPosition = Vector3.zero;

        float dashLength = 3f;
        float dashSpacing = 10f;
        float armHalf = armLength / 2f;

        for (float x = -armHalf + 2f; x < armHalf - 2f; x += dashSpacing)
        {
            CreateDash(centerLine.transform,
                new Vector3(x, 0.02f, 0f),
                new Vector3(dashLength, 0.01f, 0.15f),
                GetYellowLineMaterial());
        }

        // Edge lines
        float edgeZ = halfWidth - 0.1f;
        var topEdge = GameObject.CreatePrimitive(PrimitiveType.Cube);
        topEdge.name = "TopEdgeLine";
        topEdge.transform.SetParent(parent);
        topEdge.transform.localPosition = new Vector3(0f, 0.02f, edgeZ);
        topEdge.transform.localScale = new Vector3(armLength, 0.01f, 0.12f);
        topEdge.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
        Object.DestroyImmediate(topEdge.GetComponent<Collider>());

        var bottomEdge = GameObject.CreatePrimitive(PrimitiveType.Cube);
        bottomEdge.name = "BottomEdgeLine";
        bottomEdge.transform.SetParent(parent);
        bottomEdge.transform.localPosition = new Vector3(0f, 0.02f, -edgeZ);
        bottomEdge.transform.localScale = new Vector3(armLength, 0.01f, 0.12f);
        bottomEdge.GetComponent<Renderer>().sharedMaterial = GetWhiteLineMaterial();
        Object.DestroyImmediate(bottomEdge.GetComponent<Collider>());
    }

    private static void CreateLeftArm(Transform roadParent, float halfWidth)
    {
        var leftArm = new GameObject("LeftArm");
        leftArm.transform.SetParent(roadParent);
        // Center of arm: X = -(INTERSECTION_WIDTH/2 + ARM_LENGTH/2), Z = intersection center
        float armCenterX = -(INTERSECTION_WIDTH / 2f + ARM_LENGTH / 2f);
        leftArm.transform.localPosition = new Vector3(armCenterX, 0.01f, INTERSECTION_Z);

        CreateArmRoad(leftArm.transform, halfWidth, ARM_LENGTH);

        leftArm.SetActive(false);
    }

    private static void CreateRightArm(Transform roadParent, float halfWidth)
    {
        var rightArm = new GameObject("RightArm");
        rightArm.transform.SetParent(roadParent);
        float armCenterX = INTERSECTION_WIDTH / 2f + ARM_LENGTH / 2f;
        rightArm.transform.localPosition = new Vector3(armCenterX, 0.01f, INTERSECTION_Z);

        CreateArmRoad(rightArm.transform, halfWidth, ARM_LENGTH);

        rightArm.SetActive(false);
    }

    private static void CreateLeftAngledArm(Transform roadParent, float halfWidth)
    {
        var leftAngled = new GameObject("LeftAngledArm");
        leftAngled.transform.SetParent(roadParent);
        // 45 degree angle, offset from intersection center
        float offset = (INTERSECTION_WIDTH / 2f + ARM_LENGTH / 2f) * Mathf.Cos(45f * Mathf.Deg2Rad);
        leftAngled.transform.localPosition = new Vector3(-offset, 0.01f, INTERSECTION_Z + offset);
        leftAngled.transform.localRotation = Quaternion.Euler(0f, -45f, 0f);

        CreateArmRoad(leftAngled.transform, halfWidth, ARM_LENGTH);

        leftAngled.SetActive(false);
    }

    private static void CreateRightAngledArm(Transform roadParent, float halfWidth)
    {
        var rightAngled = new GameObject("RightAngledArm");
        rightAngled.transform.SetParent(roadParent);
        float offset = (INTERSECTION_WIDTH / 2f + ARM_LENGTH / 2f) * Mathf.Cos(45f * Mathf.Deg2Rad);
        rightAngled.transform.localPosition = new Vector3(offset, 0.01f, INTERSECTION_Z + offset);
        rightAngled.transform.localRotation = Quaternion.Euler(0f, 45f, 0f);

        CreateArmRoad(rightAngled.transform, halfWidth, ARM_LENGTH);

        rightAngled.SetActive(false);
    }

    #endregion

    private static void SetProperty(object target, string propertyName, object value)
    {
        var type = target.GetType();
        var field = type.GetField(propertyName);
        if (field != null)
        {
            field.SetValue(target, value);
        }
        else
        {
            var prop = type.GetProperty(propertyName);
            if (prop != null && prop.CanWrite)
            {
                prop.SetValue(target, value);
            }
        }
    }

    private static void AddSceneToBuildSettings(string scenePath)
    {
        var scenes = new System.Collections.Generic.List<EditorBuildSettingsScene>(
            EditorBuildSettings.scenes);

        // Check if already exists
        foreach (var scene in scenes)
        {
            if (scene.path == scenePath) return;
        }

        scenes.Add(new EditorBuildSettingsScene(scenePath, true));
        EditorBuildSettings.scenes = scenes.ToArray();
    }

    #endregion
}
