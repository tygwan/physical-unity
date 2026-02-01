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

        EditorUtility.DisplayDialog("Complete",
            "All Phase scenes created in Assets/Scenes/", "OK");
    }

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
        public int observationSize;            // 242, 254, or 260
        public bool enableLaneObservation;     // true for Phase D+
        public bool enableIntersectionObservation; // true for Phase G+
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

        // Wire up references within this training area
        WireAreaReferences(sceneManager, agent, road, goal, npcs);

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
        }

        // Configure BehaviorParameters for ML-Agents training via SerializedObject
        ConfigureBehaviorParameters(agent, config);

        return agent;
    }

    private static void ConfigureBehaviorParameters(GameObject agent, PhaseConfig config)
    {
        var bp = agent.GetComponent("BehaviorParameters") as UnityEngine.Component;
        if (bp == null) return;

        var so = new SerializedObject(bp);
        so.FindProperty("m_BehaviorName").stringValue = "E2EDrivingAgent";

        int obsSize = config.observationSize > 0 ? config.observationSize : 242;
        so.FindProperty("m_BrainParameters.VectorObservationSize").intValue = obsSize;
        so.FindProperty("m_BrainParameters.NumStackedVectorObservations").intValue = 1;
        so.FindProperty("m_BrainParameters.ActionSpec.m_NumContinuousActions").intValue = 2;

        // Clear discrete branches (set size to 0)
        var branchProp = so.FindProperty("m_BrainParameters.ActionSpec.BranchSizes");
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

    private static void WireAreaReferences(GameObject manager, GameObject agent,
        GameObject road, GameObject goal, GameObject[] npcs)
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
