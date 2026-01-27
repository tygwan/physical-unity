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
            roadLength = 300f
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
            roadLength = 300f
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
            roadLength = 400f
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
            roadLength = 500f
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
            roadLength = 500f
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
            roadLength = 300f
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
    }

    private static void CreatePhaseScene(PhaseConfig config)
    {
        // Ensure scenes folder exists
        if (!Directory.Exists(SCENES_PATH))
        {
            Directory.CreateDirectory(SCENES_PATH);
            AssetDatabase.Refresh();
        }

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

        // Ground for this area
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.SetParent(trainingArea.transform);
        ground.transform.localPosition = Vector3.zero;

        float width = Mathf.Max(50f, config.numLanes * 5f + 30f);
        ground.transform.localScale = new Vector3(width / 10f, 1, config.roadLength / 10f + 10f);

        var groundRenderer = ground.GetComponent<Renderer>();
        var groundMaterial = new Material(Shader.Find("Standard"));
        groundMaterial.color = new Color(0.15f, 0.4f, 0.15f);
        groundRenderer.material = groundMaterial;

        // Road with WaypointManager
        var road = CreateRoadForArea(config, trainingArea.transform);

        // DrivingSceneManager for this area
        var sceneManager = CreateDrivingSceneManagerForArea(config, trainingArea.transform);

        // Agent Vehicle
        var agent = CreateAgentVehicleForArea(config.roadLength, areaIndex, trainingArea.transform);

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

        // Road surface
        var roadSurface = GameObject.CreatePrimitive(PrimitiveType.Plane);
        roadSurface.name = "RoadSurface";
        roadSurface.transform.SetParent(road.transform);
        roadSurface.transform.localPosition = new Vector3(0, 0.01f, 0);

        float roadWidth = config.numLanes * 3.5f + 1f;
        roadSurface.transform.localScale = new Vector3(roadWidth / 10f, 1, config.roadLength / 10f);

        var renderer = roadSurface.GetComponent<Renderer>();
        var material = new Material(Shader.Find("Standard"));
        material.color = new Color(0.25f, 0.25f, 0.25f);
        renderer.material = material;

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

    private static GameObject CreateAgentVehicleForArea(float roadLength, int areaIndex, Transform parent)
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
            agent.AddComponent(agentType);
        }

        // Configure BehaviorParameters for ML-Agents training
        var behaviorParams = agent.GetComponent("BehaviorParameters");
        if (behaviorParams != null)
        {
            SetProperty(behaviorParams, "m_BehaviorName", "E2EDrivingAgent");
        }

        return agent;
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
