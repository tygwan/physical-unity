using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;

/// <summary>
/// Editor utility to create the DrivingScene from scratch.
/// Use: Tools > Create Driving Scene
/// </summary>
public class DrivingSceneCreator
{
    [MenuItem("Tools/Create Driving Scene")]
    public static void CreateDrivingScene()
    {
        // Create new scene
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        // 1. Create Main Camera
        var camera = new GameObject("Main Camera");
        camera.tag = "MainCamera";
        var cam = camera.AddComponent<Camera>();
        cam.clearFlags = CameraClearFlags.Skybox;
        camera.transform.position = new Vector3(0, 10, -10);
        camera.transform.rotation = Quaternion.Euler(30, 0, 0);
        camera.AddComponent<AudioListener>();

        // Add FollowCamera if available
        var followCameraType = System.Type.GetType("FollowCamera");
        if (followCameraType != null)
        {
            camera.AddComponent(followCameraType);
        }

        // 2. Create Directional Light
        var light = new GameObject("Directional Light");
        var lightComp = light.AddComponent<Light>();
        lightComp.type = LightType.Directional;
        lightComp.intensity = 1f;
        lightComp.shadows = LightShadows.Soft;
        light.transform.rotation = Quaternion.Euler(50, -30, 0);

        // 3. Create Ground Plane
        var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
        ground.name = "Ground";
        ground.transform.position = Vector3.zero;
        ground.transform.localScale = new Vector3(50, 1, 50);

        // Set ground material to dark gray
        var groundRenderer = ground.GetComponent<Renderer>();
        var groundMaterial = new Material(Shader.Find("Standard"));
        groundMaterial.color = new Color(0.2f, 0.2f, 0.2f);
        groundRenderer.material = groundMaterial;

        // 4. Create DrivingSceneManager
        var sceneManager = new GameObject("DrivingSceneManager");
        var managerType = System.Type.GetType("DrivingSceneManager");
        if (managerType != null)
        {
            sceneManager.AddComponent(managerType);
        }

        // 5. Create Road
        var road = new GameObject("Road");
        var roadPlane = GameObject.CreatePrimitive(PrimitiveType.Plane);
        roadPlane.name = "RoadSurface";
        roadPlane.transform.SetParent(road.transform);
        roadPlane.transform.localPosition = new Vector3(0, 0.01f, 0);
        roadPlane.transform.localScale = new Vector3(2, 1, 50);

        var roadRenderer = roadPlane.GetComponent<Renderer>();
        var roadMaterial = new Material(Shader.Find("Standard"));
        roadMaterial.color = new Color(0.3f, 0.3f, 0.3f);
        roadRenderer.material = roadMaterial;

        // Add WaypointManager
        var waypointType = System.Type.GetType("WaypointManager");
        if (waypointType != null)
        {
            road.AddComponent(waypointType);
        }

        // 6. Create Agent Vehicle placeholder
        var agentVehicle = GameObject.CreatePrimitive(PrimitiveType.Cube);
        agentVehicle.name = "E2EDrivingAgent";
        agentVehicle.transform.position = new Vector3(0, 0.5f, -40);
        agentVehicle.transform.localScale = new Vector3(2, 1, 4);

        var agentRenderer = agentVehicle.GetComponent<Renderer>();
        var agentMaterial = new Material(Shader.Find("Standard"));
        agentMaterial.color = Color.blue;
        agentRenderer.material = agentMaterial;

        // Add Rigidbody
        var rb = agentVehicle.AddComponent<Rigidbody>();
        rb.mass = 1500;
        rb.linearDamping = 0.5f;
        rb.angularDamping = 0.5f;

        // Add E2EDrivingAgent component
        var agentType = System.Type.GetType("E2EDrivingAgent, Assembly-CSharp");
        if (agentType != null)
        {
            agentVehicle.AddComponent(agentType);
        }

        // 7. Create NPC Spawner
        var npcSpawner = new GameObject("NPCSpawner");
        npcSpawner.transform.position = new Vector3(0, 0, 0);

        // 8. Create Training Areas parent
        var trainingAreas = new GameObject("TrainingAreas");

        // Save scene
        string scenePath = "Assets/Scenes/DrivingScene.unity";
        EditorSceneManager.SaveScene(scene, scenePath);

        // Add to build settings
        AddSceneToBuildSettings(scenePath);

        Debug.Log("DrivingScene created successfully at: " + scenePath);
        EditorUtility.DisplayDialog("Success", "DrivingScene created at:\n" + scenePath, "OK");
    }

    private static void AddSceneToBuildSettings(string scenePath)
    {
        var scenes = new System.Collections.Generic.List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);

        // Check if scene already exists
        foreach (var scene in scenes)
        {
            if (scene.path == scenePath)
            {
                return;
            }
        }

        // Add scene
        scenes.Add(new EditorBuildSettingsScene(scenePath, true));
        EditorBuildSettings.scenes = scenes.ToArray();

        Debug.Log("Added DrivingScene to Build Settings");
    }
}
