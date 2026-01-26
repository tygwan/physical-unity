using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Policies;
using System.Collections.Generic;

/// <summary>
/// Editor utility to batch-update observation Space Size for all agents.
/// Usage: Tools > ML-Agents > Update Observation Size
/// </summary>
public class ObservationSizeUpdater : EditorWindow
{
    // Predefined observation configurations
    private static readonly Dictionary<string, int> ObservationConfigs = new Dictionary<string, int>
    {
        { "Phase B/C (242D)", 242 },   // Ego(8) + History(40) + Agents(160) + Route(30) + Speed(4)
        { "Phase D (254D)", 254 },      // + Lane(12)
        { "Full (266D)", 266 },         // Future expansion
        { "Custom", -1 }
    };

    private string selectedConfig = "Phase D (254D)";
    private int customSize = 254;
    private bool includeInactive = true;
    private bool includePrefabs = false;
    private Vector2 scrollPos;
    private List<BehaviorParameters> foundAgents = new List<BehaviorParameters>();

    [MenuItem("Tools/ML-Agents/Update Observation Size")]
    public static void ShowWindow()
    {
        var window = GetWindow<ObservationSizeUpdater>("Observation Size Updater");
        window.minSize = new Vector2(400, 300);
    }

    private void OnGUI()
    {
        EditorGUILayout.Space(10);
        EditorGUILayout.LabelField("Observation Space Size Updater", EditorStyles.boldLabel);
        EditorGUILayout.HelpBox(
            "This tool updates the Space Size in BehaviorParameters for all agents.\n" +
            "IMPORTANT: Space Size must match the agent's CollectObservations() output.",
            MessageType.Info);

        EditorGUILayout.Space(10);

        // Configuration selection
        EditorGUILayout.LabelField("Configuration", EditorStyles.boldLabel);

        string[] configNames = new string[ObservationConfigs.Count];
        ObservationConfigs.Keys.CopyTo(configNames, 0);

        int currentIndex = System.Array.IndexOf(configNames, selectedConfig);
        int newIndex = EditorGUILayout.Popup("Observation Config", currentIndex, configNames);
        selectedConfig = configNames[newIndex];

        int targetSize;
        if (selectedConfig == "Custom")
        {
            customSize = EditorGUILayout.IntField("Custom Size", customSize);
            targetSize = customSize;
        }
        else
        {
            targetSize = ObservationConfigs[selectedConfig];
            EditorGUILayout.LabelField($"Target Size: {targetSize}D");
        }

        EditorGUILayout.Space(5);
        includeInactive = EditorGUILayout.Toggle("Include Inactive Objects", includeInactive);
        includePrefabs = EditorGUILayout.Toggle("Update Prefabs", includePrefabs);

        EditorGUILayout.Space(10);

        // Find agents button
        if (GUILayout.Button("Find Agents in Scene", GUILayout.Height(30)))
        {
            FindAllAgents();
        }

        // Display found agents
        if (foundAgents.Count > 0)
        {
            EditorGUILayout.Space(10);
            EditorGUILayout.LabelField($"Found {foundAgents.Count} agents:", EditorStyles.boldLabel);

            scrollPos = EditorGUILayout.BeginScrollView(scrollPos, GUILayout.Height(150));
            foreach (var agent in foundAgents)
            {
                if (agent == null) continue;

                var currentSize = agent.BrainParameters.VectorObservationSize;
                string status = currentSize == targetSize ? "✓" : $"({currentSize} → {targetSize})";

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField($"  {agent.gameObject.name}", GUILayout.Width(200));
                EditorGUILayout.LabelField(status);
                EditorGUILayout.EndHorizontal();
            }
            EditorGUILayout.EndScrollView();

            EditorGUILayout.Space(10);

            // Update button
            GUI.backgroundColor = Color.green;
            if (GUILayout.Button($"Update All to {targetSize}D", GUILayout.Height(40)))
            {
                UpdateAllAgents(targetSize);
            }
            GUI.backgroundColor = Color.white;
        }

        EditorGUILayout.Space(10);
        EditorGUILayout.HelpBox(
            "Observation Dimension Reference:\n" +
            "• Ego State: 8D (pos, vel, heading, accel)\n" +
            "• History: 40D (10 frames × 4D)\n" +
            "• Agents: 160D (8 agents × 20D)\n" +
            "• Route: 30D (10 waypoints × 3D)\n" +
            "• Speed: 4D (current, target, zone, diff)\n" +
            "• Lane: 12D (3 lanes × 4D)",
            MessageType.None);
    }

    private void FindAllAgents()
    {
        foundAgents.Clear();

        // Find in scene
        var behaviors = includeInactive
            ? Resources.FindObjectsOfTypeAll<BehaviorParameters>()
            : FindObjectsOfType<BehaviorParameters>();

        foreach (var behavior in behaviors)
        {
            // Skip prefabs unless requested
            if (!includePrefabs && PrefabUtility.IsPartOfPrefabAsset(behavior))
                continue;

            // Only include scene objects
            if (behavior.gameObject.scene.IsValid() || includePrefabs)
            {
                foundAgents.Add(behavior);
            }
        }

        Debug.Log($"[ObservationSizeUpdater] Found {foundAgents.Count} agents");
    }

    private void UpdateAllAgents(int targetSize)
    {
        int updatedCount = 0;

        foreach (var agent in foundAgents)
        {
            if (agent == null) continue;

            Undo.RecordObject(agent, "Update Observation Size");

            var brainParams = agent.BrainParameters;
            if (brainParams.VectorObservationSize != targetSize)
            {
                brainParams.VectorObservationSize = targetSize;
                EditorUtility.SetDirty(agent);
                updatedCount++;
                Debug.Log($"[ObservationSizeUpdater] Updated {agent.gameObject.name} to {targetSize}D");
            }
        }

        if (updatedCount > 0)
        {
            AssetDatabase.SaveAssets();
            Debug.Log($"[ObservationSizeUpdater] Updated {updatedCount} agents to {targetSize}D observation size");
            EditorUtility.DisplayDialog("Update Complete",
                $"Updated {updatedCount} agents to {targetSize}D observation size.", "OK");
        }
        else
        {
            EditorUtility.DisplayDialog("No Changes",
                "All agents already have the correct observation size.", "OK");
        }
    }
}
