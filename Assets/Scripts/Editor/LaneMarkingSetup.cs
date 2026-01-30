using UnityEditor;
using UnityEngine;
using ADPlatform.Environment;
using ADPlatform.Agents;

/// <summary>
/// Phase D preparation: Adds lane marking GameObjects to all TrainingAreas
/// and configures agents for 254D lane observation.
///
/// Menu Items:
///   Tools > Phase D > Setup Lane Markings     - Creates LeftEdge/RightEdge per area
///   Tools > Phase D > Configure Agents        - Sets enableLaneObservation + laneMarkingLayer
///   Tools > Phase D > Full Setup (Silent)     - Both steps, no dialogs (for MCP)
///   Tools > Phase D > Verify Setup            - Counts lane markings and checks agents
/// </summary>
public class LaneMarkingSetup
{
    private const int NUM_TRAINING_AREAS = 16;
    private const float ROAD_HALF_WIDTH = 2.25f;  // Single lane: (1 * 3.5 + 1) / 2 = 2.25m
    private const float ROAD_LENGTH = 300f;
    private const float MARKING_HEIGHT = 0.5f;     // Y position (must match agent's laneDetectHeight)
    private const float MARKING_THICKNESS = 0.1f;  // X scale (thin wall)
    private const float MARKING_WALL_HEIGHT = 1.0f; // Y scale
    private const string LANE_MARKING_LAYER = "LaneMarking";

    [MenuItem("Tools/Phase D/Setup Lane Markings")]
    public static void SetupLaneMarkings()
    {
        int created = CreateLaneMarkingsInScene();
        EditorUtility.DisplayDialog("Lane Markings Setup",
            $"Created {created} lane marking objects across {NUM_TRAINING_AREAS} TrainingAreas.\n\n" +
            "Next: Run 'Tools > Phase D > Configure Agents' to enable lane observation.",
            "OK");
    }

    [MenuItem("Tools/Phase D/Configure Agents for Phase D")]
    public static void ConfigureAgents()
    {
        int configured = ConfigureAgentsForPhaseD();
        EditorUtility.DisplayDialog("Agent Configuration",
            $"Configured {configured} agents with enableLaneObservation=true and laneMarkingLayer set.\n\n" +
            "Remember to save the scene (Ctrl+S).",
            "OK");
    }

    [MenuItem("Tools/Phase D/Full Setup (Silent)")]
    public static void FullSetupSilent()
    {
        // Ensure layer exists first
        EnsureLaneMarkingLayer();

        int markings = CreateLaneMarkingsInScene();
        int agents = ConfigureAgentsForPhaseD();

        Debug.Log($"[Phase D Setup] Complete: {markings} lane markings created, {agents} agents configured.");
    }

    [MenuItem("Tools/Phase D/Verify Setup")]
    public static void VerifySetup()
    {
        // Count LaneMarking components
        var markings = Object.FindObjectsByType<LaneMarking>(FindObjectsInactive.Include, FindObjectsSortMode.None);

        // Count configured agents
        var agents = Object.FindObjectsByType<E2EDrivingAgentBv2>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        int laneEnabled = 0;
        int layerSet = 0;
        foreach (var agent in agents)
        {
            if (agent.enableLaneObservation) laneEnabled++;
            if (agent.laneMarkingLayer != 0) layerSet++;
        }

        // Check layer exists
        int layerIndex = LayerMask.NameToLayer(LANE_MARKING_LAYER);
        string layerStatus = layerIndex >= 0 ? $"Found (index {layerIndex})" : "MISSING";

        // Check lane marking layers
        int correctLayer = 0;
        foreach (var marking in markings)
        {
            if (layerIndex >= 0 && marking.gameObject.layer == layerIndex)
                correctLayer++;
        }

        string report = $"=== Phase D Setup Verification ===\n\n" +
            $"LaneMarking Layer: {layerStatus}\n" +
            $"LaneMarking objects: {markings.Length} (expected: {NUM_TRAINING_AREAS * 2})\n" +
            $"  On correct layer: {correctLayer}/{markings.Length}\n\n" +
            $"E2EDrivingAgentBv2: {agents.Length}\n" +
            $"  enableLaneObservation: {laneEnabled}/{agents.Length}\n" +
            $"  laneMarkingLayer set: {layerSet}/{agents.Length}\n";

        Debug.Log($"[Phase D Verify]\n{report}");
        EditorUtility.DisplayDialog("Phase D Verification", report, "OK");
    }

    /// <summary>
    /// Create LeftEdge and RightEdge lane markings under each TrainingArea's Road object.
    /// </summary>
    private static int CreateLaneMarkingsInScene()
    {
        int created = 0;
        int layerIndex = LayerMask.NameToLayer(LANE_MARKING_LAYER);

        if (layerIndex < 0)
        {
            Debug.LogWarning($"[Phase D] Layer '{LANE_MARKING_LAYER}' not found. " +
                "Add it via Edit > Project Settings > Tags and Layers, or run manage_editor add_layer first.");
        }

        Undo.SetCurrentGroupName("Phase D: Setup Lane Markings");
        int undoGroup = Undo.GetCurrentGroup();

        for (int i = 0; i < NUM_TRAINING_AREAS; i++)
        {
            string areaName = $"TrainingArea_{i}";
            var area = GameObject.Find($"TrainingAreas/{areaName}");
            if (area == null)
            {
                Debug.LogWarning($"[Phase D] {areaName} not found, skipping.");
                continue;
            }

            // Find Road child
            Transform road = area.transform.Find("Road");
            if (road == null)
            {
                Debug.LogWarning($"[Phase D] Road not found under {areaName}, skipping.");
                continue;
            }

            // Check if LaneMarkings already exist
            Transform existing = road.Find("LaneMarkings");
            if (existing != null)
            {
                Debug.Log($"[Phase D] LaneMarkings already exist under {areaName}/Road, skipping.");
                continue;
            }

            // Create LaneMarkings parent
            var laneMarkingsParent = new GameObject("LaneMarkings");
            Undo.RegisterCreatedObjectUndo(laneMarkingsParent, "Create LaneMarkings");
            laneMarkingsParent.transform.SetParent(road);
            laneMarkingsParent.transform.localPosition = Vector3.zero;
            laneMarkingsParent.transform.localRotation = Quaternion.identity;

            // Left Edge (WhiteSolid - road boundary)
            created += CreateEdgeMarking(laneMarkingsParent.transform, "LeftEdge",
                -ROAD_HALF_WIDTH, LaneMarkingType.WhiteSolid, layerIndex);

            // Right Edge (WhiteSolid - road boundary)
            created += CreateEdgeMarking(laneMarkingsParent.transform, "RightEdge",
                ROAD_HALF_WIDTH, LaneMarkingType.WhiteSolid, layerIndex);
        }

        Undo.CollapseUndoOperations(undoGroup);
        Debug.Log($"[Phase D] Created {created} lane marking objects.");
        return created;
    }

    /// <summary>
    /// Create a single edge lane marking strip (thin trigger collider wall along road).
    /// </summary>
    private static int CreateEdgeMarking(Transform parent, string name,
        float xOffset, LaneMarkingType markingType, int layerIndex)
    {
        var marking = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Undo.RegisterCreatedObjectUndo(marking, $"Create {name}");

        marking.name = name;
        marking.transform.SetParent(parent);
        marking.transform.localPosition = new Vector3(xOffset, MARKING_HEIGHT, 0f);
        marking.transform.localRotation = Quaternion.identity;
        marking.transform.localScale = new Vector3(MARKING_THICKNESS, MARKING_WALL_HEIGHT, ROAD_LENGTH);

        // Set layer
        if (layerIndex >= 0)
        {
            marking.layer = layerIndex;
        }

        // Configure BoxCollider as trigger (required for raycast detection)
        var collider = marking.GetComponent<BoxCollider>();
        if (collider != null)
        {
            collider.isTrigger = true;
        }

        // Remove MeshRenderer - invisible wall (no visual needed, agent uses raycasts)
        var renderer = marking.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            renderer.enabled = false;
        }

        // Add LaneMarking component
        var laneMarking = marking.AddComponent<LaneMarking>();
        laneMarking.markingType = markingType;
        laneMarking.crossingPenalty = -2.0f;
        laneMarking.terminatesEpisode = false;

        return 1;
    }

    /// <summary>
    /// Configure all E2EDrivingAgentBv2 instances for Phase D lane observation.
    /// </summary>
    private static int ConfigureAgentsForPhaseD()
    {
        int layerIndex = LayerMask.NameToLayer(LANE_MARKING_LAYER);
        if (layerIndex < 0)
        {
            Debug.LogError($"[Phase D] Layer '{LANE_MARKING_LAYER}' not found! Cannot configure agents.");
            return 0;
        }

        LayerMask laneMarkingMask = 1 << layerIndex;

        var agents = Object.FindObjectsByType<E2EDrivingAgentBv2>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (agents.Length == 0)
        {
            Debug.LogWarning("[Phase D] No E2EDrivingAgentBv2 found in scene.");
            return 0;
        }

        Undo.SetCurrentGroupName("Phase D: Configure Agents");
        int undoGroup = Undo.GetCurrentGroup();
        int configured = 0;

        foreach (var agent in agents)
        {
            Undo.RecordObject(agent, "Configure agent for Phase D");

            agent.enableLaneObservation = true;
            agent.laneMarkingLayer = laneMarkingMask;

            EditorUtility.SetDirty(agent);
            configured++;
        }

        Undo.CollapseUndoOperations(undoGroup);
        Debug.Log($"[Phase D] Configured {configured} agents: enableLaneObservation=true, laneMarkingLayer={laneMarkingMask.value} (layer index {layerIndex})");
        return configured;
    }

    /// <summary>
    /// Ensure the LaneMarking layer exists. Logs warning if not found.
    /// Layer must be added via Unity Editor or manage_editor add_layer API.
    /// </summary>
    private static void EnsureLaneMarkingLayer()
    {
        int layerIndex = LayerMask.NameToLayer(LANE_MARKING_LAYER);
        if (layerIndex < 0)
        {
            Debug.LogWarning($"[Phase D] Layer '{LANE_MARKING_LAYER}' does not exist. " +
                "Please add it via: Edit > Project Settings > Tags and Layers");
        }
        else
        {
            Debug.Log($"[Phase D] Layer '{LANE_MARKING_LAYER}' found at index {layerIndex}.");
        }
    }
}
