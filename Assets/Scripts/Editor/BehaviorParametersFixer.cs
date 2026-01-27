using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Policies;
using Unity.MLAgents;
using ADPlatform.Agents;

/// <summary>
/// Fixes BehaviorParameters and E2EDrivingAgent settings across all scenes.
/// Observation sizes by Phase:
///   - Phase A/B/C: 242D (base)
///   - Phase D/E/F: 254D (base + 12D lane)
///   - Phase G:     260D (base + 12D lane + 6D intersection)
/// Use: Tools > Fix BehaviorParameters > Fix Phase A (242D) etc.
/// </summary>
public class BehaviorParametersFixer
{
    // Phase observation configurations
    private struct PhaseConfig
    {
        public int observationSize;
        public bool enableLane;
        public bool enableIntersection;

        public PhaseConfig(int obs, bool lane, bool intersection)
        {
            observationSize = obs;
            enableLane = lane;
            enableIntersection = intersection;
        }
    }

    private static readonly PhaseConfig PhaseABC = new PhaseConfig(242, false, false);
    private static readonly PhaseConfig PhaseDEF = new PhaseConfig(254, true, false);
    private static readonly PhaseConfig PhaseG = new PhaseConfig(260, true, true);

    // ========== Phase A (242D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase A (242D)")]
    public static void FixPhaseA()
    {
        FixSceneWithConfig("PhaseA_DenseOvertaking", PhaseABC);
    }

    // ========== Phase B (242D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase B (242D)")]
    public static void FixPhaseB()
    {
        FixSceneWithConfig("PhaseB_DecisionLearning", PhaseABC);
    }

    // ========== Phase C (242D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase C (242D)")]
    public static void FixPhaseC()
    {
        FixSceneWithConfig("PhaseC_MultiNPC", PhaseABC);
    }

    // ========== Phase D (254D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase D (254D)")]
    public static void FixPhaseD()
    {
        FixSceneWithConfig("PhaseD_LaneObservation", PhaseDEF);
    }

    // ========== Phase E (254D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase E (254D)")]
    public static void FixPhaseE()
    {
        FixSceneWithConfig("PhaseE_CurvedRoads", PhaseDEF);
    }

    // ========== Phase F (254D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase F (254D)")]
    public static void FixPhaseF()
    {
        FixSceneWithConfig("PhaseF_MultiLane", PhaseDEF);
    }

    // ========== Phase G (260D) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Phase G (260D)")]
    public static void FixPhaseG()
    {
        FixSceneWithConfig("PhaseG_Intersection", PhaseG);
    }

    // ========== Fix Current Scene (auto-detect) ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix Current Scene (Auto-detect)")]
    public static void FixCurrentScene()
    {
        var sceneName = UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene().name;
        PhaseConfig config = GetConfigForScene(sceneName);

        int count = FixAgentsInCurrentScene(config);

        Debug.Log($"[BehaviorParametersFixer] Fixed {count} agents in {sceneName} with {config.observationSize}D observation.");

        if (count > 0)
        {
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
                UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene());
        }
    }

    // ========== Fix All Phase Scenes ==========
    [MenuItem("Tools/Fix BehaviorParameters/Fix All Phase Scenes")]
    public static void FixAllPhaseScenes()
    {
        var sceneConfigs = new (string sceneName, PhaseConfig config)[]
        {
            ("PhaseA_DenseOvertaking", PhaseABC),
            ("PhaseB_DecisionLearning", PhaseABC),
            ("PhaseC_MultiNPC", PhaseABC),
            ("PhaseE_CurvedRoads", PhaseDEF),
            ("PhaseF_MultiLane", PhaseDEF),
            ("PhaseG_Intersection", PhaseG)
        };

        int totalCount = 0;

        foreach (var (sceneName, config) in sceneConfigs)
        {
            string scenePath = $"Assets/Scenes/{sceneName}.unity";

            if (!System.IO.File.Exists(scenePath))
            {
                Debug.LogWarning($"Scene not found: {scenePath}");
                continue;
            }

            var scene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene(scenePath);
            int count = FixAgentsInCurrentScene(config);

            if (count > 0)
            {
                UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene);
                Debug.Log($"[BehaviorParametersFixer] Fixed {count} agents in {sceneName} ({config.observationSize}D)");
                totalCount += count;
            }
        }

        EditorUtility.DisplayDialog("Complete",
            $"Fixed {totalCount} agents across all Phase scenes.\n\n" +
            "Phase A/B/C: 242D\nPhase D/E/F: 254D\nPhase G: 260D", "OK");
    }

    private static PhaseConfig GetConfigForScene(string sceneName)
    {
        if (sceneName.Contains("PhaseA") || sceneName.Contains("PhaseB") || sceneName.Contains("PhaseC"))
            return PhaseABC;
        if (sceneName.Contains("PhaseD") || sceneName.Contains("PhaseE") || sceneName.Contains("PhaseF"))
            return PhaseDEF;
        if (sceneName.Contains("PhaseG"))
            return PhaseG;

        // Default to Phase A config
        Debug.LogWarning($"Unknown scene '{sceneName}', using Phase A config (242D)");
        return PhaseABC;
    }

    private static void FixSceneWithConfig(string sceneName, PhaseConfig config)
    {
        string scenePath = $"Assets/Scenes/{sceneName}.unity";

        if (!System.IO.File.Exists(scenePath))
        {
            Debug.LogError($"Scene not found: {scenePath}");
            return;
        }

        var scene = UnityEditor.SceneManagement.EditorSceneManager.OpenScene(scenePath);
        int count = FixAgentsInCurrentScene(config);

        if (count > 0)
        {
            UnityEditor.SceneManagement.EditorSceneManager.SaveScene(scene);
            Debug.Log($"[BehaviorParametersFixer] Fixed {count} agents in {sceneName} ({config.observationSize}D)");
        }

        EditorUtility.DisplayDialog("Complete",
            $"Fixed {count} agents in {sceneName}.\n" +
            $"Observation Size: {config.observationSize}D\n" +
            $"Lane Observation: {config.enableLane}\n" +
            $"Intersection Observation: {config.enableIntersection}", "OK");
    }

    private static int FixAgentsInCurrentScene(PhaseConfig config)
    {
        int count = 0;
        var behaviorParams = Object.FindObjectsByType<BehaviorParameters>(FindObjectsSortMode.None);

        foreach (var bp in behaviorParams)
        {
            if (bp.gameObject.name == "E2EDrivingAgent")
            {
                // Fix BehaviorParameters
                FixBehaviorParameters(bp, config.observationSize);

                // Fix E2EDrivingAgent observation flags
                var agent = bp.GetComponent<E2EDrivingAgent>();
                if (agent != null)
                {
                    FixAgentObservationFlags(agent, config);
                }

                // Ensure DecisionRequester exists and is configured
                EnsureDecisionRequester(bp.gameObject);

                EditorUtility.SetDirty(bp);
                EditorUtility.SetDirty(bp.gameObject);
                count++;
            }
        }

        return count;
    }

    private static void EnsureDecisionRequester(GameObject agentObject)
    {
        var decisionRequester = agentObject.GetComponent<DecisionRequester>();
        if (decisionRequester == null)
        {
            decisionRequester = agentObject.AddComponent<DecisionRequester>();
            Debug.Log($"[BehaviorParametersFixer] Added DecisionRequester to {agentObject.name}");
        }

        // Configure DecisionRequester
        var serialized = new SerializedObject(decisionRequester);

        var decisionPeriodProp = serialized.FindProperty("DecisionPeriod");
        if (decisionPeriodProp != null)
        {
            decisionPeriodProp.intValue = 5;
        }

        var takeActionsProp = serialized.FindProperty("TakeActionsBetweenDecisions");
        if (takeActionsProp != null)
        {
            takeActionsProp.boolValue = true;
        }

        serialized.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(decisionRequester);
    }

    private static void FixBehaviorParameters(BehaviorParameters bp, int observationSize)
    {
        var serializedObject = new SerializedObject(bp);

        // Behavior Name
        var behaviorNameProp = serializedObject.FindProperty("m_BehaviorName");
        if (behaviorNameProp != null)
        {
            behaviorNameProp.stringValue = "E2EDrivingAgent";
        }

        // Brain Parameters
        var brainParams = serializedObject.FindProperty("m_BrainParameters");
        if (brainParams != null)
        {
            // Vector Observation Size
            var vectorObsSize = brainParams.FindPropertyRelative("VectorObservationSize");
            if (vectorObsSize != null)
            {
                vectorObsSize.intValue = observationSize;
            }

            // Stacked observations
            var numStacked = brainParams.FindPropertyRelative("NumStackedVectorObservations");
            if (numStacked != null)
            {
                numStacked.intValue = 1;
            }

            // Action Spec: 2 continuous (steering, acceleration), 0 discrete
            var actionSpec = brainParams.FindPropertyRelative("m_ActionSpec");
            if (actionSpec != null)
            {
                var numContinuous = actionSpec.FindPropertyRelative("m_NumContinuousActions");
                if (numContinuous != null)
                {
                    numContinuous.intValue = 2;
                }

                var branchSizes = actionSpec.FindPropertyRelative("BranchSizes");
                if (branchSizes != null && branchSizes.isArray)
                {
                    branchSizes.arraySize = 0;
                }
            }
        }

        serializedObject.ApplyModifiedPropertiesWithoutUndo();
    }

    private static void FixAgentObservationFlags(E2EDrivingAgent agent, PhaseConfig config)
    {
        var serializedAgent = new SerializedObject(agent);

        var enableLaneProp = serializedAgent.FindProperty("enableLaneObservation");
        if (enableLaneProp != null)
        {
            enableLaneProp.boolValue = config.enableLane;
        }

        var enableIntersectionProp = serializedAgent.FindProperty("enableIntersectionObservation");
        if (enableIntersectionProp != null)
        {
            enableIntersectionProp.boolValue = config.enableIntersection;
        }

        serializedAgent.ApplyModifiedPropertiesWithoutUndo();
        EditorUtility.SetDirty(agent);
    }
}
