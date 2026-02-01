using UnityEngine;
using UnityEditor;
using UnityEditor.SceneManagement;
using Unity.MLAgents;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Actuators;

/// <summary>
/// Utility to configure all E2EDrivingAgent BehaviorParameters in the current scene.
/// Run via Tools > Configure Agents > Phase G (260D)
/// Also ensures DecisionRequester (period=5) is present on each agent.
/// </summary>
public class ConfigurePhaseGAgents
{
    [MenuItem("Tools/Configure Agents/Phase G (260D)")]
    public static void ConfigureForPhaseG()
    {
        ConfigureAllAgents(260, true, true);
    }

    [MenuItem("Tools/Configure Agents/Phase F (254D)")]
    public static void ConfigureForPhaseF()
    {
        ConfigureAllAgents(254, true, false);
    }

    [MenuItem("Tools/Configure Agents/Phase A-C (242D)")]
    public static void ConfigureForPhaseAC()
    {
        ConfigureAllAgents(242, false, false);
    }

    private static void ConfigureAllAgents(int obsSize, bool laneObs, bool intersectionObs)
    {
        var agents = Object.FindObjectsByType<BehaviorParameters>(FindObjectsSortMode.None);
        int count = 0;

        foreach (var bp in agents)
        {
            // Configure BehaviorParameters directly
            bp.BehaviorName = "E2EDrivingAgent";
            bp.BrainParameters.VectorObservationSize = obsSize;
            bp.BrainParameters.NumStackedVectorObservations = 1;
            bp.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(2);
            EditorUtility.SetDirty(bp);

            // Ensure DecisionRequester exists with period=5
            var dr = bp.GetComponent<DecisionRequester>();
            if (dr == null)
            {
                dr = bp.gameObject.AddComponent<DecisionRequester>();
            }
            dr.DecisionPeriod = 5;
            dr.TakeActionsBetweenDecisions = true;
            EditorUtility.SetDirty(dr);

            // Configure E2EDrivingAgent component
            var agentType = System.Type.GetType("ADPlatform.Agents.E2EDrivingAgent, Assembly-CSharp");
            if (agentType != null)
            {
                var agentComp = bp.GetComponent(agentType);
                if (agentComp != null)
                {
                    var agentSo = new SerializedObject(agentComp);
                    agentSo.FindProperty("enableLaneObservation").boolValue = laneObs;
                    agentSo.FindProperty("enableIntersectionObservation").boolValue = intersectionObs;
                    agentSo.ApplyModifiedPropertiesWithoutUndo();
                    EditorUtility.SetDirty(agentComp);
                }
            }

            count++;
        }

        // Mark scene dirty
        EditorSceneManager.MarkSceneDirty(
            EditorSceneManager.GetActiveScene());

        Debug.Log($"[ConfigureAgents] Configured {count} agents: obs={obsSize}D, lane={laneObs}, intersection={intersectionObs}, DecisionPeriod=5");
        EditorUtility.DisplayDialog("Configure Agents",
            $"Configured {count} agents for {obsSize}D observation.\nDecisionRequester: period=5\n\nRemember to save the scene (Ctrl+S)!",
            "OK");
    }
}
