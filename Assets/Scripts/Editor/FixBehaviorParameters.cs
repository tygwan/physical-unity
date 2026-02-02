using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Policies;

public class FixBehaviorParameters : Editor
{
    [MenuItem("Tools/Fix Phase J BehaviorParameters")]
    public static void Fix()
    {
        var agents = FindObjectsByType<BehaviorParameters>(FindObjectsSortMode.None);
        int fixed_count = 0;

        foreach (var bp in agents)
        {
            if (bp.BehaviorName != "E2EDrivingAgent") continue;

            var so = new SerializedObject(bp);

            // Set continuous actions to 2
            var contProp = so.FindProperty("m_BrainParameters.m_ActionSpec.m_NumContinuousActions");
            if (contProp != null) contProp.intValue = 2;

            // Clear branch sizes array (removes discrete actions)
            var branchProp = so.FindProperty("m_BrainParameters.m_ActionSpec.BranchSizes");
            if (branchProp != null)
            {
                Debug.Log($"[FixBP] BranchSizes array size before: {branchProp.arraySize}");
                branchProp.ClearArray();
            }

            so.ApplyModifiedPropertiesWithoutUndo();
            fixed_count++;
        }

        Debug.Log($"[FixBP] Fixed {fixed_count} agents");

        // Verify
        foreach (var bp in agents)
        {
            if (bp.BehaviorName != "E2EDrivingAgent") continue;
            var so = new SerializedObject(bp);
            var c = so.FindProperty("m_BrainParameters.m_ActionSpec.m_NumContinuousActions");
            var b = so.FindProperty("m_BrainParameters.m_ActionSpec.BranchSizes");
            Debug.Log($"[FixBP-Verify] ContinuousActions={c?.intValue}, BranchSizes.Length={b?.arraySize}");
            break;
        }

        // Mark scene dirty for save
        UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
            UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene());
    }
}
