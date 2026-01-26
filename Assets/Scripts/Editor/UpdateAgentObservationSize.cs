using UnityEngine;
using UnityEditor;
using Unity.MLAgents.Policies;

namespace ADPlatform.Editor
{
    /// <summary>
    /// Editor utility to update BehaviorParameters VectorObservationSize for all agents.
    /// Phase C update: 242D -> 254D (added 12D lane info)
    /// </summary>
    public static class UpdateAgentObservationSize
    {
        [MenuItem("AD Platform/Update Observation Size (254D for Phase C)")]
        public static void UpdateObservationSizeTo254()
        {
            int updated = 0;

            // Find all BehaviorParameters in the scene
            BehaviorParameters[] allBehaviors = Object.FindObjectsByType<BehaviorParameters>(FindObjectsSortMode.None);

            foreach (var bp in allBehaviors)
            {
                if (bp.BehaviorName == "E2EDrivingAgent")
                {
                    // Access the serialized object to modify the brain parameters
                    SerializedObject so = new SerializedObject(bp);
                    SerializedProperty brainParams = so.FindProperty("m_BrainParameters");
                    SerializedProperty vectorObsSize = brainParams.FindPropertyRelative("VectorObservationSize");

                    if (vectorObsSize.intValue != 254)
                    {
                        vectorObsSize.intValue = 254;
                        so.ApplyModifiedProperties();
                        EditorUtility.SetDirty(bp);
                        updated++;
                        Debug.Log($"[UpdateAgentObservationSize] Updated {bp.gameObject.name} to 254D");
                    }
                }
            }

            if (updated > 0)
            {
                // Mark scene dirty
                UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(
                    UnityEditor.SceneManagement.EditorSceneManager.GetActiveScene());
                Debug.Log($"[UpdateAgentObservationSize] Updated {updated} agents to 254D observation space.");
            }
            else
            {
                Debug.Log("[UpdateAgentObservationSize] All agents already at 254D or no E2EDrivingAgent found.");
            }
        }

        [MenuItem("AD Platform/Check Observation Size")]
        public static void CheckObservationSize()
        {
            BehaviorParameters[] allBehaviors = Object.FindObjectsByType<BehaviorParameters>(FindObjectsSortMode.None);

            foreach (var bp in allBehaviors)
            {
                if (bp.BehaviorName == "E2EDrivingAgent")
                {
                    SerializedObject so = new SerializedObject(bp);
                    SerializedProperty brainParams = so.FindProperty("m_BrainParameters");
                    SerializedProperty vectorObsSize = brainParams.FindPropertyRelative("VectorObservationSize");

                    Debug.Log($"[CheckObservationSize] {bp.gameObject.name}: VectorObservationSize = {vectorObsSize.intValue}");
                }
            }
        }
    }
}
