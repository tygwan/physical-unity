using UnityEditor;
using UnityEngine;
using ADPlatform.Agents;
using System.Reflection;
using System.Collections.Generic;
using Unity.MLAgents;

/// <summary>
/// Editor utility for batch-swapping agent components across all Training Areas.
/// 
/// Usage:
///   Tools > Phase B v2 > Swap All → E2EDrivingAgentBv2   (for Phase B v2 training)
///   Tools > Phase B v2 > Swap All → E2EDrivingAgent       (restore original)
///
/// Copies all shared serialized fields (environment refs, layer masks, physics settings).
/// Reward values are overridden by Initialize() at runtime, so Inspector values
/// for rewards will show v2 defaults after swap.
///
/// Supports Undo (Ctrl+Z to revert).
/// </summary>
public class AgentSwapUtility
{
    [MenuItem("Tools/Phase B v2/Swap All → E2EDrivingAgentBv2")]
    static void SwapToV2()
    {
        SwapAgents<E2EDrivingAgent, E2EDrivingAgentBv2>();
    }

    [MenuItem("Tools/Phase B v2/Swap All → E2EDrivingAgent (Restore)")]
    static void SwapToOriginal()
    {
        SwapAgents<E2EDrivingAgentBv2, E2EDrivingAgent>();
    }

    // Silent versions for MCP/automation (no confirmation dialog)
    [MenuItem("Tools/Phase B v2/Auto Swap → Bv2 (Silent)")]
    static void SwapToV2Silent()
    {
        SwapAgentsSilent<E2EDrivingAgent, E2EDrivingAgentBv2>();
    }

    [MenuItem("Tools/Phase B v2/Auto Restore → Original (Silent)")]
    static void SwapToOriginalSilent()
    {
        SwapAgentsSilent<E2EDrivingAgentBv2, E2EDrivingAgent>();
    }

    [MenuItem("Tools/Phase B v2/Count Agents in Scene")]
    static void CountAgents()
    {
        var original = Object.FindObjectsByType<E2EDrivingAgent>(FindObjectsSortMode.None);
        var v2 = Object.FindObjectsByType<E2EDrivingAgentBv2>(FindObjectsSortMode.None);
        EditorUtility.DisplayDialog("Agent Count",
            $"E2EDrivingAgent: {original.Length}\nE2EDrivingAgentBv2: {v2.Length}\nTotal: {original.Length + v2.Length}",
            "OK");
    }

    static void SwapAgents<TFrom, TTo>()
        where TFrom : Agent
        where TTo : Agent
    {
        var agents = Object.FindObjectsByType<TFrom>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (agents.Length == 0)
        {
            EditorUtility.DisplayDialog("Agent Swap",
                $"No {typeof(TFrom).Name} found in scene.", "OK");
            return;
        }

        if (!EditorUtility.DisplayDialog("Agent Swap",
            $"Swap {agents.Length} agents:\n{typeof(TFrom).Name} → {typeof(TTo).Name}\n\nThis supports Undo (Ctrl+Z).",
            "Swap All", "Cancel"))
            return;

        Undo.SetCurrentGroupName($"Swap to {typeof(TTo).Name}");
        int undoGroup = Undo.GetCurrentGroup();
        int swapped = 0;
        int errors = 0;

        foreach (var agent in agents)
        {
            try
            {
                var go = agent.gameObject;

                // Capture all public instance fields
                var fromFields = typeof(TFrom).GetFields(BindingFlags.Public | BindingFlags.Instance);
                var capturedValues = new Dictionary<string, object>();

                foreach (var field in fromFields)
                {
                    // Skip the trainingVersion enum (different enum types between classes)
                    if (field.Name == "trainingVersion") continue;

                    try
                    {
                        capturedValues[field.Name] = field.GetValue(agent);
                    }
                    catch (System.Exception)
                    {
                        // Skip fields that can't be read
                    }
                }

                // Remove old component (with Undo support)
                Undo.DestroyObjectImmediate(agent);

                // Add new component (with Undo support)
                var newAgent = Undo.AddComponent<TTo>(go);

                // Copy matching fields
                var toFields = typeof(TTo).GetFields(BindingFlags.Public | BindingFlags.Instance);
                foreach (var field in toFields)
                {
                    if (field.Name == "trainingVersion") continue;

                    if (capturedValues.TryGetValue(field.Name, out var val))
                    {
                        try
                        {
                            if (val == null) continue;
                            if (field.FieldType.IsAssignableFrom(val.GetType()))
                            {
                                field.SetValue(newAgent, val);
                            }
                        }
                        catch (System.Exception)
                        {
                            // Skip incompatible fields silently
                        }
                    }
                }

                EditorUtility.SetDirty(go);
                swapped++;
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[AgentSwap] Failed to swap agent on {agent.gameObject.name}: {ex.Message}");
                errors++;
            }
        }

        Undo.CollapseUndoOperations(undoGroup);

        string resultMsg = $"Swapped {swapped}/{agents.Length} agents to {typeof(TTo).Name}.";
        if (errors > 0) resultMsg += $"\n{errors} errors (see Console).";
        resultMsg += "\n\nRemember to save the scene (Ctrl+S).";

        Debug.Log($"[AgentSwap] {resultMsg}");
        EditorUtility.DisplayDialog("Agent Swap Complete", resultMsg, "OK");
    }

    /// <summary>
    /// Silent swap without confirmation dialogs (for MCP/automation).
    /// Logs results to console only.
    /// </summary>
    static void SwapAgentsSilent<TFrom, TTo>()
        where TFrom : Agent
        where TTo : Agent
    {
        var agents = Object.FindObjectsByType<TFrom>(FindObjectsInactive.Include, FindObjectsSortMode.None);
        if (agents.Length == 0)
        {
            Debug.Log($"[AgentSwap] No {typeof(TFrom).Name} found in scene. Nothing to swap.");
            return;
        }

        Undo.SetCurrentGroupName($"Silent swap to {typeof(TTo).Name}");
        int undoGroup = Undo.GetCurrentGroup();
        int swapped = 0;
        int errors = 0;

        foreach (var agent in agents)
        {
            try
            {
                var go = agent.gameObject;

                var fromFields = typeof(TFrom).GetFields(BindingFlags.Public | BindingFlags.Instance);
                var capturedValues = new Dictionary<string, object>();
                foreach (var field in fromFields)
                {
                    if (field.Name == "trainingVersion") continue;
                    try { capturedValues[field.Name] = field.GetValue(agent); }
                    catch { }
                }

                Undo.DestroyObjectImmediate(agent);
                var newAgent = Undo.AddComponent<TTo>(go);

                var toFields = typeof(TTo).GetFields(BindingFlags.Public | BindingFlags.Instance);
                foreach (var field in toFields)
                {
                    if (field.Name == "trainingVersion") continue;
                    if (capturedValues.TryGetValue(field.Name, out var val))
                    {
                        try
                        {
                            if (val != null && field.FieldType.IsAssignableFrom(val.GetType()))
                                field.SetValue(newAgent, val);
                        }
                        catch { }
                    }
                }

                EditorUtility.SetDirty(go);
                swapped++;
            }
            catch (System.Exception ex)
            {
                Debug.LogError($"[AgentSwap] Failed on {agent.gameObject.name}: {ex.Message}");
                errors++;
            }
        }

        Undo.CollapseUndoOperations(undoGroup);
        Debug.Log($"[AgentSwap] Silent swap complete: {swapped}/{agents.Length} → {typeof(TTo).Name}. Errors: {errors}");
    }
}