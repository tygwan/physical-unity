using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using System.IO;

/// <summary>
/// Builds Unity training environments for ML-Agents multi-env training.
/// Use: Build > Build [Phase] Training Environment
///
/// After building, set env_path in YAML config and increase num_envs.
/// Example:
///   env_settings:
///     env_path: Builds/PhaseH/PhaseH.exe
///     num_envs: 3
///     no_graphics: true
/// </summary>
public class BuildHelper
{
    private const string BUILDS_ROOT = "Builds";

    // ====== Phase-specific builds ======

    [MenuItem("Build/Build Phase K (Dense Urban)")]
    public static void BuildPhaseK()
    {
        BuildPhase("PhaseK", "Assets/Scenes/PhaseK_DenseUrban.unity");
    }

    [MenuItem("Build/Build Phase J (Traffic Signals)")]
    public static void BuildPhaseJ()
    {
        BuildPhase("PhaseJ", "Assets/Scenes/PhaseJ_TrafficSignals.unity");
    }

    [MenuItem("Build/Build Phase H (NPC Intersection)")]
    public static void BuildPhaseH()
    {
        BuildPhase("PhaseH", "Assets/Scenes/PhaseH_NPCIntersection.unity");
    }

    [MenuItem("Build/Build Phase G (Intersection)")]
    public static void BuildPhaseG()
    {
        BuildPhase("PhaseG", "Assets/Scenes/PhaseG_Intersection.unity");
    }

    [MenuItem("Build/Build Phase F (Multi-Lane)")]
    public static void BuildPhaseF()
    {
        BuildPhase("PhaseF", "Assets/Scenes/PhaseF_MultiLane.unity");
    }

    // ====== Active Scene Build ======

    [MenuItem("Build/Build Active Scene")]
    public static void BuildActiveScene()
    {
        var activeScene = UnityEngine.SceneManagement.SceneManager.GetActiveScene();
        if (string.IsNullOrEmpty(activeScene.path))
        {
            Debug.LogError("[BuildHelper] No active scene found. Save the scene first.");
            return;
        }

        string sceneName = Path.GetFileNameWithoutExtension(activeScene.path);
        BuildPhase(sceneName, activeScene.path);
    }

    // ====== Core Build Method ======

    private static void BuildPhase(string phaseName, string scenePath)
    {
        string buildDir = Path.Combine(BUILDS_ROOT, phaseName);
        string buildPath = Path.Combine(buildDir, $"{phaseName}.exe");

        if (!Directory.Exists(buildDir))
            Directory.CreateDirectory(buildDir);

        Debug.Log($"[BuildHelper] Building {phaseName}...");
        Debug.Log($"[BuildHelper] Scene: {scenePath}");
        Debug.Log($"[BuildHelper] Output: {Path.GetFullPath(buildPath)}");

        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { scenePath },
            locationPathName = buildPath,
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            long sizeMB = (long)(summary.totalSize / (1024 * 1024));
            Debug.Log($"[BuildHelper] Build SUCCEEDED: {sizeMB} MB");
            Debug.Log($"[BuildHelper] Executable: {Path.GetFullPath(buildPath)}");
            Debug.Log($"[BuildHelper] YAML config:");
            Debug.Log($"  env_path: {buildPath.Replace('\\', '/')}");
            Debug.Log($"  num_envs: 3");

            EditorUtility.DisplayDialog("Build Succeeded",
                $"{phaseName} build completed ({sizeMB} MB)\n\n" +
                $"Path: {buildPath}\n\n" +
                "Update YAML:\n" +
                $"  env_path: {buildPath.Replace('\\', '/')}\n" +
                "  num_envs: 3\n" +
                "  no_graphics: true",
                "OK");
        }
        else
        {
            Debug.LogError($"[BuildHelper] Build FAILED for {phaseName}");
            foreach (var step in report.steps)
            {
                foreach (var message in step.messages)
                {
                    if (message.type == LogType.Error)
                        Debug.LogError($"  {message.content}");
                }
            }

            EditorUtility.DisplayDialog("Build Failed",
                $"{phaseName} build failed. Check Console for errors.", "OK");
        }
    }
}
