using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;
using System.IO;

public class BuildHelper
{
    private const string BUILD_PATH = "Builds/DrivingEnv/DrivingEnv.exe";
    private const string SCENE_PATH = "Assets/Scenes/DrivingScene.unity/DrivingScene.unity";

    [MenuItem("Build/Build Training Environment")]
    public static void BuildTrainingEnvironment()
    {
        // Ensure build directory exists
        string buildDir = Path.GetDirectoryName(BUILD_PATH);
        if (!Directory.Exists(buildDir))
        {
            Directory.CreateDirectory(buildDir);
        }

        // Build options
        BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions
        {
            scenes = new[] { SCENE_PATH },
            locationPathName = BUILD_PATH,
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.None
        };

        // Build
        BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"Build succeeded: {summary.totalSize / (1024 * 1024)} MB");
            Debug.Log($"Build path: {Path.GetFullPath(BUILD_PATH)}");
        }
        else if (summary.result == BuildResult.Failed)
        {
            Debug.LogError("Build failed!");
            foreach (var step in report.steps)
            {
                foreach (var message in step.messages)
                {
                    if (message.type == LogType.Error)
                        Debug.LogError(message.content);
                }
            }
        }
    }

    [MenuItem("Build/Build Training Environment (Server/Headless)")]
    public static void BuildTrainingEnvironmentHeadless()
    {
        string buildPath = "Builds/DrivingEnvHeadless/DrivingEnv.exe";
        
        // Ensure build directory exists
        string buildDir = Path.GetDirectoryName(buildPath);
        if (!Directory.Exists(buildDir))
        {
            Directory.CreateDirectory(buildDir);
        }

        // Build options with server build (headless)
        BuildPlayerOptions buildPlayerOptions = new BuildPlayerOptions
        {
            scenes = new[] { SCENE_PATH },
            locationPathName = buildPath,
            target = BuildTarget.StandaloneWindows64,
            subtarget = (int)StandaloneBuildSubtarget.Server,
            options = BuildOptions.EnableHeadlessMode
        };

        // Build
        BuildReport report = BuildPipeline.BuildPlayer(buildPlayerOptions);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"Headless build succeeded: {summary.totalSize / (1024 * 1024)} MB");
            Debug.Log($"Build path: {Path.GetFullPath(buildPath)}");
        }
        else if (summary.result == BuildResult.Failed)
        {
            Debug.LogError("Headless build failed!");
        }
    }
}
