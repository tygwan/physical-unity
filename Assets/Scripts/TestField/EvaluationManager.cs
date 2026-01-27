using UnityEngine;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using ADPlatform.Agents;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Failure types for detailed analysis
    /// </summary>
    public enum FailureType
    {
        None,
        Collision,
        OffRoad,
        WrongWay,
        Timeout,
        StuckBehindNPC,
        SpeedViolation,
        LaneViolation,
        MissedCheckpoint
    }

    /// <summary>
    /// Single evaluation run result
    /// </summary>
    [System.Serializable]
    public class EvaluationResult
    {
        public int runId;
        public int seed;
        public string modelName;
        public TestFieldDifficulty difficulty;

        // Success metrics
        public bool completed;
        public float completionRate;        // 0-1
        public float totalDistance;         // meters
        public float totalTime;             // seconds

        // Safety metrics
        public int collisionCount;
        public int offRoadCount;
        public int wrongWayCount;
        public float nearMissCount;         // TTC < 2s

        // Efficiency metrics
        public float averageSpeed;          // m/s
        public float speedComplianceRate;   // 0-1 (time within limit)
        public float overtakeCount;

        // Comfort metrics
        public float maxJerk;               // m/s^3
        public float averageJerk;
        public float maxLateralAccel;       // m/s^2

        // Failure analysis
        public FailureType failureType = FailureType.None;
        public Vector3 failurePosition;
        public string failureContext;
        public int failureSegmentIndex;

        // Checkpoint tracking
        public int checkpointsReached;
        public int totalCheckpoints;
        public List<float> checkpointTimes = new List<float>();

        // Score calculation
        public float ComputeScore()
        {
            float score = 0f;

            // Completion (40%)
            score += completionRate * 40f;

            // Safety (30%)
            float safetyPenalty = (collisionCount * 10f + offRoadCount * 5f + wrongWayCount * 5f);
            score += Mathf.Max(0f, 30f - safetyPenalty);

            // Efficiency (20%)
            score += speedComplianceRate * 20f;

            // Comfort (10%)
            float comfortScore = 10f;
            if (averageJerk > 2f) comfortScore -= 5f;
            if (maxLateralAccel > 4f) comfortScore -= 5f;
            score += Mathf.Max(0f, comfortScore);

            return score;
        }
    }

    /// <summary>
    /// Aggregated statistics across multiple runs
    /// </summary>
    [System.Serializable]
    public class EvaluationStatistics
    {
        public string modelName;
        public int totalRuns;
        public TestFieldDifficulty difficulty;

        public float successRate;
        public float avgCompletionRate;
        public float avgScore;

        public float avgCollisions;
        public float avgOffRoad;
        public float avgSpeedCompliance;
        public float avgComfort;

        // Failure breakdown
        public Dictionary<FailureType, int> failureCounts = new Dictionary<FailureType, int>();
        public Dictionary<SegmentType, int> failuresBySegment = new Dictionary<SegmentType, int>();

        // Best/worst runs
        public int bestRunId;
        public float bestScore;
        public int worstRunId;
        public float worstScore;
    }

    /// <summary>
    /// Manages evaluation runs and collects metrics
    /// </summary>
    public class EvaluationManager : MonoBehaviour
    {
        [Header("References")]
        public TestFieldGenerator fieldGenerator;
        public E2EDrivingAgent agent;
        public NPCVehicleController[] npcPrefabs;

        [Header("Evaluation Settings")]
        public string modelName = "PhaseE_v12";
        public int numRuns = 10;
        public float maxEpisodeTime = 300f;     // 5 minutes max
        public bool autoAdvance = true;

        [Header("Current Run")]
        public int currentRunIndex = 0;
        public EvaluationResult currentResult;
        public bool isRunning = false;

        [Header("Results")]
        public List<EvaluationResult> allResults = new List<EvaluationResult>();
        public EvaluationStatistics statistics;

        [Header("Output")]
        public string outputDirectory = "Evaluations";
        public bool saveResults = true;
        public bool generateReport = true;

        // Runtime state
        private float runStartTime;
        private int lastCheckpointIndex;
        private Vector3 lastPosition;
        private float lastSpeed;
        private List<float> jerkSamples = new List<float>();
        private List<float> lateralAccelSamples = new List<float>();
        private List<NPCVehicleController> spawnedNPCs = new List<NPCVehicleController>();

        void Start()
        {
            if (fieldGenerator == null)
                fieldGenerator = FindObjectOfType<TestFieldGenerator>();
            if (agent == null)
                agent = FindObjectOfType<E2EDrivingAgent>();
        }

        /// <summary>
        /// Start evaluation batch
        /// </summary>
        public void StartEvaluation()
        {
            allResults.Clear();
            currentRunIndex = 0;
            StartNextRun();
        }

        /// <summary>
        /// Start a single evaluation run
        /// </summary>
        public void StartNextRun()
        {
            if (currentRunIndex >= numRuns)
            {
                FinishEvaluation();
                return;
            }

            isRunning = true;

            // Generate new test field with unique seed
            int seed = fieldGenerator.config.seed + currentRunIndex;
            var field = fieldGenerator.Generate(seed);

            // Initialize result
            currentResult = new EvaluationResult
            {
                runId = currentRunIndex,
                seed = seed,
                modelName = modelName,
                difficulty = fieldGenerator.config.difficulty,
                totalCheckpoints = field.checkpoints.Count
            };

            // Setup agent
            SetupAgent(field);

            // Spawn NPCs
            SpawnNPCs(field);

            // Reset tracking
            runStartTime = Time.time;
            lastCheckpointIndex = 0;
            lastPosition = agent.transform.position;
            lastSpeed = 0f;
            jerkSamples.Clear();
            lateralAccelSamples.Clear();

            Debug.Log($"[EvaluationManager] Starting run {currentRunIndex + 1}/{numRuns}, seed={seed}");
        }

        private void SetupAgent(TestFieldData field)
        {
            // Reset agent position
            agent.transform.position = field.allWaypoints[0];
            agent.transform.rotation = Quaternion.identity;

            // Connect waypoints
            agent.routeWaypoints = field.allWaypoints.Select(wp =>
            {
                var go = new GameObject($"WP_{field.allWaypoints.IndexOf(wp)}");
                go.transform.position = wp;
                go.transform.SetParent(transform);
                return go.transform;
            }).ToArray();

            // Set goal
            if (field.checkpoints.Count > 0)
            {
                var goalGO = new GameObject("Goal");
                goalGO.transform.position = field.checkpoints[field.checkpoints.Count - 1];
                goalGO.transform.SetParent(transform);
                agent.goalTarget = goalGO.transform;
            }
        }

        private void SpawnNPCs(TestFieldData field)
        {
            // Clean up old NPCs
            foreach (var npc in spawnedNPCs)
            {
                if (npc != null) Destroy(npc.gameObject);
            }
            spawnedNPCs.Clear();

            if (npcPrefabs == null || npcPrefabs.Length == 0) return;

            foreach (var spawnPoint in field.npcSpawnPoints)
            {
                var prefab = npcPrefabs[Random.Range(0, npcPrefabs.Length)];
                var npc = Instantiate(prefab, spawnPoint, Quaternion.identity, transform);

                // Set speed based on zone
                float speedLimit = 16.67f;  // Default 60 km/h
                npc.SpawnAt(spawnPoint, Quaternion.identity, speedLimit * Random.Range(0.4f, 0.9f));

                spawnedNPCs.Add(npc);
            }
        }

        void FixedUpdate()
        {
            if (!isRunning || agent == null) return;

            // Update metrics
            UpdateMetrics();

            // Check termination conditions
            CheckTermination();
        }

        private void UpdateMetrics()
        {
            float dt = Time.fixedDeltaTime;
            Vector3 currentPos = agent.transform.position;
            float currentSpeed = agent.GetCurrentSpeed();

            // Distance
            currentResult.totalDistance += Vector3.Distance(currentPos, lastPosition);

            // Time
            currentResult.totalTime = Time.time - runStartTime;

            // Speed tracking
            currentResult.averageSpeed = currentResult.totalDistance / Mathf.Max(currentResult.totalTime, 0.1f);

            // Jerk calculation (rate of acceleration change)
            float accel = (currentSpeed - lastSpeed) / dt;
            if (jerkSamples.Count > 0)
            {
                float jerk = Mathf.Abs(accel - (jerkSamples.Count > 0 ? jerkSamples[jerkSamples.Count - 1] : 0f)) / dt;
                jerkSamples.Add(jerk);
                currentResult.maxJerk = Mathf.Max(currentResult.maxJerk, jerk);
            }
            jerkSamples.Add(accel);

            // Lateral acceleration
            Vector3 velocity = (currentPos - lastPosition) / dt;
            if (velocity.magnitude > 0.1f)
            {
                Vector3 forward = agent.transform.forward;
                float latAccel = Mathf.Abs(Vector3.Dot(velocity, agent.transform.right));
                lateralAccelSamples.Add(latAccel);
                currentResult.maxLateralAccel = Mathf.Max(currentResult.maxLateralAccel, latAccel);
            }

            // Checkpoint tracking
            if (fieldGenerator.generatedField != null)
            {
                var checkpoints = fieldGenerator.generatedField.checkpoints;
                while (lastCheckpointIndex < checkpoints.Count)
                {
                    if (Vector3.Distance(currentPos, checkpoints[lastCheckpointIndex]) < 10f)
                    {
                        currentResult.checkpointsReached++;
                        currentResult.checkpointTimes.Add(currentResult.totalTime);
                        lastCheckpointIndex++;
                        Debug.Log($"[Eval] Checkpoint {currentResult.checkpointsReached}/{currentResult.totalCheckpoints}");
                    }
                    else break;
                }
            }

            // Completion rate
            if (fieldGenerator.generatedField != null)
            {
                currentResult.completionRate = currentResult.totalDistance / fieldGenerator.generatedField.totalArcLength;
            }

            lastPosition = currentPos;
            lastSpeed = currentSpeed;
        }

        private void CheckTermination()
        {
            // Timeout
            if (currentResult.totalTime > maxEpisodeTime)
            {
                EndRun(FailureType.Timeout, "Time limit exceeded");
                return;
            }

            // Goal reached
            if (currentResult.completionRate >= 0.99f)
            {
                currentResult.completed = true;
                EndRun(FailureType.None, "Goal reached!");
                return;
            }

            // All checkpoints reached
            if (currentResult.checkpointsReached >= currentResult.totalCheckpoints)
            {
                currentResult.completed = true;
                EndRun(FailureType.None, "All checkpoints reached!");
            }
        }

        /// <summary>
        /// Called by agent when collision occurs
        /// </summary>
        public void OnAgentCollision(Vector3 position)
        {
            currentResult.collisionCount++;
            if (currentResult.collisionCount >= 3)
            {
                EndRun(FailureType.Collision, $"Too many collisions ({currentResult.collisionCount})");
            }
        }

        /// <summary>
        /// Called by agent when going off-road
        /// </summary>
        public void OnAgentOffRoad(Vector3 position)
        {
            currentResult.offRoadCount++;
            EndRun(FailureType.OffRoad, "Went off-road");
        }

        /// <summary>
        /// Called by agent when driving wrong way
        /// </summary>
        public void OnAgentWrongWay(Vector3 position)
        {
            currentResult.wrongWayCount++;
            EndRun(FailureType.WrongWay, "Wrong-way driving");
        }

        private void EndRun(FailureType failure, string context)
        {
            isRunning = false;

            currentResult.failureType = failure;
            currentResult.failureContext = context;
            currentResult.failurePosition = agent.transform.position;

            // Find failure segment
            if (fieldGenerator.generatedField != null)
            {
                currentResult.failureSegmentIndex = FindNearestSegment(agent.transform.position);
            }

            // Calculate averages
            if (jerkSamples.Count > 0)
                currentResult.averageJerk = jerkSamples.Average();
            if (lateralAccelSamples.Count > 0)
                currentResult.maxLateralAccel = lateralAccelSamples.Max();

            allResults.Add(currentResult);

            Debug.Log($"[EvaluationManager] Run {currentRunIndex + 1} ended: {failure} - {context}, " +
                     $"completion={currentResult.completionRate:P1}, score={currentResult.ComputeScore():F1}");

            // Auto advance
            currentRunIndex++;
            if (autoAdvance && currentRunIndex < numRuns)
            {
                Invoke(nameof(StartNextRun), 2f);  // 2 second delay between runs
            }
            else if (currentRunIndex >= numRuns)
            {
                FinishEvaluation();
            }
        }

        private int FindNearestSegment(Vector3 position)
        {
            if (fieldGenerator.generatedField == null) return -1;

            float minDist = float.MaxValue;
            int nearestIdx = 0;

            for (int i = 0; i < fieldGenerator.generatedField.segments.Count; i++)
            {
                foreach (var wp in fieldGenerator.generatedField.segments[i].waypoints)
                {
                    float dist = Vector3.Distance(position, wp);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        nearestIdx = i;
                    }
                }
            }

            return nearestIdx;
        }

        private void FinishEvaluation()
        {
            Debug.Log($"[EvaluationManager] Evaluation complete! {allResults.Count} runs");

            // Compute statistics
            ComputeStatistics();

            // Save results
            if (saveResults)
                SaveResults();

            // Generate report
            if (generateReport)
                GenerateReport();
        }

        private void ComputeStatistics()
        {
            statistics = new EvaluationStatistics
            {
                modelName = modelName,
                totalRuns = allResults.Count,
                difficulty = fieldGenerator.config.difficulty
            };

            if (allResults.Count == 0) return;

            // Success rate
            statistics.successRate = allResults.Count(r => r.completed) / (float)allResults.Count;

            // Averages
            statistics.avgCompletionRate = allResults.Average(r => r.completionRate);
            statistics.avgScore = allResults.Average(r => r.ComputeScore());
            statistics.avgCollisions = (float)allResults.Average(r => r.collisionCount);
            statistics.avgOffRoad = (float)allResults.Average(r => r.offRoadCount);
            statistics.avgSpeedCompliance = allResults.Average(r => r.speedComplianceRate);

            // Failure breakdown
            foreach (FailureType ft in System.Enum.GetValues(typeof(FailureType)))
            {
                statistics.failureCounts[ft] = allResults.Count(r => r.failureType == ft);
            }

            // Failure by segment type
            if (fieldGenerator.generatedField != null)
            {
                foreach (var result in allResults.Where(r => r.failureType != FailureType.None))
                {
                    if (result.failureSegmentIndex >= 0 &&
                        result.failureSegmentIndex < fieldGenerator.generatedField.segments.Count)
                    {
                        var segType = fieldGenerator.generatedField.segments[result.failureSegmentIndex].config.type;
                        if (!statistics.failuresBySegment.ContainsKey(segType))
                            statistics.failuresBySegment[segType] = 0;
                        statistics.failuresBySegment[segType]++;
                    }
                }
            }

            // Best/worst
            var sorted = allResults.OrderByDescending(r => r.ComputeScore()).ToList();
            if (sorted.Count > 0)
            {
                statistics.bestRunId = sorted[0].runId;
                statistics.bestScore = sorted[0].ComputeScore();
                statistics.worstRunId = sorted[sorted.Count - 1].runId;
                statistics.worstScore = sorted[sorted.Count - 1].ComputeScore();
            }
        }

        private void SaveResults()
        {
            string dir = Path.Combine(Application.dataPath, "..", outputDirectory);
            Directory.CreateDirectory(dir);

            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string filename = $"{modelName}_{fieldGenerator.config.difficulty}_{timestamp}.json";
            string filepath = Path.Combine(dir, filename);

            var data = new
            {
                model = modelName,
                difficulty = fieldGenerator.config.difficulty.ToString(),
                statistics = statistics,
                runs = allResults
            };

            string json = JsonUtility.ToJson(data, true);
            File.WriteAllText(filepath, json);

            Debug.Log($"[EvaluationManager] Results saved to: {filepath}");
        }

        private void GenerateReport()
        {
            string report = $@"
================================================================================
                     EVALUATION REPORT: {modelName}
================================================================================

Test Configuration:
  - Difficulty: {fieldGenerator.config.difficulty}
  - Total Runs: {statistics.totalRuns}
  - Field Length: {fieldGenerator.config.totalLength}m

--------------------------------------------------------------------------------
                           SUMMARY STATISTICS
--------------------------------------------------------------------------------

  Success Rate:       {statistics.successRate:P1}
  Avg Completion:     {statistics.avgCompletionRate:P1}
  Avg Score:          {statistics.avgScore:F1} / 100

  Avg Collisions:     {statistics.avgCollisions:F2} per run
  Avg Off-road:       {statistics.avgOffRoad:F2} per run
  Speed Compliance:   {statistics.avgSpeedCompliance:P1}

--------------------------------------------------------------------------------
                           FAILURE ANALYSIS
--------------------------------------------------------------------------------

By Type:
";
            foreach (var kv in statistics.failureCounts.Where(kv => kv.Value > 0))
            {
                report += $"  - {kv.Key}: {kv.Value} ({kv.Value * 100f / statistics.totalRuns:F1}%)\n";
            }

            report += "\nBy Segment Type:\n";
            foreach (var kv in statistics.failuresBySegment.OrderByDescending(kv => kv.Value))
            {
                report += $"  - {kv.Key}: {kv.Value} failures\n";
            }

            report += $@"
--------------------------------------------------------------------------------
                           BEST / WORST RUNS
--------------------------------------------------------------------------------

  Best:  Run #{statistics.bestRunId + 1}, Score: {statistics.bestScore:F1}
  Worst: Run #{statistics.worstRunId + 1}, Score: {statistics.worstScore:F1}

================================================================================
";

            Debug.Log(report);

            // Save report
            string dir = Path.Combine(Application.dataPath, "..", outputDirectory);
            string timestamp = System.DateTime.Now.ToString("yyyyMMdd_HHmmss");
            string filepath = Path.Combine(dir, $"report_{modelName}_{timestamp}.txt");
            File.WriteAllText(filepath, report);
        }

        /// <summary>
        /// Quick test with single run
        /// </summary>
        [ContextMenu("Quick Test (1 Run)")]
        public void QuickTest()
        {
            numRuns = 1;
            StartEvaluation();
        }

        /// <summary>
        /// Full benchmark with 10 runs
        /// </summary>
        [ContextMenu("Full Benchmark (10 Runs)")]
        public void FullBenchmark()
        {
            numRuns = 10;
            StartEvaluation();
        }
    }
}
