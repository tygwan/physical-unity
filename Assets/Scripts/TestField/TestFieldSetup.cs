using UnityEngine;
using ADPlatform.Agents;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Helper script to quickly set up a test field scene
    /// Add to an empty GameObject and use context menu to create the test environment
    /// </summary>
    public class TestFieldSetup : MonoBehaviour
    {
        [Header("Model Selection")]
        [Tooltip("ONNX model file to test (drag from Assets/Resources/Models or models/planning)")]
        public UnityEngine.Object modelFile;

        [Header("Quick Presets")]
        public TestFieldDifficulty difficulty = TestFieldDifficulty.Medium;

        [Header("Generated Components")]
        public TestFieldGenerator fieldGenerator;
        public EvaluationManager evaluationManager;
        public E2EDrivingAgent agent;

        /// <summary>
        /// Create complete test field setup
        /// </summary>
        [ContextMenu("1. Create Test Field Setup")]
        public void CreateTestFieldSetup()
        {
            // Create Test Field Generator
            var generatorGO = new GameObject("TestFieldGenerator");
            generatorGO.transform.SetParent(transform);
            fieldGenerator = generatorGO.AddComponent<TestFieldGenerator>();
            fieldGenerator.config = new TestFieldConfig
            {
                seed = Random.Range(10000, 99999),
                totalLength = 2000f,
                difficulty = difficulty
            };

            // Create Evaluation Manager
            var evalGO = new GameObject("EvaluationManager");
            evalGO.transform.SetParent(transform);
            evaluationManager = evalGO.AddComponent<EvaluationManager>();
            evaluationManager.fieldGenerator = fieldGenerator;

            // Find or create agent
            agent = FindObjectOfType<E2EDrivingAgent>();
            if (agent == null)
            {
                Debug.LogWarning("[TestFieldSetup] No E2EDrivingAgent found in scene. Please add one with your vehicle prefab.");
            }
            else
            {
                evaluationManager.agent = agent;
            }

            Debug.Log("[TestFieldSetup] Test field setup created! Now:\n" +
                     "1. Assign your ONNX model to the agent's BehaviorParameters\n" +
                     "2. Set BehaviorType to 'Inference Only'\n" +
                     "3. Right-click EvaluationManager > 'Full Benchmark' to start");
        }

        /// <summary>
        /// Generate preview of the test field
        /// </summary>
        [ContextMenu("2. Generate Preview")]
        public void GeneratePreview()
        {
            if (fieldGenerator == null)
            {
                Debug.LogError("Please run 'Create Test Field Setup' first!");
                return;
            }

            fieldGenerator.config.difficulty = difficulty;
            fieldGenerator.Generate();
            Debug.Log("[TestFieldSetup] Test field preview generated. Check Scene view with Gizmos enabled.");
        }

        /// <summary>
        /// Run single quick test
        /// </summary>
        [ContextMenu("3. Quick Test (Single Run)")]
        public void QuickTest()
        {
            if (evaluationManager == null)
            {
                Debug.LogError("Please run 'Create Test Field Setup' first!");
                return;
            }

            if (agent == null || !agent.gameObject.activeInHierarchy)
            {
                Debug.LogError("Agent not found or inactive!");
                return;
            }

            evaluationManager.QuickTest();
        }

        /// <summary>
        /// Run full benchmark
        /// </summary>
        [ContextMenu("4. Full Benchmark (10 Runs)")]
        public void FullBenchmark()
        {
            if (evaluationManager == null)
            {
                Debug.LogError("Please run 'Create Test Field Setup' first!");
                return;
            }

            if (agent == null || !agent.gameObject.activeInHierarchy)
            {
                Debug.LogError("Agent not found or inactive!");
                return;
            }

            evaluationManager.FullBenchmark();
        }

        /// <summary>
        /// Compare multiple models
        /// </summary>
        [ContextMenu("5. Compare All Phase Models")]
        public void CompareAllModels()
        {
            string[] modelPaths = new string[]
            {
                "models/planning/E2EDrivingAgent_phaseD_254d.onnx",
                "models/planning/E2EDrivingAgent_phaseE_254d.onnx"
                // Add more as phases complete
            };

            Debug.Log("[TestFieldSetup] Model comparison feature coming soon!\n" +
                     "For now, manually swap models and run benchmarks:\n" +
                     "1. Assign model to BehaviorParameters\n" +
                     "2. Run 'Full Benchmark'\n" +
                     "3. Check Evaluations/ folder for results\n" +
                     "4. Repeat with different model");
        }

        void OnDrawGizmos()
        {
            // Draw setup info in scene
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(transform.position, 5f);

#if UNITY_EDITOR
            UnityEditor.Handles.Label(transform.position + Vector3.up * 6f,
                $"Test Field Setup\nDifficulty: {difficulty}\nModel: {(modelFile != null ? modelFile.name : "Not assigned")}");
#endif
        }
    }
}
