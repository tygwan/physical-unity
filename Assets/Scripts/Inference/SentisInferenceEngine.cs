using UnityEngine;

using System.Collections.Generic;

namespace ADPlatform.Inference
{
    /// <summary>
    /// Unity Sentis inference engine for E2E driving model.
    ///
    /// Loads an ONNX model exported from PyTorch and runs inference
    /// to produce steering and acceleration outputs.
    ///
    /// Pipeline:
    ///   PyTorch training → ONNX export → Unity Sentis inference → Vehicle control
    ///
    /// Usage:
    ///   1. Place .onnx model in Assets/Resources/Models/
    ///   2. Attach this component to the vehicle GameObject
    ///   3. Assign the E2EDrivingAgent component reference
    ///   4. Set modelAsset to the ONNX model
    /// </summary>
    public class SentisInferenceEngine : MonoBehaviour
    {
        [Header("Model")]
        [Tooltip("ONNX model asset exported from PyTorch E2E model")]
        public Unity.InferenceEngine.ModelAsset modelAsset;

        [Tooltip("Backend type for inference")]
        public Unity.InferenceEngine.BackendType backendType = Unity.InferenceEngine.BackendType.GPUCompute;

        [Header("Inference Settings")]
        [Tooltip("Run inference every N fixed updates")]
        public int inferenceInterval = 1;

        [Tooltip("Enable async inference for non-blocking execution")]
        public bool asyncInference = false;

        [Header("Output")]
        [Tooltip("Steering output [-0.5, 0.5] rad")]
        public float currentSteering;

        [Tooltip("Acceleration output [-4.0, 2.0] m/s²")]
        public float currentAcceleration;

        [Header("Debug")]
        public bool showDebugInfo = false;
        public float inferenceTimeMs;

        // Internal
        private Unity.InferenceEngine.Model runtimeModel;
        private Unity.InferenceEngine.Worker worker;
        private bool isModelLoaded = false;
        private int frameCount = 0;

        // Observation buffer
        private float[] observationBuffer;
        private const int OBS_DIM = 238;

        void Start()
        {
            LoadModel();
            observationBuffer = new float[OBS_DIM];
        }

        void OnDestroy()
        {
            DisposeWorker();
        }

        /// <summary>
        /// Load ONNX model and create inference worker
        /// </summary>
        public void LoadModel()
        {
            if (modelAsset == null)
            {
                Debug.LogWarning("[SentisInference] No model asset assigned");
                return;
            }

            try
            {
                runtimeModel = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
                worker = new Unity.InferenceEngine.Worker(runtimeModel, backendType);
                isModelLoaded = true;

                Debug.Log($"[SentisInference] Model loaded successfully");
                Debug.Log($"  Backend: {backendType}");
                Debug.Log($"  Inputs: {runtimeModel.inputs.Count}");
                Debug.Log($"  Outputs: {runtimeModel.outputs.Count}");

                // Log input/output details
                foreach (var input in runtimeModel.inputs)
                {
                    Debug.Log($"  Input: {input.name}, shape: {input.shape}");
                }
                foreach (var output in runtimeModel.outputs)
                {
                    Debug.Log($"  Output: {output.name}");
                }
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[SentisInference] Failed to load model: {e.Message}");
                isModelLoaded = false;
            }
        }

        /// <summary>
        /// Run inference with the given observation vector
        /// </summary>
        /// <param name="observations">238D observation vector</param>
        /// <returns>(steering, acceleration) tuple</returns>
        public (float steering, float acceleration) RunInference(float[] observations)
        {
            if (!isModelLoaded || worker == null)
            {
                return (0f, 0f);
            }

            float startTime = Time.realtimeSinceStartup;

            // Create input tensor [1, 238]
            using var inputTensor = new Unity.InferenceEngine.Tensor<float>(new Unity.InferenceEngine.TensorShape(1, OBS_DIM), observations);

            // Execute inference
            worker.Schedule(inputTensor);

            // Get outputs
            var steeringTensor = worker.PeekOutput("steering") as Unity.InferenceEngine.Tensor<float>;
            var accelerationTensor = worker.PeekOutput("acceleration") as Unity.InferenceEngine.Tensor<float>;

            if (steeringTensor != null && accelerationTensor != null)
            {
                // Download results
                steeringTensor.ReadbackRequest();
                accelerationTensor.ReadbackRequest();

                steeringTensor.ReadbackAndClone();
                accelerationTensor.ReadbackAndClone();

                currentSteering = steeringTensor[0];
                currentAcceleration = accelerationTensor[0];
            }

            inferenceTimeMs = (Time.realtimeSinceStartup - startTime) * 1000f;

            if (showDebugInfo)
            {
                Debug.Log($"[SentisInference] Steer={currentSteering:F4}, " +
                         $"Accel={currentAcceleration:F4}, " +
                         $"Time={inferenceTimeMs:F2}ms");
            }

            return (currentSteering, currentAcceleration);
        }

        /// <summary>
        /// Set observation buffer from E2EDrivingAgent's observation data
        /// </summary>
        public void SetObservations(float[] obs)
        {
            if (obs.Length != OBS_DIM)
            {
                Debug.LogWarning($"[SentisInference] Expected {OBS_DIM}D obs, got {obs.Length}D");
                return;
            }
            System.Array.Copy(obs, observationBuffer, OBS_DIM);
        }

        /// <summary>
        /// Run inference using the internal observation buffer
        /// </summary>
        public (float steering, float acceleration) Infer()
        {
            return RunInference(observationBuffer);
        }

        /// <summary>
        /// Load model from a file path at runtime
        /// </summary>
        public void LoadModelFromPath(string path)
        {
            try
            {
                DisposeWorker();
                runtimeModel = Unity.InferenceEngine.ModelLoader.Load(path);
                worker = new Unity.InferenceEngine.Worker(runtimeModel, backendType);
                isModelLoaded = true;
                Debug.Log($"[SentisInference] Model loaded from: {path}");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[SentisInference] Failed to load model from path: {e.Message}");
                isModelLoaded = false;
            }
        }

        public bool IsModelLoaded => isModelLoaded;

        private void DisposeWorker()
        {
            if (worker != null)
            {
                worker.Dispose();
                worker = null;
            }
            isModelLoaded = false;
        }
    }
}
