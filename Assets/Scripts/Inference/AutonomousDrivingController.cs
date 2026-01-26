using UnityEngine;
using ADPlatform.Agents;

namespace ADPlatform.Inference
{
    /// <summary>
    /// Autonomous Driving Controller
    ///
    /// Bridges E2EDrivingAgent (observation collection) with
    /// SentisInferenceEngine (ONNX model inference) for deployment.
    ///
    /// Modes:
    ///   - Training: Uses ML-Agents for RL/IL training
    ///   - Inference: Uses Sentis for autonomous driving
    ///   - Manual: Keyboard control for testing
    ///
    /// Pipeline:
    ///   Observations → ONNX Model (Sentis) → Actions → Vehicle Physics
    /// </summary>
    [RequireComponent(typeof(Rigidbody))]
    public class AutonomousDrivingController : MonoBehaviour
    {
        public enum ControlMode
        {
            Training,   // ML-Agents handles actions
            Inference,  // Sentis model drives
            Manual      // Keyboard input
        }

        [Header("Mode")]
        public ControlMode mode = ControlMode.Inference;

        [Header("Components")]
        public SentisInferenceEngine inferenceEngine;

        [Header("Vehicle Physics")]
        public float maxSpeed = 30f;
        public float maxAcceleration = 4f;
        public float maxBraking = 8f;
        public float wheelBase = 2.7f;

        [Header("Observation Settings")]
        public int maxAgents = 20;
        public float detectionRange = 50f;
        public Transform goalTarget;
        public Transform[] routeWaypoints;

        [Header("Safety")]
        [Tooltip("Emergency brake if collision imminent")]
        public bool enableEmergencyBrake = true;
        public float emergencyBrakeDistance = 5f;

        [Header("Debug")]
        public bool showGizmos = true;
        public float currentSteering;
        public float currentAcceleration;
        public float currentSpeed;

        // Internal
        private Rigidbody rb;
        private float[] observations;
        private const int OBS_DIM = 238;

        // History
        private float[][] egoHistory;
        private int historyIndex;
        private int historySteps = 5;
        private Vector3 prevVelocity;
        private Vector3 startPosition;

        void Start()
        {
            rb = GetComponent<Rigidbody>();
            observations = new float[OBS_DIM];
            startPosition = transform.position;

            egoHistory = new float[historySteps][];
            for (int i = 0; i < historySteps; i++)
                egoHistory[i] = new float[8];

            if (inferenceEngine == null)
                inferenceEngine = GetComponent<SentisInferenceEngine>();

            if (mode == ControlMode.Inference && inferenceEngine == null)
            {
                Debug.LogWarning("[ADController] No inference engine found, switching to Manual mode");
                mode = ControlMode.Manual;
            }
        }

        void FixedUpdate()
        {
            if (mode == ControlMode.Training)
                return; // ML-Agents handles everything

            // Collect observations
            CollectObservations();

            // Get actions
            float steering = 0f;
            float acceleration = 0f;

            if (mode == ControlMode.Inference && inferenceEngine != null && inferenceEngine.IsModelLoaded)
            {
                var result = inferenceEngine.RunInference(observations);
                steering = result.steering;
                acceleration = result.acceleration;
            }
            else if (mode == ControlMode.Manual)
            {
                steering = Input.GetAxis("Horizontal") * 0.5f;
                acceleration = Input.GetAxis("Vertical") * maxAcceleration;
            }

            // Safety override
            if (enableEmergencyBrake && IsCollisionImminent())
            {
                acceleration = -maxBraking;
            }

            // Apply physics
            ApplyControl(steering, acceleration);

            // Update debug values
            currentSteering = steering;
            currentAcceleration = acceleration;
            currentSpeed = rb.linearVelocity.magnitude;
        }

        private void CollectObservations()
        {
            System.Array.Clear(observations, 0, OBS_DIM);
            int idx = 0;

            // === EGO STATE (8D) ===
            Vector3 localPos = transform.position - startPosition;
            observations[idx++] = localPos.x / 100f;
            observations[idx++] = localPos.z / 100f;

            Vector3 vel = rb.linearVelocity;
            observations[idx++] = vel.x / maxSpeed;
            observations[idx++] = vel.z / maxSpeed;

            float heading = transform.eulerAngles.y * Mathf.Deg2Rad;
            observations[idx++] = Mathf.Cos(heading);
            observations[idx++] = Mathf.Sin(heading);

            Vector3 accel = (vel - prevVelocity) / Mathf.Max(Time.fixedDeltaTime, 0.001f);
            observations[idx++] = Mathf.Clamp(accel.x / maxAcceleration, -1f, 1f);
            observations[idx++] = Mathf.Clamp(accel.z / maxAcceleration, -1f, 1f);
            prevVelocity = vel;

            // Store history
            System.Array.Copy(observations, 0, egoHistory[historyIndex], 0, 8);
            historyIndex = (historyIndex + 1) % historySteps;

            // === EGO HISTORY (40D) ===
            for (int step = 0; step < historySteps; step++)
            {
                int hIdx = (historyIndex + step) % historySteps;
                System.Array.Copy(egoHistory[hIdx], 0, observations, idx, 8);
                idx += 8;
            }

            // === SURROUNDING AGENTS (160D) ===
            idx = CollectNearbyAgents(idx);

            // === ROUTE (30D) ===
            idx = CollectRoute(idx);
        }

        private int CollectNearbyAgents(int startIdx)
        {
            int idx = startIdx;
            Collider[] nearby = Physics.OverlapSphere(transform.position, detectionRange);

            int agentCount = 0;
            foreach (var col in nearby)
            {
                if (col.gameObject == gameObject) continue;
                if (agentCount >= maxAgents) break;

                Rigidbody otherRb = col.GetComponent<Rigidbody>();
                if (otherRb == null) continue;

                Vector3 relPos = transform.InverseTransformPoint(col.transform.position);
                float dist = Vector3.Distance(transform.position, col.transform.position);

                observations[idx++] = relPos.x / detectionRange;
                observations[idx++] = relPos.z / detectionRange;

                Vector3 relVel = transform.InverseTransformDirection(
                    otherRb.linearVelocity - rb.linearVelocity);
                observations[idx++] = relVel.x / maxSpeed;
                observations[idx++] = relVel.z / maxSpeed;

                float relHeading = (col.transform.eulerAngles.y - transform.eulerAngles.y) * Mathf.Deg2Rad;
                observations[idx++] = Mathf.Cos(relHeading);
                observations[idx++] = Mathf.Sin(relHeading);

                observations[idx++] = otherRb.linearVelocity.magnitude / maxSpeed;
                observations[idx++] = dist / detectionRange;

                agentCount++;
            }

            // Pad remaining
            idx = startIdx + maxAgents * 8;
            return idx;
        }

        private int CollectRoute(int startIdx)
        {
            int idx = startIdx;
            int numWaypoints = 10;

            if (goalTarget != null)
            {
                Vector3 dirToGoal = (goalTarget.position - transform.position).normalized;
                float totalDist = Vector3.Distance(transform.position, goalTarget.position);

                for (int i = 0; i < numWaypoints; i++)
                {
                    float t = (float)(i + 1) / numWaypoints;
                    Vector3 wp = transform.position + dirToGoal * (totalDist * t);
                    Vector3 relPos = transform.InverseTransformPoint(wp);

                    observations[idx++] = relPos.x / 100f;
                    observations[idx++] = relPos.z / 100f;
                    observations[idx++] = (totalDist * t) / 500f;
                }
            }
            else
            {
                for (int i = 0; i < numWaypoints; i++)
                {
                    observations[idx++] = 0f;
                    observations[idx++] = (i + 1) * 10f / 100f;
                    observations[idx++] = (i + 1) * 10f / 500f;
                }
            }

            return idx;
        }

        private void ApplyControl(float steering, float acceleration)
        {
            float speed = rb.linearVelocity.magnitude;

            // Acceleration
            Vector3 force = transform.forward * acceleration * rb.mass;
            rb.AddForce(force);

            // Steering (bicycle model)
            if (speed > 0.5f)
            {
                float turnRadius = wheelBase / Mathf.Tan(Mathf.Abs(steering) + 0.001f);
                float angularVel = speed / turnRadius * Mathf.Sign(steering);
                float turnAngle = angularVel * Time.fixedDeltaTime * Mathf.Rad2Deg;
                transform.Rotate(0f, turnAngle, 0f);
            }

            // Speed limiting
            if (rb.linearVelocity.magnitude > maxSpeed)
            {
                rb.linearVelocity = rb.linearVelocity.normalized * maxSpeed;
            }
        }

        private bool IsCollisionImminent()
        {
            float speed = rb.linearVelocity.magnitude;
            if (speed < 1f) return false;

            return Physics.Raycast(transform.position + Vector3.up * 0.5f,
                                   transform.forward,
                                   emergencyBrakeDistance);
        }

        void OnDrawGizmos()
        {
            if (!showGizmos) return;

            // Draw detection range
            Gizmos.color = new Color(0, 1, 0, 0.1f);
            Gizmos.DrawWireSphere(transform.position, detectionRange);

            // Draw goal direction
            if (goalTarget != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawLine(transform.position, goalTarget.position);
            }

            // Draw forward direction with steering
            Gizmos.color = Color.blue;
            Vector3 steerDir = Quaternion.Euler(0, currentSteering * Mathf.Rad2Deg * 60f, 0) * transform.forward;
            Gizmos.DrawRay(transform.position, steerDir * 10f);
        }
    }
}
