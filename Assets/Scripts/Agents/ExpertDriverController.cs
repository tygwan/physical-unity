using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Demonstrations;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Expert driver controller for recording demonstrations.
    /// Follows waypoints using PID-like control to produce smooth expert trajectories.
    /// 
    /// Usage:
    ///   1. Add DemonstrationRecorder component to Vehicle
    ///   2. Set BehaviorParameters to "Heuristic Only"
    ///   3. Enter Play mode - expert drives automatically
    ///   4. Demonstrations saved to Assets/Demonstrations/
    ///
    /// The recorded demonstrations can be used for:
    ///   - GAIL training (vehicle_gail.yaml)
    ///   - Hybrid BC+RL training (vehicle_hybrid.yaml)
    /// </summary>
    [RequireComponent(typeof(E2EDrivingAgent))]
    public class ExpertDriverController : MonoBehaviour
    {
        [Header("Control Parameters")]
        public float lookAheadDistance = 20f;      // meters ahead to target
        public float steeringGain = 2.0f;          // P-gain for steering
        public float steeringDamping = 0.5f;       // D-gain for steering
        public float targetSpeed = 12f;            // m/s target speed
        public float accelerationGain = 0.5f;      // P-gain for speed control

        [Header("Safety")]
        public float safeFollowDistance = 15f;
        public float emergencyBrakeDistance = 5f;
        public LayerMask vehicleLayer;

        [Header("Recording")]
        public bool autoRecord = true;
        public int maxDemoEpisodes = 50;

        private E2EDrivingAgent agent;
        private int episodesRecorded = 0;
        private float prevSteeringError = 0f;

        void Start()
        {
            agent = GetComponent<E2EDrivingAgent>();

            if (autoRecord)
            {
                SetupDemonstrationRecorder();
            }
        }

        private void SetupDemonstrationRecorder()
        {
            var recorder = GetComponent<DemonstrationRecorder>();
            if (recorder == null)
            {
                recorder = gameObject.AddComponent<DemonstrationRecorder>();
            }
            recorder.DemonstrationName = "expert_driving";
            recorder.DemonstrationDirectory = "Assets/Demonstrations";
            recorder.Record = true;
        }

        /// <summary>
        /// Called by E2EDrivingAgent.Heuristic when in Heuristic mode.
        /// Provides expert-quality actions for demonstration recording.
        /// </summary>
        public void GetExpertAction(in ActionBuffers actionsOut)
        {
            var continuousActions = actionsOut.ContinuousActions;

            float steering = ComputeSteering();
            float acceleration = ComputeAcceleration();

            continuousActions[0] = Mathf.Clamp(steering, -1f, 1f);
            continuousActions[1] = Mathf.Clamp(acceleration, -1f, 1f);
        }

        private float ComputeSteering()
        {
            if (agent.routeWaypoints == null || agent.routeWaypoints.Length == 0)
                return 0f;

            // Find target waypoint ahead
            Transform targetWP = GetLookAheadWaypoint();
            if (targetWP == null) return 0f;

            // Calculate steering angle using PD control
            Vector3 localTarget = transform.InverseTransformPoint(targetWP.position);
            float steeringError = Mathf.Atan2(localTarget.x, localTarget.z);

            // PD controller
            float steeringRate = (steeringError - prevSteeringError) / Time.fixedDeltaTime;
            float steering = steeringGain * steeringError + steeringDamping * steeringRate;
            prevSteeringError = steeringError;

            return steering;
        }

        private float ComputeAcceleration()
        {
            Rigidbody rb = GetComponent<Rigidbody>();
            float currentSpeed = Vector3.Dot(rb.linearVelocity, transform.forward);

            // Check for vehicles ahead
            float desiredSpeed = targetSpeed;
            if (Physics.Raycast(transform.position, transform.forward, out RaycastHit hit,
                                safeFollowDistance, vehicleLayer))
            {
                if (hit.distance < emergencyBrakeDistance)
                {
                    desiredSpeed = 0f;  // Emergency brake
                }
                else
                {
                    // Slow down proportionally
                    float ratio = hit.distance / safeFollowDistance;
                    desiredSpeed = targetSpeed * ratio;
                }
            }

            // P-controller for speed
            float speedError = desiredSpeed - currentSpeed;
            float acceleration = accelerationGain * speedError;

            return acceleration;
        }

        private Transform GetLookAheadWaypoint()
        {
            if (agent.routeWaypoints == null) return null;

            // Find the waypoint closest to lookAheadDistance
            float bestDist = float.MaxValue;
            Transform best = null;

            for (int i = 0; i < agent.routeWaypoints.Length; i++)
            {
                if (agent.routeWaypoints[i] == null) continue;

                Vector3 toWP = agent.routeWaypoints[i].position - transform.position;
                float forwardDist = Vector3.Dot(toWP, transform.forward);

                // Only consider waypoints ahead
                if (forwardDist < 5f) continue;

                float diff = Mathf.Abs(forwardDist - lookAheadDistance);
                if (diff < bestDist)
                {
                    bestDist = diff;
                    best = agent.routeWaypoints[i];
                }
            }

            // Fallback: use the furthest ahead waypoint
            if (best == null && agent.routeWaypoints.Length > 0)
            {
                for (int i = agent.routeWaypoints.Length - 1; i >= 0; i--)
                {
                    if (agent.routeWaypoints[i] != null)
                    {
                        best = agent.routeWaypoints[i];
                        break;
                    }
                }
            }

            return best;
        }

        public void OnEpisodeCompleted()
        {
            episodesRecorded++;
            if (episodesRecorded >= maxDemoEpisodes)
            {
                Debug.Log($"[ExpertDriver] Recorded {episodesRecorded} episodes. Stopping.");
                var recorder = GetComponent<DemonstrationRecorder>();
                if (recorder != null)
                    recorder.Record = false;
            }
        }
    }
}
