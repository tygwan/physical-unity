using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Demonstrations;
using ADPlatform.Environment;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Expert driver controller for recording demonstrations.
    /// Follows waypoints using Pure Pursuit control to produce smooth expert trajectories.
    ///
    /// Supports both linear road and grid network (Phase M) environments.
    /// In grid mode, uses sequential waypoint index-based lookahead for correct
    /// cyclic route following, with traffic light and intersection awareness.
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
        public float lookAheadDistance = 15f;      // meters ahead to target
        public float steeringGain = 2.5f;          // P-gain for steering
        public float steeringDamping = 0.3f;       // D-gain for steering
        public float targetSpeed = 11f;            // m/s (~40 km/h for grid)
        public float accelerationGain = 0.6f;      // P-gain for speed control

        [Header("Intersection")]
        public float intersectionSlowDistance = 30f;  // start slowing at this distance
        public float intersectionMinSpeed = 5f;       // minimum speed at intersection
        public float turnSpeedFactor = 0.5f;          // speed factor during turns

        [Header("Traffic Light")]
        public float redLightStopDistance = 5f;        // stop this far from stop line
        public float yellowLightSlowDistance = 25f;    // start slowing for yellow

        [Header("Safety")]
        public float safeFollowDistance = 15f;
        public float emergencyBrakeDistance = 5f;
        public LayerMask vehicleLayer;

        [Header("Recording")]
        public bool autoRecord = false;
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
            // Lazy init: component may be added at runtime before Start() runs
            if (agent == null)
                agent = GetComponent<E2EDrivingAgent>();

            var continuousActions = actionsOut.ContinuousActions;

            if (agent == null || agent.routeWaypoints == null)
            {
                continuousActions[0] = 0f;
                continuousActions[1] = 0f;
                return;
            }

            float steering = ComputeSteering();
            float acceleration = ComputeAcceleration();

            continuousActions[0] = Mathf.Clamp(steering, -1f, 1f);
            continuousActions[1] = Mathf.Clamp(acceleration, -1f, 1f);
        }

        private float ComputeSteering()
        {
            if (agent.routeWaypoints == null || agent.routeWaypoints.Length == 0)
                return 0f;

            // Find target waypoint using sequential index-based lookahead
            Transform targetWP = GetSequentialLookAheadWaypoint();
            if (targetWP == null) return 0f;

            // Pure Pursuit: steering angle from local target position
            Vector3 localTarget = transform.InverseTransformPoint(targetWP.position);
            float steeringError = Mathf.Atan2(localTarget.x, localTarget.z);

            // PD controller
            float dt = Mathf.Max(Time.fixedDeltaTime, 0.001f);
            float steeringRate = (steeringError - prevSteeringError) / dt;
            float steering = steeringGain * steeringError + steeringDamping * steeringRate;
            prevSteeringError = steeringError;

            return steering;
        }

        private float ComputeAcceleration()
        {
            Rigidbody rb = GetComponent<Rigidbody>();
            float currentSpeed = rb != null ? Vector3.Dot(rb.linearVelocity, transform.forward) : 0f;

            float desiredSpeed = targetSpeed;

            // 1. Traffic light: stop at red, slow at yellow
            desiredSpeed = ApplyTrafficLightSpeed(desiredSpeed);

            // 2. Intersection speed reduction
            desiredSpeed = ApplyIntersectionSpeed(desiredSpeed);

            // 3. Vehicle ahead: slow down or emergency brake
            if (Physics.Raycast(transform.position + Vector3.up * 0.5f, transform.forward,
                                out RaycastHit hit, safeFollowDistance, vehicleLayer))
            {
                if (hit.distance < emergencyBrakeDistance)
                {
                    desiredSpeed = 0f;
                }
                else
                {
                    float ratio = hit.distance / safeFollowDistance;
                    desiredSpeed = Mathf.Min(desiredSpeed, targetSpeed * ratio);
                }
            }

            // P-controller for speed
            float speedError = desiredSpeed - currentSpeed;
            float acceleration = accelerationGain * speedError;

            // Stronger braking when far from desired speed
            if (speedError < -2f)
                acceleration *= 1.5f;

            return acceleration;
        }

        /// <summary>
        /// Sequential index-based lookahead for cyclic grid routes.
        /// Starts from agent.CurrentWaypointIndex and walks forward along the
        /// route waypoint sequence until accumulated distance >= lookAheadDistance.
        /// </summary>
        private Transform GetSequentialLookAheadWaypoint()
        {
            var wps = agent.routeWaypoints;
            if (wps == null || wps.Length == 0) return null;

            int startIdx = agent.CurrentWaypointIndex;
            int numWps = wps.Length;

            // Walk forward from current index
            float accDist = 0f;
            Vector3 prevPos = transform.position;

            for (int step = 0; step < Mathf.Min(numWps, 40); step++)
            {
                int idx = (startIdx + step) % numWps;
                if (wps[idx] == null) continue;

                if (step == 0)
                {
                    accDist = Vector3.Distance(transform.position, wps[idx].position);
                }
                else
                {
                    accDist += Vector3.Distance(prevPos, wps[idx].position);
                }

                prevPos = wps[idx].position;

                if (accDist >= lookAheadDistance)
                    return wps[idx];
            }

            // Fallback: use a few waypoints ahead of current
            int fallbackIdx = (startIdx + 3) % numWps;
            if (wps[fallbackIdx] != null)
                return wps[fallbackIdx];

            // Last resort: first non-null waypoint from current
            for (int step = 0; step < numWps; step++)
            {
                int idx = (startIdx + step) % numWps;
                if (wps[idx] != null) return wps[idx];
            }

            return null;
        }

        /// <summary>
        /// Reduce speed for traffic lights.
        /// </summary>
        private float ApplyTrafficLightSpeed(float desiredSpeed)
        {
            if (agent.trafficLight == null) return desiredSpeed;

            var state = agent.trafficLight.GetCurrentState();
            bool isRed = state == TrafficLightController.LightState.Red;
            bool isYellow = state == TrafficLightController.LightState.Yellow;

            if (!isRed && !isYellow) return desiredSpeed;

            // Get distance to intersection (from waypointManager proxy)
            float distToIntersection = 200f;
            if (agent.waypointManager != null)
                distToIntersection = agent.waypointManager.intersectionDistance;

            if (isRed)
            {
                if (distToIntersection < redLightStopDistance)
                    return 0f;  // Full stop
                if (distToIntersection < intersectionSlowDistance)
                {
                    // Gradual slow-down to stop
                    float ratio = (distToIntersection - redLightStopDistance) /
                                  (intersectionSlowDistance - redLightStopDistance);
                    return desiredSpeed * Mathf.Clamp01(ratio) * 0.5f;
                }
            }

            if (isYellow && distToIntersection < yellowLightSlowDistance)
            {
                float ratio = distToIntersection / yellowLightSlowDistance;
                return desiredSpeed * Mathf.Clamp01(ratio);
            }

            return desiredSpeed;
        }

        /// <summary>
        /// Reduce speed near intersections, especially before turns.
        /// </summary>
        private float ApplyIntersectionSpeed(float desiredSpeed)
        {
            if (agent.waypointManager == null) return desiredSpeed;

            float distToIntersection = agent.waypointManager.intersectionDistance;
            int turnDir = agent.waypointManager.turnDirection;

            if (distToIntersection > intersectionSlowDistance)
                return desiredSpeed;

            // Approaching intersection - reduce speed
            float slowFactor;
            if (turnDir != 0)
            {
                // Turning: reduce more aggressively
                slowFactor = Mathf.Lerp(turnSpeedFactor, 1f,
                    Mathf.Clamp01(distToIntersection / intersectionSlowDistance));
            }
            else
            {
                // Straight through: mild reduction
                slowFactor = Mathf.Lerp(0.7f, 1f,
                    Mathf.Clamp01(distToIntersection / intersectionSlowDistance));
            }

            return Mathf.Max(desiredSpeed * slowFactor, intersectionMinSpeed);
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
