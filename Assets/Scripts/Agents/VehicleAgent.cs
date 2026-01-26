using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Basic Vehicle Agent for Autonomous Driving RL training.
    /// Observations: position, velocity, heading, distance to waypoints
    /// Actions: acceleration, steering
    /// </summary>
    public class VehicleAgent : Agent
    {
        [Header("Vehicle Settings")]
        public float maxSpeed = 20f;           // m/s
        public float maxAcceleration = 4f;     // m/s²
        public float maxSteering = 30f;        // degrees
        public float dragCoefficient = 0.5f;

        [Header("Reward Settings")]
        public float progressReward = 0.1f;
        public float collisionPenalty = -10f;
        public float offRoadPenalty = -5f;
        public float goalReward = 10f;

        [Header("References")]
        public Transform goalTarget;
        public Transform roadCenter;

        private Rigidbody rb;
        private Vector3 startPosition;
        private Quaternion startRotation;
        private float currentSpeed;
        private float previousDistanceToGoal;

        public override void Initialize()
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }
            
            rb.mass = 1500f;  // 1.5 ton vehicle
            rb.linearDamping = dragCoefficient;
            rb.angularDamping = 2f;
            rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

            startPosition = transform.position;
            startRotation = transform.rotation;
        }

        public override void OnEpisodeBegin()
        {
            // Reset vehicle position and velocity
            transform.position = startPosition;
            transform.rotation = startRotation;
            rb.linearVelocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            currentSpeed = 0f;

            // Calculate initial distance to goal
            if (goalTarget != null)
            {
                previousDistanceToGoal = Vector3.Distance(transform.position, goalTarget.position);
            }
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Ego state (8 observations)
            // Position (normalized)
            sensor.AddObservation(transform.localPosition.x / 50f);
            sensor.AddObservation(transform.localPosition.z / 250f);
            
            // Velocity (normalized)
            Vector3 localVelocity = transform.InverseTransformDirection(rb.linearVelocity);
            sensor.AddObservation(localVelocity.x / maxSpeed);
            sensor.AddObservation(localVelocity.z / maxSpeed);
            
            // Heading (sin/cos for continuity)
            float heading = transform.eulerAngles.y * Mathf.Deg2Rad;
            sensor.AddObservation(Mathf.Sin(heading));
            sensor.AddObservation(Mathf.Cos(heading));
            
            // Current speed (normalized)
            currentSpeed = rb.linearVelocity.magnitude;
            sensor.AddObservation(currentSpeed / maxSpeed);
            
            // Angular velocity
            sensor.AddObservation(rb.angularVelocity.y / 5f);

            // Goal direction (4 observations)
            if (goalTarget != null)
            {
                Vector3 directionToGoal = (goalTarget.position - transform.position).normalized;
                Vector3 localGoalDir = transform.InverseTransformDirection(directionToGoal);
                sensor.AddObservation(localGoalDir.x);
                sensor.AddObservation(localGoalDir.z);
                
                float distanceToGoal = Vector3.Distance(transform.position, goalTarget.position);
                sensor.AddObservation(distanceToGoal / 500f);  // Normalized distance
                sensor.AddObservation(Mathf.Clamp01(1f - distanceToGoal / 500f));  // Progress indicator
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(1f);  // Default forward
                sensor.AddObservation(1f);
                sensor.AddObservation(0f);
            }

            // Road alignment (2 observations)
            if (roadCenter != null)
            {
                float lateralOffset = transform.position.x - roadCenter.position.x;
                sensor.AddObservation(lateralOffset / 10f);  // Normalized lateral offset
                sensor.AddObservation(Mathf.Abs(lateralOffset) > 10f ? 1f : 0f);  // Off-road indicator
            }
            else
            {
                sensor.AddObservation(0f);
                sensor.AddObservation(0f);
            }
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            // Continuous actions
            float acceleration = Mathf.Clamp(actions.ContinuousActions[0], -1f, 1f);
            float steering = Mathf.Clamp(actions.ContinuousActions[1], -1f, 1f);

            // Debug: log non-zero actions
            if (Mathf.Abs(acceleration) > 0.01f || Mathf.Abs(steering) > 0.01f)
            {
                Debug.Log($"[VehicleAgent] Action: Accel={acceleration:F2}, Steer={steering:F2}");
            }

            // Apply acceleration
            float targetAccel = acceleration * maxAcceleration;
            Vector3 force = transform.forward * targetAccel * rb.mass;
            rb.AddForce(force);

            // Apply steering (only when moving)
            if (currentSpeed > 0.5f)
            {
                float turnAngle = steering * maxSteering * Time.fixedDeltaTime;
                transform.Rotate(0f, turnAngle, 0f);
            }

            // Speed limiting
            if (rb.linearVelocity.magnitude > maxSpeed)
            {
                rb.linearVelocity = rb.linearVelocity.normalized * maxSpeed;
            }

            // Calculate rewards
            CalculateRewards();
        }

        private void CalculateRewards()
        {
            // Progress reward
            if (goalTarget != null)
            {
                float currentDistanceToGoal = Vector3.Distance(transform.position, goalTarget.position);
                float progressMade = previousDistanceToGoal - currentDistanceToGoal;
                AddReward(progressMade * progressReward);
                previousDistanceToGoal = currentDistanceToGoal;

                // Goal reached
                if (currentDistanceToGoal < 5f)
                {
                    AddReward(goalReward);
                    EndEpisode();
                }
            }

            // Off-road penalty
            if (roadCenter != null)
            {
                float lateralOffset = Mathf.Abs(transform.position.x - roadCenter.position.x);
                if (lateralOffset > 10f)
                {
                    AddReward(offRoadPenalty * Time.fixedDeltaTime);
                }
            }

            // Small time penalty to encourage faster completion
            AddReward(-0.001f);
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.gameObject.CompareTag("Obstacle") || 
                collision.gameObject.CompareTag("Vehicle"))
            {
                AddReward(collisionPenalty);
                EndEpisode();
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Manual control for testing
            var continuousActions = actionsOut.ContinuousActions;
            float vertical = Input.GetAxis("Vertical");
            float horizontal = Input.GetAxis("Horizontal");

            continuousActions[0] = vertical;   // W/S or Up/Down
            continuousActions[1] = horizontal; // A/D or Left/Right

            // Debug logging
            if (Mathf.Abs(vertical) > 0.01f || Mathf.Abs(horizontal) > 0.01f)
            {
                Debug.Log($"[VehicleAgent] Input: V={vertical:F2}, H={horizontal:F2}");
            }
        }
    }
}
