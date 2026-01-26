using UnityEngine;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Simple NPC vehicle controller for traffic simulation.
    /// Moves forward at constant speed with basic lane-keeping.
    /// Used to create realistic traffic for RL training.
    /// </summary>
    public class NPCVehicleController : MonoBehaviour
    {
        [Header("Movement")]
        public float speed = 10f;           // m/s (constant speed)
        public float speedVariation = 2f;   // Random speed variation
        public bool isOncoming = false;     // Moves in -Z direction

        [Header("Behavior")]
        public float resetDistance = 300f;   // Reset when this far from origin
        public float spawnDistance = 250f;   // Respawn at this distance ahead
        public Vector3 spawnOffset;          // Offset from center on respawn

        [Header("Avoidance")]
        public float frontDetectDistance = 15f;
        public float slowDownFactor = 0.3f;
        public LayerMask vehicleLayer;

        private Rigidbody rb;
        private float actualSpeed;
        private Vector3 moveDirection;
        private Vector3 initialPosition;
        private Vector3 currentVelocity;

        void Start()
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }

            // NPC vehicles are kinematic (moved via transform)
            // but we track velocity manually for agent detection
            rb.isKinematic = true;
            rb.useGravity = false;

            // Set initial speed with variation
            actualSpeed = speed + Random.Range(-speedVariation, speedVariation);
            moveDirection = isOncoming ? Vector3.back : Vector3.forward;
            initialPosition = transform.position;
            spawnOffset = new Vector3(transform.position.x, transform.position.y, 0);
        }

        void FixedUpdate()
        {
            // Simple forward movement
            float currentSpeed = actualSpeed;

            // Front vehicle detection
            if (vehicleLayer != 0)
            {
                Vector3 rayDir = isOncoming ? -transform.forward : transform.forward;
                if (Physics.Raycast(transform.position, rayDir, frontDetectDistance, vehicleLayer))
                {
                    currentSpeed *= slowDownFactor;
                }
            }

            // Move via MovePosition (reports correct velocity on kinematic rb)
            Vector3 movement = moveDirection * currentSpeed * Time.fixedDeltaTime;
            rb.MovePosition(transform.position + movement);

            // Track velocity for agent detection
            currentVelocity = moveDirection * currentSpeed;

            // Reset/respawn when too far from initial position (area-relative)
            float distFromOrigin = Mathf.Abs(transform.position.z - initialPosition.z);
            if (distFromOrigin > resetDistance)
            {
                Respawn();
            }
        }

        private void Respawn()
        {
            // Respawn relative to initial position (supports multi-area)
            float newZ = isOncoming
                ? initialPosition.z + spawnDistance
                : initialPosition.z - spawnDistance;
            transform.position = new Vector3(spawnOffset.x, spawnOffset.y, newZ);
            actualSpeed = speed + Random.Range(-speedVariation, speedVariation);
        }

        /// <summary>
        /// Set speed externally (for traffic manager)
        /// </summary>
        public void SetSpeed(float newSpeed)
        {
            speed = newSpeed;
            actualSpeed = speed + Random.Range(-speedVariation, speedVariation);
        }

        /// <summary>
        /// Spawn NPC at a specific position with given speed.
        /// Used by DrivingSceneManager for random NPC placement each episode.
        /// </summary>
        public void SpawnAt(Vector3 position, Quaternion rotation, float newSpeed)
        {
            transform.position = position;
            transform.rotation = rotation;
            initialPosition = position;
            spawnOffset = new Vector3(position.x, position.y, 0);
            speed = newSpeed;
            actualSpeed = speed + Random.Range(-speedVariation, speedVariation);
            moveDirection = isOncoming ? Vector3.back : Vector3.forward;
        }

        public float GetCurrentSpeed() => actualSpeed;
        public Vector3 GetVelocity() => currentVelocity;
    }
}
