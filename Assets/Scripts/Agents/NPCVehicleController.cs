using UnityEngine;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Simple NPC vehicle controller for traffic simulation.
    /// Moves forward at constant speed with basic lane-keeping.
    /// Supports waypoint-following mode for intersection navigation.
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

        [Header("Waypoint Following")]
        public Transform[] waypoints;
        public int currentWaypointIndex = 0;
        public float waypointReachDistance = 5f;
        public bool useWaypointFollowing = false;

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
            if (useWaypointFollowing && waypoints != null && waypoints.Length > 0)
            {
                FollowWaypoints();
                return;
            }

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

        /// <summary>
        /// Follow waypoints through intersections.
        /// Uses rb.MovePosition so agent's OverlapSphere reads correct rb.linearVelocity.
        /// </summary>
        private void FollowWaypoints()
        {
            if (currentWaypointIndex >= waypoints.Length)
            {
                RespawnAtEarlyWaypoint();
                return;
            }

            Transform targetWP = waypoints[currentWaypointIndex];
            if (targetWP == null)
            {
                currentWaypointIndex++;
                return;
            }

            // Check if reached current waypoint
            Vector3 toWaypoint = targetWP.position - transform.position;
            toWaypoint.y = 0f;
            float distToWP = toWaypoint.magnitude;

            if (distToWP < waypointReachDistance)
            {
                currentWaypointIndex++;
                if (currentWaypointIndex >= waypoints.Length)
                {
                    RespawnAtEarlyWaypoint();
                    return;
                }
                targetWP = waypoints[currentWaypointIndex];
                if (targetWP == null) return;
                toWaypoint = targetWP.position - transform.position;
                toWaypoint.y = 0f;
            }

            // Front vehicle detection (slow down if vehicle ahead)
            float currentSpeed = actualSpeed;
            if (vehicleLayer != 0)
            {
                if (Physics.Raycast(transform.position, transform.forward, frontDetectDistance, vehicleLayer))
                {
                    currentSpeed *= slowDownFactor;
                }
            }

            // Move toward current waypoint via MovePosition
            Vector3 direction = toWaypoint.normalized;
            Vector3 movement = direction * currentSpeed * Time.fixedDeltaTime;
            rb.MovePosition(transform.position + movement);

            // Smooth rotation toward waypoint
            if (direction.sqrMagnitude > 0.001f)
            {
                Quaternion targetRot = Quaternion.LookRotation(direction);
                rb.MoveRotation(Quaternion.Slerp(transform.rotation, targetRot, 5f * Time.fixedDeltaTime));
            }

            // Track velocity for agent detection
            currentVelocity = direction * currentSpeed;
        }

        /// <summary>
        /// Respawn NPC at a random early waypoint (first third of path).
        /// </summary>
        private void RespawnAtEarlyWaypoint()
        {
            if (waypoints == null || waypoints.Length < 2) return;

            int earlyRange = Mathf.Max(1, waypoints.Length / 3);
            int wpIdx = Random.Range(0, earlyRange);
            currentWaypointIndex = wpIdx + 1;

            transform.position = waypoints[wpIdx].position;

            // Face next waypoint
            if (currentWaypointIndex < waypoints.Length && waypoints[currentWaypointIndex] != null)
            {
                Vector3 dir = waypoints[currentWaypointIndex].position - waypoints[wpIdx].position;
                dir.y = 0f;
                if (dir.sqrMagnitude > 0.01f)
                    transform.rotation = Quaternion.LookRotation(dir);
            }

            actualSpeed = speed + Random.Range(-speedVariation, speedVariation);
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

        /// <summary>
        /// Spawn NPC at a specific waypoint with waypoint-following enabled.
        /// Used by DrivingSceneManager for intersection NPC placement.
        /// </summary>
        public void SpawnAtWaypoint(int wpIndex, float newSpeed, Transform[] wps)
        {
            waypoints = wps;
            currentWaypointIndex = wpIndex + 1; // target next waypoint
            useWaypointFollowing = true;

            // Position at waypoint
            if (wpIndex < wps.Length && wps[wpIndex] != null)
            {
                transform.position = wps[wpIndex].position;

                // Face next waypoint
                if (currentWaypointIndex < wps.Length && wps[currentWaypointIndex] != null)
                {
                    Vector3 dir = wps[currentWaypointIndex].position - wps[wpIndex].position;
                    dir.y = 0f;
                    if (dir.sqrMagnitude > 0.01f)
                        transform.rotation = Quaternion.LookRotation(dir);
                }
            }

            speed = newSpeed;
            actualSpeed = speed + Random.Range(-speedVariation, speedVariation);
        }

        public float GetCurrentSpeed() => actualSpeed;
        public Vector3 GetVelocity() => currentVelocity;
    }
}
