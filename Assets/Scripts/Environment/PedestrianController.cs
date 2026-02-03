using UnityEngine;

namespace ADPlatform.Environment
{
    /// <summary>
    /// Kinematic pedestrian that crosses the road at a crosswalk.
    /// Follows NPCVehicleController pattern: kinematic Rigidbody + MovePosition.
    ///
    /// Lifecycle:
    ///   1. DrivingSceneManager calls SpawnAtCrosswalk() at episode start
    ///   2. Pedestrian walks perpendicular across road at walkSpeed
    ///   3. After crossing, deactivates itself
    ///   4. DrivingSceneManager respawns next episode
    /// </summary>
    public class PedestrianController : MonoBehaviour
    {
        public enum PedestrianState { Waiting, Crossing, Exited }

        [Header("Movement")]
        public float walkSpeed = 1.2f;         // m/s (typical pedestrian speed)

        private float crosswalkZ;               // Z position of assigned crosswalk
        private float crossingDirectionX;       // +1 or -1 (crossing direction)
        private float roadWidth;                // Total road width to cross
        private float startX;                   // Starting X position
        private PedestrianState state = PedestrianState.Waiting;

        private Rigidbody rb;
        private Vector3 currentVelocity;

        void Awake()
        {
            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }
            rb.isKinematic = true;
            rb.useGravity = false;
        }

        void FixedUpdate()
        {
            if (state != PedestrianState.Crossing)
            {
                currentVelocity = Vector3.zero;
                return;
            }

            // Move perpendicular across road
            Vector3 movement = new Vector3(crossingDirectionX, 0f, 0f) * walkSpeed * Time.fixedDeltaTime;
            rb.MovePosition(transform.position + movement);
            currentVelocity = new Vector3(crossingDirectionX * walkSpeed, 0f, 0f);

            // Check if crossed entire road
            float distanceCrossed = Mathf.Abs(transform.position.x - startX);
            if (distanceCrossed >= roadWidth + 2f)
            {
                state = PedestrianState.Exited;
                currentVelocity = Vector3.zero;
                gameObject.SetActive(false);
            }
        }

        /// <summary>
        /// Spawn pedestrian at the edge of the road at given crosswalk Z position.
        /// </summary>
        public void SpawnAtCrosswalk(float crosswalkZPos, float totalRoadWidth, float speed)
        {
            crosswalkZ = crosswalkZPos;
            roadWidth = totalRoadWidth;
            walkSpeed = speed;

            // Random crossing direction: left-to-right or right-to-left
            crossingDirectionX = Random.value > 0.5f ? 1f : -1f;

            // Start at road edge (outside road + 1m margin)
            startX = -crossingDirectionX * (roadWidth / 2f + 1f);
            transform.position = new Vector3(
                transform.parent.position.x + startX,
                0.9f,  // Capsule center height
                transform.parent.position.z + crosswalkZPos
            );

            state = PedestrianState.Crossing;
            currentVelocity = Vector3.zero;
            gameObject.SetActive(true);
        }

        /// <summary>
        /// Deactivate pedestrian (called by DrivingSceneManager on episode reset).
        /// </summary>
        public void Deactivate()
        {
            state = PedestrianState.Waiting;
            currentVelocity = Vector3.zero;
            gameObject.SetActive(false);
        }

        // === Public API for observations ===

        public Vector3 GetPosition() => transform.position;

        public Vector3 GetVelocity() => currentVelocity;

        public bool IsActive() => gameObject.activeSelf && state == PedestrianState.Crossing;

        public PedestrianState GetState() => state;
    }
}
