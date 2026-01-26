using UnityEngine;

namespace ADPlatform.Agents
{
    /// <summary>
    /// Simple keyboard controller for testing vehicle movement without ML-Agents.
    /// </summary>
    public class SimpleVehicleController : MonoBehaviour
    {
        [Header("Movement Settings")]
        public float maxSpeed = 20f;
        public float acceleration = 15f;  // m/s² (reasonable for vehicle)
        public float turnSpeed = 50f;

        private Rigidbody rb;

        void Start()
        {
            Debug.Log("[SimpleController] Start() called!");

            rb = GetComponent<Rigidbody>();
            if (rb == null)
            {
                rb = gameObject.AddComponent<Rigidbody>();
            }
            rb.mass = 1500f;
            rb.linearDamping = 0.5f;
            rb.angularDamping = 2f;
            rb.constraints = RigidbodyConstraints.FreezeRotationX | RigidbodyConstraints.FreezeRotationZ;

            Debug.Log($"[SimpleController] Configured: mass={rb.mass}");
        }

        void Update()
        {
            // Also check in Update for input
            if (Input.GetKeyDown(KeyCode.W))
            {
                Debug.Log("[SimpleController] W key pressed!");
            }
        }

        void FixedUpdate()
        {
            float vertical = Input.GetAxis("Vertical");
            float horizontal = Input.GetAxis("Horizontal");

            // Apply forward/backward force (ForceMode.Acceleration ignores mass)
            Vector3 accel = transform.forward * vertical * acceleration;
            rb.AddForce(accel, ForceMode.Acceleration);

            // Apply turning (only when moving)
            if (rb.linearVelocity.magnitude > 0.5f)
            {
                float turn = horizontal * turnSpeed * Time.fixedDeltaTime;
                transform.Rotate(0f, turn, 0f);
            }

            // Limit speed
            if (rb.linearVelocity.magnitude > maxSpeed)
            {
                rb.linearVelocity = rb.linearVelocity.normalized * maxSpeed;
            }

            // Debug
            if (Mathf.Abs(vertical) > 0.01f || Mathf.Abs(horizontal) > 0.01f)
            {
                Debug.Log($"[SimpleController] V={vertical:F2}, H={horizontal:F2}, Speed={rb.linearVelocity.magnitude:F1}");
            }
        }
    }
}
