using UnityEngine;

namespace ADPlatform.DebugTools
{
    /// <summary>
    /// Third-person follow camera for the ego vehicle.
    /// Smoothly follows the target with configurable offset.
    /// </summary>
    public class FollowCamera : MonoBehaviour
    {
        [Header("Target")]
        public Transform target;

        [Header("Position")]
        public Vector3 offset = new Vector3(0, 8, -15);
        public float followSpeed = 5f;
        public float rotationSpeed = 3f;

        [Header("View Modes")]
        public bool topDownView = false;
        public Vector3 topDownOffset = new Vector3(0, 30, 0);

        private Vector3 velocity = Vector3.zero;

        void LateUpdate()
        {
            if (target == null) return;

            Vector3 desiredPosition;
            Quaternion desiredRotation;

            if (topDownView)
            {
                desiredPosition = target.position + topDownOffset;
                desiredRotation = Quaternion.Euler(90, 0, 0);
            }
            else
            {
                desiredPosition = target.position + target.TransformDirection(offset);
                desiredRotation = Quaternion.LookRotation(target.position - transform.position + Vector3.up * 2f);
            }

            // Smooth follow
            transform.position = Vector3.SmoothDamp(
                transform.position, desiredPosition, ref velocity, 1f / followSpeed);
            transform.rotation = Quaternion.Slerp(
                transform.rotation, desiredRotation, rotationSpeed * Time.deltaTime);
        }

        void Update()
        {
            // Toggle view mode with V key
            if (Input.GetKeyDown(KeyCode.V))
            {
                topDownView = !topDownView;
            }
        }
    }
}
