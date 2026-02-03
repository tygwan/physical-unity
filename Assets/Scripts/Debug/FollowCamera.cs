using UnityEngine;

namespace ADPlatform.DebugTools
{
    /// <summary>
    /// Third-person follow camera for the ego vehicle.
    /// Smoothly follows the target with configurable offset.
    /// Supports multi-target cycling (Tab / number keys) for Phase M test field.
    /// </summary>
    public class FollowCamera : MonoBehaviour
    {
        [Header("Target")]
        public Transform target;

        [Header("Multi-Target")]
        public Transform[] targets;
        private int currentTargetIndex = 0;

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

            // Multi-target cycling with Tab
            if (Input.GetKeyDown(KeyCode.Tab) && targets != null && targets.Length > 0)
            {
                currentTargetIndex = (currentTargetIndex + 1) % targets.Length;
                target = targets[currentTargetIndex];
            }

            // Number keys 1-9 for direct agent selection
            for (int i = 0; i < 9 && i < (targets != null ? targets.Length : 0); i++)
            {
                if (Input.GetKeyDown(KeyCode.Alpha1 + i))
                {
                    currentTargetIndex = i;
                    target = targets[currentTargetIndex];
                }
            }
        }

        /// <summary>
        /// Programmatically switch to a specific target index.
        /// </summary>
        public void SetTargetIndex(int index)
        {
            if (targets == null || targets.Length == 0) return;
            currentTargetIndex = Mathf.Clamp(index, 0, targets.Length - 1);
            target = targets[currentTargetIndex];
        }

        public int GetCurrentTargetIndex() => currentTargetIndex;
    }
}
