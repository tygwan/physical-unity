using UnityEngine;

namespace ADPlatform.DebugTools
{
    /// <summary>
    /// WASD + mouse free-fly camera for scene inspection.
    /// Toggle with F key. When active, disables FollowCamera.
    /// </summary>
    public class FreeFlyCamera : MonoBehaviour
    {
        public float moveSpeed = 20f;
        public float fastMultiplier = 3f;
        public float lookSensitivity = 2f;

        private bool isActive = false;
        private FollowCamera followCamera;
        private float yaw;
        private float pitch;

        void Start()
        {
            followCamera = GetComponent<FollowCamera>();
            yaw = transform.eulerAngles.y;
            pitch = transform.eulerAngles.x;
        }

        void Update()
        {
            if (Input.GetKeyDown(KeyCode.F))
            {
                isActive = !isActive;
                if (followCamera != null)
                    followCamera.enabled = !isActive;
                Cursor.lockState = isActive ? CursorLockMode.Locked : CursorLockMode.None;
                Cursor.visible = !isActive;
            }

            if (!isActive) return;

            // Mouse look
            yaw += Input.GetAxis("Mouse X") * lookSensitivity;
            pitch -= Input.GetAxis("Mouse Y") * lookSensitivity;
            pitch = Mathf.Clamp(pitch, -90f, 90f);
            transform.rotation = Quaternion.Euler(pitch, yaw, 0f);

            // WASD movement
            float speed = moveSpeed * (Input.GetKey(KeyCode.LeftShift) ? fastMultiplier : 1f);
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.W)) move += transform.forward;
            if (Input.GetKey(KeyCode.S)) move -= transform.forward;
            if (Input.GetKey(KeyCode.A)) move -= transform.right;
            if (Input.GetKey(KeyCode.D)) move += transform.right;
            if (Input.GetKey(KeyCode.E)) move += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) move -= Vector3.up;

            transform.position += move.normalized * speed * Time.unscaledDeltaTime;
        }
    }
}
