using UnityEngine;

namespace ADPlatform.Environment
{
    /// <summary>
    /// Controls traffic light state cycling for a Training Area intersection.
    /// State machine: Red → Green → Yellow → Red
    ///
    /// Phase J: Agent must learn to stop at red lights and proceed on green.
    /// Each Training Area has its own independent TrafficLightController.
    ///
    /// Usage:
    ///   - Attached to a TrafficLight GameObject within each Training Area
    ///   - DrivingSceneManager configures enabled/timing on episode reset
    ///   - E2EDrivingAgent reads state via GetCurrentState() for observations
    /// </summary>
    public class TrafficLightController : MonoBehaviour
    {
        public enum LightState { Red, Yellow, Green, None }

        [Header("Signal Timing")]
        [Tooltip("Duration of green phase in seconds")]
        public float greenDuration = 12f;
        [Tooltip("Duration of yellow phase in seconds (Korean standard: 3s)")]
        public float yellowDuration = 3f;
        [Tooltip("Duration of red phase in seconds")]
        public float redDuration = 12f;

        [Header("State")]
        [Tooltip("Whether traffic signal is active (curriculum-controlled)")]
        public bool signalEnabled = false;

        [Header("Stop Line")]
        [Tooltip("Z position of the stop line in local Training Area space")]
        public float stopLineZ = 93f;
        [Tooltip("Tolerance for 'stopped at line' detection (meters)")]
        public float stopLineTolerance = 5f;

        [Header("Visual References")]
        public Renderer redLight;
        public Renderer yellowLight;
        public Renderer greenLight;

        // Current state
        private LightState currentState = LightState.None;
        private float stateTimer = 0f;
        private float currentStateDuration = 0f;

        // Materials for visual feedback
        private static readonly Color RED_ON = new Color(1f, 0.1f, 0.1f);
        private static readonly Color YELLOW_ON = new Color(1f, 0.9f, 0.1f);
        private static readonly Color GREEN_ON = new Color(0.1f, 1f, 0.2f);
        private static readonly Color LIGHT_OFF = new Color(0.15f, 0.15f, 0.15f);

        void Update()
        {
            if (!signalEnabled || currentState == LightState.None)
                return;

            stateTimer += Time.deltaTime;

            if (stateTimer >= currentStateDuration)
            {
                AdvanceState();
            }
        }

        /// <summary>
        /// Reset signal for a new episode with randomized start phase.
        /// Called by DrivingSceneManager.ResetEpisode().
        /// </summary>
        public void ResetSignal(bool enabled, float greenRatio = 0.5f)
        {
            signalEnabled = enabled;

            if (!enabled)
            {
                currentState = LightState.None;
                UpdateVisuals();
                return;
            }

            // Adjust green/red timing based on ratio
            // greenRatio 0.7 = 70% green (easy), 0.4 = 40% green (hard)
            float cycleTime = greenDuration + yellowDuration + redDuration;
            greenDuration = cycleTime * greenRatio;
            redDuration = cycleTime * (1f - greenRatio) - yellowDuration;
            redDuration = Mathf.Max(redDuration, 3f); // Minimum 3s red

            // Randomize start state and timer position
            float totalCycle = greenDuration + yellowDuration + redDuration;
            float randomOffset = Random.Range(0f, totalCycle);

            if (randomOffset < greenDuration)
            {
                currentState = LightState.Green;
                currentStateDuration = greenDuration;
                stateTimer = randomOffset;
            }
            else if (randomOffset < greenDuration + yellowDuration)
            {
                currentState = LightState.Yellow;
                currentStateDuration = yellowDuration;
                stateTimer = randomOffset - greenDuration;
            }
            else
            {
                currentState = LightState.Red;
                currentStateDuration = redDuration;
                stateTimer = randomOffset - greenDuration - yellowDuration;
            }

            UpdateVisuals();
        }

        private void AdvanceState()
        {
            switch (currentState)
            {
                case LightState.Green:
                    currentState = LightState.Yellow;
                    currentStateDuration = yellowDuration;
                    break;
                case LightState.Yellow:
                    currentState = LightState.Red;
                    currentStateDuration = redDuration;
                    break;
                case LightState.Red:
                    currentState = LightState.Green;
                    currentStateDuration = greenDuration;
                    break;
            }

            stateTimer = 0f;
            UpdateVisuals();
        }

        private void UpdateVisuals()
        {
            if (redLight != null)
            {
                redLight.material.color = currentState == LightState.Red ? RED_ON : LIGHT_OFF;
                redLight.material.SetColor("_EmissionColor",
                    currentState == LightState.Red ? RED_ON * 2f : Color.black);
            }
            if (yellowLight != null)
            {
                yellowLight.material.color = currentState == LightState.Yellow ? YELLOW_ON : LIGHT_OFF;
                yellowLight.material.SetColor("_EmissionColor",
                    currentState == LightState.Yellow ? YELLOW_ON * 2f : Color.black);
            }
            if (greenLight != null)
            {
                greenLight.material.color = currentState == LightState.Green ? GREEN_ON : LIGHT_OFF;
                greenLight.material.SetColor("_EmissionColor",
                    currentState == LightState.Green ? GREEN_ON * 2f : Color.black);
            }
        }

        // ============================================================
        // PUBLIC API (for E2EDrivingAgent observations)
        // ============================================================

        /// <summary>
        /// Get current traffic light state.
        /// Returns None if signal is disabled.
        /// </summary>
        public LightState GetCurrentState()
        {
            return signalEnabled ? currentState : LightState.None;
        }

        /// <summary>
        /// Get normalized time remaining in current state [0, 1].
        /// 1.0 = just started current phase, 0.0 = about to transition.
        /// </summary>
        public float GetTimeRemainingNormalized()
        {
            if (!signalEnabled || currentStateDuration <= 0f)
                return 0f;
            return Mathf.Clamp01(1f - stateTimer / currentStateDuration);
        }

        /// <summary>
        /// Get signed distance along the approach axis from vehicle to stop line.
        /// Uses local space so it works regardless of traffic light orientation.
        /// Positive = vehicle is before the stop line, Negative = vehicle has passed it.
        /// Backward compatible: existing scenes with Z+ facing lights produce identical results.
        /// </summary>
        public float GetDistanceAlongApproach(Vector3 vehiclePosition)
        {
            Vector3 localPos = transform.InverseTransformPoint(vehiclePosition);
            return stopLineZ - localPos.z;
        }

        /// <summary>
        /// Get the Z position of the stop line in world space.
        /// Kept for backward compatibility with existing training scenes.
        /// </summary>
        public float GetStopLineWorldZ()
        {
            return transform.position.z + stopLineZ;
        }

        /// <summary>
        /// Check if a vehicle at the given position should stop.
        /// True if red/yellow AND vehicle is behind the stop line.
        /// Uses local space for direction-independent detection.
        /// </summary>
        public bool ShouldStop(Vector3 vehiclePosition)
        {
            if (!signalEnabled) return false;
            if (currentState != LightState.Red && currentState != LightState.Yellow) return false;

            return GetDistanceAlongApproach(vehiclePosition) > 0f;
        }

        /// <summary>
        /// Check if a vehicle is stopped at the stop line (within tolerance).
        /// </summary>
        public bool IsStoppedAtLine(Vector3 vehiclePosition, float vehicleSpeed)
        {
            if (!signalEnabled) return false;

            float dist = GetDistanceAlongApproach(vehiclePosition);
            return dist > 0f && dist < stopLineTolerance && vehicleSpeed < 0.5f;
        }

        /// <summary>
        /// Check if a vehicle has violated the red light (crossed stop line while red).
        /// </summary>
        public bool HasViolatedRedLight(Vector3 vehiclePosition)
        {
            if (!signalEnabled) return false;
            if (currentState != LightState.Red) return false;

            return GetDistanceAlongApproach(vehiclePosition) < 0f;
        }

        /// <summary>
        /// Get distance from vehicle to stop line (positive = behind line, negative = past line).
        /// Normalized by 200m for observation space.
        /// </summary>
        public float GetDistanceToStopLineNormalized(Vector3 vehiclePosition)
        {
            float dist = GetDistanceAlongApproach(vehiclePosition);
            return Mathf.Clamp01(dist / 200f);
        }
    }
}
