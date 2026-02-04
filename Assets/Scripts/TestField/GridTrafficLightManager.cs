using UnityEngine;
using ADPlatform.Environment;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Manages traffic light groups for grid intersections.
    /// Each intersection has N-S and E-W paired lights that alternate.
    /// 12 intersections have signals; 13 are unsignalized.
    /// </summary>
    public class GridTrafficLightManager : MonoBehaviour
    {
        [System.Serializable]
        public struct IntersectionLights
        {
            public int col, row;
            public TrafficLightController nsLight;  // N-S direction (Z axis)
            public TrafficLightController ewLight;  // E-W direction (X axis)
        }

        [Header("Intersections")]
        public IntersectionLights[] intersections;

        [Header("Timing")]
        public float greenDuration = 12f;
        public float yellowDuration = 3f;
        public float redDuration = 12f;
        public float phaseOffset = 2f;

        private GridRoadNetwork gridNetwork;

        /// <summary>
        /// Initialize all traffic light coordination.
        /// NS and EW lights at each intersection are opposite phases.
        /// Adjacent intersections have a phase offset for green wave.
        /// </summary>
        public void InitializeCoordination(GridRoadNetwork network)
        {
            gridNetwork = network;

            if (intersections == null) return;

            float totalCycle = greenDuration + yellowDuration + redDuration;

            for (int i = 0; i < intersections.Length; i++)
            {
                var intersection = intersections[i];

                // Phase offset based on position in grid (green wave effect)
                float offset = (intersection.col + intersection.row) * phaseOffset;

                // NS light: starts green (with offset)
                if (intersection.nsLight != null)
                {
                    intersection.nsLight.greenDuration = greenDuration;
                    intersection.nsLight.yellowDuration = yellowDuration;
                    intersection.nsLight.redDuration = redDuration;
                    intersection.nsLight.signalEnabled = true;
                    InitializeLightWithOffset(intersection.nsLight, offset, totalCycle);
                }

                // EW light: starts red (offset by half cycle for opposing phase)
                if (intersection.ewLight != null)
                {
                    intersection.ewLight.greenDuration = greenDuration;
                    intersection.ewLight.yellowDuration = yellowDuration;
                    intersection.ewLight.redDuration = redDuration;
                    intersection.ewLight.signalEnabled = true;
                    float ewOffset = offset + greenDuration + yellowDuration;
                    InitializeLightWithOffset(intersection.ewLight, ewOffset, totalCycle);
                }
            }
        }

        private void InitializeLightWithOffset(TrafficLightController light, float offset, float totalCycle)
        {
            // Use ResetSignal to enable, then the light will cycle from a deterministic start
            light.ResetSignal(true, greenDuration / (greenDuration + yellowDuration + redDuration));
        }

        /// <summary>
        /// Get the relevant traffic light for a vehicle at the given position
        /// heading in the given direction.
        /// Returns the NS or EW light of the nearest ahead intersection.
        /// </summary>
        public TrafficLightController GetRelevantLight(Vector3 position, Vector3 forward)
        {
            if (intersections == null || intersections.Length == 0 || gridNetwork == null)
                return null;

            float bestDist = float.MaxValue;
            TrafficLightController bestLight = null;

            for (int i = 0; i < intersections.Length; i++)
            {
                Vector3 intersectionPos = gridNetwork.GetIntersectionPosition(
                    intersections[i].col, intersections[i].row);

                Vector3 toIntersection = intersectionPos - position;
                float dist = toIntersection.magnitude;

                // Only consider intersections within 150m and ahead of the vehicle
                if (dist > 150f) continue;
                float dot = Vector3.Dot(toIntersection.normalized, forward.normalized);
                if (dot < 0.3f) continue;  // Must be roughly ahead

                if (dist < bestDist)
                {
                    bestDist = dist;

                    // Determine if vehicle is approaching N-S or E-W
                    float absX = Mathf.Abs(forward.x);
                    float absZ = Mathf.Abs(forward.z);

                    if (absZ > absX)
                    {
                        // Primarily traveling N-S -> use NS light
                        bestLight = intersections[i].nsLight;
                    }
                    else
                    {
                        // Primarily traveling E-W -> use EW light
                        bestLight = intersections[i].ewLight;
                    }
                }
            }

            return bestLight;
        }

        /// <summary>
        /// Indices of the 12 signalized intersections in the 5x5 grid.
        /// Corners (4) + edge centers (4) + inner (4).
        /// </summary>
        public static readonly Vector2Int[] SignalizedIntersections = new Vector2Int[]
        {
            // Corners
            new Vector2Int(0, 0), new Vector2Int(4, 0),
            new Vector2Int(0, 4), new Vector2Int(4, 4),
            // Edge centers
            new Vector2Int(2, 0), new Vector2Int(0, 2),
            new Vector2Int(4, 2), new Vector2Int(2, 4),
            // Inner
            new Vector2Int(1, 1), new Vector2Int(3, 1),
            new Vector2Int(1, 3), new Vector2Int(3, 3),
        };
    }
}
