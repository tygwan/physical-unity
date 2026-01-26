using UnityEngine;

namespace ADPlatform.Environment
{
    /// <summary>
    /// Lane marking types based on Korean traffic regulations.
    /// Used for lane change policy enforcement in RL training.
    /// </summary>
    public enum LaneMarkingType
    {
        None = 0,           // No marking detected
        WhiteDashed = 1,    // 백색 점선 - Lane change allowed
        WhiteSolid = 2,     // 백색 실선 - Lane change prohibited
        YellowDashed = 3,   // 황색 점선 (중앙선) - Overtaking possible but risky
        YellowSolid = 4,    // 황색 실선 (중앙선) - Never cross
        DoubleYellow = 5    // 이중 황색 실선 - Absolutely prohibited (fatal)
    }

    /// <summary>
    /// Marks a lane boundary with specific crossing rules.
    /// Attach to road lane marking GameObjects with BoxCollider (trigger).
    /// </summary>
    public class LaneMarking : MonoBehaviour
    {
        [Header("Lane Marking Configuration")]
        public LaneMarkingType markingType = LaneMarkingType.WhiteDashed;

        [Header("Penalty Values (for reference)")]
        [Tooltip("Penalty applied when crossing this marking")]
        public float crossingPenalty = 0f;

        [Tooltip("Whether crossing this marking ends the episode")]
        public bool terminatesEpisode = false;

        [Header("Visual Settings")]
        public Color gizmoColor = Color.white;

        private void OnValidate()
        {
            // Auto-set penalty and termination based on marking type
            switch (markingType)
            {
                case LaneMarkingType.None:
                    crossingPenalty = 0f;
                    terminatesEpisode = false;
                    gizmoColor = Color.clear;
                    break;
                case LaneMarkingType.WhiteDashed:
                    crossingPenalty = 0f;  // Allowed
                    terminatesEpisode = false;
                    gizmoColor = Color.white;
                    break;
                case LaneMarkingType.WhiteSolid:
                    crossingPenalty = -2.0f;
                    terminatesEpisode = false;
                    gizmoColor = Color.gray;
                    break;
                case LaneMarkingType.YellowDashed:
                    crossingPenalty = -3.0f;
                    terminatesEpisode = false;
                    gizmoColor = Color.yellow;
                    break;
                case LaneMarkingType.YellowSolid:
                    crossingPenalty = -5.0f;
                    terminatesEpisode = false;
                    gizmoColor = new Color(1f, 0.8f, 0f);  // Orange-yellow
                    break;
                case LaneMarkingType.DoubleYellow:
                    crossingPenalty = -10.0f;
                    terminatesEpisode = true;
                    gizmoColor = Color.red;
                    break;
            }
        }

        private void Awake()
        {
            // Ensure collider is set as trigger
            var collider = GetComponent<Collider>();
            if (collider != null)
            {
                collider.isTrigger = true;
            }

            // Set layer based on marking type
            SetLayerByType();
        }

        private void SetLayerByType()
        {
            string layerName = markingType switch
            {
                LaneMarkingType.WhiteDashed => "LaneDashed",
                LaneMarkingType.WhiteSolid => "LaneSolid",
                LaneMarkingType.YellowDashed => "CenterLine",
                LaneMarkingType.YellowSolid => "CenterLine",
                LaneMarkingType.DoubleYellow => "CenterLine",
                _ => "Default"
            };

            int layer = LayerMask.NameToLayer(layerName);
            if (layer >= 0)
            {
                gameObject.layer = layer;
            }
        }

        /// <summary>
        /// Check if lane change is allowed across this marking.
        /// </summary>
        public bool IsLaneChangeAllowed()
        {
            return markingType == LaneMarkingType.WhiteDashed;
        }

        /// <summary>
        /// Check if this is a center line marking.
        /// </summary>
        public bool IsCenterLine()
        {
            return markingType == LaneMarkingType.YellowDashed ||
                   markingType == LaneMarkingType.YellowSolid ||
                   markingType == LaneMarkingType.DoubleYellow;
        }

        /// <summary>
        /// Get the severity level of crossing (0-3).
        /// 0 = allowed, 1 = minor, 2 = major, 3 = fatal
        /// </summary>
        public int GetSeverityLevel()
        {
            return markingType switch
            {
                LaneMarkingType.WhiteDashed => 0,
                LaneMarkingType.WhiteSolid => 1,
                LaneMarkingType.YellowDashed => 2,
                LaneMarkingType.YellowSolid => 2,
                LaneMarkingType.DoubleYellow => 3,
                _ => 0
            };
        }

        /// <summary>
        /// Get one-hot encoding for observation space (4D).
        /// [WhiteDashed, WhiteSolid, YellowDashed/Solid, DoubleYellow]
        /// </summary>
        public float[] GetOneHotEncoding()
        {
            float[] encoding = new float[4];

            switch (markingType)
            {
                case LaneMarkingType.WhiteDashed:
                    encoding[0] = 1f;
                    break;
                case LaneMarkingType.WhiteSolid:
                    encoding[1] = 1f;
                    break;
                case LaneMarkingType.YellowDashed:
                case LaneMarkingType.YellowSolid:
                    encoding[2] = 1f;
                    break;
                case LaneMarkingType.DoubleYellow:
                    encoding[3] = 1f;
                    break;
            }

            return encoding;
        }

        private void OnDrawGizmos()
        {
            Gizmos.color = gizmoColor;
            var collider = GetComponent<BoxCollider>();
            if (collider != null)
            {
                Gizmos.matrix = transform.localToWorldMatrix;
                Gizmos.DrawWireCube(collider.center, collider.size);
            }
        }
    }
}
