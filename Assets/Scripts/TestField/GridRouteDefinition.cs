using UnityEngine;

namespace ADPlatform.TestField
{
    /// <summary>
    /// Defines a cyclic route through the 4x4 city block grid.
    /// Each route is a sequence of (row, col) intersections with turn types.
    /// Used by GridRoadNetwork to generate waypoints for agents and NPCs.
    /// </summary>
    [System.Serializable]
    public class GridRouteDefinition
    {
        /// <summary>Sequence of intersections as (row, col) pairs in the 5x5 grid.</summary>
        public Vector2Int[] waypoints;

        /// <summary>Turn type at each intersection: 0=straight, 1=left, 2=right.</summary>
        public int[] turnTypes;

        /// <summary>Whether this route loops back to the start.</summary>
        public bool isCyclic = true;

        public GridRouteDefinition(Vector2Int[] waypoints, int[] turnTypes, bool isCyclic = true)
        {
            this.waypoints = waypoints;
            this.turnTypes = turnTypes;
            this.isCyclic = isCyclic;
        }
    }

    /// <summary>
    /// Predefined routes for 12 agents with diverse patterns through the grid.
    /// Routes cover outer loops, inner loops, diagonals, and mixed paths.
    /// </summary>
    public static class GridRoutes
    {
        /// <summary>
        /// Get all 12 predefined agent routes.
        /// </summary>
        public static GridRouteDefinition[] GetAgentRoutes()
        {
            return new GridRouteDefinition[]
            {
                // Route 0: Outer clockwise
                // I00 -> I10 -> I20 -> I30 -> I40 -> I41 -> I42 -> I43 -> I44
                // -> I34 -> I24 -> I14 -> I04 -> I03 -> I02 -> I01 -> I00
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(0,0), v(1,0), v(2,0), v(3,0), v(4,0),
                        v(4,1), v(4,2), v(4,3), v(4,4),
                        v(3,4), v(2,4), v(1,4), v(0,4),
                        v(0,3), v(0,2), v(0,1)
                    },
                    new int[] {
                        0, 0, 0, 0, 1, // east then left-turn north
                        0, 0, 0, 1,    // north then left-turn west
                        0, 0, 0, 1,    // west then left-turn south
                        0, 0, 0        // south (loops back)
                    }),

                // Route 1: Outer counter-clockwise
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(0,0), v(0,1), v(0,2), v(0,3), v(0,4),
                        v(1,4), v(2,4), v(3,4), v(4,4),
                        v(4,3), v(4,2), v(4,1), v(4,0),
                        v(3,0), v(2,0), v(1,0)
                    },
                    new int[] {
                        0, 0, 0, 0, 2, // south then right-turn east
                        0, 0, 0, 2,    // east then right-turn north
                        0, 0, 0, 2,    // north then right-turn west
                        0, 0, 0        // west (loops back)
                    }),

                // Route 2: Inner clockwise (I11->I31->I33->I13)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(1,1), v(2,1), v(3,1),
                        v(3,2), v(3,3),
                        v(2,3), v(1,3),
                        v(1,2)
                    },
                    new int[] {
                        0, 0, 1,    // east then left-turn north
                        0, 1,       // north then left-turn west
                        0, 1,       // west then left-turn south
                        0           // south (loops back)
                    }),

                // Route 3: Inner counter-clockwise (I11->I13->I33->I31)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(1,1), v(1,2), v(1,3),
                        v(2,3), v(3,3),
                        v(3,2), v(3,1),
                        v(2,1)
                    },
                    new int[] {
                        0, 0, 2,    // south then right-turn east
                        0, 2,       // east then right-turn north
                        0, 2,       // north then right-turn west
                        0           // west (loops back)
                    }),

                // Route 4: Diagonal zigzag NE
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(0,0), v(1,0), v(2,0),
                        v(2,1), v(2,2),
                        v(3,2), v(4,2),
                        v(4,3), v(4,4),
                        v(3,4), v(2,4),
                        v(2,3), v(2,2),
                        v(1,2), v(0,2),
                        v(0,1)
                    },
                    new int[] {
                        0, 0, 1,    // east then left-turn north
                        0, 0,       // north straight
                        0, 1,       // east then left-turn north
                        0, 1,       // north then left-turn west
                        0, 2,       // west then right-turn south
                        0, 0,       // south straight
                        0, 2,       // west then right-turn south
                        0           // south (loops back)
                    }),

                // Route 5: Diagonal zigzag SW
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(4,4), v(3,4), v(2,4),
                        v(2,3), v(2,2),
                        v(1,2), v(0,2),
                        v(0,1), v(0,0),
                        v(1,0), v(2,0),
                        v(2,1), v(2,2),
                        v(3,2), v(4,2),
                        v(4,3)
                    },
                    new int[] {
                        0, 0, 1,
                        0, 0,
                        0, 1,
                        0, 1,
                        0, 2,
                        0, 0,
                        0, 2,
                        0
                    }),

                // Route 6: H-shape (I01->I41->I43->I03)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(0,1), v(1,1), v(2,1), v(3,1), v(4,1),
                        v(4,2), v(4,3),
                        v(3,3), v(2,3), v(1,3), v(0,3),
                        v(0,2)
                    },
                    new int[] {
                        0, 0, 0, 0, 1,  // east then left-turn north
                        0, 1,            // north then left-turn west
                        0, 0, 0, 1,     // west then left-turn south
                        0                // south (loops back)
                    }),

                // Route 7: Reverse H-shape (I10->I14->I34->I30)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(1,0), v(1,1), v(1,2), v(1,3), v(1,4),
                        v(2,4), v(3,4),
                        v(3,3), v(3,2), v(3,1), v(3,0),
                        v(2,0)
                    },
                    new int[] {
                        0, 0, 0, 0, 2,  // south then right-turn east
                        0, 2,            // east then right-turn north
                        0, 0, 0, 2,     // north then right-turn west
                        0                // west (loops back)
                    }),

                // Route 8: Cross vertical (I20->I24->I44->I40->I20)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(2,0), v(2,1), v(2,2), v(2,3), v(2,4),
                        v(3,4), v(4,4),
                        v(4,3), v(4,2), v(4,1), v(4,0),
                        v(3,0)
                    },
                    new int[] {
                        0, 0, 0, 0, 2,
                        0, 2,
                        0, 0, 0, 2,
                        0
                    }),

                // Route 9: Cross horizontal (I02->I42->I44->I04->I02)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(0,2), v(1,2), v(2,2), v(3,2), v(4,2),
                        v(4,3), v(4,4),
                        v(3,4), v(2,4), v(1,4), v(0,4),
                        v(0,3)
                    },
                    new int[] {
                        0, 0, 0, 0, 1,
                        0, 1,
                        0, 0, 0, 1,
                        0
                    }),

                // Route 10: Center small loop (I12->I22->I23->I13)
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(1,2), v(2,2),
                        v(2,3),
                        v(1,3)
                    },
                    new int[] {
                        0, 1,  // east then left-turn north
                        1,     // north then left-turn west
                        1      // west then left-turn south (loops back)
                    }),

                // Route 11: Outer + center cut
                new GridRouteDefinition(
                    new Vector2Int[] {
                        v(0,0), v(1,0), v(2,0),
                        v(2,1), v(2,2), v(2,3), v(2,4),
                        v(1,4), v(0,4),
                        v(0,3), v(0,2),
                        v(1,2), v(2,2),
                        v(2,1), v(2,0),
                        v(1,0)
                    },
                    new int[] {
                        0, 0, 1,
                        0, 0, 0, 1,
                        0, 1,
                        0, 2,
                        0, 2,
                        0, 2,
                        0
                    }),
            };
        }

        /// <summary>
        /// Get simple NPC routes (straight lines along grid edges).
        /// </summary>
        public static GridRouteDefinition[] GetNPCRoutes()
        {
            return new GridRouteDefinition[]
            {
                // Horizontal routes (east-bound)
                new GridRouteDefinition(new Vector2Int[] { v(0,0), v(1,0), v(2,0), v(3,0), v(4,0) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(0,1), v(1,1), v(2,1), v(3,1), v(4,1) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(0,2), v(1,2), v(2,2), v(3,2), v(4,2) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(0,3), v(1,3), v(2,3), v(3,3), v(4,3) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(0,4), v(1,4), v(2,4), v(3,4), v(4,4) },
                    new int[] { 0, 0, 0, 0, 0 }),
                // Vertical routes (north-bound)
                new GridRouteDefinition(new Vector2Int[] { v(0,0), v(0,1), v(0,2), v(0,3), v(0,4) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(1,0), v(1,1), v(1,2), v(1,3), v(1,4) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(2,0), v(2,1), v(2,2), v(2,3), v(2,4) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(3,0), v(3,1), v(3,2), v(3,3), v(3,4) },
                    new int[] { 0, 0, 0, 0, 0 }),
                new GridRouteDefinition(new Vector2Int[] { v(4,0), v(4,1), v(4,2), v(4,3), v(4,4) },
                    new int[] { 0, 0, 0, 0, 0 }),
            };
        }

        private static Vector2Int v(int col, int row) => new Vector2Int(col, row);
    }
}
