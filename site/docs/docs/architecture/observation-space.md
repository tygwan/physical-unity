# Observation Space

## Current: 254D (Phase D)

The observation vector provides a comprehensive view of the driving environment.

### Breakdown

| Segment | Dimensions | Content |
|---------|-----------|---------|
| Ego State | 8D | position (2), velocity (2), heading (1), acceleration (2), speed (1) |
| Route Info | 30D | waypoint positions, distances, headings |
| NPC Vehicles | 192D | 8 vehicles x 24 features (position, velocity, heading, relative state) |
| Lane Markings | 12D | left/right edge raycasts (distance, normal, type) |
| **Total** | **242D + 12D = 254D** | |

### Evolution

```
Phase A-C: 242D (ego + route + NPC)
Phase D:   254D (+ 12D lane markings)
Phase H+:  259D (+ 5D curvature/SOTIF, proposed)
```

### Lane Observation Detail (12D)

Added in Phase D, using raycasts to detect road boundaries:

| Feature | Dims | Description |
|---------|------|-------------|
| Left distance | 1 | Distance to left edge marking |
| Left normal | 3 | Surface normal of left marking |
| Left type | 2 | Marking type (solid/dashed, one-hot) |
| Right distance | 1 | Distance to right edge marking |
| Right normal | 3 | Surface normal of right marking |
| Right type | 2 | Marking type (solid/dashed, one-hot) |

### NPC Vehicle Features (24D per vehicle)

Each of the 8 nearest NPC vehicles provides:

- Relative position (3D)
- Relative velocity (3D)
- Absolute heading (1D)
- Relative heading (1D)
- Distance (1D)
- Time-to-collision (1D)
- Is-in-front flag (1D)
- Lane offset (1D)
- Size (3D)
- Additional state features (8D)
