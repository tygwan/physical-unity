# Phase D v3: Lane Observation - Training Analysis Report

**Status**: SUCCESS
**Run ID**: phase-D-v3-254d
**Steps**: 5,000,103 / 5,000,000
**Final Reward**: +895.5

## Executive Summary

Phase D v3 successfully validated that 12D lane observations provide a measurable +7.2% reward improvement over the 242D baseline. By applying P-009 (Observation-Curriculum Coupling Ban), the training converged smoothly without any collapse events, in contrast to D v1/v2 which both failed from simultaneous observation+curriculum changes.

The two-run approach (accidental 242D baseline + intentional 254D run) provided a clean A/B comparison that conclusively demonstrates the value of lane marking observations.

## Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Final Reward | > +800 | +895.5 | PASS |
| Lane Obs Benefit | > 0% | +7.2% | PASS |
| Curriculum Collapse | None | None | PASS |
| Peak Reward | > +800 | +895.5 @ 5.0M | PASS |
| Convergence Point | < 4M | ~3.0M steps | PASS |

## Reward Trajectory

| Step | Mean Reward | Notes |
|-----:|----------:|-------|
| 500K | -36.5 | Exploration phase |
| 1.0M | -392.0 | Exploration burst (expected for fresh start) |
| 1.5M | +105.8 | Breakthrough - first positive reward |
| 2.0M | +384.3 | Rapid improvement |
| 2.5M | +553.1 | Steady climb |
| 3.0M | +878.8 | Near peak convergence |
| 3.5M | +883.2 | Plateau |
| 4.0M | +855.4 | Minor dip (within variance) |
| 4.5M | +849.6 | Stable |
| 5.0M | +895.5 | Final peak |

### Learning Curve Shape
- **0-1.5M**: Exploration and initial learning (negative rewards expected)
- **1.5-3.0M**: S-curve rapid improvement (+105 -> +878)
- **3.0-5.0M**: Convergence plateau with minor fluctuations

## 242D vs 254D Comparison

| Metric | 242D (No Lane Obs) | 254D (Lane Obs) | Delta |
|--------|-------------------|-----------------|-------|
| Final Reward | +835.0 | +895.5 | **+60.5 (+7.2%)** |
| Peak Reward | +838.7 @ 3M | +895.5 @ 5M | +56.8 |
| Breakthrough | 1.5M steps | 1.5M steps | Same |
| Convergence | 3.0M steps | 3.0M steps | Same |
| Stability | Smooth | Smooth | Same |

## Checkpoints

| Steps | Reward | File |
|------:|-------:|------|
| 999,880 | -392.0 | E2EDrivingAgent-999880.pt |
| 1,499,818 | +105.8 | E2EDrivingAgent-1499818.pt |
| 1,999,962 | +384.3 | E2EDrivingAgent-1999962.pt |
| 2,499,849 | +553.1 | E2EDrivingAgent-2499849.pt |
| 2,999,965 | +878.8 | E2EDrivingAgent-2999965.pt |
| 3,499,956 | +883.2 | E2EDrivingAgent-3499956.pt |
| 3,999,957 | +855.4 | E2EDrivingAgent-3999957.pt |
| 4,499,900 | +849.6 | E2EDrivingAgent-4499900.pt |
| 4,999,947 | +895.5 | E2EDrivingAgent-4999947.pt (best) |
| 5,000,103 | +895.5 | Final checkpoint |

## Policy Discoveries

### P-009 Validated: Observation-Curriculum Coupling Ban
- D v1/v2: Changed observations + curriculum -> collapse
- D v3: Changed observations + fixed environment -> smooth convergence
- Conclusion: **Never change observation space and curriculum simultaneously**

### P-010 Confirmed: Scene-Config-Code Consistency
- Run 1 accidentally trained with VectorObservationSize=242 (lane obs disabled)
- Run 2 fixed to VectorObservationSize=254 (lane obs enabled)
- Lesson: **Always verify Unity scene config matches expected observation dimension**

## Recommendations

1. **Use 254D as the new baseline** for subsequent phases (E, F, etc.)
2. **Lane observations provide diminishing returns** in simple environments (+7.2%) but may be more valuable in multi-lane scenarios (Phase F)
3. **Fresh start from zero is viable** for observation space changes - smooth S-curve learning within 5M steps

---
**Analysis Confidence**: HIGH
**Recommended Action**: Proceed to Phase E with 254D observation space
