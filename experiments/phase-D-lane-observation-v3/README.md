# Phase D v3: Lane Observation - Fixed Environment (254D)

## Status: COMPLETED

## Quick Reference

| Item | Value |
|------|-------|
| Run ID | `phase-D-v3-254d` |
| Config | `python/configs/planning/vehicle_ppo_phase-D.yaml` |
| Scene | PhaseB_DecisionLearning |
| Agent | E2EDrivingAgentBv2 |
| Observation | 254D (242D + 12D lane) |
| Max Steps | 5,000,000 |
| Init Path | None (fresh start, 242D->254D incompatible) |

## Results

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Final Reward | +800 | +895.5 | PASS |
| Lane Obs Benefit | > 0% | +7.2% vs 242D baseline | PASS |
| Curriculum Collapse | None | None | PASS |
| Peak Reward | +800 | +895.5 @ 5.0M | PASS |

## Strategy

Phase D v1/v2 both failed from simultaneous observation+curriculum changes.
v3 applies **P-009 (Observation-Curriculum Coupling Ban)**: when observation space changes (242D->254D), remove ALL curriculum complexity.

- **No curriculum**: Fixed environment (3 NPC, 0.6 speed ratio)
- **No checkpoint transfer**: 242D->254D dimension mismatch prevents --initialize-from
- **Fresh start**: Agent learns from scratch with 254D in a stable environment
- **Validation**: Compare 254D reward against 242D baseline (+835) to measure lane observation benefit

## Checkpoints

| File | Step | Reward | Notes |
|------|------|--------|-------|
| E2EDrivingAgent-999880.pt | 1.0M | -392.0 | Early learning |
| E2EDrivingAgent-1499818.pt | 1.5M | +105.8 | Positive reward |
| E2EDrivingAgent-1999962.pt | 2.0M | +384.3 | Rapid improvement |
| E2EDrivingAgent-2499849.pt | 2.5M | +553.1 | Steady climb |
| E2EDrivingAgent-2999965.pt | 3.0M | +878.8 | Near peak |
| E2EDrivingAgent-3499956.pt | 3.5M | +883.2 | Plateau |
| E2EDrivingAgent-3999957.pt | 4.0M | +855.4 | Minor dip |
| E2EDrivingAgent-4499900.pt | 4.5M | +849.6 | Stable |
| E2EDrivingAgent-4999947.pt | 5.0M | **+895.5** | **Peak / Final** |

## Key Findings

1. **P-009 validated**: Removing curriculum when observation changes = no collapse (vs D v1/v2 both collapsed)
2. **Lane observation benefit**: +895.5 (254D) vs +835 (242D) = +7.2% improvement
3. **Learning trajectory**: Smooth S-curve from -392 to +895.5 without any collapse events
4. **Convergence**: ~3.0M steps to reach near-peak performance

## Related

- Design: `../phase-D-lane-observation-v2/DESIGN_V3.md`
- Previous (failed): `../phase-D-lane-observation-v2/`
- Next: `../phase-E-curved-roads/`
