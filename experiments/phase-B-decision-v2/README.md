# Phase B v2: Decision Learning - Hybrid Approach

## Status: SUCCESS

## Quick Reference

| Item | Value |
|------|-------|
| Run ID | `phase-B-decision-v2` |
| Config | `python/configs/planning/vehicle_ppo_phase-B-v2.yaml` |
| Scene | PhaseB_DecisionLearning |
| Agent | E2EDrivingAgentBv2 |
| Observation | 242D |
| Max Steps | 3,500,000 |
| Init Path | `results/phase-A-overtaking/E2EDrivingAgent/E2EDrivingAgent-2500155.pt` |
| Duration | 11 min 48 sec |

## Results

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Final Reward | +600 | **+877** | EXCEED |
| Peak Reward | +800 | **+897** | EXCEED |
| Speed | >12 m/s | **12.9 m/s** | PASS |
| Collision Rate | <10% | **~0%** | PASS |
| Curriculum | All 4 stages | **All 4 stages** | COMPLETE |

## v1 Failure → v2 Fix

Phase B v1 failed with -108 reward (agent stopped completely). v2 applied:
1. **Phase A checkpoint init** (v1 used Phase 0 - no overtaking skill)
2. **Zero hyperparameter changes** from Phase A (v1 changed 7 parameters)
3. **Gradual curriculum** 0→1→2→3 NPCs (v1 started at 2 NPCs immediately)
4. **Fixed reward function**: speedUnderPenalty -0.1→-0.02, blocked detection added

## Curriculum Progression

| Stage | NPCs | Steps | Reward |
|-------|------|-------|--------|
| 0: Solo Warmup | 0 | 2,505K-2,710K | +1,340 |
| 1: Single Slow NPC | 1 | 2,710K-3,015K | -594→+500 |
| 2: Two Mixed NPCs | 2 | 3,015K-3,320K | +630→+845 |
| 3: Three Mixed NPCs | 3 | 3,320K-3,500K | +825→+897 |

## Model Artifacts

- Final: `results/phase-B-decision-v2/E2EDrivingAgent/E2EDrivingAgent-3500347.onnx`
- ONNX: `results/phase-B-decision-v2/E2EDrivingAgent.onnx`

## Related

- v1 failure: `../phase-B-decision/ROOT-CAUSE-ANALYSIS.md`
- Design: `DESIGN.md`
- Analysis: `ANALYSIS.md`
- Next: Phase C Multi-NPC (`../phase-C-multi-npc/`)
