# Phase C Design: Multi-NPC Interaction & Advanced Decision-Making

Created: 2026-01-29
Status: Ready for Training
Expected Duration: 3.6M steps (~45-50 minutes)
Expected Reward: +850 (Stage 0) to +1500 (Stage 4)

## Executive Summary

Phase C builds on Phase B v2 success (+877 reward, 3 NPCs) by scaling to 5-8 NPCs while encouraging selective overtaking decisions.

Key Innovation: Dynamic overtaking incentive + blocked-detection reward suspension.

## Problem Statement

Phase B v2 succeeded with 3 NPCs (+877 reward) but showed OvertakePhase_Active=0 (no overtaking observed). This is actually correct behavior—in constrained 3-NPC scenario, overtaking risk exceeds reward.

Phase C addresses this by:
1. Gradually scaling NPC count (3→4→5→6→8)
2. Implementing dynamic overtaking bonus based on blocked duration
3. Suspending speed penalties when truly blocked (don't penalize for being stuck)

## Initialization: Phase B v2 Checkpoint

Decision: Use Phase B v2 (NOT Phase A)

Rationale:
- Phase B v2 has 3-NPC experience + decision-making
- Phase A has raw overtaking speed but no multi-NPC judgment
- Phase C is fundamentally about DECISIONS in crowded scenarios
- Transfer learning should be more efficient from B v2

