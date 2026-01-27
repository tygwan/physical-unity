# Progress Tracker

## Project: Autonomous Driving ML Platform

**Focus**: Planning (RL/IL 모션 플래닝)

---

## Current Status

| Metric | Value |
|--------|-------|
| **Current Phase** | Phase G - Intersection Navigation |
| **Current Training** | v12_phaseG (교차로 학습) |
| **Steps** | ~750K / 8,000,000 (9.4%) |
| **Current Reward** | +492 |
| **Overall Progress** | 75% |
| **Last Updated** | 2026-01-27 14:30 |

---

## Training Dashboard

### Active Training 🔄

```
Phase G: Intersection Navigation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Progress: [████░░░░░░░░░░░░░░░░░░░░░░░░░░] 9.4%
Steps:    750,000 / 8,000,000
Reward:   +492 (target: +800 for curriculum transition)
Time:     ~14 minutes elapsed
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Curriculum State

| Parameter | Current Lesson | Value | Next Threshold |
|-----------|----------------|-------|----------------|
| intersection_type | NoIntersection | 0 | +800 → T-Junction |
| turn_direction | StraightOnly | 0 | +700 → LeftTurn |
| num_active_npcs | NoNPCs | 0 | +700 → OneNPC |
| goal_distance | ShortGoal | 120m | +600 → MediumGoal |

---

## Phase History

| Phase | Steps | Best Reward | Final Reward | Status | Date |
|-------|-------|-------------|--------------|--------|------|
| v10g | 8M | +95 | +40 | ⚠️ Plateau | 2026-01-23 |
| v11 | 8M | +51 | +41 | ❌ Failed | 2026-01-24 |
| **Phase A** | 2M | **+937** | +714 | ✅ Complete | 2026-01-25 |
| **Phase B** | 2M | **+994** | +903 | ✅ Complete | 2026-01-25 |
| **Phase C** | 4M | **+1086** | +961 | ✅ Complete | 2026-01-26 |
| Phase D | 6M | +402 | +332 | ⏭️ Skipped | 2026-01-26 |
| **Phase E** | 6M | **+931** | +931 | ✅ Complete | 2026-01-27 |
| **Phase F** | 6M | **+988** | +988 | ✅ Complete | 2026-01-27 |
| **Phase G** | 8M | +492 | 🔄 | 🔄 In Progress | 2026-01-27 |

---

## Recent Training Progress (Phase G)

| Step | Reward | Std | Curriculum | Time |
|------|--------|-----|------------|------|
| 100K | +439 | 5 | NoIntersection | 6min |
| 200K | +442 | 6 | NoIntersection | 7min |
| 300K | +456 | 8 | NoIntersection | 9min |
| 400K | +467 | 6 | NoIntersection | 10min |
| 500K | +480 | 15 | NoIntersection | 11min |
| 600K | +496 | 16 | NoIntersection | 13min |
| 700K | +474 | 94 | NoIntersection | 14min |
| **750K** | **+492** | - | NoIntersection | **14min** |

**Trend**: 점진적 상승 중 (+423 → +492), threshold 800까지 약 300 gap

---

## Upcoming Milestones

| Milestone | Expected Step | Condition |
|-----------|---------------|-----------|
| T-Junction 도입 | ~1-1.5M | reward > 800 |
| Cross 교차로 | ~2-3M | T-Junction reward > 600 |
| Y-Junction | ~4-5M | Cross reward > 500 |
| 좌회전 학습 | ~3-4M | turn_direction curriculum |
| 우회전 학습 | ~5-6M | turn_direction curriculum |
| Phase G 완료 | ~8M | 모든 curriculum 완료 |

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [TRAINING-LOG.md](./TRAINING-LOG.md) | 실시간 학습 데이터 상세 기록 |
| [LEARNING-ROADMAP.md](./LEARNING-ROADMAP.md) | 전략/분석/다음 계획 |
| [README.md](../README.md) | 프로젝트 개요 |

---

## Notes

- Phase G는 도로 곡률을 0으로 단순화하여 교차로 학습에 집중
- NPC 수를 2대로 제한하여 복잡도 관리
- Phase F checkpoint에서 초기화하여 기존 능력 유지

---

*이 문서는 학습 진행에 따라 자동으로 업데이트됩니다.*
