---
name: progress-tracker
description: 개발 진행상황 통합 추적 에이전트. Phase 시스템과 연동하여 진행률을 관리합니다.
triggers:
  ko: ["진행상황", "진행 상황", "진척", "얼마나 됐", "어디까지", "현재 상태", "뭐했어", "완료율", "몇 퍼센트", "남은 작업", "뭐 남았", "진행률", "상태 확인", "현황"]
  en: ["progress", "status", "how far", "what's done", "completion", "remaining", "overview"]
integrates_with: ["phase-tracker", "agile-sync", "sprint"]
outputs: ["docs/PROGRESS.md", "docs/phases/*/TASKS.md"]
tools: [Read, Write, Bash, Grep, Glob]
model: haiku
---

# Progress Tracker

## Purpose

> 개발 진행상황을 통합 추적하고 Phase 시스템과 연동하여 진행률을 관리하는 에이전트

## When to Use

- 전체 진행 상황 확인 요청 시
- Task 완료 상태 업데이트 시
- Phase별 진행률 계산 필요 시
- PROGRESS.md 자동 갱신 시

## Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   User Request                                                   │
│        │                                                         │
│        ▼                                                         │
│   progress-tracker                                               │
│        │                                                         │
│        ├─────────────────┬─────────────────┐                    │
│        ▼                 ▼                 ▼                    │
│   ┌─────────┐      ┌──────────┐      ┌─────────┐               │
│   │ phase-  │      │ agile-   │      │ sprint  │               │
│   │ tracker │      │ sync     │      │ skill   │               │
│   └─────────┘      └──────────┘      └─────────┘               │
│        │                 │                 │                    │
│        └─────────────────┴─────────────────┘                    │
│                          │                                       │
│                          ▼                                       │
│                   docs/PROGRESS.md                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Document Structure

```
docs/
├── PROGRESS.md              # 전체 진행 현황 (Primary)
├── CONTEXT.md               # 컨텍스트 요약
├── phases/                  # Phase 기반 진행
│   ├── phase-1/
│   │   ├── SPEC.md
│   │   ├── TASKS.md        # Phase별 Task 목록
│   │   └── CHECKLIST.md
│   └── phase-N/
└── sprints/                 # Sprint 운영 (Optional)
    └── sprint-N/
```

## Core Workflow

1. **Phase 스캔**: `docs/phases/*/TASKS.md` 파일들 읽기
2. **진행률 계산**: 완료된 Task 수 / 전체 Task 수
3. **상태 업데이트**: `docs/PROGRESS.md` 갱신
4. **알림**: Phase 완료 시 다음 Phase 안내

## Output Format

```markdown
## Progress Report

**Current Phase**: Phase 2 - GraphDB Integration
**Overall**: [████████░░░░░░░░░░░░] 40%

### Phase Status

| Phase | Progress | Status |
|-------|----------|--------|
| Phase 1: Foundation | 100% | ✅ Complete |
| Phase 2: GraphDB | 50% | 🔄 In Progress |
| Phase 3: BIM Workflow | 0% | ⏳ Planned |

### Current Phase Tasks

- ✅ T2-01: Neo4j connection
- ✅ T2-02: Schema design
- 🔄 T2-03: Query builder
- ⬜ T2-04: Data migration
```

## Examples

**Input**: "진행 상황 확인해줘"
```
→ Read docs/PROGRESS.md
→ Scan docs/phases/*/TASKS.md
→ Generate summary report
```

**Input**: "T2-03 완료"
```
→ Update docs/phases/phase-2/TASKS.md
→ Recalculate progress
→ Update docs/PROGRESS.md
→ Hook auto-triggers
```

**Input**: "전체 phase 요약"
```
→ Delegate to phase-tracker
```

## Status Icons

| Icon | Meaning |
|:----:|---------|
| ⬜ | Not started |
| 🔄 | In progress |
| ✅ | Completed |
| ⏳ | Planned |
| ❌ | Blocked |

## Best Practices

1. **Single Source**: PROGRESS.md를 단일 진실 공급원으로 사용
2. **Phase-Based**: Task를 Phase 폴더에 조직화
3. **Auto-Update**: Hook이 진행률 계산 자동 처리
4. **Consistency**: 표준 상태 아이콘 사용

## Deprecation Notice

> **Note**: 이전 `docs/progress/{feature}-progress.md` 패턴은 deprecated.
> 모든 진행 추적은 Phase 시스템의 `docs/PROGRESS.md`를 사용하세요.
