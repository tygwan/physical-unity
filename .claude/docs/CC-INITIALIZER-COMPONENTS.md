# cc-initializer Components Overview

**Source**: [cc-initializer](https://github.com/tygwan/cc-initializer)
**Project**: physical-unity (Autonomous Driving ML Platform)
**Last Updated**: 2026-01-29 (Updated: Added experiment-documenter & forensic-analyst agents)

---

## Table of Contents

1. [Agents (38)](#agents-38)
2. [Skills (22)](#skills-22)
3. [Commands (6)](#commands-6)
4. [Hooks (6)](#hooks-6)
5. [Configuration](#configuration)
6. [Usage Examples](#usage-examples)

---

## Agents (38)

### Core Framework Agents (26)

#### Documentation (7)
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| dev-docs-writer | 프로젝트 개시 시 개발 문서 생성 (PRD, 기술 설계서, 진행상황) | "프로젝트 시작", "개발 문서", "PRD", "기술 설계" |
| doc-generator | 기술 문서 생성 (README, API docs, 아키텍처) | "README", "API docs", "사용자 가이드" |
| doc-splitter | 대형 문서 분할 및 Phase 구조 생성 | "document", "split", "phase structure" |
| doc-validator | 개발 문서 완성도 검증 | "문서 검증", "검토", "완성도", "품질 체크" |
| prd-writer | PRD 작성 및 요구사항 정의 | "PRD", "요구사항", "기능 정의" |
| tech-spec-writer | 기술 설계서 작성 | "기술 설계", "아키텍처", "API 설계" |
| readme-helper | README 작성 및 개선 | "README", "배지", "구조 최적화" |

#### Project Management (5)
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| progress-tracker | 개발 진행상황 통합 추적 | "진행 상황", "progress", "tracking" |
| phase-tracker | Phase별 개발 진행 추적 및 관리 | "phase", "단계", "페이즈", "현재 단계" |
| project-analyzer | 프로젝트 구조, 기술 스택, 패턴 분석 | "analyze project", "프로젝트 분석", "구조 분석" |
| project-discovery | 새 프로젝트 시작 전 요구사항 파악 | "/init --full", "/init --discover", "프로젝트 시작" |
| work-unit-manager | 작업 단위 추적 및 그룹화 | "work unit", "세션", "changes", "그룹화" |

#### Code Quality (3)
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| code-reviewer | 코드 리뷰, 품질 검사, 보안 분석 | "review", "리뷰", "코드 리뷰", "검토", "분석" |
| refactor-assistant | 리팩토링 전문 | "refactor", "리팩토링", "코드 개선", "구조 개선" |
| test-helper | 테스트 작성 및 커버리지 분석 | "test", "테스트", "unit test", "coverage" |

#### Git/GitHub (5)
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| branch-manager | GitHub Flow 브랜치 관리 | "브랜치", "branch", "remote", "upstream" |
| commit-helper | Conventional Commits 커밋 메시지 작성 | "커밋", "커밋해", "저장해", "commit" |
| git-troubleshooter | Git 문제 해결 (충돌, 히스토리 복구) | "충돌", "conflict", "git 문제", "복구" |
| github-manager | GitHub 통합 관리 (gh CLI) | "github", "gh", "이슈", "CI", "워크플로우" |
| pr-creator | PR 생성 및 설명 작성 | "PR", "PR 만들어", "pull request" |

#### Analytics & Knowledge (2)
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| analytics-reporter | Agent/Skill 사용 통계 및 성과 리포트 | "통계", "사용량", "analytics", "성과", "리포트" |
| obsidian-sync | Obsidian vault 동기화 | "obsidian", "옵시디언", "vault", "노트" |

#### Infrastructure (4)
| Agent | Purpose | Trigger Keywords |
|-------|---------|------------------|
| config-validator | cc-initializer 설정 검증 | "설정 검증", "config check", "validate" |
| file-explorer | 프로젝트 정리 및 파일 분석 | "cleanup", "analyze files", "gitignore" |
| google-searcher | Google 웹 검색 및 정보 수집 | "검색", "search", "찾아봐", "구글" |
| agent-writer | Claude Code Agent 작성 전문가 | "create agent", "agent 작성" |

### ML/AD-Specific Agents (12)

| Agent | Purpose | Trigger Keywords | Tools |
|-------|---------|------------------|-------|
| **ad-experiment-manager** | AD 실험 생성, 실행, 비교, 추적 | "experiment", "실험", "training run", "학습 실행", "compare models", "MLflow", "W&B" | Read, Glob, Grep, Bash |
| **benchmark-evaluator** | nuPlan 벤치마크 실행, 메트릭 계산 | "evaluate", "평가", "benchmark", "벤치마크", "metrics", "collision rate", "nuPlan score" | Read, Glob, Grep, Bash |
| **dataset-curator** | 데이터셋 다운로드, 전처리, 큐레이션 | "dataset", "데이터셋", "nuPlan", "Waymo", "download data", "preprocessing" | Read, Glob, Grep, Bash |
| **experiment-documenter** | 자동 실험 문서화 및 결과 기록 (Codex 조율자) | "실험 문서화", "document experiment", "학습 완료", "training completed", "결과 기록", "update docs" | Read, Bash, Glob, Grep |
| **forensic-analyst** | ML 학습 실패 근본 원인 분석 (Codex 조율자, 수학적 검증) | "근본 원인", "root cause", "forensic", "분석 보고서", "왜 실패", "수학적 증명" | Read, Bash, Glob, Grep |
| **model-trainer** | RL/IL 학습 시작 및 관리 | "train", "학습", "PPO", "SAC", "GAIL", "BC", "start training", "학습 시작" | Read, Glob, Grep, Bash |
| **training-analyst** | 학습 결과 분석, 성공/실패 판정 | "결과 분석", "리포트", "왜 실패", "원인 분석", "학습 완료", "분석해줘" | Bash |
| **training-doc-manager** | 학습 문서 동기화, 아카이브 관리 | "문서 동기화", "아카이브", "진행 업데이트", "로그 정리", "문서 업데이트" | Bash |
| **training-monitor** | 실시간 학습 상태 모니터링 | "학습 상태", "진행률", "모니터링", "현재 reward", "몇 스텝", "training status" | Read, Bash, Glob, Grep |
| **training-orchestrator** | 학습 워크플로우 총괄 조율 | "다음 단계", "워크플로우", "전체 상태", "뭐 해야 해", "next step", "what's next" | Read, Write, Glob, Grep, Bash |
| **training-planner** | 실험 설계 및 Config 생성 | "실험 설계", "다음 버전", "config 생성", "Phase 계획", "새 실험", "v11", "v12", "phaseA" | Bash |
| **training-site-publisher** | GitHub Pages 사이트 발행 | "사이트 업데이트", "gh-pages", "웹 발행", "갤러리", "publish", "site update" | Bash |

---

## Skills (22)

### Core Skills (18)

#### Agile & Project Management (4)
| Skill | Description | Keywords | Command |
|-------|-------------|----------|---------|
| **agile-sync** | CHANGELOG, README, 진행상황 동기화 | "동기화", "sync", "changelog", "readme" | `/agile-sync` |
| **sprint** | Sprint lifecycle 관리 (velocity, burndown) | "sprint", "velocity", "burndown", "스프린트" | `/sprint` |
| **quality-gate** | 개발 lifecycle Quality Gate | "pre-commit", "pre-merge", "quality", "release" | `/quality-gate` |
| **sync-fix** | Phase/Sprint/문서 동기화 문제 해결 | "sync fix", "불일치", "동기화 문제" | `/sync-fix` |

#### Documentation & Knowledge (4)
| Skill | Description | Keywords | Command |
|-------|-------------|----------|---------|
| **dev-doc-system** | 개발 문서 통합 관리 | "문서 시스템", "개발 기록", "방향 설정", "회고" | `/dev-doc-system` |
| **feedback-loop** | 피드백 수집 및 ADR 생성 | "feedback", "learning", "retrospective", "ADR" | `/feedback-loop` |
| **obsidian** | Obsidian vault 동기화 | "obsidian", "vault", "지식 동기화", "노트" | `/obsidian` |
| **readme-sync** | README 자동 동기화 | "readme sync", "update readme", "README 업데이트" | `/readme-sync` |

#### Development Tools (6)
| Skill | Description | Keywords | Command |
|-------|-------------|----------|---------|
| **brainstorming** | 아이디어 구체화 및 대안 탐색 | "brainstorm", "아이디어", "alternative", "explore" | `/brainstorming` |
| **context-optimizer** | 컨텍스트 로딩 최적화 | "context", "token", "optimize", "summarize" | `/context-optimizer` |
| **gh** | GitHub CLI 통합 | "github", "issue", "CI", "workflow", "release" | `/gh` |
| **prompt-enhancer** | 프로젝트 컨텍스트 기반 프롬프트 향상 | "enhance prompt", "context-aware" | `/prompt-enhancer` |
| **analytics** | Tool/Agent 사용 통계 시각화 | "통계", "사용량", "analytics", "metrics", "리포트" | `/analytics` |
| **repair** | cc-initializer 자동 복구 | "repair", "fix", "troubleshoot", "복구" | `/repair` |

#### Meta-Development (4)
| Skill | Description | Keywords | Command |
|-------|-------------|----------|---------|
| **hook-creator** | Claude Code Hook 생성 | "create hook", "configure hook", "hook 만들기" | `/hook-creator` |
| **skill-creator** | 새로운 Skill 생성 가이드 | "create skill", "new skill", "skill 만들기" | `/skill-creator` |
| **subagent-creator** | 커스텀 Sub-agent 생성 | "create agent", "new agent", "agent 만들기" | `/subagent-creator` |
| **validate** | cc-initializer 설정 검증 | "validate", "검증", "check config", "설정 확인" | `/validate` |

### ML-Specific Skills (4)

| Skill | Description | Keywords | Command |
|-------|-------------|----------|---------|
| **dataset** | 데이터셋 다운로드, 전처리, 분할 | "dataset", "데이터셋", "download", "preprocessing" | `/dataset` |
| **experiment** | ML 실험 생성, 실행, 비교, 추적 | "experiment", "실험", "training run", "compare" | `/experiment` |
| **evaluate** | 모델 평가 및 벤치마크 실행 | "evaluate", "평가", "benchmark", "metrics" | `/evaluate` |
| **train** | RL/IL 학습 시작 및 모니터링 | "train", "학습", "PPO", "SAC", "GAIL" | `/train` |

---

## Commands (6)

### Workflow Commands (3)
| Command | Purpose | Integration | Key Features |
|---------|---------|-------------|--------------|
| **/bugfix** | 버그 수정 워크플로우 | Git + Phase + Sprint + Docs | 이슈 분석 → 코드 수정 → 테스트 → PR 생성 |
| **/feature** | 기능 개발 워크플로우 | Phase + Sprint + Git + Docs | Phase 전환 → Sprint 태스크 → 구현 → 문서화 → PR |
| **/release** | 릴리스 워크플로우 | Git + Docs + Archive | 버전 관리 → CHANGELOG → 문서 정리 → 배포 |

### Management Commands (2)
| Command | Purpose | Integration | Key Features |
|---------|---------|-------------|--------------|
| **/phase** | Phase 상태 확인, 전환, 진행률 업데이트 | Phase 시스템 | 현재 Phase 확인, Phase 전환, 체크리스트 업데이트 |
| **/git-workflow** | Git 워크플로우 관리 | GitHub Flow + Conventional Commits | 브랜치 전략, 커밋 규칙, PR 템플릿 |

### Planning Commands (1)
| Command | Purpose | Integration | Key Features |
|---------|---------|-------------|--------------|
| **/dev-doc-planner** | 개발 문서 작성 플래너 | PRD/TECH-SPEC/PROGRESS Templates | 문서 템플릿, 작성 가이드, 구조화 |

---

## Hooks (6)

### Pre-Tool Hooks (1)

#### pre-tool-use-safety.sh
- **Trigger**: Bash, Write, Edit 호출 전
- **Purpose**: 안전성 검사
- **Features**:
  - 위험 명령어 차단 (`rm -rf`, `git reset --hard`, `sudo rm` 등)
  - 민감 파일 보호 (`.env`, `credentials.json`, `.git/` 등)
  - 파괴적 작업 확인 요청

### Post-Tool Hooks (3)

#### auto-doc-sync.sh
- **Trigger**: Bash, Write, Edit 호출 후
- **Purpose**: 문서 자동 동기화
- **Features**:
  - CHANGELOG.md 자동 업데이트
  - README.md 통계 동기화
  - PROGRESS.md 진행 상황 업데이트

#### phase-progress.sh
- **Trigger**: Write, Edit 호출 후
- **Purpose**: Phase 진행 상황 업데이트
- **Features**:
  - Phase 문서 변경 감지
  - 체크리스트 완료율 계산
  - Phase 상태 자동 업데이트

#### post-tool-use-tracker.sh
- **Trigger**: Bash, Write, Edit 호출 후
- **Purpose**: Tool 사용 추적 (Analytics)
- **Features**:
  - Tool 호출 로깅 (JSONL)
  - 사용 통계 집계
  - 성과 메트릭 계산

### Notification Hooks (1)

#### notification-handler.sh
- **Trigger**: 모든 알림
- **Purpose**: 알림 처리 및 라우팅
- **Features**:
  - Phase 완료 알림
  - Sprint 마감 알림
  - Quality Gate 실패 알림

### Utility Hooks (1)

#### error-recovery.sh
- **Trigger**: Hook 실패 시
- **Purpose**: 자동 복구
- **Features**:
  - Hook 권한 자동 수정
  - 누락 디렉토리 생성
  - 로그 로테이션

---

## Configuration

### settings.json Highlights

```json
{
  "hooks": {
    "PreToolUse": ["pre-tool-use-safety.sh"],
    "PostToolUse": ["auto-doc-sync.sh", "phase-progress.sh", "post-tool-use-tracker.sh"],
    "Notification": ["notification-handler.sh"]
  },
  "agile": {
    "auto_changelog": true,
    "auto_readme_sync": true,
    "sprint_tracking": true
  },
  "phase": {
    "enabled": true,
    "auto_progress_update": true,
    "auto_check_completion": true
  },
  "quality-gate": {
    "pre-commit": { "enabled": true, "checks": ["lint", "format", "types", "secrets"] },
    "pre-merge": { "coverage_threshold": 80, "require_review": true }
  },
  "analytics": {
    "enabled": true,
    "track_tool_usage": true,
    "track_agent_calls": true
  },
  "sync": {
    "framework_source": "cc-initializer",
    "preserve_project_customizations": true
  }
}
```

---

## Usage Examples

### ML Training Workflow

```bash
# 1. 실험 계획
/experiment create --name phase-G --type RL --algorithm PPO

# 2. Config 생성
# training-planner agent가 자동으로 호출됨

# 3. 학습 시작
/train start --config python/configs/planning/vehicle_ppo_phase-G.yaml

# 4. 모니터링
# training-monitor agent가 실시간 로그 파싱

# 5. 결과 분석
# training-analyst agent가 성공/실패 판정

# 6. 문서 동기화
# training-doc-manager agent가 TRAINING-LOG.md 업데이트
```

### Feature Development Workflow

```bash
# 1. Phase 확인
/phase status

# 2. 기능 개발 시작
/feature start --name "intersection-navigation"

# 3. 코드 작성
# ... (개발 작업)

# 4. 커밋
# commit-helper agent가 Conventional Commits 메시지 생성

# 5. PR 생성
# pr-creator agent가 PR 설명 작성

# 6. 문서 동기화
# auto-doc-sync.sh hook이 자동 실행
```

### Quality Check Workflow

```bash
# 1. 코드 리뷰
# code-reviewer agent 호출

# 2. 테스트
# test-helper agent가 테스트 작성 지원

# 3. Pre-commit 검사
# quality-gate skill 실행

# 4. Pre-merge 검증
# coverage_threshold 80% 확인

# 5. Release 준비
# release command로 버전 관리
```

---

## Project-Specific Customizations

### ML/AD Agents
- **12개 AD 특화 Agent 추가**: 실험, 학습, 평가, 문서화 전문화
- **Codex 조율자 패턴**: experiment-documenter, forensic-analyst (haiku가 codex에 복잡한 작업 위임)
- **training-orchestrator**: 학습 워크플로우 총괄 (다른 agent 조율)

### ML Skills
- **4개 ML 특화 Skill**: dataset, experiment, evaluate, train
- **TensorBoard/MLflow 통합**: 실시간 모니터링

### Hooks
- **post-training-log.md**: 학습 완료 시 자동 로그 기록 (미구현)
- **error-recovery.sh**: Hook 실패 시 graceful degradation

### Analytics
- **JSONL 기반 메트릭 수집**: `.claude/analytics/metrics.jsonl`
- **CLI 차트 시각화**: ascii 차트로 통계 표시

---

## Sync & Update

### Framework Source
- **Source**: `https://github.com/tygwan/cc-initializer.git`
- **Merge Strategy**: `add_missing` (기존 커스터마이징 보존)
- **Auto-validation**: 동기화 후 자동 검증 실행

### Update Command
```bash
# cc-initializer 업데이트
/update --source cc-initializer

# 동기화 문제 해결
/sync-fix

# 설정 검증
/validate
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        cc-initializer Framework                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐          │
│  │   Agents (36) │   │  Skills (22)  │   │  Commands (6) │          │
│  │               │   │               │   │               │          │
│  │ Core:     26  │   │ Core:     18  │   │ Workflow:  3  │          │
│  │ ML/AD:    10  │   │ ML:        4  │   │ Mgmt:      2  │          │
│  │               │   │               │   │ Planning:  1  │          │
│  └───────────────┘   └───────────────┘   └───────────────┘          │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                        Hooks (6)                               │  │
│  │  PreToolUse (1) → PostToolUse (3) → Notification (1) → Error  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     settings.json                              │  │
│  │  Agile | Phase | Sprint | Quality-Gate | Analytics | Sync     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                       Integration Points                             │
│                                                                       │
│  Git/GitHub ◄──► Phase System ◄──► Sprint ◄──► Documentation        │
│       │                │               │              │              │
│       └────────────────┴───────────────┴──────────────┘              │
│                            │                                          │
│                       Analytics JSONL                                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    physical-unity Project                            │
│                                                                       │
│  Unity ML-Agents + PyTorch + ROS2 + AD Algorithms                    │
│  RL/IL Training + TensorBoard + MLflow + GitHub Pages                │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Benefits

### Automation
✅ Phase/Sprint 자동 진행 추적
✅ Git 워크플로우 자동화 (Conventional Commits)
✅ 문서 자동 동기화 (CHANGELOG, README, PROGRESS)
✅ Quality Gate 자동 실행

### ML/AD Specific
✅ 실험 추적 및 비교 (MLflow/W&B 통합)
✅ TensorBoard 모니터링 자동화
✅ 학습 문서 자동 생성 및 아카이브
✅ GitHub Pages 자동 발행

### Safety & Recovery
✅ 위험 명령어 차단 (`rm -rf`, `git reset --hard` 등)
✅ Hook 실패 시 자동 복구 (graceful degradation)
✅ 문서 손상 감지 및 복구

### Analytics
✅ Tool/Agent/Skill 사용 통계 (JSONL)
✅ CLI 차트 시각화
✅ 30일 데이터 보관 및 rollup

---

## Related Documentation

- [cc-initializer GitHub](https://github.com/tygwan/cc-initializer)
- [Project README](../../README.md)
- [Development Workflow](.claude/docs/ARCHITECTURE.md)
- [Hook Documentation](.claude/hooks/README.md)
- [Analytics Guide](.claude/scripts/analytics-visualizer.sh)

---

**Generated**: 2026-01-29
**Framework Version**: cc-initializer latest
**Project**: physical-unity (Autonomous Driving ML Platform)
