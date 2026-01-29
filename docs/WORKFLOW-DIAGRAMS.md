# Physical-Unity Workflow Diagrams

> Powered by [cc-initializer](https://github.com/tygwan/cc-initializer) | 38 Agents, 22 Skills, 6 Hooks, 6 Commands

---

## 1. System Overview (Top-Level Architecture)

```mermaid
graph TB
    subgraph USER["User / Developer"]
        CMD["/commands"]
        CHAT["Natural Language"]
    end

    subgraph CC["cc-initializer Framework"]
        direction TB
        subgraph AGENTS["Agents (38)"]
            CORE_AG["Core (26)"]
            ML_AG["ML/AD (12)"]
        end
        subgraph SKILLS["Skills (22)"]
            CORE_SK["Core (18)"]
            ML_SK["ML (4)"]
        end
        subgraph HOOKS["Hooks (6)"]
            PRE["PreToolUse (1)"]
            POST["PostToolUse (3)"]
            NOTI["Notification (1)"]
            ERR["Error Recovery (1)"]
        end
        subgraph CMDS["Commands (6)"]
            FEAT["/feature"]
            BUG["/bugfix"]
            REL["/release"]
            PH["/phase"]
            GIT["/git-workflow"]
            DOC["/dev-doc-planner"]
        end
        SETTINGS["settings.json"]
    end

    subgraph PROJECT["physical-unity Project"]
        UNITY["Unity 6 + ML-Agents"]
        PYTHON["Python + PyTorch"]
        RESULTS["results/ (TensorBoard)"]
        MODELS["models/ (.onnx)"]
        DOCS["docs/ (Markdown)"]
    end

    subgraph EXTERNAL["External Services"]
        GH["GitHub (gh CLI)"]
        TB["TensorBoard"]
        MLFLOW["MLflow"]
        GHPAGES["GitHub Pages"]
    end

    CMD --> CMDS
    CHAT --> AGENTS
    CHAT --> SKILLS
    CMDS --> AGENTS
    SKILLS --> AGENTS
    AGENTS --> HOOKS
    HOOKS --> PROJECT
    AGENTS --> PROJECT
    AGENTS --> EXTERNAL
    SETTINGS -.->|configure| HOOKS
    SETTINGS -.->|configure| AGENTS
    SETTINGS -.->|configure| SKILLS

    style USER fill:#e1f5fe
    style CC fill:#fff3e0
    style PROJECT fill:#e8f5e9
    style EXTERNAL fill:#fce4ec
```

---

## 2. ML Training Workflow (Core Loop)

```mermaid
flowchart TD
    START((Start)) --> PLAN

    subgraph PLAN_PHASE["1. PLAN"]
        PLAN["training-planner"]
        PLAN --> |"실험 설계"| CONFIG["YAML Config 생성"]
        CONFIG --> |"python/configs/planning/"| YAML["vehicle_ppo_phase-X.yaml"]
    end

    subgraph TRAIN_PHASE["2. TRAIN"]
        YAML --> TRAIN_CMD["mlagents-learn"]
        TRAIN_CMD --> |"results/phase-X/"| TRAINING["model-trainer"]
        TRAINING --> MONITOR["training-monitor"]
        MONITOR --> |"TensorBoard\nReward, Loss"| CHECK{Converged?}
    end

    subgraph ANALYZE_PHASE["3. ANALYZE"]
        CHECK -->|Yes| ANALYST["training-analyst"]
        CHECK -->|No, Continue| MONITOR
        CHECK -->|Stuck/Failed| FORENSIC["forensic-analyst"]
        FORENSIC --> |"Root Cause\n수학적 검증"| DECISION
        ANALYST --> |"Success/Fail\n판정"| DECISION
    end

    subgraph DECIDE_PHASE["4. DECIDE (training-orchestrator)"]
        DECISION{Phase Success?}
        DECISION -->|Success| DOC_PHASE
        DECISION -->|Fail: Retry| PLAN
        DECISION -->|Fail: Redesign| PLAN
    end

    subgraph DOC_PHASE["5. DOCUMENT"]
        DOC_PHASE["experiment-documenter"]
        DOC_PHASE --> UPDATE_LOG["docs/TRAINING-LOG.md"]
        DOC_PHASE --> UPDATE_PROG["docs/PROGRESS.md"]
        DOC_PHASE --> UPDATE_README["README.md"]
        DOC_PHASE --> ANALYSIS_MD["experiments/phase-X/ANALYSIS.md"]
    end

    subgraph PUBLISH_PHASE["6. PUBLISH"]
        UPDATE_LOG --> SITE["training-site-publisher"]
        SITE --> |"gh-pages"| GHPAGES["GitHub Pages"]
    end

    ANALYSIS_MD --> NEXT_PHASE["Next Phase"]
    NEXT_PHASE --> PLAN

    style PLAN_PHASE fill:#e3f2fd
    style TRAIN_PHASE fill:#fff8e1
    style ANALYZE_PHASE fill:#fce4ec
    style DECIDE_PHASE fill:#f3e5f5
    style DOC_PHASE fill:#e8f5e9
    style PUBLISH_PHASE fill:#e0f2f1
```

---

## 3. Training Orchestrator State Machine

```mermaid
stateDiagram-v2
    [*] --> IDLE

    IDLE --> PLANNING: "새 실험" / "다음 Phase"
    PLANNING --> TRAINING: Config 완료

    state PLANNING {
        [*] --> DesignExperiment
        DesignExperiment --> GenerateConfig
        GenerateConfig --> ValidateConfig
        ValidateConfig --> [*]
        note right of DesignExperiment: training-planner agent
    }

    state TRAINING {
        [*] --> StartTraining
        StartTraining --> Monitoring
        Monitoring --> CheckProgress
        CheckProgress --> Monitoring: Not converged
        CheckProgress --> [*]: Converged / Failed
        note right of Monitoring: training-monitor agent\n(TensorBoard parsing)
    }

    TRAINING --> ANALYZING: 학습 완료/중단

    state ANALYZING {
        [*] --> ParseResults
        ParseResults --> EvaluateMetrics
        EvaluateMetrics --> JudgeSuccess
        JudgeSuccess --> [*]
        note right of EvaluateMetrics: training-analyst agent
    }

    ANALYZING --> DECIDING

    state DECIDING {
        [*] --> AssessOutcome
        AssessOutcome --> NextPhase: Success
        AssessOutcome --> RetryWithFix: Recoverable Failure
        AssessOutcome --> ForensicAnalysis: Unknown Failure
        ForensicAnalysis --> RetryWithFix
        NextPhase --> [*]
        RetryWithFix --> [*]
        note right of ForensicAnalysis: forensic-analyst agent\n(Codex orchestrator)
    }

    DECIDING --> DOCUMENTING: 결정 완료

    state DOCUMENTING {
        [*] --> GenerateReport
        GenerateReport --> UpdateDocs
        UpdateDocs --> SyncReadme
        SyncReadme --> PublishSite
        PublishSite --> [*]
        note right of GenerateReport: experiment-documenter\n(Codex orchestrator)
    }

    DOCUMENTING --> IDLE: Phase 완료
    DECIDING --> PLANNING: Retry
```

---

## 4. Feature Development Workflow (/feature)

```mermaid
flowchart TD
    subgraph START_CMD["/feature start"]
        A1["1. branch-manager\nfeat/feature-name 생성"] --> A2["2. phase-tracker\nPhase Task 연결"]
        A2 --> A3["3. Sprint Item 추가\n(/sprint add)"]
        A3 --> A4["4. progress-tracker\nPROGRESS.md 업데이트"]
        A4 --> A5["5. context-optimizer\n관련 파일 자동 로드"]
    end

    subgraph DEV["개발 작업"]
        A5 --> CODE["코드 작성\n(Bash/Write/Edit)"]

        CODE --> PRE_HOOK{{"PreToolUse Hook\npre-tool-use-safety.sh"}}
        PRE_HOOK -->|Safe| EXECUTE["도구 실행"]
        PRE_HOOK -->|Dangerous| BLOCK["차단 + 경고"]

        EXECUTE --> POST_HOOK{{"PostToolUse Hooks"}}
        POST_HOOK --> SYNC["auto-doc-sync.sh\n문서 자동 동기화"]
        POST_HOOK --> PHASE["phase-progress.sh\nPhase 진행률 업데이트"]
        POST_HOOK --> TRACK["post-tool-use-tracker.sh\nAnalytics 기록"]
    end

    subgraph PROGRESS_CMD["/feature progress"]
        SYNC --> P1["진행상황 리포트"]
        PHASE --> P1
        P1 --> P2["Phase 체크리스트\n완료율 계산"]
    end

    subgraph COMPLETE_CMD["/feature complete"]
        P2 --> C1["1. code-reviewer\n코드 리뷰"]
        C1 --> C2["2. test-helper\n테스트 작성/실행"]
        C2 --> C3{{"Quality Gate\npre-commit checks"}}
        C3 -->|lint, format, types, secrets| C4["3. commit-helper\nConventional Commits"]
        C4 --> C5["4. pr-creator\nPR 생성 (gh pr create)"]
        C5 --> C6["5. phase-tracker\nPhase 진행률 업데이트"]
        C6 --> C7["6. Sprint 포인트 반영"]
    end

    style START_CMD fill:#e3f2fd
    style DEV fill:#fff8e1
    style PROGRESS_CMD fill:#e8f5e9
    style COMPLETE_CMD fill:#f3e5f5
```

---

## 5. Bug Fix Workflow (/bugfix)

```mermaid
flowchart TD
    subgraph BUGFIX_START["/bugfix start"]
        B1["branch-manager\nfix/issue-name-123"] --> B2["gh issue view 123\n이슈 정보 로드"]
        B2 --> B3["/sprint add\n긴급 항목 추가"]
        B3 --> B4["Root Cause 분석\ncode-reviewer / forensic-analyst"]
    end

    subgraph BUGFIX_ANALYZE["/bugfix analyze"]
        B4 --> BA1["project-analyzer\n관련 코드 탐색"]
        BA1 --> BA2["code-reviewer\n문제 코드 식별"]
        BA2 --> BA3["forensic-analyst\n근본 원인 분석\n(수학적 검증)"]
    end

    subgraph BUGFIX_FIX["수정 작업"]
        BA3 --> FIX["코드 수정"]
        FIX --> PRE{{"pre-tool-use-safety.sh"}}
        PRE --> TEST["test-helper\n회귀 테스트 작성"]
        TEST --> POST{{"PostToolUse Hooks\nauto-doc-sync\nphase-progress\ntracker"}}
    end

    subgraph BUGFIX_COMPLETE["/bugfix complete"]
        POST --> BC1{{"Quality Gate"}}
        BC1 --> BC2["commit-helper\nfix: 커밋 메시지"]
        BC2 --> BC3["pr-creator\nPR 생성"]
        BC3 --> BC4["feedback-loop\n학습 기록 (ADR)"]
        BC4 --> BC5["Sprint 포인트 반영"]
    end

    style BUGFIX_START fill:#ffebee
    style BUGFIX_ANALYZE fill:#fff3e0
    style BUGFIX_FIX fill:#e8f5e9
    style BUGFIX_COMPLETE fill:#e3f2fd
```

---

## 6. Release Workflow (/release)

```mermaid
flowchart TD
    subgraph PREPARE["/release prepare v1.x.0"]
        R1["버전 결정\n(major/minor/patch)"] --> R2["phase-tracker\n모든 Phase 검증"]
        R2 --> R3["/sprint end\n현재 Sprint 종료"]
        R3 --> R4{{"Quality Gate\npre-release"}}
        R4 --> |"coverage ≥80%\nsecurity scan\nall docs"| R5["CHANGELOG 정리\nUnreleased → v1.x.0"]
        R5 --> R6["릴리스 체크리스트 생성"]
    end

    subgraph CREATE["/release create"]
        R6 --> C1["git tag v1.x.0"]
        C1 --> C2["gh release create"]
        C2 --> C3["Release Notes 생성"]
    end

    subgraph PUBLISH["/release publish"]
        C3 --> P1["Sprint 아카이브"]
        P1 --> P2["Retrospective 생성"]
        P2 --> P3["README 업데이트\n(readme-sync)"]
        P3 --> P4["training-site-publisher\ngh-pages 발행"]
    end

    subgraph POST_RELEASE["Post-Release Hooks"]
        P4 --> PR1["feedback-loop\nRetro 프롬프트"]
        PR1 --> PR2["obsidian-sync\n지식 동기화"]
        PR2 --> PR3["analytics-reporter\n통계 리포트"]
    end

    style PREPARE fill:#e3f2fd
    style CREATE fill:#fff8e1
    style PUBLISH fill:#e8f5e9
    style POST_RELEASE fill:#f3e5f5
```

---

## 7. Hook Execution Flow

```mermaid
flowchart LR
    subgraph TRIGGER["Tool Invocation"]
        BASH["Bash"]
        WRITE["Write"]
        EDIT["Edit"]
    end

    subgraph PRE["PreToolUse"]
        SAFETY["pre-tool-use-safety.sh"]
        SAFETY --> CHECK_CMD{"Dangerous?"}
        CHECK_CMD -->|"rm -rf, git reset\n--hard, sudo rm"| BLOCK["BLOCK"]
        CHECK_CMD -->|"Safe"| PASS["PASS"]
        SAFETY --> CHECK_FILE{"Sensitive File?"}
        CHECK_FILE -->|".env, credentials\n.git/"| WARN["WARN"]
        CHECK_FILE -->|"Normal"| PASS
    end

    subgraph EXEC["Tool Execution"]
        PASS --> RUN["Execute Tool"]
    end

    subgraph POST["PostToolUse"]
        RUN --> DOC_SYNC["auto-doc-sync.sh\nCHANGELOG, README\nPROGRESS 동기화"]
        RUN --> PHASE_PROG["phase-progress.sh\nPhase 체크리스트\n완료율 계산"]
        RUN --> TRACKER["post-tool-use-tracker.sh\nmetrics.jsonl\n사용 통계"]
    end

    subgraph NOTIFY["Notification"]
        DOC_SYNC --> NOTI_HANDLER["notification-handler.sh"]
        PHASE_PROG --> NOTI_HANDLER
        NOTI_HANDLER --> PHASE_DONE["Phase 완료 알림"]
        NOTI_HANDLER --> SPRINT_WARN["Sprint 마감 알림"]
        NOTI_HANDLER --> QUALITY_FAIL["Quality 실패 알림"]
    end

    subgraph RECOVERY["Error Recovery"]
        SAFETY -.->|"Hook Fail"| ERR_REC["error-recovery.sh"]
        DOC_SYNC -.->|"Hook Fail"| ERR_REC
        PHASE_PROG -.->|"Hook Fail"| ERR_REC
        TRACKER -.->|"Hook Fail"| ERR_REC
        ERR_REC --> FIX_PERM["권한 수정"]
        ERR_REC --> FIX_DIR["디렉토리 생성"]
        ERR_REC --> FIX_LOG["로그 로테이션"]
    end

    BASH --> SAFETY
    WRITE --> SAFETY
    EDIT --> SAFETY

    style TRIGGER fill:#e3f2fd
    style PRE fill:#ffebee
    style EXEC fill:#fff8e1
    style POST fill:#e8f5e9
    style NOTIFY fill:#f3e5f5
    style RECOVERY fill:#fce4ec
```

---

## 8. Phase-Based Development Lifecycle

```mermaid
flowchart TD
    subgraph PHASE_SYSTEM["Phase System (7 Phases)"]
        PH1["Phase 1\nFoundation"] --> PH2["Phase 2\nData Infra"]
        PH2 --> PH3["Phase 3\nPerception"]
        PH3 --> PH4["Phase 4\nPrediction"]
        PH4 --> PH5["Phase 5\nPlanning\n(PRIMARY)"]
        PH5 --> PH6["Phase 6\nIntegration"]
        PH6 --> PH7["Phase 7\nAdvanced"]
    end

    subgraph PHASE5_DETAIL["Phase 5: Planning (Current Focus)"]
        direction TB
        PA["Phase A\nDense Overtaking\n+937"] --> PB["Phase B\nOvertake/Follow\n+994"]
        PB --> PC["Phase C\nMulti-NPC\n+1086"]
        PC --> PD["Phase D\nLane Obs 254D\n+402"]
        PD --> PE["Phase E\nCurved Roads\n+931"]
        PE --> PF["Phase F\nMulti-Lane\n+988"]
        PF --> PG["Phase G\nIntersection\n+461 (WIP)"]
        PG --> PH["Phase H\nTraffic Lights"]
        PH --> PI["Phase I\nU-turn"]
    end

    subgraph EACH_PHASE["Each Phase Lifecycle"]
        direction TB
        SPEC["SPEC.md\n기술 명세"] --> TASKS["TASKS.md\n작업 목록"]
        TASKS --> CHECKLIST["CHECKLIST.md\n체크리스트"]
        CHECKLIST --> DEV["개발 & 학습"]
        DEV --> VALIDATE["phase-tracker\n완료 검증"]
        VALIDATE --> TRANSITION["Phase 전환"]
    end

    subgraph PHASE_AGENTS["Phase Agent Coordination"]
        direction LR
        PT["phase-tracker\n진행 추적"] --- PP["progress-tracker\n통합 진행"]
        PP --- DW["dev-docs-writer\n문서 생성"]
    end

    PH5 -.-> PHASE5_DETAIL
    PG -.-> EACH_PHASE
    EACH_PHASE -.-> PHASE_AGENTS

    style PHASE_SYSTEM fill:#e3f2fd
    style PHASE5_DETAIL fill:#fff8e1
    style EACH_PHASE fill:#e8f5e9
    style PHASE_AGENTS fill:#f3e5f5
```

---

## 9. Quality Gate Pipeline

```mermaid
flowchart LR
    subgraph PRE_COMMIT["Pre-Commit Gate"]
        PC1["Lint Check"] --> PC2["Format Check"]
        PC2 --> PC3["Type Check"]
        PC3 --> PC4["Secrets Scan"]
        PC4 --> PC_RESULT{Pass?}
    end

    subgraph PRE_MERGE["Pre-Merge Gate"]
        PM1["Coverage ≥ 80%"] --> PM2["Code Review\n(code-reviewer)"]
        PM2 --> PM3["CHANGELOG\nUpdated?"]
        PM3 --> PM_RESULT{Pass?}
    end

    subgraph PRE_RELEASE["Pre-Release Gate"]
        PR1["Coverage ≥ 80%"] --> PR2["Security Scan"]
        PR2 --> PR3["All Docs\nComplete?"]
        PR3 --> PR4["Phase Checklist\nComplete?"]
        PR4 --> PR_RESULT{Pass?}
    end

    subgraph POST_RELEASE["Post-Release"]
        PO1["Archive Sprint"] --> PO2["Release Notes"]
        PO2 --> PO3["Retrospective\nPrompt"]
    end

    PC_RESULT -->|Yes| COMMIT["git commit"]
    PC_RESULT -->|No| FIX_PRE["Fix & Retry"]
    COMMIT --> PR_CREATE["PR Create"]
    PR_CREATE --> PM_RESULT
    PM_RESULT -->|Yes| MERGE["Merge"]
    PM_RESULT -->|No| FIX_MERGE["Fix & Retry"]
    MERGE --> TAG["git tag"]
    TAG --> PR_RESULT
    PR_RESULT -->|Yes| RELEASE["gh release create"]
    PR_RESULT -->|No| FIX_REL["Fix & Retry"]
    RELEASE --> POST_RELEASE

    style PRE_COMMIT fill:#e3f2fd
    style PRE_MERGE fill:#fff8e1
    style PRE_RELEASE fill:#ffebee
    style POST_RELEASE fill:#e8f5e9
```

---

## 10. Agent Interaction Map

```mermaid
graph TB
    subgraph ORCHESTRATORS["Orchestrators"]
        TO["training-orchestrator\n(sonnet)"]
        ED["experiment-documenter\n(haiku → codex)"]
        FA["forensic-analyst\n(haiku → codex)"]
    end

    subgraph ML_AGENTS["ML Agents"]
        TP["training-planner"]
        TM["training-monitor"]
        TA["training-analyst"]
        MT["model-trainer"]
        DC["dataset-curator"]
        BE["benchmark-evaluator"]
        AEM["ad-experiment-manager"]
    end

    subgraph DOC_AGENTS["Documentation Agents"]
        DDW["dev-docs-writer"]
        DG["doc-generator"]
        DS["doc-splitter"]
        DV["doc-validator"]
        TDM["training-doc-manager"]
        TSP["training-site-publisher"]
        RH["readme-helper"]
        PW["prd-writer"]
        TSW["tech-spec-writer"]
    end

    subgraph DEV_AGENTS["Development Agents"]
        CR["code-reviewer"]
        RA["refactor-assistant"]
        TH["test-helper"]
        WUM["work-unit-manager"]
    end

    subgraph GIT_AGENTS["Git/GitHub Agents"]
        BM["branch-manager"]
        CH["commit-helper"]
        GT["git-troubleshooter"]
        GM["github-manager"]
        PC["pr-creator"]
    end

    subgraph INFRA_AGENTS["Infrastructure Agents"]
        PT["phase-tracker"]
        PRG["progress-tracker"]
        PA["project-analyzer"]
        PD["project-discovery"]
        CV["config-validator"]
        FE["file-explorer"]
        GS["google-searcher"]
        AW["agent-writer"]
        AR["analytics-reporter"]
        OS["obsidian-sync"]
    end

    TO -->|"Plan"| TP
    TO -->|"Monitor"| TM
    TO -->|"Analyze"| TA
    TO -->|"Document"| ED
    TO -->|"Investigate"| FA

    ED -.->|"Codex\ndelegate"| TDM
    FA -.->|"Codex\ndelegate"| TA

    TP --> MT
    TM --> TA
    TA --> TDM
    TDM --> TSP

    CR --> TH
    CH --> PC
    BM --> GM

    PT --> PRG
    DDW --> DV

    style ORCHESTRATORS fill:#f3e5f5
    style ML_AGENTS fill:#e3f2fd
    style DOC_AGENTS fill:#e8f5e9
    style DEV_AGENTS fill:#fff8e1
    style GIT_AGENTS fill:#fce4ec
    style INFRA_AGENTS fill:#e0f2f1
```

---

## 11. Codex Orchestrator Pattern

```mermaid
sequenceDiagram
    participant U as User
    participant H as Haiku Agent<br/>(Orchestrator)
    participant C as Codex<br/>(gpt-5-codex)
    participant F as Files

    Note over U,F: experiment-documenter or forensic-analyst

    U->>H: "학습 완료, 문서화해줘"
    activate H
    Note right of H: Token: ~200-500

    H->>F: ls results/phase-X/ (quick scan)
    F-->>H: .onnx file, training.log tail

    H->>C: codex exec "Generate report..."
    activate C
    Note right of C: Token: 15,000+<br/>workspace-write sandbox

    C->>F: Read: TRAINING-LOG.md, PROGRESS.md
    C->>F: Read: TensorBoard events
    C->>F: Parse: configs/*.yaml
    C->>F: Write: experiments/phase-X/ANALYSIS.md
    C->>F: Update: docs/TRAINING-LOG.md
    C->>F: Update: docs/PROGRESS.md
    C->>F: Update: README.md stats
    C-->>H: Done (exit 0)
    deactivate C

    H-->>U: "Phase X 문서화 완료"
    deactivate H

    Note over U,F: Total: Haiku ~300 + Codex ~15K tokens<br/>vs Sonnet alone: ~50K+ tokens
```

---

## 12. Git Workflow (/git-workflow)

```mermaid
gitgraph
    commit id: "v1.0.0" tag: "release"
    branch feature/phase-G
    checkout feature/phase-G
    commit id: "feat: intersection env"
    commit id: "feat: T-junction reward"
    commit id: "feat: cross intersection"
    checkout main
    merge feature/phase-G id: "PR #42" tag: "squash"
    branch fix/collision-bug
    checkout fix/collision-bug
    commit id: "fix: collision detection"
    commit id: "test: add regression"
    checkout main
    merge fix/collision-bug id: "PR #43" tag: "squash"
    commit id: "v1.1.0" tag: "release"
```

```mermaid
flowchart LR
    subgraph BRANCH["Branch Strategy (GitHub Flow)"]
        MAIN["main\n(production)"]
        FEAT["feature/*"]
        FIX["fix/*"]
        REL["release/*"]
    end

    subgraph COMMIT["Commit Convention"]
        FEAT_C["feat: 새 기능"]
        FIX_C["fix: 버그 수정"]
        DOCS_C["docs: 문서 변경"]
        REFACTOR_C["refactor: 리팩토링"]
        TEST_C["test: 테스트"]
        CHORE_C["chore: 기타"]
    end

    subgraph AGENTS_GIT["Agent Coordination"]
        BM2["branch-manager\n브랜치 생성/삭제"]
        CH2["commit-helper\n메시지 작성"]
        PC2["pr-creator\nPR 생성"]
        GT2["git-troubleshooter\n충돌 해결"]
        GM2["github-manager\nCI/CD 관리"]
    end

    FEAT --> MAIN
    FIX --> MAIN
    REL --> MAIN
    BM2 --> FEAT
    BM2 --> FIX
    CH2 --> COMMIT
    PC2 --> MAIN

    style BRANCH fill:#e3f2fd
    style COMMIT fill:#fff8e1
    style AGENTS_GIT fill:#e8f5e9
```

---

## 13. Documentation Automation Flow

```mermaid
flowchart TD
    subgraph TRIGGERS["Change Triggers"]
        CODE_CHANGE["Code Change\n(Write/Edit)"]
        TRAIN_COMPLETE["Training Complete"]
        PHASE_CHANGE["Phase Transition"]
        SPRINT_END["Sprint End"]
        RELEASE_TAG["Release Tag"]
    end

    subgraph AUTO_HOOKS["Automatic (Hooks)"]
        ADS["auto-doc-sync.sh"]
        PPS["phase-progress.sh"]
        PTT["post-tool-use-tracker.sh"]
    end

    subgraph AGENTS_DOC["Agent-Driven"]
        ED2["experiment-documenter"]
        TDM2["training-doc-manager"]
        RH2["readme-helper"]
        PT2["phase-tracker"]
        PRG2["progress-tracker"]
    end

    subgraph DOCUMENTS["Updated Documents"]
        CL["CHANGELOG.md"]
        RM["README.md"]
        PG["PROGRESS.md"]
        TL["TRAINING-LOG.md"]
        PHASE_DOCS["phases/phase-X/*"]
        ANALYSIS["experiments/*/ANALYSIS.md"]
        METRICS["analytics/metrics.jsonl"]
    end

    CODE_CHANGE --> ADS
    CODE_CHANGE --> PPS
    CODE_CHANGE --> PTT
    TRAIN_COMPLETE --> ED2
    TRAIN_COMPLETE --> TDM2
    PHASE_CHANGE --> PT2
    PHASE_CHANGE --> PRG2
    SPRINT_END --> PRG2
    RELEASE_TAG --> RH2

    ADS --> CL
    ADS --> RM
    ADS --> PG
    PPS --> PHASE_DOCS
    PTT --> METRICS
    ED2 --> TL
    ED2 --> ANALYSIS
    TDM2 --> TL
    RH2 --> RM
    PT2 --> PHASE_DOCS
    PRG2 --> PG

    style TRIGGERS fill:#e3f2fd
    style AUTO_HOOKS fill:#fff8e1
    style AGENTS_DOC fill:#f3e5f5
    style DOCUMENTS fill:#e8f5e9
```

---

## 14. Analytics & Monitoring Pipeline

```mermaid
flowchart LR
    subgraph COLLECTION["Data Collection"]
        HOOK_DATA["post-tool-use-tracker.sh\n→ metrics.jsonl"]
        TB_DATA["TensorBoard\n→ events.out.tfevents"]
        GH_DATA["GitHub Actions\n→ CI/CD logs"]
    end

    subgraph PROCESSING["Processing"]
        HOURLY["Hourly Rollup"]
        DAILY["Daily Summary"]
        PHASE_AGG["Phase Aggregation"]
    end

    subgraph VISUALIZATION["Visualization"]
        CLI_CHART["CLI ASCII Charts\n(/analytics)"]
        TB_DASH["TensorBoard\nDashboard"]
        GH_PAGES["GitHub Pages\nTraining History"]
    end

    subgraph AGENTS_ANALYTICS["Agent Layer"]
        AR2["analytics-reporter"]
        TM2["training-monitor"]
        TSP2["training-site-publisher"]
    end

    HOOK_DATA --> HOURLY
    HOOK_DATA --> DAILY
    TB_DATA --> PHASE_AGG
    GH_DATA --> DAILY

    HOURLY --> AR2
    DAILY --> AR2
    PHASE_AGG --> TM2

    AR2 --> CLI_CHART
    TM2 --> TB_DASH
    TSP2 --> GH_PAGES

    style COLLECTION fill:#e3f2fd
    style PROCESSING fill:#fff8e1
    style VISUALIZATION fill:#e8f5e9
    style AGENTS_ANALYTICS fill:#f3e5f5
```

---

## 15. End-to-End Scenario: New Phase Training

```mermaid
sequenceDiagram
    participant User
    participant Orch as training-orchestrator
    participant Plan as training-planner
    participant Train as model-trainer
    participant Mon as training-monitor
    participant Ana as training-analyst
    participant Doc as experiment-documenter
    participant Phase as phase-tracker
    participant Hooks as PostToolUse Hooks

    User->>Orch: "Phase H 시작하자"
    activate Orch

    Orch->>Phase: Phase G 완료 확인
    Phase-->>Orch: Phase G: +461, 목표 미달

    Orch->>Plan: Phase G 재설계 요청
    activate Plan
    Plan->>Plan: Curriculum 조정
    Plan->>Plan: Reward shaping 수정
    Plan-->>Orch: vehicle_ppo_phase-G-v2.yaml
    deactivate Plan

    Orch->>Train: 학습 시작
    activate Train
    Train->>Train: mlagents-learn --run-id=phase-G-v2
    Train-->>Mon: 학습 진행중
    deactivate Train

    loop Every 100K steps
        Mon->>Mon: TensorBoard 파싱
        Mon-->>Orch: Reward: +XXX, Steps: XXXK
    end

    Mon-->>Orch: 학습 완료 (8M steps)

    Orch->>Ana: 결과 분석
    activate Ana
    Ana->>Ana: Success/Fail 판정
    Ana-->>Orch: SUCCESS (+800, collision <5%)
    deactivate Ana

    Orch->>Doc: 문서화
    activate Doc
    Doc->>Doc: Codex에 위임
    Doc-->>Hooks: Write/Edit 실행
    Hooks-->>Hooks: auto-doc-sync
    Hooks-->>Hooks: phase-progress
    Hooks-->>Hooks: tracker
    Doc-->>Orch: 문서 완료
    deactivate Doc

    Orch->>Phase: Phase G → Complete
    Phase-->>User: Phase G 완료!

    Orch-->>User: "Phase G 완료. Phase H 시작 준비됨."
    deactivate Orch
```

---

## Summary

| Diagram | Content | Key Components |
|---------|---------|----------------|
| #1 System Overview | 전체 아키텍처 | 38 Agents, 22 Skills, 6 Hooks |
| #2 ML Training | 학습 메인 루프 | Plan → Train → Analyze → Document |
| #3 State Machine | Orchestrator 상태 | PLAN → TRAIN → ANALYZE → DECIDE → DOCUMENT |
| #4 Feature Dev | /feature 워크플로우 | start → dev → progress → complete |
| #5 Bug Fix | /bugfix 워크플로우 | start → analyze → fix → complete |
| #6 Release | /release 워크플로우 | prepare → create → publish |
| #7 Hook Flow | Hook 실행 흐름 | Pre → Execute → Post → Notify → Recovery |
| #8 Phase Lifecycle | Phase 기반 개발 | 7 Phases, Phase 5 detail (A~I) |
| #9 Quality Gate | 품질 파이프라인 | pre-commit → pre-merge → pre-release |
| #10 Agent Map | Agent 상호작용 | 6 groups, orchestrator pattern |
| #11 Codex Pattern | Codex 위임 패턴 | Haiku → Codex delegation |
| #12 Git Workflow | Git 전략 | GitHub Flow + Conventional Commits |
| #13 Doc Automation | 문서 자동화 | Hooks + Agents → Documents |
| #14 Analytics | 분석 파이프라인 | Collection → Processing → Visualization |
| #15 E2E Scenario | 전체 시나리오 | New Phase Training (sequence) |

---

**Generated**: 2026-01-29 | **Framework**: [cc-initializer](https://github.com/tygwan/cc-initializer) | **Project**: physical-unity
