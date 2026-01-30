# Claude Code Agents - Documentation

이 문서는 physical-unity 프로젝트에서 사용하는 모든 Claude Code 서브에이전트를 설명합니다.

## 🎯 Agent 구조 개요

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW AGENTS                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  training-planner (Opus)                                         │
│    ↓ (designs)                                                   │
│  Experiment Config + DESIGN.md                                   │
│    ↓                                                             │
│  training-orchestrator (Haiku)                                   │
│    ↓ (executes)                                                  │
│  mlagents-learn (Unity Training)                                 │
│    ↓ (monitors)                                                  │
│  training-monitor (Haiku)                                        │
│    ↓ (completes)                                                 │
│  training-analyst (Haiku) ──┬─→ SUCCESS?                         │
│                             │     ↓                               │
│                             │   experiment-documenter (Opus)      │
│                             │                                     │
│                             └─→ FAILURE?                          │
│                                   ↓                               │
│                                 forensic-analyst (Opus)           │
│                                   ↓                               │
│                                 experiment-documenter (Opus)      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📚 Agent 분류

### 🔬 Analysis & Investigation (분석 및 조사)

#### forensic-analyst
- **Model**: Haiku (orchestrator) + **Codex** (deep analysis)
- **Purpose**: ML 학습 실패 근본 원인 분석
- **Capabilities**:
  - Codex에 위임: TensorBoard 파싱, 수학적 검증
  - Codex에 위임: Unity C# 코드 검사 (reward function)
  - Codex에 위임: Config vs 실제 구현 차이 발견
  - Codex에 위임: 100% 신뢰도 증명 생성
- **Output**: `ROOT-CAUSE-ANALYSIS.md` (15페이지, Codex 생성)
- **Token Efficiency**: Haiku ~400 tokens, Codex 20K+ tokens
- **Triggers**:
  - "근본 원인", "root cause"
  - "forensic", "분석 보고서"
  - "왜 실패", "why failed"
  - "수학적 증명", "mathematical proof"

**Example Use Case**:
```
User: "Phase B가 -108로 실패했는데 왜 그런지 상세 분석해줘"
→ forensic-analyst (Haiku)
  1. Quick failure detection
  2. Delegate to Codex:
     - TensorBoard 파싱: Episode/Length=501, Stats/Speed=0.0
     - Unity 코드 검사: speedUnderPenalty = -0.1 * (2.0 - speedRatio * 2.0)
     - 수학적 검증: -0.2/step × 501 steps = -100.2 ✓
     - ROOT-CAUSE-ANALYSIS.md 생성 (15 pages)
  3. Return: "🔍 Root cause: Speed penalty. Confidence: 100%"
```

---

### 📝 Documentation (문서화)

#### experiment-documenter
- **Model**: Haiku (orchestrator) + **Codex** (documentation)
- **Purpose**: 실험 완료 후 자동 문서화
- **Capabilities**:
  - Codex에 위임: TensorBoard 메트릭 수집 및 파싱
  - Codex에 위임: ANALYSIS.md 생성 (실험 결과 리포트)
  - Codex에 위임: TRAINING-LOG.md 업데이트 (시간순 기록)
  - Codex에 위임: PROGRESS.md 동기화 (Phase 진행 상황)
  - Codex에 위임: SPEC.md 업데이트 (Success criteria 체크)
  - Codex에 위임: 문서 간 상호 참조 유지
- **Output**: 5-7개 문서 자동 업데이트 (Codex 생성)
- **Token Efficiency**: Haiku ~400 tokens, Codex 15K+ tokens
- **Triggers**:
  - "실험 완료", "training completed"
  - "document experiment"
  - "결과 기록", "record results"
  - "update docs", "문서 업데이트"

**Example Use Case**:
```
User: "Phase A 학습 완료했어, 문서화해줘"
→ experiment-documenter (Haiku)
  1. Quick info collection (metadata only)
  2. Delegate to Codex:
     - TensorBoard 파싱: final reward, convergence, episode length
     - ANALYSIS.md 생성 (success/failure, metrics, recommendations)
     - TRAINING-LOG.md 엔트리 추가
     - PROGRESS.md Phase A → ✅ Complete
     - SPEC.md success criteria checkboxes
  3. Return: "✅ Documentation complete. Status: SUCCESS. Files: 4"
```

---

#### training-doc-manager
- **Model**: Haiku (fast operations)
- **Purpose**: 학습 중 문서 실시간 동기화
- **Capabilities**:
  - PROGRESS.md 업데이트
  - TRAINING-LOG.md 실시간 기록
  - Phase 문서 간 일관성 유지
- **Output**: 실시간 문서 동기화
- **Triggers**:
  - "문서 동기화", "sync docs"
  - "아카이브", "archive"
  - "진행 업데이트", "update progress"

---

### 🧪 Training Lifecycle (학습 생명주기)

#### training-planner
- **Model**: Opus (strategic planning)
- **Purpose**: 실험 설계 및 Config 생성
- **Capabilities**:
  - 다음 Phase 계획 수립
  - YAML config 생성 (hyperparameters, curriculum)
  - DESIGN.md 작성 (설계 근거)
  - Phase 전환 전략 수립
- **Output**: Config YAML + DESIGN.md + HYPOTHESIS.md
- **Triggers**:
  - "실험 설계", "design experiment"
  - "다음 버전", "next version"
  - "config 생성", "create config"
  - "Phase 계획", "plan phase"

---

#### training-orchestrator
- **Model**: Haiku (coordination)
- **Purpose**: 학습 워크플로우 총괄 조율
- **Capabilities**:
  - 전체 진행 상황 파악
  - 다음 단계 결정
  - 다른 Agent 호출 조율
  - Workflow 관리
- **Output**: 전체 학습 프로세스 조율
- **Triggers**:
  - "다음 단계", "next step"
  - "워크플로우", "workflow"
  - "전체 상태", "overall status"
  - "뭐 해야 해", "what's next"

---

#### training-monitor
- **Model**: Haiku (real-time monitoring)
- **Purpose**: 학습 진행 상황 실시간 추적
- **Capabilities**:
  - TensorBoard 로그 실시간 파싱
  - 진행률 계산
  - 이상 징후 감지
  - Alert 생성
- **Output**: 진행 상황 리포트
- **Triggers**:
  - "학습 상태", "training status"
  - "진행률", "progress"
  - "모니터링", "monitor"
  - "현재 reward", "current reward"

---

#### training-analyst
- **Model**: Haiku + Codex (orchestrator)
- **Purpose**: 학습 결과 분석 조율
- **Capabilities**:
  - 성공/실패 판정
  - 빠른 메트릭 분석
  - forensic-analyst 호출 (실패 시)
  - experiment-documenter 호출 (완료 시)
- **Output**: 간단 분석 + Agent 위임
- **Triggers**:
  - "결과 분석", "analyze results"
  - "리포트", "report"
  - "왜 실패", "why failed"
  - "학습 완료", "training done"

---

### 🔧 Infrastructure (인프라)

#### training-site-publisher
- **Model**: Haiku (automation)
- **Purpose**: GitHub Pages 사이트 발행
- **Capabilities**:
  - gh-pages 브랜치 업데이트
  - Phase 문서 발행
  - 갤러리 관리
- **Output**: 웹사이트 업데이트
- **Triggers**:
  - "사이트 업데이트", "update site"
  - "gh-pages", "publish"
  - "웹 발행", "site update"

---

## 🛠️ Skills

### codex
- **Source**: https://github.com/tygwan/skills/tree/master/codex
- **Purpose**: OpenAI Codex CLI 실행
- **Capabilities**:
  - 코드 분석, 리팩토링
  - 자동화된 편집
  - Deep reasoning (gpt-5, gpt-5-codex)
- **Sandbox Modes**:
  - `read-only`: 분석만
  - `workspace-write`: 파일 편집
  - `danger-full-access`: 네트워크 접근
- **Triggers**:
  - "codex", "코덱스"
  - "deep reasoning"
  - "code analysis", "refactoring"

---

## 🔄 Typical Workflows

### Workflow 1: 새로운 실험 시작

```
User: "Phase B 실험 설계해줘"
  ↓
training-planner (Opus)
  - Config YAML 생성
  - DESIGN.md 작성
  - HYPOTHESIS.md 생성
  ↓
User: "학습 시작"
  ↓
training-orchestrator (Haiku)
  - mlagents-learn 실행
  ↓
training-monitor (Haiku)
  - 실시간 진행 상황 모니터링
```

### Workflow 2: 학습 완료 (성공)

```
Training Completed
  ↓
training-analyst (Haiku + Codex)
  - Quick metrics: +1800 reward ✓ SUCCESS
  ↓
experiment-documenter (Opus)
  - ANALYSIS.md 생성
  - TRAINING-LOG.md 업데이트
  - PROGRESS.md Phase 완료 표시
  - SPEC.md success criteria ✓
```

### Workflow 3: 학습 실패 (근본 원인 분석)

```
Training Failed: -108 reward
  ↓
training-analyst (Haiku + Codex)
  - Quick assessment: FAILURE detected
  ↓
forensic-analyst (Opus)
  - TensorBoard 파싱: Episode/Length=501, Speed=0.0
  - Unity 코드 검사: speedUnderPenalty logic
  - 수학적 검증: -0.2 × 501 = -100.2 ✓
  - ROOT-CAUSE-ANALYSIS.md 생성
  ↓
experiment-documenter (Opus)
  - ANALYSIS.md + ROOT-CAUSE-ANALYSIS.md 통합
  - TRAINING-LOG.md 실패 기록
  - PROGRESS.md 실패 표시
  - 재시도 전략 제안
```

---

## 📋 Model Usage Guidelines

### When to Use Codex (Primary for Heavy Work)
- **Deep analysis**: TensorBoard parsing, mathematical verification (forensic-analyst)
- **Long-form writing**: 10-15 page reports, comprehensive documentation (experiment-documenter)
- **Code inspection**: Unity C# reward function analysis (forensic-analyst)
- **Multi-document updates**: 5-7 files cross-referencing (experiment-documenter)
- **Strategic planning**: Phase design, config generation (training-planner)

### When to Use Haiku (Orchestrators)
- **Fast operations**: monitoring, quick assessment, coordination
- **Agent delegation**: calling Codex, parsing outputs, user response
- **Repetitive tasks**: status checks, simple document syncing
- **Real-time updates**: training progress tracking

### When to Use Opus (Rare, Legacy)
- **Not recommended**: Use Codex instead for better efficiency
- **Only if**: Codex unavailable or specific Opus-only features needed

---

## 🎯 Best Practices

### 1. Agent Selection
- **명확한 실패 원인 필요** → forensic-analyst (Opus)
- **실험 완료 후 문서화** → experiment-documenter (Opus)
- **빠른 진행 상황 확인** → training-monitor (Haiku)
- **다음 실험 계획** → training-planner (Opus)

### 2. Trigger Keywords
각 Agent는 특정 키워드에 자동 반응하도록 설정되어 있습니다:
- 한국어 + 영어 키워드 모두 지원
- 명확한 intent 전달 위해 정확한 키워드 사용 권장

### 3. Output Quality
- **Opus agents**: 10-15 페이지 상세 보고서, 수학적 증명 포함
- **Haiku agents**: 간결한 상태 요약, 빠른 응답
- **Codex**: 코드 중심 분석, 자동화된 편집

---

## 📊 Agent Performance Metrics

| Agent | Model | Avg Response Time | Output Size | Token Usage (You/Codex) |
|-------|-------|-------------------|-------------|------------------------|
| forensic-analyst | **Haiku+Codex** | 2-3 min | 15 pages | ~400 / 20K |
| experiment-documenter | **Haiku+Codex** | 1-2 min | 5-7 docs | ~400 / 15K |
| training-planner | **Haiku+Codex** | 1-2 min | Config + 3 docs | ~500 / 12K |
| training-analyst | Haiku+Codex | 10-30 sec | Summary | ~300 / 5K |
| training-monitor | Haiku | 5-10 sec | Progress | ~200 / - |
| training-orchestrator | Haiku | 10-20 sec | Coordination | ~300 / - |

**Token Efficiency Comparison**:
- **Old (Opus direct)**: forensic-analyst uses 15K tokens in Claude
- **New (Haiku+Codex)**: ~400 tokens in Claude + 20K in Codex
- **Benefit**: 97% reduction in Claude token usage, delegated to specialized Codex

---

## 🔍 Troubleshooting

### Agent가 응답하지 않을 때
1. Trigger keyword 확인
2. 명확한 요청 문구 사용
3. 수동으로 Agent 호출: `/Task subagent_type=[agent-name]`

### 잘못된 Agent가 호출될 때
1. 더 구체적인 키워드 사용
2. Agent 이름 직접 지정
3. Trigger 키워드 충돌 확인

### Output 품질이 낮을 때
1. Opus model 사용 여부 확인 (forensic, planner, documenter)
2. 충분한 context 제공
3. 명확한 요구사항 명시

---

**Last Updated**: 2026-01-29
**Total Agents**: 8 (6 training + 2 specialized)
**Total Skills**: 1 (codex)
