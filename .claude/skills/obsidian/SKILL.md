---
name: obsidian
description: Obsidian vault 동기화 및 지식 관리. 프로젝트 문서를 Obsidian으로 내보내고 연결합니다. "obsidian", "옵시디언", "vault", "지식", "노트" 키워드에 반응.
---

# Obsidian Skill

Obsidian vault synchronization and knowledge management. Export project documents, sessions, decisions, and learnings to your Obsidian vault.

## Usage

```bash
/obsidian <command> [options]
```

### Commands

| Command | Description |
|---------|-------------|
| `init` | Initialize Obsidian vault structure |
| `sync` | Sync current project to vault |
| `status` | Show sync status |
| `export session` | Export current session summary |
| `export project` | Export entire project |
| `capture decision` | Capture a decision as ADR |
| `capture learning` | Capture a learning note |

## Command Details

### /obsidian init

Initialize the Obsidian vault with standard structure:

```bash
/obsidian init [--vault PATH]
```

Creates:
```
~/Obsidian/Dev-Knowledge/
├── Projects/
├── Decisions/
├── Learnings/
├── Sessions/
└── Analytics/
```

### /obsidian sync

Synchronize current project to vault:

```bash
/obsidian sync [--full]
```

**Default sync:**
- PROGRESS.md → Progress tracking
- CONTEXT.md → Project context

**Full sync (--full):**
- All docs including PRD, TECH-SPEC
- Phase documents
- Sprint history

### /obsidian status

Show current sync status:

```bash
/obsidian status
```

Output:
```
Obsidian Sync Status
════════════════════════════════════════
Vault: ~/Obsidian/Dev-Knowledge
Project: my-project

Last Sync: 2026-01-21 14:30:00
Files Synced: 12
Pending Changes: 3

Sync Required:
  - docs/PROGRESS.md (modified)
  - docs/phases/phase-2/TASKS.md (modified)
```

### /obsidian export session

Export current session summary:

```bash
/obsidian export session [--include-analytics]
```

Creates a session note with:
- Session duration
- Tools used (from analytics)
- Files modified
- Key decisions
- Learnings captured

### /obsidian export project

Export entire project:

```bash
/obsidian export project [--archive]
```

Exports:
- All documentation
- Decision history
- Learning notes
- Session summaries
- Analytics data (optional)

### /obsidian capture decision

Capture an architecture decision:

```bash
/obsidian capture decision "Use PostgreSQL for main database"
```

Interactive prompts:
1. Decision context
2. Alternatives considered
3. Consequences
4. Related decisions

### /obsidian capture learning

Capture a learning note:

```bash
/obsidian capture learning "React useCallback optimization"
```

Interactive prompts:
1. What was learned
2. How it was discovered
3. Related topics
4. Code examples (optional)

## Vault Structure

```
~/Obsidian/Dev-Knowledge/
├── Projects/
│   └── my-project/
│       ├── _Index.md           # Project overview
│       ├── Progress.md         # From PROGRESS.md
│       ├── Context.md          # From CONTEXT.md
│       ├── PRD.md              # From docs/PRD.md
│       ├── Tech-Spec.md        # From docs/TECH-SPEC.md
│       ├── Phases/
│       │   ├── Phase-1.md
│       │   └── Phase-2.md
│       └── Sessions/
│           ├── 2026-01-21.md
│           └── 2026-01-20.md
├── Decisions/
│   ├── ADR-001-database-choice.md
│   └── ADR-002-auth-strategy.md
├── Learnings/
│   ├── LRN-001-react-optimization.md
│   └── LRN-002-typescript-generics.md
├── Sessions/
│   └── (cross-project sessions)
└── Analytics/
    ├── Weekly-2026-W03.md
    └── Monthly-2026-01.md
```

## Obsidian Features Used

### Wikilinks
All internal links use Obsidian wikilinks:
```markdown
Related: [[Projects/my-project/_Index|My Project]]
Decision: [[Decisions/ADR-001-database-choice]]
```

### Dataview Metadata
Frontmatter optimized for Dataview:
```yaml
---
tags: [project, typescript, react]
status: active
progress: 45
created: 2026-01-21
updated: 2026-01-21
---
```

### Tags
Hierarchical tag structure:
- `project/my-project`
- `tech/typescript`
- `decision/architecture`
- `learning/optimization`

## Configuration

Settings in `.claude/settings.json`:

```json
{
  "obsidian": {
    "enabled": true,
    "vault_path": "~/Obsidian/Dev-Knowledge",
    "auto_sync": false,
    "export_format": {
      "use_wikilinks": true,
      "add_dataview_metadata": true,
      "include_frontmatter": true
    },
    "structure": {
      "projects": "Projects",
      "decisions": "Decisions",
      "learnings": "Learnings",
      "sessions": "Sessions",
      "analytics": "Analytics"
    },
    "sync_options": {
      "include_phases": true,
      "include_sprints": true,
      "include_analytics": true
    }
  }
}
```

## Templates

Templates are in `.claude/templates/obsidian/`:

| Template | Purpose |
|----------|---------|
| `project.md` | Project index page |
| `session.md` | Session summary |
| `learning.md` | Learning note |
| `decision.md` | ADR template |
| `analytics.md` | Analytics summary |

## Related

- **Agent**: `obsidian-sync` - Detailed sync operations
- **Analytics**: `/analytics` - Usage data for export
