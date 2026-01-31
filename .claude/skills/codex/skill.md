---
skill: codex
description: Execute OpenAI Codex CLI for code analysis, refactoring, and automated editing
triggers:
  - codex
  - "코덱스"
  - "deep reasoning"
  - "code analysis"
  - "refactoring"
model: opus
---

# Codex Skill Guide

## Overview
The Codex skill enables running Codex CLI commands (`codex exec`, `codex resume`) and leverages OpenAI Codex for code analysis, refactoring, and automated editing tasks.

## Execution Flow

### Step 1: Model Selection
Use `AskUserQuestion` to ask the user:
- **gpt-5** (Recommended) - Balanced capability and speed
- **gpt-5-codex** - Optimized for code tasks

### Step 2: Reasoning Effort
Use `AskUserQuestion` to ask:
- **low** - Quick tasks (formatting, simple edits)
- **medium** (Recommended) - Standard analysis and refactoring
- **high** - Complex architecture, deep reasoning

### Step 3: Sandbox Mode
Determine automatically based on task type, or ask user:
- **read-only** - Analysis, review, explanation (default)
- **workspace-write** - File edits, code generation
- **danger-full-access** - Network access, package installs

### Step 4: Command Assembly & Execution
Assemble the command using Bash tool:

```bash
# Standard execution (with project directory)
codex exec \
  -m {model} \
  --reasoning-effort {effort} \
  --sandbox {sandbox_mode} \
  --full-auto \
  -C "{project_path}" \
  "{user_prompt}"
```

#### Execution Templates

**Analysis (read-only)**:
```bash
codex exec -m gpt-5 --reasoning-effort medium --sandbox read-only --full-auto -C "{project_path}" "{prompt}"
```

**Code editing (workspace-write)**:
```bash
codex exec -m gpt-5 --reasoning-effort medium --sandbox workspace-write --full-auto -C "{project_path}" "{prompt}"
```

**Full access (danger-full-access)**:
```bash
codex exec -m gpt-5-codex --reasoning-effort high --sandbox danger-full-access --full-auto -C "{project_path}" "{prompt}"
```

**Subdirectory targeting** (e.g., Astro site within a larger project):
```bash
codex exec -m gpt-5 --reasoning-effort medium --sandbox workspace-write --full-auto -C "{project_path}/site/portfolio" "{prompt}"
```

### Step 5: Session Resumption
For continuing previous work:
```bash
echo "{follow_up_prompt}" | codex exec resume --last
```
Resumed sessions inherit all original settings (model, reasoning, sandbox).

## Project Path Binding

The `-C` flag is critical for targeting specific projects:

| Scenario | Command |
|----------|---------|
| Full project | `-C "C:\Users\user\Desktop\dev\physical-unity"` |
| Astro site only | `-C "C:\Users\user\Desktop\dev\physical-unity\site\portfolio"` |
| Submodule | `-C "{path_to_submodule}"` |
| Skip git check | Add `--skip-git-repo-check` if git detection issues |

## Flag Reference

| Flag | Purpose | Values |
|------|---------|--------|
| `-m` | Model selection | `gpt-5`, `gpt-5-codex` |
| `--reasoning-effort` | Thinking depth | `low`, `medium`, `high` |
| `--sandbox` | Permission scope | `read-only`, `workspace-write`, `danger-full-access` |
| `--full-auto` | Skip confirmations | (no value) |
| `-C` | Working directory | `"{path}"` |
| `--config` | Custom config file | `"{config_path}"` |
| `--skip-git-repo-check` | Bypass git detection | (no value) |

## Follow-Up Protocol

After each command execution, use `AskUserQuestion` to confirm next steps:
- Continue with same session? (resume)
- New task with same settings?
- Adjust model/reasoning/sandbox?
- Exit codex workflow?

## Error Handling

- Stop on non-zero exit codes and report the error
- Request user permission before `danger-full-access` sandbox
- On timeout, suggest reducing `--reasoning-effort`
- On permission errors, suggest upgrading sandbox mode
- Summarize any warnings for adjustment guidance
