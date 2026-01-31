---
skill: json-codex
description: Delegate JSON file editing, creation, and validation to Codex CLI for precise structural changes
triggers:
  - "json edit"
  - "json 수정"
  - "json 생성"
  - "json create"
  - "json 편집"
  - "config edit"
  - "설정 파일"
  - "package.json"
  - "tsconfig"
  - ".json 수정"
model: opus
---

# JSON-Codex Skill

## Overview
Delegates JSON file operations (create, modify, validate, merge) to Codex CLI for precise structural edits. Codex excels at JSON because it can reason about the full structure, handle nested paths, and preserve formatting.

## When to Use This Skill
- Editing deeply nested JSON structures
- Merging multiple JSON configurations
- Creating JSON files from natural language descriptions
- Refactoring large JSON files (package.json, tsconfig.json, astro.config)
- Validating JSON schema compliance
- Bulk property additions/removals across JSON files

## Execution Flow

### Step 1: Identify Target Files
```
Glob: **/*.json
Glob: **/*.jsonc
```

Determine which JSON files need modification.

### Step 2: Determine Operation Type
Use `AskUserQuestion` if ambiguous:
- **Create**: Generate new JSON file from requirements
- **Modify**: Edit existing JSON properties/structure
- **Validate**: Check JSON syntax and schema compliance
- **Merge**: Combine multiple JSON sources
- **Transform**: Restructure or migrate JSON format

### Step 3: Assemble Codex Command

#### Create JSON
```bash
codex exec -m gpt-5 \
  --reasoning-effort medium \
  --sandbox workspace-write \
  --full-auto \
  -C "{project_path}" \
  "Create {filename}.json with the following structure: {description}. Output valid JSON only."
```

#### Modify JSON
```bash
codex exec -m gpt-5 \
  --reasoning-effort medium \
  --sandbox workspace-write \
  --full-auto \
  -C "{project_path}" \
  "In {filepath}, {modification_description}. Preserve existing structure and formatting."
```

#### Validate JSON
```bash
codex exec -m gpt-5 \
  --reasoning-effort low \
  --sandbox read-only \
  --full-auto \
  -C "{project_path}" \
  "Validate all JSON files in the project. Report syntax errors, missing required fields, and schema violations."
```

#### Merge JSON
```bash
codex exec -m gpt-5 \
  --reasoning-effort high \
  --sandbox workspace-write \
  --full-auto \
  -C "{project_path}" \
  "Merge {source_file} into {target_file}. Deep merge objects, concatenate arrays, resolve conflicts by preferring {preference}."
```

### Step 4: Post-Execution Validation
After Codex completes, verify the output:
```bash
node -e "JSON.parse(require('fs').readFileSync('{filepath}', 'utf8')); console.log('Valid JSON')"
```

## Common JSON Targets in Astro Projects

| File | Typical Operations |
|------|--------------------|
| `package.json` | Add/remove dependencies, update scripts, modify metadata |
| `tsconfig.json` | Adjust compiler options, path aliases, strictness |
| `.vscode/settings.json` | Editor preferences, extension configs |
| `src/content/config.ts` | Not JSON but schemas are JSON-like |

## Reasoning Effort Guide

| Task Complexity | Effort | Example |
|----------------|--------|---------|
| Single property change | `low` | Add a script to package.json |
| Multi-property edit | `medium` | Restructure tsconfig paths |
| Deep structural change | `high` | Merge two complex configs |

## Error Recovery
If Codex produces invalid JSON:
1. Report the error to user
2. Resume session with correction prompt:
```bash
echo "The previous output had invalid JSON. Fix the syntax error and output valid JSON." | codex exec resume --last
```
