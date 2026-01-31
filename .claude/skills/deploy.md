---
name: deploy
description: GitHub Pages deployment management. Check status, trigger deploy, configure Pages source, cleanup legacy branches. "deploy", "배포", "pages", "사이트" keywords.
---

# Deploy Skill

Manages GitHub Pages deployment for the Astro portfolio site.

## Usage
```
/deploy [subcommand]
```

### Subcommands
- `/deploy` or `/deploy status` - Check deployment status
- `/deploy trigger` - Trigger manual deployment
- `/deploy setup` - Configure Pages source to GitHub Actions
- `/deploy cleanup` - Remove legacy branches (gh-pages) and validate config
- `/deploy verify` - Build locally and verify site works

## Prerequisites

- `gh` CLI authenticated (`gh auth login`)
- Repository: `tygwan/physical-unity`
- Site source: `site/portfolio/` (Astro)
- Workflow: `.github/workflows/deploy-site.yml`

## Workflow

### Step 1: Check Current State
```bash
# Pages configuration
gh api repos/{owner}/{repo}/pages --jq '{source: .source, build_type: .build_type, url: .html_url, status: .status}'

# Recent workflow runs
gh run list --workflow=deploy-site.yml --limit=3

# Check for legacy branches
git branch -r | grep gh-pages
```

### Step 2: Safety Checks

**CRITICAL**: Verify Pages source is set to "GitHub Actions", NOT "Deploy from a branch".

| Setting | Safe | Dangerous |
|---------|------|-----------|
| build_type | `workflow` | `legacy` |
| source.branch | (none) | `gh-pages` |

If `build_type` is `legacy`:
1. **WARN**: "Pages source is 'legacy' (branch-based). Jekyll will override Astro deployments."
2. **FIX**: Switch to GitHub Actions:
   ```bash
   gh api -X PUT repos/{owner}/{repo}/pages \
     -f build_type=workflow
   ```

### Step 3: Actions per Subcommand

#### `status`
```bash
# 1. Pages config
gh api repos/{owner}/{repo}/pages

# 2. Latest deployment
gh run list --workflow=deploy-site.yml --limit=1 --json status,conclusion,startedAt,url

# 3. Site accessibility
curl -sI https://tygwan.github.io/physical-unity/ | head -5
```

Report: Pages source type, last deploy status, site HTTP status.

#### `trigger`
```bash
# 1. Verify workflow exists
gh workflow view deploy-site.yml

# 2. Trigger
gh workflow run deploy-site.yml

# 3. Wait and check
sleep 10
gh run list --workflow=deploy-site.yml --limit=1
```

#### `setup`
```bash
# 1. Switch Pages source to GitHub Actions
gh api -X PUT repos/{owner}/{repo}/pages \
  -f build_type=workflow

# 2. Verify
gh api repos/{owner}/{repo}/pages --jq '.build_type'
# Should output: "workflow"
```

#### `cleanup`
```bash
# 1. Check for gh-pages branch
git branch -r | grep gh-pages

# 2. Delete remote gh-pages branch
gh api -X DELETE repos/{owner}/{repo}/git/refs/heads/gh-pages

# 3. Prune local tracking
git remote prune origin

# 4. Verify Pages source is correct
gh api repos/{owner}/{repo}/pages --jq '.build_type'

# 5. Check no legacy configs remain
echo "Checking for deprecated configs..."
grep -l "DEPRECATED" site/docs/mkdocs.yml docs/site/_config.yml 2>/dev/null
```

#### `verify`
```bash
# 1. Local build test
cd site/portfolio && npm ci && npx astro build

# 2. Check output
ls -la dist/
du -sh dist/

# 3. Check for broken links (if available)
npx astro check 2>&1 || true
```

## Deployment Architecture

```
master branch push (site/portfolio/**)
  -> GitHub Actions (deploy-site.yml)
    -> npm ci + npx astro build
      -> upload-pages-artifact (site/portfolio/dist)
        -> deploy-pages
          -> https://tygwan.github.io/physical-unity/
```

**Single deployment path**: Astro only. No MkDocs, no Jekyll.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Site shows Jekyll theme | Pages source = legacy | `/deploy setup` |
| 404 on site | Workflow failed or not triggered | `/deploy trigger` |
| Old content after push | Workflow not triggered by paths | Check `paths:` filter in workflow |
| Build fails | Astro dependency issue | `/deploy verify` locally first |
| gh-pages branch exists | Legacy remnant | `/deploy cleanup` |

## Output Format

```markdown
## Deployment Status

| Item | Status |
|------|--------|
| Pages Source | GitHub Actions |
| Last Deploy | 2026-01-31 14:30 (success) |
| Site URL | https://tygwan.github.io/physical-unity/ |
| HTTP Status | 200 OK |
| gh-pages branch | Deleted |
| Legacy configs | All marked DEPRECATED |
```
