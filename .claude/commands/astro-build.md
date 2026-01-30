---
command: astro-build
description: Build and validate Astro site with pre-build checks
triggers:
  - "astro-build"
  - "astro build"
  - "build site"
  - "사이트 빌드"
---

# /astro-build Command

## Usage
```
/astro-build [--check] [--analyze] [--preview]
```

## Workflow

### 1. Pre-Build Validation
```bash
# Type checking
npx astro check

# Lint (if configured)
npm run lint 2>/dev/null || echo "No lint script configured"
```

### 2. Build
```bash
npm run build
```

### 3. Post-Build Analysis
```bash
# Output size
du -sh dist/
find dist -type f | wc -l

# Check for large files
find dist -type f -size +500k -exec ls -lh {} \;

# Verify critical pages exist
ls dist/index.html
ls dist/404.html 2>/dev/null || echo "WARNING: No 404 page"
```

### 4. Preview (optional)
```bash
npm run preview
```

### 5. Report
```markdown
## Build Report

| Metric | Value |
|--------|-------|
| Status | Success/Failed |
| Output Size | X MB |
| File Count | N files |
| Build Time | X.Xs |
| Errors | 0 |
| Warnings | N |
```

## Flags
- `--check`: Run `astro check` before build (default: true)
- `--analyze`: Show output size analysis (default: true)
- `--preview`: Start preview server after build (default: false)
