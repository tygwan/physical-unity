---
command: astro-preview
description: Start Astro development or preview server
triggers:
  - "astro-preview"
  - "astro preview"
  - "astro dev"
  - "사이트 미리보기"
---

# /astro-preview Command

## Usage
```
/astro-preview [--dev | --build-preview] [--port <port>] [--host]
```

## Workflow

### Development Mode (default)
```bash
npm run dev
# Starts at http://localhost:4321 with HMR
```

### Production Preview
```bash
npm run build && npm run preview
# Serves built output at http://localhost:4321
```

### Custom Port
```bash
npm run dev -- --port 3000
```

### Network Access (for mobile testing)
```bash
npm run dev -- --host 0.0.0.0
```

## Modes

| Mode | Command | Use Case |
|------|---------|----------|
| Dev | `npm run dev` | Active development with HMR |
| Preview | `npm run preview` | Test production build locally |
| Check | `npx astro check` | Type-check without building |

## Output
```markdown
## Server Started

| Setting | Value |
|---------|-------|
| Mode | Development / Preview |
| URL | http://localhost:4321 |
| Network | http://192.168.x.x:4321 |
| Hot Reload | Enabled / N/A |

Press Ctrl+C to stop the server.
```
