---
skill: astro
description: Astro framework development - project setup, components, content collections, i18n, and deployment
triggers:
  - astro
  - "아스트로"
  - "static site"
  - "정적 사이트"
  - "ssg"
  - "astro build"
  - "astro deploy"
model: sonnet
---

# Astro Framework Skill

## Overview
Comprehensive Astro development skill covering project initialization, component architecture, Content Collections, internationalization, and deployment automation.

## Quick Reference

| Sub-skill | Trigger | Purpose |
|-----------|---------|---------|
| `/astro-new` | New project | Scaffold Astro project with integrations |
| `/astro-build` | Build site | Build, optimize, and validate |
| `/astro-preview` | Preview | Local dev server with HMR |

## Core Workflow

### 1. Project Analysis
Before any work, analyze the existing Astro project:
```
Read: astro.config.mjs
Read: package.json
Read: tsconfig.json
Glob: src/**/*.astro
Glob: src/content/**/*
```

### 2. Architecture Decision
Determine the rendering strategy:
- **SSG (Static)** - Default, best for content sites
- **SSR (Server)** - Dynamic content, auth required
- **Hybrid** - Mix of static and server routes

### 3. Component Creation Pattern
```astro
---
// Component Script (runs at build time)
interface Props {
  title: string;
  description?: string;
}
const { title, description } = Astro.props;
---

<!-- Component Template -->
<section class="component">
  <h2>{title}</h2>
  {description && <p>{description}</p>}
  <slot />
</section>

<style>
  .component {
    /* Scoped CSS - no leaks */
  }
</style>
```

### 4. React Island Pattern
For interactive components:
```astro
---
import InteractiveWidget from '../components/react/Widget';
---
<InteractiveWidget client:visible data={staticData} />
```

Client directives:
- `client:load` - Hydrate immediately
- `client:idle` - Hydrate when idle
- `client:visible` - Hydrate when visible (recommended)
- `client:media` - Hydrate on media query match
- `client:only="react"` - Skip SSR, client-only

### 5. Content Collections
```typescript
// src/content/config.ts
import { defineCollection, z } from 'astro:content';

const blog = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    date: z.date(),
    tags: z.array(z.string()).optional(),
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog };
```

### 6. i18n Setup
```javascript
// astro.config.mjs
export default defineConfig({
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ko'],
    routing: { prefixDefaultLocale: false },
  },
});
```

### 7. Deployment
GitHub Pages:
```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages
on:
  push:
    branches: [main]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with: { path: dist }
  deploy:
    needs: build
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

## Integration with Other Skills

| Skill | When to Use |
|-------|-------------|
| `frontend-design` | Visual design, CSS, animations |
| `react-expert` | Complex interactive islands |
| `d3js-skill` | Data visualizations |
| `codex` | Deep code analysis with OpenAI models |
| `publishing-astro-websites` | Advanced Astro patterns |

## Tailwind CSS v4 Integration
```javascript
// astro.config.mjs
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({
  vite: { plugins: [tailwindcss()] },
});
```

```css
/* src/styles/global.css */
@import "tailwindcss";
```

## Performance Checklist
- [ ] Images use `<Image>` component (auto-optimization)
- [ ] Scripts use `client:visible` where possible
- [ ] Static pages use SSG (no unnecessary SSR)
- [ ] CSS is scoped per component
- [ ] Fonts use `preload` link
- [ ] Build output analyzed with `astro check`
