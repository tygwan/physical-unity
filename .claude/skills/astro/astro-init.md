---
skill: astro-init
description: Initialize new Astro projects with integrations, tailwind, and deployment config
triggers:
  - "astro init"
  - "astro 초기화"
  - "new astro project"
  - "새 아스트로"
model: sonnet
---

# Astro Project Initialization

## Workflow

### Step 1: Scaffold Project
```bash
npm create astro@latest -- --template minimal --install --git --typescript strict
```

### Step 2: Add Integrations
Based on project needs:
```bash
# React (for interactive islands)
npx astro add react

# Tailwind CSS v4
npm install @tailwindcss/vite
# Then configure in astro.config.mjs

# MDX (for content)
npx astro add mdx

# Sitemap
npx astro add sitemap
```

### Step 3: Configure astro.config.mjs
```javascript
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';
import react from '@astrojs/react';
import sitemap from '@astrojs/sitemap';

export default defineConfig({
  site: 'https://yourdomain.com',
  base: '/project-name/',
  vite: { plugins: [tailwindcss()] },
  integrations: [react(), sitemap()],
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ko'],
  },
});
```

### Step 4: Project Structure
```
src/
├── components/       # .astro components
│   ├── layout/       # Layout components
│   ├── ui/           # Reusable UI
│   └── react/        # React islands
├── content/          # Content Collections
│   └── config.ts     # Collection schemas
├── layouts/          # Page layouts
├── pages/            # File-based routing
│   ├── index.astro
│   └── [...slug].astro
├── styles/           # Global styles
│   └── global.css
└── utils/            # Utilities
```

### Step 5: TypeScript Config
```json
{
  "extends": "astro/tsconfigs/strict",
  "compilerOptions": {
    "baseUrl": ".",
    "paths": { "@/*": ["src/*"] }
  }
}
```

## Post-Init Checklist
- [ ] `astro.config.mjs` configured with site URL and base path
- [ ] TypeScript strict mode enabled
- [ ] Tailwind CSS v4 working
- [ ] React integration added (if needed)
- [ ] Content Collections schema defined
- [ ] `.github/workflows/deploy.yml` created
- [ ] `.gitignore` includes `node_modules/`, `dist/`, `.astro/`
