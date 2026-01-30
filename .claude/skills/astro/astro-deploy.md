---
skill: astro-deploy
description: Deploy Astro sites to GitHub Pages, Vercel, Netlify, or Firebase
triggers:
  - "astro deploy"
  - "아스트로 배포"
  - "github pages"
  - "deploy site"
model: sonnet
---

# Astro Deployment

## GitHub Pages (Recommended for OSS)

### 1. Configure astro.config.mjs
```javascript
export default defineConfig({
  site: 'https://{username}.github.io',
  base: '/{repo-name}/',
});
```

### 2. GitHub Actions Workflow
```yaml
# .github/workflows/deploy.yml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: './package-lock.json'
      - run: npm ci
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: ./dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - uses: actions/deploy-pages@v4
        id: deployment
```

### 3. Enable in Repository Settings
Settings > Pages > Source: GitHub Actions

## Vercel

```bash
npx astro add vercel
npm run build
npx vercel deploy --prod
```

## Netlify

```bash
npx astro add netlify
# netlify.toml
# [build]
#   command = "npm run build"
#   publish = "dist"
```

## Pre-Deploy Checklist
- [ ] `astro check` passes with no errors
- [ ] `npm run build` succeeds locally
- [ ] `site` and `base` correctly set in astro.config.mjs
- [ ] All internal links use `base` prefix
- [ ] Images and assets load with correct paths
- [ ] 404 page exists (`src/pages/404.astro`)
- [ ] Favicon and meta tags configured
- [ ] Sitemap generated (if using `@astrojs/sitemap`)

## Troubleshooting

### Assets not loading on GitHub Pages
Ensure all paths respect the `base` config:
```astro
<img src={`${import.meta.env.BASE_URL}images/logo.png`} />
```

### Build fails on CI
Check Node.js version matches local (use `.nvmrc` or `engines` in package.json).
