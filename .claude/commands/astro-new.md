---
command: astro-new
description: Scaffold a new Astro project with integrations and deployment config
triggers:
  - "astro-new"
  - "astro new"
  - "new astro"
  - "새 아스트로 프로젝트"
---

# /astro-new Command

## Usage
```
/astro-new [project-name] [--template <template>] [--deploy <target>]
```

## Workflow

### 1. Gather Requirements
Use `AskUserQuestion` to determine:
- **Project name**: Directory name for the new project
- **Template**: `minimal`, `blog`, `starlight` (docs), or `portfolio`
- **Integrations**: React, Tailwind, MDX, Sitemap
- **Deploy target**: GitHub Pages, Vercel, Netlify
- **i18n**: Languages to support

### 2. Scaffold
```bash
npm create astro@latest -- {project-name} \
  --template {template} \
  --install \
  --git \
  --typescript strict
```

### 3. Add Integrations
```bash
cd {project-name}
npx astro add react        # If interactive components needed
npm install @tailwindcss/vite  # Tailwind CSS v4
npx astro add mdx          # If rich content needed
npx astro add sitemap       # SEO
```

### 4. Configure
Update `astro.config.mjs` with:
- `site` URL
- `base` path (for GitHub Pages subpath)
- Vite plugins (Tailwind)
- i18n locales

### 5. Create Structure
```bash
mkdir -p src/{components/{layout,ui,react},content,layouts,pages,styles,utils,i18n}
```

### 6. Deploy Setup
Create `.github/workflows/deploy.yml` if GitHub Pages selected.

### 7. Summary
Output created project details:
- Project path
- Installed integrations
- Available npm scripts
- Next steps
