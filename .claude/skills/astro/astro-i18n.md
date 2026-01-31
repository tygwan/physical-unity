---
skill: astro-i18n
description: Internationalization for Astro sites - routing, translations, locale switching
triggers:
  - "astro i18n"
  - "아스트로 국제화"
  - "다국어"
  - "multilingual"
  - "internationalization"
model: sonnet
---

# Astro Internationalization (i18n)

## Configuration

### astro.config.mjs
```javascript
export default defineConfig({
  i18n: {
    defaultLocale: 'en',
    locales: ['en', 'ko', 'ja'],
    routing: {
      prefixDefaultLocale: false, // / = English, /ko/ = Korean
    },
    fallback: {
      ko: 'en',
      ja: 'en',
    },
  },
});
```

## Directory Structure
```
src/pages/
├── index.astro           # English (default)
├── about.astro
├── ko/
│   ├── index.astro       # Korean
│   └── about.astro
└── ja/
    ├── index.astro       # Japanese
    └── about.astro
```

## Translation Utility
```typescript
// src/i18n/utils.ts
import en from './en.json';
import ko from './ko.json';

const translations = { en, ko } as const;
type Locale = keyof typeof translations;

export function t(locale: Locale, key: string): string {
  const keys = key.split('.');
  let value: any = translations[locale];
  for (const k of keys) {
    value = value?.[k];
  }
  return value ?? key;
}

export function getLocaleFromUrl(url: URL): Locale {
  const [, locale] = url.pathname.split('/');
  if (locale in translations) return locale as Locale;
  return 'en';
}
```

### Translation Files
```json
// src/i18n/en.json
{
  "nav": { "home": "Home", "about": "About", "blog": "Blog" },
  "hero": { "title": "Welcome", "subtitle": "Build something great" }
}

// src/i18n/ko.json
{
  "nav": { "home": "홈", "about": "소개", "blog": "블로그" },
  "hero": { "title": "환영합니다", "subtitle": "멋진 것을 만들어보세요" }
}
```

## Language Switcher Component
```astro
---
const { currentLocale = 'en' } = Astro.props;
const locales = [
  { code: 'en', label: 'English', flag: 'EN' },
  { code: 'ko', label: '한국어', flag: 'KO' },
];
const currentPath = Astro.url.pathname;
---

<nav class="locale-switcher">
  {locales.map(({ code, label, flag }) => {
    const isActive = code === currentLocale;
    const href = code === 'en'
      ? currentPath.replace(/^\/(ko|ja)/, '')
      : `/${code}${currentPath.replace(/^\/(ko|ja)/, '')}`;
    return (
      <a href={href} class:list={['locale-link', { active: isActive }]}>
        {flag} {label}
      </a>
    );
  })}
</nav>
```

## Content Collections with i18n
```
src/content/blog/
├── en/
│   └── first-post.md
└── ko/
    └── first-post.md
```

```typescript
// Query by locale
const posts = await getCollection('blog', ({ id }) => id.startsWith('en/'));
```

## SEO: hreflang Tags
```astro
<link rel="alternate" hreflang="en" href={`${site}/page`} />
<link rel="alternate" hreflang="ko" href={`${site}/ko/page`} />
<link rel="alternate" hreflang="x-default" href={`${site}/page`} />
```
