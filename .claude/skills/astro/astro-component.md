---
skill: astro-component
description: Create Astro components, layouts, and React islands with best practices
triggers:
  - "astro component"
  - "아스트로 컴포넌트"
  - "create component"
  - "astro layout"
model: sonnet
---

# Astro Component Creation

## Component Types

### 1. Astro Component (.astro)
Static, zero-JS by default:
```astro
---
interface Props {
  title: string;
  variant?: 'primary' | 'secondary';
  class?: string;
}
const { title, variant = 'primary', class: className } = Astro.props;
---

<div class:list={['card', `card--${variant}`, className]}>
  <h3>{title}</h3>
  <slot />
</div>

<style>
  .card {
    border-radius: 0.5rem;
    padding: 1.5rem;
  }
  .card--primary { background: var(--color-primary); }
  .card--secondary { background: var(--color-secondary); }
</style>
```

### 2. Layout Component
```astro
---
interface Props {
  title: string;
  description?: string;
}
const { title, description = 'Default description' } = Astro.props;
---

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content={description} />
  <title>{title}</title>
</head>
<body>
  <slot name="header" />
  <main>
    <slot />
  </main>
  <slot name="footer" />
</body>
</html>
```

### 3. React Island Component
For interactive UI requiring client-side JavaScript:
```tsx
// src/components/react/Counter.tsx
import { useState } from 'react';

interface CounterProps {
  initial?: number;
  label: string;
}

export default function Counter({ initial = 0, label }: CounterProps) {
  const [count, setCount] = useState(initial);
  return (
    <div>
      <span>{label}: {count}</span>
      <button onClick={() => setCount(c => c + 1)}>+</button>
    </div>
  );
}
```

Usage in Astro:
```astro
---
import Counter from '../components/react/Counter';
---
<Counter client:visible label="Views" initial={0} />
```

## Naming Conventions
- Astro components: `PascalCase.astro`
- Layouts: `BaseLayout.astro`, `PostLayout.astro`
- React components: `PascalCase.tsx`
- Pages: `kebab-case.astro` or `[slug].astro`

## Slot Patterns

### Named Slots
```astro
<!-- Parent -->
<BaseLayout>
  <header slot="header">Nav here</header>
  <p>Main content</p>
  <footer slot="footer">Footer here</footer>
</BaseLayout>
```

### Slot Fallbacks
```astro
<slot name="sidebar">
  <p>Default sidebar content</p>
</slot>
```

## Props Validation
Always use TypeScript `interface Props` in the frontmatter for type safety and autocompletion.
