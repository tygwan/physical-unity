---
skill: astro-content
description: Astro Content Collections - schema definition, querying, rendering, and MDX integration
triggers:
  - "content collections"
  - "콘텐츠 컬렉션"
  - "astro content"
  - "markdown collection"
model: sonnet
---

# Astro Content Collections

## Schema Definition
```typescript
// src/content/config.ts
import { defineCollection, z, reference } from 'astro:content';

const blog = defineCollection({
  type: 'content', // Markdown/MDX
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.date(),
    updatedDate: z.date().optional(),
    author: z.string().default('Anonymous'),
    tags: z.array(z.string()).default([]),
    image: z.string().optional(),
    draft: z.boolean().default(false),
    relatedPosts: z.array(reference('blog')).optional(),
  }),
});

const projects = defineCollection({
  type: 'content',
  schema: z.object({
    title: z.string(),
    description: z.string(),
    tech: z.array(z.string()),
    url: z.string().url().optional(),
    repo: z.string().url().optional(),
    featured: z.boolean().default(false),
    order: z.number().default(0),
  }),
});

const data = defineCollection({
  type: 'data', // JSON/YAML
  schema: z.object({
    name: z.string(),
    value: z.number(),
  }),
});

export const collections = { blog, projects, data };
```

## Querying Collections

### Get All Entries
```astro
---
import { getCollection } from 'astro:content';

const allPosts = await getCollection('blog', ({ data }) => !data.draft);
const sortedPosts = allPosts.sort((a, b) =>
  b.data.date.valueOf() - a.data.date.valueOf()
);
---
```

### Get Single Entry
```astro
---
import { getEntry } from 'astro:content';
const post = await getEntry('blog', 'my-post');
const { Content } = await post.render();
---
<Content />
```

### Dynamic Routes
```astro
---
// src/pages/blog/[...slug].astro
import { getCollection } from 'astro:content';

export async function getStaticPaths() {
  const posts = await getCollection('blog');
  return posts.map(post => ({
    params: { slug: post.slug },
    props: { post },
  }));
}

const { post } = Astro.props;
const { Content } = await post.render();
---

<article>
  <h1>{post.data.title}</h1>
  <time datetime={post.data.date.toISOString()}>
    {post.data.date.toLocaleDateString()}
  </time>
  <Content />
</article>
```

## MDX Integration
```bash
npx astro add mdx
```

MDX files can import and use components:
```mdx
---
title: Interactive Post
---
import Chart from '../../components/react/Chart';

# My Post

Here's an interactive chart:

<Chart client:visible data={[1,2,3]} />
```

## Content File Format
```markdown
---
title: "My First Post"
date: 2025-01-15
tags: ["astro", "web"]
draft: false
---

# Content starts here

Regular markdown with **formatting**.
```

## Type Safety
Collections are fully typed. TypeScript will error if:
- Required fields are missing in frontmatter
- Field types don't match schema
- Referenced entries don't exist (with `reference()`)
