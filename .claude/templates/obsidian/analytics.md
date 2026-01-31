---
tags: [analytics, {{period_type}}]
date: {{date}}
period: {{period}}
project: "[[Projects/{{project_name}}/_Index|{{project_name}}]]"
---

# Analytics: {{period}}

## Overview

**Period**: {{period_start}} ~ {{period_end}}
**Project**: [[Projects/{{project_name}}/_Index|{{project_name}}]]

## Summary Stats

| Metric | Value |
|--------|-------|
| Total Tool Calls | {{total_calls}} |
| Success Rate | {{success_rate}}% |
| Sessions | {{session_count}} |
| Files Modified | {{files_modified}} |
| Decisions Made | {{decision_count}} |
| Learnings Captured | {{learning_count}} |

## Tool Usage

```
{{tool_usage_chart}}
```

| Tool | Count | % |
|------|-------|---|
{{tool_usage_table}}

## Activity Pattern

```
{{activity_sparkline}}
```

### Peak Hours
{{peak_hours}}

### Most Active Days
{{active_days}}

## Category Distribution

```
{{category_chart}}
```

## Agent Usage

| Agent | Invocations |
|-------|-------------|
{{agent_usage_table}}

## Top Files

Most frequently modified:

{{top_files_list}}

## Insights

### Productivity Patterns

{{productivity_insights}}

### Tool Efficiency

{{tool_efficiency}}

### Recommendations

{{recommendations}}

## Sessions This Period

{{session_links}}

## Decisions Made

{{decision_links}}

## Learnings

{{learning_links}}

---

**Previous**: [[Analytics/{{prev_period}}|{{prev_period}}]]
**Next**: [[Analytics/{{next_period}}|{{next_period}}]]
