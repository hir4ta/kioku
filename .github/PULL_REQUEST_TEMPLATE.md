<!-- Replace placeholder text. Keep the section headings intact. -->

## Summary

<!-- One-line summary of what this PR does and why. -->

## Motivation

<!-- What problem does this solve? Reference issue numbers if applicable. -->

## Approach

<!-- High-level description of the implementation strategy. -->

## Affected components

<!-- Check all that apply. -->

- [ ] `lib/` (Python core)
- [ ] `hooks/` (bash wrappers + `hooks.json`)
- [ ] `agents/` (plugin agents)
- [ ] `skills/` (plugin skills)
- [ ] `schemas/` (JSON Schemas)
- [ ] `templates/` (vault templates rendered by `/kioku:init`)
- [ ] `cron/` (background jobs)
- [ ] `dashboard/` (Web dashboard)
- [ ] `docs/` (architecture / decisions / threat model / getting-started)
- [ ] `.github/workflows/` (CI)
- [ ] `tests/`
- [ ] `.claude-plugin/plugin.json` (manifest)

## Test plan

<!-- What did you run? Include outputs or paste relevant logs. -->

- [ ] `task lint`
- [ ] `task test:unit`
- [ ] `task test:integration` (if relevant)
- [ ] `task test:bats` (if hooks changed)
- [ ] `task dashboard:typecheck && task dashboard:test` (if dashboard changed)
- [ ] Manual test: <describe>

## Pre-merge checklist

- [ ] `docs/kioku-decisions.md` updated if architecture / scope changed (MADR entry)
- [ ] `README.md` updated if user-facing behavior changed
- [ ] `schemas/` updated and `dashboard` types regenerated if schema changed
- [ ] `.claude/rules/` updated if conventions changed
- [ ] No secrets, `.env`, credentials, or absolute home paths in diff (`git diff --cached | grep -iE 'api[_-]?key|secret|token'` returns nothing)
- [ ] Voyage / Anthropic API tokens never logged or echoed by any new code

## Breaking change?

<!-- Yes / No. If yes, describe migration path. For memory.schema.json changes,
     link to the new DEC-* entry in docs/kioku-decisions.md and the migration
     script under lib/migrations/. -->
