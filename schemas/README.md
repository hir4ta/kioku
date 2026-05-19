# kioku JSON Schemas

These schemas define the data contracts between kioku's Python core, the Markdown vault, and the dashboard.

## Files

| File | Consumed by | Purpose |
|---|---|---|
| `memory.schema.json` | vault Markdown frontmatter, `lib/vault.py`, dashboard | Memory record metadata (frontmatter spec for every `.md` in the vault) |
| `settings.schema.json` | `~/.config/kioku/config.toml`, `lib/config.py` | User settings: vault path, Voyage model selection, scoring weights, cron schedule |
| `conflict.schema.json` | `store.conflicts` table, `lib/classify.py` | Memory conflict records (STALE-style, arxiv 2605.06527) |
| `benchmark-result.schema.json` | `_meta/benchmark/*.json`, `lib/benchmark.py` | LongMemEval-style weekly evaluation output (arxiv 2410.10813) |

## Validation flow

Every Python entry point that reads/writes a record covered by one of these schemas calls `jsonschema.validate()` with the corresponding loaded schema:

- `lib/vault.py::read_memory_file` → validates against `memory.schema.json`
- `lib/vault.py::write_memory_file` → validates against `memory.schema.json` before writing
- `lib/config.py::load_settings` → validates against `settings.schema.json`
- `lib/classify.py::record_conflict` → validates against `conflict.schema.json`
- `lib/benchmark.py::write_result` → validates against `benchmark-result.schema.json`

Validation failures raise `KiokuSchemaError` (defined in `lib/errors.py`).

## Dashboard type generation

The dashboard (`dashboard/`) generates TypeScript types from these schemas via `npm run generate-types`. **Never edit `dashboard/src/types/*.ts` by hand** — CI will fail on drift. The schemas are the single source of truth for shared types.

## Schema evolution

Breaking changes to `memory.schema.json` require a migration plan documented in `docs/kioku-decisions.md` (new DEC entry, status=accepted, with `supersedes` pointing at the previous schema decision). `lib/migrations/` (planned for Phase 1+) holds the migration scripts.

Non-breaking additions (new optional fields, new enum values that fall back to a default) can be made freely.
