# kioku

> **記憶** (kioku, "memory") — persistent long-term memory for Claude Code via an Obsidian vault, Voyage embeddings, and lifecycle hooks.

`kioku` is a [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) plugin that gives your AI a memory that survives sessions, `/compact`, `/clear`, and resumes. Session summaries, decisions, and patterns are written to a local [Obsidian](https://obsidian.md/) vault you own. Voyage embeddings + `rerank-2.5` index them. Eight lifecycle hooks capture and re-inject the most relevant memory into every new session.

## Status

> **Pre-alpha (v0.0.1)** — Phase 0 scaffolding. Not installable yet. The architecture is decided and the scaffolding is being built in public.

## Why kioku exists

Claude Code already ships with `CLAUDE.md` and an auto-memory directory. They are flat Markdown, capped at 200 lines for auto-load, and lose context at every `/compact`. Anthropic's own research ([context rot](https://www.trychroma.com/research/context-rot), [NoLiMa](https://arxiv.org/abs/2502.05167)) shows that even 1M-token models degrade sharply when irrelevant context is in scope.

`kioku` is the missing layer:

- **Multi-tier memory** (working / episodic / semantic / procedural) backed by Markdown files in your Obsidian vault — the cognitive-architecture split from [CoALA (Sumers et al., 2023)](https://arxiv.org/abs/2309.02427).
- **Hybrid retrieval** (FTS5 + `sqlite-vec` + `voyage-4-large` + `rerank-2.5`) instead of cosine-only top-K.
- **Just-in-time injection** through eight Claude Code hooks (`SessionStart × 4 matchers`, `PreCompact`, `SessionEnd`, `Stop`, `UserPromptSubmit`).
- **MADR-style decision records** designed to be read back by future Claude sessions: decision first, context last, so chunked retrieval surfaces the conclusion.
- **Obsidian-native**: the vault is the source of truth. SQLite and DuckDB are derived indexes that can be rebuilt from the vault at any time.
- **No new MCP server, no daemon** — kioku runs as Claude Code hooks plus a few `cron` jobs.

## How it works

```
┌────────────────────────────────────────────────────────────────────┐
│  Obsidian vault  (source of truth, human-readable Markdown)        │
│  ~/Documents/Obsidian/Kioku/                                       │
│    working/, episodic/, semantic/, procedural/, projects/          │
└────────────────────────────────────────────────────────────────────┘
              ▲                                          │
              │ Claude writes (SessionEnd, PreCompact)   │ ETL
              │                                          ▼
┌────────────────────────────┐         ┌─────────────────────────────┐
│  Claude Code hooks         │ ◀────── │  SQLite + sqlite-vec        │
│  - SessionStart (×4)       │  inject │  Hybrid (BM25 + dense)      │
│  - PreCompact              │         │  Voyage rerank-2.5          │
│  - SessionEnd / Stop       │         └─────────────────────────────┘
│  - UserPromptSubmit        │
└────────────────────────────┘
```

## Prerequisites

- macOS or Linux (Windows untested)
- Python 3.12+
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) (Pro / Max plan or API key)
- [Obsidian](https://obsidian.md/) (free)
- [Voyage AI API key](https://www.voyageai.com/) (free tier sufficient to bootstrap)

You do **not** need a separate Anthropic API key. `kioku` invokes `claude -p` for in-hook summarization, which uses your existing Claude Code billing.

## Install

> Not yet shippable. Once Phase 0 lands:
>
> ```bash
> claude plugin install kioku@hir4ta/kioku
> /kioku:init   # interactive setup: vault path, Voyage key, project defaults
> ```

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Repo scaffolding, plugin manifest, vault templates, decision log | In progress |
| 1 | L4 → L3 ETL (vault.py, chunk.py, embed.py, store_sqlite.py) | Pending |
| 2 | SessionStart × 4 matchers + inject.py (XML format + cache layers) | Pending |
| 3 | Hybrid retrieval + Voyage rerank + UserPromptSubmit prefetch | Pending |
| 4 | SessionEnd + Stop hooks + Claude-driven `working/` updates | Pending |
| 5 | PreCompact + SessionStart(compact) + structured extraction via `claude -p` | Pending |
| 6 | Cron: nightly consolidate, decay pass, conflict scan, weekly review | Pending |
| 7 | Memory poisoning hardening + LongMemEval-style benchmark suite | Pending |
| 8 | DuckDB analytical layer + CLI dashboard | Pending |
| 9 | Web dashboard (Vite + React 19 + Tailwind v4 + shadcn/ui) | Pending |

## Architecture

See [docs/architecture.md](docs/architecture.md) (coming).

## Threat model

`kioku` reads your transcripts and sends embedding payloads to Voyage AI. See [docs/threat-model.md](docs/threat-model.md) for the data-flow and memory-poisoning mitigations ([MINJA, arXiv 2503.03704](https://arxiv.org/abs/2503.03704); [OWASP LLM01:2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)).

## Sibling projects

- [mumei](https://github.com/hir4ta/mumei) — a Claude Code SDD harness (`/mumei:plan`, `/mumei:review`). Different shape, same author. `kioku` and `mumei` are fully independent and can be installed side by side.

## License

MIT — see [LICENSE](LICENSE).
