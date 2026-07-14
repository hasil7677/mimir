     <div align="center">

# Mimir

**Your AI agent's memory is a folder of markdown files.**
Open it in Obsidian. Edit a fact by hand. The agent sees your edit on its next thought — no sync step, because there's nothing to sync.

[![CI](https://github.com/hasil7677/mimir/actions/workflows/ci.yml/badge.svg)](https://github.com/hasil7677/mimir/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![MCP](https://img.shields.io/badge/protocol-MCP-6b5ca5)](docs/CLIENTS.md)

</div>

<!-- 📸 REPLACE ME: a GIF or screenshot of the Obsidian graph view showing
     scenes → entities linked by [[wikilinks]], grown from a real conversation.
     This is the single highest-leverage thing in this README — the whole
     pitch is "your agent's brain is a vault you can see," so show it. -->

---

Every agent framework has the same problem: **conversations start from zero.** Full-context-stuffing burns your token budget, naive summarization throws facts away forever, and flat vector RAG can't tell a memory from yesterday apart from one from six months ago.

Mimir is a memory engine that fixes this **without asking you to run anything.** No cloud account, no Docker, no API key. Point it at nothing and it works — a local file becomes the transcript, another becomes the searchable facts, and a folder of markdown becomes the part you can actually read, understand, and correct. Add a local model (or a cloud one) later and every layer gets sharper automatically. Nothing you built against ever has to change.

## Why a vault, not a database

Every other memory system stores your agent's knowledge as opaque rows you'd need a script to inspect. Mimir writes it as **OKF** (Open Knowledge Format) — YAML frontmatter, markdown prose, `[[wikilinks]]` between facts and the entities they mention. That means:

- **You can open it.** `~/.mimir/vault` is a real Obsidian vault. Graph view, backlinks, search — all free, all native.
- **You can fix it.** The agent got something wrong? Edit the note. The fix takes effect on the next read. No re-ingestion, no cache invalidation dance.
- **It's yours.** `git init` your vault if you want history. Copy the folder to a new machine and your agent remembers everything, everywhere.
- **The databases are just indexes.** DuckDB and Qdrant exist to make search fast — the vault is the source of truth, always.

## What's actually happening under the hood

Four stores, each one optional except the first, each degrading independently if it's missing:

| Store | Holds | Without it |
|---|---|---|
| **DuckDB** (`~/.mimir/memories.db`) | Raw transcripts, extracted facts, audit log | Nothing works — this is the one file that has to exist |
| **Vault** (`~/.mimir/vault/`) | Human-readable markdown: scenes, entities, persona | No graph signal, no linked-note enrichment — facts still return |
| **Qdrant** (`~/.mimir/qdrant/`, embedded) | Fact vectors for semantic search | Keyword-only recall — still real, just literal-match |
| **Redis** (optional server) | Hot recent turns + query cache | No recent-turn context, cache misses every time — capture still lands |

A conversation flows through: **capture** (raw turns land in DuckDB, best-effort push to Redis) → **flush** at session end (an LLM — or a deterministic offline digest — turns the session into an Obsidian scene note, extracts atomic facts, dedupes against what's already known, flags contradictions instead of silently picking a winner, refreshes a running persona doc every N facts). **Recall** runs the other way: semantic cache check → BM25 + vector hybrid search → reciprocal rank fusion → a four-signal score (semantic relevance, recency decay, access frequency, graph proximity through your vault's own wikilinks) → a context string ready to inject into your agent's prompt.

Full diagrams and a module-by-module walkthrough: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## Quickstart

```bash
git clone https://github.com/hasil7677/mimir.git
cd mimir
pip install -e ".[dev]"
uvicorn app.main:app --port 8080
```

That's it — `GET /health` works with zero config. No `mimir.yaml`, no Redis, no API key required. Copy `mimir.yaml.example` to `mimir.yaml` when you want to point at a local Ollama model, a cloud LLM, or a Redis instance; every setting has a sane default until then.

## Use it from Claude Code (or any MCP client) right now

```bash
claude mcp add mimir --scope user -e MIMIR_USER_ID=you -- python /path/to/mimir/engine/adapters/mcp_embedded.py
```

No gateway to run — the embedded adapter imports the engine directly, so a process only exists while your agent session is open. Three tools show up: `mimir_recall`, `mimir_remember`, `mimir_flush`. Point your `CLAUDE.md` at them and your agent starts building a memory of you, one conversation at a time.

Also documented: OpenCode (native MCP), Pi (via the community `pi-mcp-adapter`), and a plain HTTP contract for anything else — see **[docs/CLIENTS.md](docs/CLIENTS.md)**.

## Status

This is early — built fast, tested hard, not yet battle-tested by anyone but me. Here's the honest split:

**Solid and tested (83 tests — 80 pass with zero services running, 3 need a live Redis — real HTTP layer, real filesystem, real dedup logic):**
hybrid recall with a 4-signal scoring formula · semantic caching with measured cache hits · L1.5 fact consolidation (exact-dup detection needs zero LLM calls; an LLM present gets you store/skip/supersede/contradiction-flag decisions, with hallucinated target IDs rejected) · GDPR-style erasure and export across every store · a self-healing recovery path for orphaned sessions (found live, fixed same day — see the commit log if you want to watch that happen) · MCP support verified end-to-end inside real Claude Code sessions.

**Known gaps, not hidden:**
- No real graph database yet — [KuZu](https://kuzudb.com) has no Python 3.11+ wheels as of this writing, so entity relationships live as vault `[[wikilinks]]` with hop-distance scoring instead of a Cypher-traversable graph. The scoring interface is already hop-based, so KuZu slots in without a rewrite once it's installable.
- Entity extraction is a regex heuristic (capitalized-run detection), not a real NER model. It works, and it also occasionally wikilinks a stray proper noun it shouldn't. spaCy is the planned fix.
- No benchmark numbers yet — no PersonaMem run, no measured recall accuracy. Every design claim here is architectural reasoning, not a published score.
- LangChain / OpenAI Agents adapters aren't built. The HTTP contract they'd need already exists.

If you're looking for something production-hardened with a support contract, this isn't it yet. If you want to see what a memory system looks like when the databases are treated as caches and the filesystem is treated as the truth, open the vault.

## Development

```bash
pip install -e ".[dev]"
pytest              # 83 tests; Redis-backed ones auto-skip without a server
```

CI runs the suite on Python 3.11–3.13, on both Ubuntu and Windows — the Windows leg is not decorative, it's what caught a real timezone bug during development.

## License

MIT — see [LICENSE](LICENSE). Built on the idea that the core memory engine should always be free and open; anything resembling a hosted/managed offering is a separate conversation for another day.
