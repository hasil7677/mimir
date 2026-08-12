
<div align="center">

# Mimir

**Your AI agent's memory is a folder of markdown files.**
Open it in Obsidian. Edit a fact by hand. The agent sees your edit on its next thought — no sync step, because there's nothing to sync.

[![CI](https://github.com/hasil7677/mimir/actions/workflows/ci.yml/badge.svg)](https://github.com/hasil7677/mimir/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![MCP](https://img.shields.io/badge/protocol-MCP-6b5ca5)](docs/CLIENTS.md)
[![GitHub stars](https://img.shields.io/github/stars/hasil7677/mimir?style=social)](https://github.com/hasil7677/mimir/stargazers)

</div>

<p align="center">
  <img src="docs/images/vault-demo.gif" alt="The Obsidian vault graph view, grown from a real conversation" width="640">
</p>

---

**[Skip to: What's under the hood](#whats-actually-happening-under-the-hood) · [Quickstart](#quickstart) · [Benchmarks](#benchmarks) · [Roadmap](#roadmap-closing-the-gap) · [Status / known gaps](#status)**

---

Every agent framework has the same problem: **conversations start from zero.** Full-context-stuffing burns your token budget, naive summarization throws facts away forever, and flat vector RAG can't tell a memory from yesterday apart from one from six months ago.

Mimir is a memory engine that fixes this **without asking you to run anything.** No cloud account, no Docker, no API key. Point it at nothing and it works — a local file becomes the transcript, another becomes the searchable facts, and a folder of markdown becomes the part you can actually read, understand, and correct. Add a local model (or a cloud one) later and every layer gets sharper automatically. Nothing you built against ever has to change.

| | |
|---|---|
| **Public benchmark** | [Agent Memory Benchmark](https://agentmemorybenchmark.ai), `personamem/32k`: **59.6%** (351/589), real number, no cherry-picking. [Details ↓](#benchmarks) |
| **Setup** | `pip install -e ".[dev]"` — that's it, no server, no API key |
| **Talks to** | Claude Code / any MCP client, or a plain HTTP contract for everything else |

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
claude mcp add mimir --scope user -e MIMIR_USER_ID=you -- python /path/to/mimir/adapters/mcp_embedded.py
```

No gateway to run — the embedded adapter imports the engine directly, so a process only exists while your agent session is open. Three tools show up: `mimir_recall`, `mimir_remember`, `mimir_flush`. Point your `CLAUDE.md` at them and your agent starts building a memory of you, one conversation at a time.

Also documented: OpenCode (native MCP), Pi (via the community `pi-mcp-adapter`), and a plain HTTP contract for anything else — see **[docs/CLIENTS.md](docs/CLIENTS.md)**.

## Status

This is early — built fast, tested hard, not yet battle-tested by anyone but me. Here's the honest split:

**Solid and tested (113 tests, 109 pass with zero services/extras installed, 4 skip: 3 need a live Redis, 1 needs the optional `fastembed` extra, real HTTP layer, real filesystem, real dedup logic):**
hybrid recall with a 4-signal scoring formula · semantic caching with measured cache hits · L1.5 fact consolidation (exact-dup detection needs zero LLM calls; an LLM present gets you store/skip/supersede/contradiction-flag decisions, with hallucinated target IDs rejected) · GDPR-style erasure and export across every store · a self-healing recovery path for orphaned sessions (found live, fixed same day — see the commit log if you want to watch that happen) · MCP support verified end-to-end inside real Claude Code sessions.

**Known gaps, not hidden:**
- No real graph database yet. [KuZu](https://kuzudb.com) has no Python 3.11+ wheels as of this writing, so entity relationships live as vault `[[wikilinks]]` with hop-distance scoring instead of a Cypher-traversable graph. The scoring interface is already hop-based, so KuZu slots in without a rewrite once it's installable.
- Entity extraction now runs real NER (spaCy `en_core_web_sm`, optional `mimir-engine[ner]` extra) instead of the old regex capitalized-run heuristic, which remains as the fallback when spaCy isn't installed.
- Benchmark accuracy is 59.6% on the full public AMB run below (up from 43.3% first-pass). The single biggest lever wasn't a Mimir code change at all: the AMB harness silently defaults to a weak `gemini-2.5-flash-lite` model for answering the benchmark questions. Pointing it at the same `gemini-2.5-flash` tier already used for Mimir's own extraction/synthesis moved accuracy 48.7% to 59.6%, and an oracle-mode diagnostic (gold docs only, zero retrieval noise) landed at the *exact same* 59.6% ceiling under `flash`. That convergence is the real signal: once the answer model is competent, real Mimir retrieval and theoretically-perfect retrieval score identically, so retrieval/graph quality is no longer the constraint. The answer model's own reasoning is. (Also tried `gemini-2.5-pro`, expecting it to do even better. It scored *worse*, 46.0%, second-guessing itself on this benchmark's forced-choice-between-near-identical-paraphrases format in a way `flash` doesn't. Bigger isn't automatically better here.)
- LangChain / OpenAI Agents adapters aren't built. The HTTP contract they'd need already exists.

If you're looking for something production-hardened with a support contract, this isn't it yet. If you want to see what a memory system looks like when the databases are treated as caches and the filesystem is treated as the truth, open the vault.

## Benchmarks

**[Agent Memory Benchmark](https://agentmemorybenchmark.ai)**: public, reproducible. Full `personamem/32k` split: 195 real sessions, 589 questions, multiple-choice (exact-letter-match scored, no LLM judge involved for this task type). Mimir's own extraction/synthesis always runs on `gemini-2.5-flash`; the table below varies only the model the *benchmark harness* uses to answer the MCQ questions, since that turned out to matter more than anything in Mimir's own retrieval pipeline.

| Retrieval | Answer model | Accuracy |
|---|---|---|
| Mimir (real) | `gemini-2.5-flash-lite` (AMB harness's silent default) | 48.7% (287/589) |
| Mimir (real) | `gemini-2.5-pro` | 46.0% (271/589) |
| Mimir (real) | **`gemini-2.5-flash`** | **59.6% (351/589)** |
| Oracle mode (gold docs only, zero retrieval noise) | `gemini-2.5-flash-lite` | 50.8% (299/589) |
| Oracle mode (gold docs only, zero retrieval noise) | `gemini-2.5-flash` | 59.6% (351/589) |

Headline number is real Mimir retrieval plus `gemini-2.5-flash` answering: **59.6%**. Not cherry-picked: the table shows all three answer models tried, including the one that scored worse (`pro`). See the [live leaderboard](https://agentmemorybenchmark.ai) for how this compares to other systems (their configured answer model per-provider isn't independently verified here yet, a fairness pass on that is still open, see roadmap).

The oracle-mode rows are the key diagnostic. Oracle mode bypasses retrieval entirely (only gold-relevant documents get ingested, so retrieval/graph scoring literally cannot cost you points), and under `flash` it lands at **the exact same 59.6%** as real retrieval. Under a weak answer model there was a real ~2-point retrieval gap (48.7% vs. 50.8%); under a competent one, that gap vanishes completely. In other words: Mimir's own retrieval and 4-signal graph scoring are no longer where the accuracy is being lost. The constraint moved entirely to the answer model's own reasoning quality on this benchmark's question format.

## Roadmap: closing the gap

59.6% is up from a 43.3% first pass, not a finished number. Here's the actual order things got worked, not a wishlist:

- **Shipped**: entity notes were getting created for names an LLM synthesis step returned even when that name never literally showed up in the note it was supposedly linked from, orphaned single-node clutter in the graph with no edge to anything. Fixed: only entities that actually got wikilinked get a note now. Also widened the regex entity extractor's filler-word list to cut false-positive nodes (conversational filler like "Sure", "Actually" getting mistaken for named entities).
- **Shipped**: graph traversal was structurally dead. Entity notes had no outgoing links, so hop-1/2 scoring and the LINKED NOTES prompt section never had anything to walk. Entity notes now backlink to every scene that mentions them, and two knock-on bugs (a wasted first traversal hop, multi-word entities keyed by slug instead of display text) got fixed alongside it. Also fixed a scoring bug where an exact BM25 keyword hit the vector leg missed was forced to semantic=0.0 instead of keeping its RRF-derived relevance. Together: 43.3% to 49.4%.
- **Shipped**: ran an oracle-mode diagnostic (ingest only gold documents, bypassing retrieval noise entirely) to find out how much of the remaining gap is retrieval versus extraction/generation quality. Result at the time: 50.8% vs. 49.4%, only a 1.4-point move, so retrieval and graph scoring looked close to their ceiling. (This conclusion held up but for a different reason than expected, see below.)
- **Shipped, no measurable win**: swapped the regex capitalized-run entity extractor for real NER (spaCy `en_core_web_sm`, optional `mimir-engine[ner]` extra, degrades to the old regex heuristic if it isn't installed). Also drops numeric/temporal entity labels (dates, quantities, money) and lowercase false positives the small model occasionally tags, neither of which are wikilink-worthy. Re-ran the full benchmark: 48.7% vs. the pre-swap 49.4%, flat, within noise, arguably slightly down. Kept the change anyway (strictly better entity quality, cleaner vault graph) but it confirmed entity/graph quality wasn't the lever.
- **Shipped, the actual lever**: the oracle-mode diagnostic above was quietly using a weak answer model the whole time. The AMB harness silently defaults to `gemini-2.5-flash-lite` for answering benchmark questions, completely independent of what LLM Mimir itself uses for extraction. Pointed the harness's answer step at `gemini-2.5-flash` (the same tier Mimir's extraction already uses) instead: **48.7% to 59.6%**, and oracle mode moved in lockstep to the identical 59.6%. That convergence is the real finding: under a competent answer model, real Mimir retrieval and theoretically-perfect retrieval score identically, so the roughly 2-point "retrieval gap" measured earlier wasn't really about retrieval quality at all. It was retrieval quality *as a bottleneck for a weak answer model*. Also tried `gemini-2.5-pro` expecting further gains: it scored worse (46.0%), second-guessing itself on this benchmark's near-identical-paraphrase MCQ format in a way `flash` doesn't. Not a Mimir engine change; a benchmark-harness configuration fix, disclosed as exactly that.
- **Next**: verify whether other AMB leaderboard entries are configured with a comparably competent answer model or are also sitting on a weak default. Matters for how the leaderboard comparison should be read, not addressed yet.
- **Next**: prompt-level work on the MCQ answer step itself (more directive instructions, maybe few-shot examples) now that the answer model, not retrieval, is the known ceiling. The next accuracy gains likely live here, not in the engine.
- **Then**: a real Cypher-traversable graph via [KuZu](https://kuzudb.com) once it ships Python 3.11+ wheels, replacing the current wikilink-hop-distance approximation. Still a legitimate architecture improvement, just no longer expected to move this particular benchmark number much.
- **Then**: run the same harness against LoCoMo and LongMemEval (already wired into the eval setup) to see whether the retrieval-ceiling finding above holds on other question formats too, or is specific to personamem's paraphrase-heavy MCQ style.

## Development

```bash
pip install -e ".[dev]"
pytest              # 113 tests; Redis-backed and fastembed-backed ones auto-skip without a server/extra installed
```

CI runs the suite on Python 3.11–3.13, on both Ubuntu and Windows — the Windows leg is not decorative, it's what caught a real timezone bug during development.

## License

MIT — see [LICENSE](LICENSE). Built on the idea that the core memory engine should always be free and open; anything resembling a hosted/managed offering is a separate conversation for another day.
