
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
| **Public benchmark** | [Agent Memory Benchmark](https://agentmemorybenchmark.ai), `personamem/32k`: **62.1%** (366/589), full split, extraction fix + `decay_rate=0.0` — [how we got here ↓](#benchmarks) |
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

Going deeper:
- **[docs/GUIDE.html](docs/GUIDE.html)** — a code-first walkthrough of the whole system: real snippets in the order that builds understanding, the reasoning behind each design decision, and hands-on exercises. Open it in any browser; it's a single self-contained file.
- **[docs/EVOLUTION.md](docs/EVOLUTION.md)** — where the architecture is headed: multi-tenancy, store segregation, scoring changes, and richer graph relationships.

## Quickstart

```bash
pip install mimir-engine
mimir serve --port 8080
```

That's it — `GET /health` works with zero config. No `mimir.yaml`, no Redis, no API key required. Grab `mimir.yaml.example` from the repo when you want to point at a local Ollama model, a cloud LLM, or a Redis instance; every setting has a sane default until then.

Working on the engine itself (contributing, running the test suite)? Use a checkout instead:

```bash
git clone https://github.com/hasil7677/mimir.git
cd mimir/engine
pip install -e ".[dev]"
uvicorn app.main:app --port 8080   # same as `mimir serve`, but picks up local edits
```

## Use it from Claude Code (or any MCP client) right now

```bash
claude mcp add mimir --scope user -e MIMIR_USER_ID=you -- mimir mcp
```

No gateway to run — the embedded adapter imports the engine directly, so a process only exists while your agent session is open. Three tools show up: `mimir_recall`, `mimir_remember`, `mimir_flush`. Point your `CLAUDE.md` at them and your agent starts building a memory of you, one conversation at a time.

Running more than one agent session at once? DuckDB and embedded Qdrant are single-writer, so start the gateway (`mimir serve`) and point every client at `mimir mcp --gateway` instead — same tools, same files, no lock contention.

Also documented: OpenCode (native MCP), Pi (via the community `pi-mcp-adapter`), and a plain HTTP contract for anything else — see **[docs/CLIENTS.md](docs/CLIENTS.md)**.

## Status

This is early — built fast, tested hard, not yet battle-tested by anyone but me. Here's the honest split:

**Solid and tested (113 tests, 109 pass with zero services/extras installed, 4 skip: 3 need a live Redis, 1 needs the optional `fastembed` extra, real HTTP layer, real filesystem, real dedup logic):**
hybrid recall with a 4-signal scoring formula · semantic caching with measured cache hits · L1.5 fact consolidation (exact-dup detection needs zero LLM calls; an LLM present gets you store/skip/supersede/contradiction-flag decisions, with hallucinated target IDs rejected) · GDPR-style erasure and export across every store · a self-healing recovery path for orphaned sessions (found live, fixed same day — see the commit log if you want to watch that happen) · MCP support verified end-to-end inside real Claude Code sessions.

**Known gaps, not hidden:**
- No real graph database yet. [KuZu](https://kuzudb.com) has no Python 3.11+ wheels as of this writing, so entity relationships live as vault `[[wikilinks]]` with hop-distance scoring instead of a Cypher-traversable graph. The scoring interface is already hop-based, so KuZu slots in without a rewrite once it's installable.
- Accuracy is well short of the top of the benchmark: 62.1% against 81.8% to 86.6% for the three published entries on the same split. An oracle test (perfect memory, nothing dropped) scores ~75% on this split, so most of that gap is not something better retrieval can close. See [Benchmarks](#benchmarks) for the numbers and the caveats on comparing them.
- LangChain / OpenAI Agents adapters aren't built. The HTTP contract they'd need already exists.

If you're looking for something production-hardened with a support contract, this isn't it yet. If you want to see what a memory system looks like when the databases are treated as caches and the filesystem is treated as the truth, open the vault.

## Benchmarks

**[Agent Memory Benchmark](https://agentmemorybenchmark.ai)**: public, reproducible. Full `personamem/32k` split: 195 real sessions, 589 questions, multiple-choice (exact-letter-match scored, no LLM judge involved for this task type). Mimir's own extraction/synthesis always runs on `gemini-2.5-flash`; the table below varies only the model the *benchmark harness* uses to answer the MCQ questions.

> ### Correction (2026-08-14)
>
> An earlier version of this README reported **59.6%** as the headline. That number was wrong, and it is worth being precise about how, because the data was real and the code was real — the method was not.
>
> It came from a local resume script that, by default, re-answers only the questions the *previous* run got wrong and keeps that run's correct answers untested. That is the right behavior for resuming an interrupted run. Pointed at a *different* model, it silently stops being a resume optimization and becomes a best-of-two-models score.
>
> The arithmetic is exact. `gemini-2.5-flash-lite` answered all 589 and got **287** right. `gemini-2.5-flash` was then run on only the 302 it missed, and got **64** of those right. 287 + 64 = **351** — the published "59.6%". No single configuration ever scored it.
>
> Re-run cleanly, every question answered once by one model, the real number was **47.7%**. Everything below is corrected. The engine improvements are unaffected — those were always measured flash-lite against flash-lite.

> ### Update (2026-08-18) — full split re-run with the extraction fix
>
> The table right below captures the answer-model comparison as it stood at the correction above — it's still the right read on *answer model choice*, but the retrieval store behind it predates the real fix (see [What was actually wrong ↓](#what-was-actually-wrong-extraction-was-landing-one-fact-per-session)). With that bug fixed and `decay_rate: 0.0` applied (the config found to help in the [retrieval sweep ↓](#retrieval-parameters-swept-against-a-store-that-finally-had-facts-in-it)), a clean full-589-question run — same questions, same answer model (`gemini-2.5-flash`) — scores **62.1% (366/589)**, up from **47.7% (281/589)** paired on the identical 589 questions: **+14.4 points, +85 questions flipped correct**. This is now the current headline number, and it lands inside the 58.0%–64.3% range measured on the 143-question subset further down, which is exactly the sanity check this run was for.

| Retrieval | Answer model | Accuracy |
|---|---|---|
| Mimir (real) | `gemini-2.5-flash-lite` (AMB harness's silent default) | 48.7% (287/589) |
| Mimir (real) | **`gemini-2.5-flash`** | **47.7% (281/589)** |
| Mimir (real) | `gemini-2.5-pro` | 46.0% (271/589)¹ |
| Oracle — gold *documents* only, extraction + retrieval still run | `gemini-2.5-flash-lite` | 50.8% (299/589) |
| Oracle — gold *text* injected directly, no extraction, no retrieval | `gemini-2.5-flash` | 70.7% (106/150)² |

¹ produced by the same failed-only method described above, so it is an upper bound; a clean re-run has not been done.
² 150-question stratified sample, not the full split. Projected onto the full question-type mix: **~75%**.

Superseded by the update above: the current headline is **62.1%** (post-extraction-fix, `decay_rate=0.0`). The table's row is the pre-fix retrieval store, kept because it's what the answer-model comparison above was actually measuring.

### The answer model does not matter much — that was the error

The harness picks its own model to answer the benchmark's multiple-choice questions, independent of whatever the memory provider uses internally, and it defaults to `gemini-2.5-flash-lite` without surfacing that anywhere in the provider API. That part is true and still worth knowing.

What is *not* true is the claim this section used to make: that repointing it at `gemini-2.5-flash` moved accuracy from 48.7% to 59.6%. Measured properly, on identical retrieval contexts, `flash` scores **47.7%** against flash-lite's **48.7%** — a one-point regression, i.e. a tie. The two models agree on only 64.3% of their answers, and trade wins almost evenly (flash wins 64, loses 70), which is exactly why taking the union of both looked like a large gain. Swapping the answer model is not a lever.

### Where Mimir actually stands

The three published entries on this split, from the benchmark's own `results-manifest.json`:

| System | Accuracy | Avg context tokens per query |
|---|---|---|
| hindsight | 86.6% | 15,812 |
| hybrid-search | 84.4% | 24,169 |
| cognee | 81.8% | 11,848 |
| **Mimir** | **62.1%** | **~489¹** |

¹ estimated: the full re-run measured 2,406 avg context *characters* per query directly; converted to tokens using the ~4.92 chars/token ratio the harness itself reported for the pre-fix run (875.68 chars → 178 tokens) rather than a generic chars/4 guess. Not independently re-measured in tokens.

No rank is being claimed from this. Those three ran in a `single-query` response mode that isn't registered in the installed harness (which offers `rag`, `agentic-rag`, and `agent`), and their answer-model configuration hasn't been verified. Treat it as orientation, not a leaderboard position: Mimir is roughly 20 to 25 points behind the top, and that gap is real regardless of how the caveats resolve.

The context column is still the most interesting thing in that table. Mimir ships an estimated ~489 tokens per query where the leaders ship 11,800 to 24,200 — roughly 24x to 49x less material for the answer model to reason over, even after the extraction fix more than tripled how much gets retrieved.

### The ceiling, and where the remaining gap actually is

There are two different oracle tests here and they answer different questions.

The AMB harness's own `--oracle` flag ingests only the gold-relevant documents — but Mimir's extraction and retrieval still run on top of them. It only asks "is picking the right *documents* the problem?" It scored 50.8% against 48.7%, so: no, document selection was never the bottleneck.

The second test bypasses Mimir entirely and hands the answer model the gold session text itself — no extraction, no retrieval, nothing dropped, ~36k tokens instead of Mimir's ~178. That is the ceiling of *any* memory system on this split. Paired against real retrieval on the same 150 questions:

| | Accuracy |
|---|---|
| Real retrieval | 50.0% |
| Perfect memory (all gold text, nothing compressed) | **70.7%** |

So there are about **21 points** available from better memory, and roughly **29 points that no memory system can reach** — lost in the answer step, not the retrieval step. Projected onto the full question mix, perfect memory scores ~75%.

The split by question type is lopsided, and it is the useful part:

| Question type | Real | Perfect memory | Share of split |
|---|---|---|---|
| `recall_user_shared_facts` | 30.4% | **82.6%** | 22% |
| `recalling_the_reasons_behind_previous_updates` | 72.7% | 100.0% | 17% |
| `recalling_facts_mentioned_by_the_user` | 29.4% | 47.1% | 3% |
| `generalizing_to_new_scenarios` | 63.6% | 81.8% | 10% |
| `track_full_preference_evolution` | 77.3% | 90.9% | 24% |
| `provide_preference_aligned_recommendations` | 68.2% | 63.6% | 9% |
| `suggest_new_ideas` | 4.5% | **22.7%** | 16% |

(All seven types, not a selection. `provide_preference_aligned_recommendations` going *down* with perfect context is within noise at n=22, but it is reported as measured.)

`recall_user_shared_facts` is where the work pays: a 52-point gap on nearly a quarter of the benchmark.

`suggest_new_ideas` is the opposite, and it caps everything. It scores **22.7% even with perfect, complete context** — below the 25% you would get by guessing on a four-option question. Below chance in both conditions means something is systematically selecting the wrong option rather than failing to find information, so it is not a memory problem and no amount of retrieval work will move it. It is 93 questions, 16% of the split. Diagnosing it is now the open question; the leading hypothesis is that the correct answer is the one *novel* relative to history, so grounding harder in retrieved memory actively steers toward the distractor.

Caveat: the perfect-memory run is a 150-question stratified sample, so per-type figures carry wide error bars (n=22 each). The aggregate 70.7% is the solid number; treat the ~75% projection as ±5.

### What was actually wrong: extraction was landing one fact per session

Chasing that gap turned up the real constraint, and it was not retrieval. The store behind the 47.7% run held **280 facts across 191 ingested sessions** — 172 of those sessions had produced exactly one fact. Retrieval was returning `candidates=5` because five facts were all that existed for that user, and the assembled context averaged 1,493 characters against a 6,000-character budget. The recall window was never binding.

The cause was a truncation bug. `_chat_vertex` sent `maxOutputTokens` with no `thinkingConfig`, and Gemini 2.5 models think by default — on the Vertex API those thinking tokens are charged against the same budget the answer has to fit in. Measured on a real 60k-character session:

| | finishReason | thinking tokens | usable output | facts extracted |
|---|---|---|---|---|
| Before | `MAX_TOKENS` | 1,918 | 78 tokens | **1** |
| Thinking off | `MAX_TOKENS` | — | 2,000 tokens | 32 |
| Thinking off, 8k budget | `STOP` | — | 2,695 tokens | **44** |

1,918 of a 2,000-token budget went to thinking. The 78 tokens left held one complete JSON fact object and the start of a second, and the resilient parser correctly kept the one object it could parse. This returned HTTP 200 every time, with `finishReason: MAX_TOKENS` in a response field the engine never read.

Fixed: `llm.thinking_budget` (default `0`) emitted as `thinkingConfig`; truncation now logs a warning and an empty candidate raises `LlmUnavailable` instead of `KeyError`; extraction's budget raised to 8,000; `max_memories_per_session` 20 → 50, a cap that had never bound on anything because sessions were only landing one fact.

Re-ingesting 30 sessions across 6 personas and re-answering their 143 questions — the same questions, paired:

| | Accuracy | Facts per session |
|---|---|---|
| No memory at all | 45.5% | — |
| Before the fix | 46.2% | 1.5 |
| After the fix | **58.0%** | **25.1** |

**Before the fix, the entire memory system was worth +0.7 points over having no memory at all.** After it, +12.6. Biggest movers: `recall_user_shared_facts` 34.4% → 65.6%, and `suggest_new_ideas` 15% → 30% — the first time that category has cleared random guessing.

Two caveats. This is 143 questions from 6 personas, not the full split, so the headline above stays at the last full-split measurement until a complete re-run is done. And 58.0% is still short of the 70.7% ceiling, so roughly half the available memory headroom is still on the table.

### Retrieval parameters, swept against a store that finally had facts in it

With extraction fixed, retrieval tuning became worth measuring — every query was now hitting the `max_results: 10` cap, which had never once happened when users averaged five facts. Same 143 questions, same store, paired, McNemar on the discordant pairs:

| Config | Accuracy | vs baseline | p |
|---|---|---|---|
| `max_results 10`, `threshold 0.30`, `decay 0.05` (baseline) | 58.0% | — | — |
| **`decay_rate 0.0`** | **64.3%** | **+6.3** | **0.012** |
| `decay_rate 0`, `max_results 20` | 55.9% | −2.1 | 0.648 |
| `decay_rate 0`, `max_results 20`, `threshold 0.15`, `12k chars` | 59.4% | +1.4 | 0.804 |

Two findings, both significant, pointing opposite ways.

**Recency decay was costing 6.3 points.** `exp(-0.05 * age_days)` drives a fact stated 60 days ago to 0.05, and in a multi-session benchmark every session is equally relevant — the decay was penalising facts purely for being old. Ten questions flipped right, one flipped wrong (p=0.012). `decay_rate` is config-driven, so this needs no code change: set it to `0.0` for evaluation runs. The default stays `0.05`, which is the right behaviour for an assistant where recent context genuinely matters more.

**Widening the recall window cost 8.4 points.** Going from `max_results` 10 → 20 against the `decay 0` config sent 16 questions wrong and only 4 right (p=0.012). More retrieved facts made things actively worse — the same pattern as the memory-off ablation, where sparse-but-wrong memory scored below no memory at all. This model is more easily misled by a plausible irrelevant fact than helped by a marginal relevant one.

That kills the parameter recommendations this project had been carrying (`max_results` 15–20, `threshold` 0.15–0.20, `max_context_chars` 12000). All three are wrong here, and the honest reading is that recall should get *tighter*, not wider, once the facts are good.

Best measured configuration is the extraction fix plus `decay_rate: 0.0`, everything else at defaults: **46.2% → 64.3% on this subset**, against a 70.7% perfect-memory ceiling. Confirmed on the full 589-question split at that same configuration: **62.1% (366/589)**, up from the pre-fix **47.7% (281/589)** paired on the identical questions — inside the subset's 58.0–64.3% range, as expected.

Sample-size caveat, stated plainly: n=143 means the 95% interval on any single accuracy number here is roughly ±8 points. The paired McNemar tests above are what the claims rest on, not the point estimates — two configurations that differ only trivially in what they retrieve (`top20` and `wide`) disagree on just 9 of 143 questions, which is the noise floor made visible.

## Roadmap: closing the gap

62.1% is up from a 43.3% first pass, not a finished number. Measured like-for-like — flash-lite throughout, which is how the engine changes were always evaluated — that progression is 43.3% → 48.7%. Here's the actual order things got worked, not a wishlist:

- **Shipped**: entity notes were getting created for names an LLM synthesis step returned even when that name never literally showed up in the note it was supposedly linked from, orphaned single-node clutter in the graph with no edge to anything. Fixed: only entities that actually got wikilinked get a note now. Also widened the regex entity extractor's filler-word list to cut false-positive nodes (conversational filler like "Sure", "Actually" getting mistaken for named entities).
- **Shipped**: graph traversal was structurally dead. Entity notes had no outgoing links, so hop-1/2 scoring and the LINKED NOTES prompt section never had anything to walk. Entity notes now backlink to every scene that mentions them, and two knock-on bugs (a wasted first traversal hop, multi-word entities keyed by slug instead of display text) got fixed alongside it. Also fixed a scoring bug where an exact BM25 keyword hit the vector leg missed was forced to semantic=0.0 instead of keeping its RRF-derived relevance. Together: 43.3% to 49.4%.
- **Shipped**: ran an oracle-mode diagnostic (ingest only gold documents, bypassing retrieval noise entirely) to find out how much of the remaining gap is retrieval versus extraction/generation quality. Result at the time: 50.8% vs. 49.4%, only a 1.4-point move, so document selection looked close to its ceiling.
- **Shipped, no measurable win**: swapped the regex capitalized-run entity extractor for real NER (spaCy `en_core_web_sm`, optional `mimir-engine[ner]` extra, degrades to the old regex heuristic if it isn't installed). Also drops numeric/temporal entity labels (dates, quantities, money) and lowercase false positives the small model occasionally tags, neither of which are wikilink-worthy. Re-ran the full benchmark: 48.7% vs. the pre-swap 49.4%, flat, within noise, arguably slightly down. Kept the change anyway (strictly better entity quality, cleaner vault graph) but it confirmed entity/graph quality wasn't the lever.
- **Retracted**: this list previously claimed that repointing the harness's answer model from `flash-lite` to `flash` was "the actual lever," worth 48.7% → 59.6%. It isn't. That gain was a measurement artifact, and a clean re-run puts `flash` at 47.7% — a tie with the default. Full explanation under [Benchmarks](#benchmarks). The `pro` result quoted alongside it came from the same flawed method and has not been cleanly re-run either.
- **Shipped**: measured the real ceiling by handing the answer model the gold session text directly, bypassing extraction and retrieval entirely. Perfect memory scores 70.7% where real retrieval scores 50.0% on the same questions. That bounds every remaining memory improvement at about 21 points and puts the achievable target near 75%, not 80%+.
- **Shipped, the actual lever**: extraction was silently truncated to one fact per session by Gemini thinking tokens eating the output budget. Fixed; 1.5 → 25.1 facts per session, and **+11.9 points** on a paired 143-question re-ingest. Full write-up under [Benchmarks](#what-was-actually-wrong-extraction-was-landing-one-fact-per-session). This is what every earlier retrieval-shaped hypothesis was actually running into.
- **Shipped**: re-ran the full 589-question split with the fix and `decay_rate: 0.0` in place, replacing the 47.7% headline: **62.1% (366/589)**, up from **47.7% (281/589)** paired on the identical questions (+14.4 points). Lands inside the 58.0%–64.3% range measured on the 143-question subset — the sanity check this run was for.
- **Shipped**: swept retrieval parameters now that the store actually holds facts. `decay_rate: 0.0` is worth **+6.3 points** (p=0.012) on a multi-session benchmark where every session is equally relevant; no code change needed, it was already config-driven. Widening `max_results` 10 → 20 *costs* **8.4 points** (p=0.012), which retires the wider-window/lower-threshold recommendations this project had been carrying.
- **Next**: the remaining ~6 points between 64.3% and the 70.7% perfect-memory ceiling. Given that loosening retrieval hurt, the promising direction is tightening it — better ranking rather than more results.
- **Next**: `min_priority` (currently 50) and chunked extraction are the two untested extraction-side ideas. Both change what gets stored, so both need a fresh ingest to evaluate.
- **Next**: `track_full_preference_evolution` was the one category that did *not* improve with 17x more facts (50.0% → 47.5%). Worth understanding before assuming more extraction is uniformly good.
- **Dropped**: widening the recall window. The assembled context averaged 1,493 characters against a 6,000 cap, so the window was never the constraint — this hypothesis was measuring a symptom of the extraction bug.
- **Next**: verify what answer model the other published entries actually used, and whether `single-query` mode can be run locally at all. Until both are known, the comparison in the table above is orientation rather than a ranking.
- **Then**: a real Cypher-traversable graph via [KuZu](https://kuzudb.com) once it ships Python 3.11+ wheels, replacing the current wikilink-hop-distance approximation. Still a legitimate architecture improvement, just not expected to be the thing that moves this number.
- **Then**: run the same harness against LoCoMo and LongMemEval (already wired into the eval setup) to see whether these findings hold on other question formats, or are specific to personamem's paraphrase-heavy MCQ style.

## Development

```bash
pip install -e ".[dev]"
pytest              # 113 tests; Redis-backed and fastembed-backed ones auto-skip without a server/extra installed
```

CI runs the suite on Python 3.11–3.13, on both Ubuntu and Windows — the Windows leg is not decorative, it's what caught a real timezone bug during development.

## License

MIT — see [LICENSE](LICENSE). Built on the idea that the core memory engine should always be free and open; anything resembling a hosted/managed offering is a separate conversation for another day.
