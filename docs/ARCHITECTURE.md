# Mimir — Architecture & Code Walkthrough

Mimir is a **local-first memory system for AI agents**. An agent captures what
you say, Mimir distills it into facts and human-readable notes, and hands back
a ranked "memory context" block when the agent needs to know something about
you. Everything lives in `~/.mimir` as plain files. No cloud, no daemon
required, and every external dependency (LLM, embeddings, Redis) is an
*upgrade*, never a requirement.

## The one-paragraph mental model

Think of it as four containers with one brain: **DuckDB** holds the raw
transcript and extracted facts (ground truth, always works), the **vault**
holds human-readable Obsidian markdown (the part you can open and edit),
**Qdrant** holds vectors for semantic search (only if embeddings are
configured), and **Redis** holds hot conversation turns and a query cache
(only if running). Two flows connect them: **capture/flush** (write path)
and **recall** (read path).

```
                     WRITE PATH (capture → flush)

  agent turn ──► pipeline.capture ──► DuckDB l0_conversations  (ground truth)
                       │
                       └──► Redis hot list                     (best-effort)

  session end ──► pipeline.flush_session
                       │
     ┌─────────────────┼───────────────────────────┐
     ▼                 ▼                           ▼
  synthesis        extraction ──► consolidation   persona.maybe_synthesize
  (scene note)     (L1 facts)     (L1.5 dedup/         (L3, every N facts)
     │                 │           supersede/            │
     ▼                 ▼           contradict)           ▼
  vault/scenes/*.md   DuckDB l1_memories            vault/persona.md
  + entity stubs       + Qdrant vectors (best-effort)
  [[wikilinked]]       + semantic-cache invalidation


                      READ PATH (recall)

  query ──► embed (optional) ──► semantic cache? ──hit──► return
                       │ miss
                       ▼
        ┌── BM25 keyword (DuckDB fts) ──┐
        │                               ├─► RRF merge ─► 4-signal scoring ─► top-k
        └── vector search (Qdrant) ─────┘      semantic·0.45 + freq·0.20
                       │                       + recency·0.25 + graph·0.10
                       ▼
        graph signal & enrichment: vault [[wikilink]] hop-walk
                       │
                       ▼
        [MEMORY CONTEXT] string: USER PROFILE + RECENT + MEMORIES + LINKED NOTES
```

## Design invariants (the rules everything obeys)

1. **Degrade, never fail.** Every dependency has a documented fallback: no LLM
   → verbatim fact extraction + digest scenes/persona; no embeddings →
   keyword-only recall; no Redis → no hot turns/cache, capture still lands.
   Grep for `Unavailable` / `except Exception` around the boundaries.
2. **The vault is ground truth for human-readable memory; databases are
   indexes.** Reads come from disk, so a hand edit in Obsidian is visible on
   the next call, no sync step.
3. **Supersede, never delete** (except explicit GDPR erasure). Old facts get
   `is_active=false` + `superseded_by`; persona history is backed up;
   contradictions are flagged for review, never auto-resolved.
4. **Tenant + user on every record and every query**, even in single-user
   local mode (`tenant="local"`). Prevents the retrofit-isolation nightmare.
5. **Naive UTC at the DuckDB boundary.** DuckDB localizes tz-aware datetimes
   before dropping the tz; `to_utc_naive()` guards every write.

---

## Module-by-module

### `app/config.py` — configuration
Nested pydantic models mirroring `mimir.yaml` exactly: `server`, `storage`
(redis/qdrant/kuzu/duckdb/vault), `embedding`, `llm`, `pipeline`, `recall`
(incl. the 4 scoring weights), `extraction`, `capture`, `pii`, `compliance`.
`load_config()` reads `MIMIR_CONFIG` or `./mimir.yaml`, substitutes
`${ENV_VARS}`, and **boots on all-defaults if no file exists** — config is an
override, never a requirement. Module-level `settings` singleton.

### `app/core/okf.py` — the file format (pure logic, no I/O)
Open Knowledge Format: YAML frontmatter + markdown + `[[wikilinks]]`,
Obsidian-compatible. `render_note`/`parse_note` round-trip notes — and a file
with missing/broken frontmatter parses as `({}, body)` instead of erroring,
because hand-written notes are first-class input. `extract_wikilinks` (handles
`[[Target|alias]]`), `wikify` (wraps first occurrence of each entity, longest
name first, never double-wraps), `slugify` (deterministic filename stems).

### `app/core/vault.py` — the brain on disk
Layout: `{vault}/{tenant}/{user}/scenes/*.md`, `entities/*.md`, `persona.md`,
`persona_history/`. Key functions:
- `write_scene` — wikifies entities into the body, writes the scene note, and
  creates a **stub note per entity** so Obsidian's graph never has dangling links.
- `ensure_entity_stub` — never overwrites an existing note (user edits win).
- `upsert_persona` — timestamped backup of the previous version before writing.
- `read_note` — resolves a wikilink target (title or slug) fresh from disk.
- `expand_links` — the enrichment walk: collect `[[links]]` from given bodies,
  read those notes, follow *their* links, up to N hops, cycle-safe. This is
  both recall's graph signal and its "linked notes" context.

### `app/core/llm.py` + `app/core/embeddings.py` — model access
Both hit any OpenAI-compatible endpoint (`/chat/completions`, `/embeddings`);
no API key just means no auth header, so keyless local Ollama works. Failures
raise `LlmUnavailable` / `EmbeddingsUnavailable` — typed signals that mean
"take your offline path", and every caller has one.

### `app/core/synthesis.py` — turns → scene note
`synthesize_scene(turns)` → `(title, body, entities, mode)`. LLM path asks for
JSON `{title, summary, entities}`; any failure (unreachable, garbage output)
falls back to a deterministic transcript digest marked `synthesis: digest`.
Also home to `extract_entities_naive`: spaCy NER (`en_core_web_sm`, optional
`mimir-engine[ner]` extra) when installed, numeric/temporal labels (dates,
quantities, money) dropped since they aren't wikilink-worthy. Falls back to
a capitalized-run regex heuristic with a sentence-start guard when spaCy
isn't installed (a lone capitalized word opening a sentence only counts if
seen capitalized mid-sentence elsewhere, which is what keeps `Huge!` and
`Smooth.` from becoming entities in your vault).

### `app/core/extraction.py` — turns → L1 atomic facts
`extract_facts(turns)` → `(facts, mode)`. LLM path: typed facts
(`persona|episodic|instruction`) with 0-100 priority, filtered below
`extraction.min_priority`. Offline path: each substantive user turn (>15
chars) becomes a verbatim episodic fact — keyword search over verbatim text is
still real recall, and facts can be re-extracted when an LLM exists.

### `app/core/consolidation.py` — L1.5, the dedup gate
`consolidate(tenant, user, new_facts)` runs **before** facts are inserted.
Layer 1 (free, always): exact normalized-text duplicates — against the store
and within the batch — are dropped with zero LLM spend. Layer 2 (LLM, one
batched call over all new facts + a unified candidate pool from keyword
search): per-fact `store`/`skip`/`update` decisions plus `contradicts_id`.
`update` → the survivor carries `_supersedes`; contradictions are recorded
but the fact still stores (flag-and-log). **LLM-returned ids are validated
against the candidate pool it was shown** — a hallucinated or injected id
can't supersede an arbitrary memory.

### `app/core/persona.py` — L3, who the user is
`maybe_synthesize`: count-triggered (every `l3_every_n_memories` new facts,
tracked in persona.md's own frontmatter — the vault carries its own
bookkeeping). LLM rewrites the persona doc from top-priority facts; offline
digest groups persona/instruction facts into sections. `profile_lines` feeds
recall's USER PROFILE section — read fresh from disk every call.

### `app/core/scoring.py` — the ranking math (pure functions)
`rrf_merge` (Reciprocal Rank Fusion, k=60 — scale-free, so BM25 scores and
cosines never need normalizing against each other); `recency_score`
(`e^(-decay·days)`, ~0.5 at 14 days); `frequency_score` (log-normalized
access counts); `graph_score` (hops → 1.0/0.5/0.25); `final_score` (weighted
sum, weights from config).

### `app/core/recall.py` — the read pipeline
Order matters: (0) embed query once, check semantic cache (exact-hash needs no
embeddings) — hit returns immediately; (1) hot turns from Redis (degrades to
none); (2) BM25 top-20 + vector top-20; (3) RRF merge; (4) score every
candidate with all four signals — when the vector leg didn't run, normalized
RRF stands in for the semantic signal so keyword rank still drives ordering;
(5) threshold + top-k, then `bump_access` (being recalled reinforces future
ranking); (6) vault enrichment — entities mentioned in returned memories are
hop-walked for linked notes; assemble `[MEMORY CONTEXT]`, truncating
weakest-first to `max_context_chars`; cache the result.

### `app/core/pipeline.py` — the write path, shared
`capture` (L0 insert + best-effort Redis push + audit) and `flush_session`
(scene → extraction → consolidation → insert/supersede/contradict → vector
index → cache invalidation → persona). Extracted so the HTTP routes and the
embedded MCP adapter are thin wrappers over **one** implementation.

### `app/core/semantic_cache.py` — the cost saver
Redis hash per (tenant, user): field = normalized-query hash, value =
`{vector, response}`. Exact-hash hits are free; cosine hits at
`cache_threshold` (0.92). Invalidation is blunt — any new memory wipes the
user's whole cache (stale answers are worse than re-running the pipeline).
Redis down = every lookup is a miss, never an error.

### `app/db/duckdb_client.py` — cold archive / ground truth
Single lazy connection; schema: `l0_conversations`, `l1_memories`,
`l1_contradictions`, `audit_log`. `to_utc_naive` on every timestamp write.
`erase_user` deletes content rows but **keeps the audit log** — it holds only
ids/actions, and the erasure receipt must survive the erasure. (Footgun found
live: `execute()` returns the connection, so DELETE counts must be fetched
immediately or you read the wrong statement's count.)

### `app/db/l1_store.py` — the fact store
Insert/search/supersede/contradiction/access-count over `l1_memories`. Search
is real BM25 via DuckDB's `fts` extension when it loads, else SQL
term-frequency fallback. The FTS index is static, so writes set a dirty flag
and the next search rebuilds lazily. Every query filters
`tenant_id AND user_id AND is_active`.

### `app/db/vector_store.py` — Qdrant embedded
`QdrantClient(path=...)` — a directory, no server. Collection created lazily
on first write (dimension comes from whatever embedding model is configured).
Payload duplicates tenant/user so filtering never needs a join back to DuckDB.

### `app/db/redis_client.py` — hot memory
`hot:{tenant}:{user}:{session}` lists with TTL refresh per push. Note: the
key includes tenant_id even though the original spec's table omitted it —
the spec's own isolation requirement (every store, both ids) wins over its
example.

### `app/api/` — the HTTP gateway
`deps.get_tenant_id`: tenant identity comes **only** from the Bearer key
(constant-time compare), never the request body; keyless mode pins to the
`local` tenant. `routes.py`: `/capture`, `/session/end`, `/recall`,
`/vault/notes`, `/vault/note/{target}`, `DELETE /user/{id}` (erasure across
all four stores + audit receipt), `GET /export/{id}` (data portability).
`main.py` wires lifespan schema-init and mounts under `/v1`.

### `adapters/` — how agents plug in
`mcp_embedded.py`: imports the engine in-process — no gateway, process exists
only while an MCP session is open (single-writer: one session at a time).
`mcp_server.py`: same three tools over HTTP to a running gateway (for
concurrent sessions/machines). Both expose `mimir_recall` / `mimir_remember`
/ `mimir_flush`. See `docs/CLIENTS.md` for per-agent setup.

---

## Where things live on disk

```
~/.mimir/
  memories.db        DuckDB: transcripts, facts, contradictions, audit log
  qdrant/            embedded vector index (only if embeddings configured)
  vault/
    local/<user>/
      scenes/*.md            one note per flushed session
      entities/*.md          one stub (or user-enriched note) per entity
      persona.md             current persona (+ its own trigger bookkeeping)
      persona_history/*.md   every previous persona version
```

## Known gaps

KuZu graph DB (no Python 3.14 wheels; vault wikilinks carry the graph signal
behind the same hop-based interface), L2 scene *aggregation* as a distinct
tier (scenes are per-session today), spaCy NER, LangChain/OpenAI-Agents
adapters, PersonaMem benchmarks, PII scrubbing, multi-tenant control plane.
