# Mimir: Architectural Evolution and Next Steps

This document captures the architectural decisions and future evolution path for Mimir's memory system, specifically addressing multi-tenancy, database segregation, memory scoring, and graph relationships.

## 1. Single-Device Local-First vs. Multi-Tenancy

Currently, focusing on a single-device, local-first approach is the safest and most effective strategy. 

While Mimir's database schemas are perfectly prepared for multi-tenancy (every record requires a `tenant_id` and `user_id`, defaulting to `local`), transitioning to true multi-device or multi-tenant sync introduces severe complexities:
*   **Conflict Resolution:** If a user manually edits a markdown file in their Obsidian vault offline while the agent simultaneously updates a fact in the cloud, you run into classic distributed system conflicts.
*   **Sync Complexity:** Avoiding CRDTs (Conflict-Free Replicated Data Types) in markdown is preferable. A background batch-sync job is far more stable than real-time Pub/Sub WebRTC connections, which demand both edge and cloud nodes to be online concurrently.

By sticking to single-device local-first for now, you avoid the "retrofit isolation nightmare" while bypassing synchronization headaches.

## 2. Segregating the "Multi-DB" Layers

Mimir relies on a stack of four different data stores: DuckDB, Qdrant, Redis, and a Markdown Vault. Managing this "zillion layers" of memory requires a strict hierarchy of truth to prevent state drift.

**The Hierarchy of Truth:**
1.  **The Vault (Markdown):** The ultimate, human-readable source of truth.
2.  **DuckDB:** The immutable audit log and raw transcript.
3.  **Qdrant & Redis:** Strictly disposable caches and indexes.

**Handling Segregation:**
If the layers ever drift out of sync (e.g., Qdrant vectors do not match DuckDB facts), the solution is not to write complex migration scripts. Instead, the disposable indexes (Qdrant/Redis) should be deleted and lazily rebuilt from the ground truth (DuckDB). Treating the vector and cache layers as purely ephemeral ensures stability.

## 3. The 4 Memory Retrieval Biases (Scoring)

Mimir ranks retrieved memories using four distinct biases, merged using Reciprocal Rank Fusion (RRF):
1.  **Recency Bias:** Newer memories are scored higher using an exponential decay function.
2.  **Semantic Bias:** Vector cosine similarity (via Qdrant) ensures contextual relevance.
3.  **Frequency Bias:** Memories that are accessed or recalled more often receive a logarithmic score boost.
4.  **Graph Proximity Bias:** If a retrieved memory mentions an entity (e.g., `[[Project X]]`), Mimir traverses the vault's links. Notes that are 1 or 2 hops away receive a proximity score boost, enriching the context.

## 4. Evolving the Memory System: What Comes Next?

To evolve from a static "store and retrieve" database into a true cognitive architecture, the system needs the following additions:

*   **Active Forgetting (Garbage Collection):** The system must eventually prune or compress irrelevant noise. Facts with persistently low frequency scores over long periods should be deleted or summarized into high-level semantic archetypes.
*   **Proactive Contradiction Resolution:** Mimir currently flags contradictions (e.g., "User likes X" vs. "User hates X"). The next step is a proactive agent workflow that actively asks the user for clarification to resolve the flagged discrepancy.
*   **Stateful Working Memory:** Agents require a temporary "scratchpad" for the current task. This prevents the long-term memory vault from being polluted with ephemeral context (e.g., "I am currently refactoring line 42").
*   **Qdrant Skills Integration:** Injecting Qdrant's engineering "Skills" repository directly into Mimir as `instruction` facts. This allows the agent to dynamically recall optimization strategies for its own vector database.

## 5. Entities, Relationships, and Moving Beyond KuzuDB

Currently, Mimir relies on standard OKF (Open Knowledge Format) wikilinks for relationships. Incorporating a heavy graph database like KuzuDB introduces unnecessary friction, especially lacking robust Python 3.11+ support. 

**The Alternative: In-Memory Graph via NetworkX**
Because Mimir's core identity is an Obsidian Vault, the dataset is lightweight (a folder of text files). A heavy graph DB is overkill.

*   **Knowledge Triples:** Instead of simple `[[Wikilinks]]`, the LLM should extract explicit knowledge triples (Subject -> Predicate -> Object) and save them in the YAML frontmatter of the markdown files (e.g., `User -> manages -> Project X`).
*   **In-Memory Traversal:** Upon booting or querying, the system can parse the YAML files and build a graph in RAM instantly using a lightweight library like **NetworkX**. 
*   **Benefits:** This approach keeps the system blazingly fast, entirely local, drops the dependency on a 4th database, and leverages the fact that text-based graph traversal in memory takes milliseconds.
