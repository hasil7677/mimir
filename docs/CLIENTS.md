# Hooking Mimir up to your agents

Mimir speaks MCP, so any MCP-capable agent can use the same brain. Two adapter
modes, same three tools (`mimir_recall`, `mimir_remember`, `mimir_flush`),
same files on disk:

- **Embedded** (`mimir mcp`, or `adapters/mcp_embedded.py` from a checkout) —
  no server, the agent spawns the process per session. Use this when ONE agent
  runs at a time.
- **Gateway** (`mimir mcp --gateway`, run against `mimir serve --port 8080`) —
  use this when you want MULTIPLE agents open simultaneously. DuckDB and
  embedded Qdrant are single-writer, so two embedded sessions against the same
  `~/.mimir` will hit a file lock; the gateway serializes access for everyone.

Rule of thumb: start embedded everywhere. The day you actually run Claude Code
and OpenCode at the same moment and see a lock error, start the gateway and
flip your configs to the gateway adapter. Nothing about your data changes.

Every command below shows `mimir mcp` (from `pip install mimir-engine`). If
you're working from a git checkout instead, substitute
`python <abs-path-to-mimir-engine>/adapters/mcp_embedded.py` — same tool, same files.

---

## Claude Code

```
claude mcp add mimir --scope user -e MIMIR_USER_ID=you -- mimir mcp
```

Pair it with a `~/.claude/CLAUDE.md` section telling sessions when to
recall/remember/flush (see README). Verify with `claude mcp list`.

## OpenCode

Add to `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mimir": {
      "type": "local",
      "command": ["mimir", "mcp"],
      "enabled": true,
      "environment": { "MIMIR_USER_ID": "you" }
    }
  }
}
```

OpenCode has an AGENTS.md convention analogous to CLAUDE.md — put the same
memory-discipline instructions there so the model actually uses the tools.

## Pi

Pi deliberately does **not** support MCP natively (its author considers MCP
context-overhead too high). The community `pi-mcp-adapter` bridges it with a
single ~200-token proxy tool. Install it, then in `~/.pi/agent/mcp.json`:

```json
{
  "servers": {
    "mimir": {
      "command": ["mimir", "mcp"],
      "env": { "MIMIR_USER_ID": "you" }
    }
  },
  "directTools": ["mimir_recall", "mimir_remember", "mimir_flush"]
}
```

`directTools` matters: it surfaces the three mimir tools directly in Pi's tool
list instead of behind the proxy, which is what you want for tools the agent
should reach for habitually. Pi's adapter starts servers lazily and disconnects
after idle — a perfect match for the embedded adapter's process model.

## Hermes / anything else

Any framework that can either (a) act as an MCP client or (b) make HTTP calls
can use Mimir. For (b), run the gateway and call four endpoints:

- `POST /v1/recall {user_id, query, session_id?}` → `{context_string, ...}` — inject before the prompt
- `POST /v1/capture {user_id, session_id, messages:[{role, content}]}` — after each turn
- `POST /v1/session/end {user_id, session_id}` — on conversation end
- `GET /health` — liveness

That's the whole adapter contract; every adapter in this repo is a thin
translation of a framework's lifecycle hooks onto those calls, with no
business logic. A Hermes MemoryProvider maps as: `prefetch → /recall`,
`sync_turn → /capture`, `on_session_end → /session/end`.

## One brain, many agents — identity

All clients should pass the SAME `MIMIR_USER_ID` if they're all you. That's
what makes memory written from Claude Code recallable from OpenCode. Different
values = deliberately separate brains (e.g. `me` vs `me-work`).
