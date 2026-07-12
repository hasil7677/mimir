# Study resources

Everything here is a **plain file** — no claude.ai account, no internet connection,
no dependency needed to open any of it. Download or clone this repo on any machine
and it all works.

## What's in here

| File | What it is | How to open it |
|---|---|---|
| `study-guide.html` | **Start here.** A code-first walkthrough of the codebase — real snippets pulled from the repo, in the order that builds understanding, with the reasoning behind each design decision and hands-on exercises. | Double-click it, or drag it into any browser. |
| `architecture.html` | The structural reference — diagrams of the write path and read path, the four-store architecture, the five design invariants, and a module-by-module summary. More "what connects to what," less "read this code line by line." | Same — double-click, opens in any browser. |
| `architecture-notes.md` | The same architecture reference as `architecture.html`, in plain Markdown (no diagrams, all the same content). Useful if you're somewhere that only renders Markdown, like GitHub itself or a code editor's preview pane. | Any Markdown viewer, or just read it as text. |
| `client-setup.md` | How to hook Mimir up to Claude Code, OpenCode, Pi, or anything else that speaks MCP or plain HTTP. | Same as above. |

## Suggested order

1. Read `study-guide.html` stage by stage, with the actual repo (`../app/`) open
   in an editor alongside it — every code snippet in the guide is copied verbatim
   from a real file, so you can jump straight to the source.
2. Once the code makes sense, skim `architecture.html` for the bird's-eye view —
   it'll click faster now that you've seen the actual implementation.
3. `client-setup.md` only matters once you want to actually run this against
   Claude Code or another agent.

## No internet needed

Both `.html` files are fully self-contained — all styling is inline, nothing
loads from a CDN. They're built to be opened as local files, not served from a
website. If you can open a file in a browser, you can read these, office
network restrictions or not.
