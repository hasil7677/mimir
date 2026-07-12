# Study resources

Everything here is a **plain file** — no claude.ai account, no internet connection,
no dependency needed to open any of it. Download or clone this repo on any machine
and it all works.

## What's in here

| File | What it is | How to open it |
|---|---|---|
| `mimir-guide.html` | **Start here — the whole thing in one file.** Part 1 is a code-first walkthrough (real snippets pulled from the repo, in the order that builds understanding, with the reasoning behind each design decision and hands-on exercises). Part 2 is the structural/diagram reference (write path, read path, the four-store architecture, five invariants, module-by-module summary). One table of contents links to both parts. | Double-click it, or drag it into any browser. |
| `architecture-notes.md` | The Part 2 content in plain Markdown (no diagrams, same substance). Useful if you're somewhere that only renders Markdown, like GitHub itself or a code editor's preview pane. | Any Markdown viewer, or just read it as text. |
| `client-setup.md` | How to hook Mimir up to Claude Code, OpenCode, Pi, or anything else that speaks MCP or plain HTTP. | Same as above. |

## Suggested order

1. Open `mimir-guide.html` and read Part 1 stage by stage, with the actual repo
   (`../app/`) open in an editor alongside it — every code snippet is copied
   verbatim from a real file, so you can jump straight to the source.
2. Once the code makes sense, jump to Part 2 (the table of contents links straight
   there) for the bird's-eye view — it'll click faster now that you've seen the
   actual implementation.
3. `client-setup.md` only matters once you want to actually run this against
   Claude Code or another agent.

## No internet needed

`mimir-guide.html` is fully self-contained — all styling is inline, nothing loads
from a CDN. It's built to be opened as a local file, not served from a website.
If you can open a file in a browser, you can read it, office network restrictions
or not.
