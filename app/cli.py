"""Console entry point (`pip install mimir-engine` -> the `mimir` command).

Before this existed, running Mimir required a git checkout: `uvicorn
app.main:app` and `python adapters/mcp_embedded.py` both only work from
inside the repo. This wraps the same two entry points so a plain `pip
install` is enough.
"""

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mimir")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the HTTP gateway (FastAPI + uvicorn)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    mcp_cmd = sub.add_parser("mcp", help="run as an MCP server for Claude Code or any MCP client")
    mcp_cmd.add_argument(
        "--gateway", action="store_true",
        help="talk to a running `mimir serve` over HTTP instead of embedding the engine in-process "
             "(needed for concurrent sessions; DuckDB/Qdrant are single-writer)",
    )

    args = parser.parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("app.main:app", host=args.host, port=args.port)
    elif args.command == "mcp":
        if args.gateway:
            from adapters.mcp_server import mcp
        else:
            from adapters.mcp_embedded import mcp
        mcp.run()


if __name__ == "__main__":
    main()
