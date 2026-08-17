"""`mimir serve` / `mimir mcp` argument wiring — the console-script entry
point that lets a plain `pip install` replace `uvicorn app.main:app` and
`python adapters/mcp_embedded.py`, both of which only work from a git
checkout."""

from unittest.mock import MagicMock

from app.cli import main


def test_serve_runs_uvicorn_with_parsed_host_and_port(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_run)

    main(["serve", "--host", "0.0.0.0", "--port", "9000"])

    mock_run.assert_called_once_with("app.main:app", host="0.0.0.0", port=9000)


def test_serve_defaults_to_localhost_8080(monkeypatch):
    mock_run = MagicMock()
    monkeypatch.setattr("uvicorn.run", mock_run)

    main(["serve"])

    mock_run.assert_called_once_with("app.main:app", host="127.0.0.1", port=8080)


def test_mcp_defaults_to_the_embedded_in_process_adapter(monkeypatch):
    from adapters.mcp_embedded import mcp as embedded_mcp

    mock_run = MagicMock()
    monkeypatch.setattr(embedded_mcp, "run", mock_run)

    main(["mcp"])

    mock_run.assert_called_once()


def test_mcp_gateway_flag_uses_the_http_adapter_instead(monkeypatch):
    from adapters.mcp_server import mcp as gateway_mcp

    mock_run = MagicMock()
    monkeypatch.setattr(gateway_mcp, "run", mock_run)

    main(["mcp", "--gateway"])

    mock_run.assert_called_once()
