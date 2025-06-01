import argparse
import logging
import subprocess
import datetime
from . import memory


def main():
    """
    Main entry point for the mcp-server-qdrant script defined
    in pyproject.toml. It runs the MCP server with a specific transport
    protocol.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("[main.py] Starting MCP server entrypoint...")
    # Logolom a futó git commit hash-t és a betöltés idejét a memory.py betöltésekor, így a logban mindig látszik, melyik verzió fut.
    try:
        GIT_HASH = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception:
        GIT_HASH = "unknown"
    logging.info(f"[memory.py] MCP server version: {GIT_HASH} (loaded {datetime.datetime.utcnow().isoformat()} UTC)")

    # Parse the command-line arguments to determine the transport protocol.
    parser = argparse.ArgumentParser(description="mcp-server-qdrant")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    args = parser.parse_args()
    logging.info(f"[main.py] Parsed arguments: {args}")

    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    logging.info("[main.py] Importing mcp_server_qdrant.server.mcp ...")
    from mcp_server_qdrant.server import mcp

    logging.info(f"[main.py] Running MCP server with transport: {args.transport}")
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
