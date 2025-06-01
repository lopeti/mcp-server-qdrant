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
    # Log the running git commit hash and load time
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
    logging.info(f"[main.py] Parsed arguments: {args}")    # Import is done here to make sure environment variables are loaded
    # only after we make the changes.
    logging.info("[main.py] Importing mcp_server_qdrant.server.mcp ...")
    from mcp_server_qdrant.server import mcp
    
    logging.info(f"[main.py] Running MCP server with transport: {args.transport}")    # Add additional configuration for SSE transport
    if args.transport == "sse":
        logging.info("[main.py] Using SSE transport with additional configuration")
        mcp.run(
            transport=args.transport, 
            host="0.0.0.0",      # Explicitly bind to all network interfaces
            port=8000,           # Explicitly set port
            log_level="info"
        )
    else:
        logging.info(f"[main.py] Using standard {args.transport} transport")
        mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
