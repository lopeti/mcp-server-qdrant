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
    except Exception as e:
        GIT_HASH = "unknown"
        logging.error(f"[main.py] Failed to retrieve git hash: {e}")
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
    try:
        from mcp_server_qdrant.server import mcp
    except ImportError as e:
        logging.error(f"[main.py] Failed to import mcp_server_qdrant.server.mcp: {e}")
        raise

    logging.info(f"[main.py] Running MCP server with transport: {args.transport}")

    # Add additional configuration for SSE transport
    if args.transport == "sse":
        logging.info("[main.py] Using SSE transport with additional configuration")
        try:
            mcp.run(
                transport=args.transport, 
                host="0.0.0.0",      # Explicitly bind to all network interfaces
                port=8000,           # Explicitly set port
                log_level="info"
            )
        except Exception as e:
            logging.error(f"[main.py] Failed to run MCP server with SSE transport: {e}")
            raise
    else:
        logging.info(f"[main.py] Using standard {args.transport} transport")
        try:
            mcp.run(transport=args.transport)
        except Exception as e:
            logging.error(f"[main.py] Failed to run MCP server with transport {args.transport}: {e}")
            raise


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"[main.py] Unhandled exception in main: {e}")
        raise
