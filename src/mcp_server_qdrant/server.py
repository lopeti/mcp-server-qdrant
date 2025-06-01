import os
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

# Only pass server_options if FORCE_ASGI_SERVER is set (for HTTP/SSE mode)
if os.environ.get("FORCE_ASGI_SERVER") == "1":
    mcp = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider_settings=EmbeddingProviderSettings(),
        server_options={
            "host": "0.0.0.0",
            "port": 8000,
            "lifespan": "on",
            "workers": 1,
            "debug": True,
            "log_level": "info"
        },
    )
else:
    mcp = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings(),
        embedding_provider_settings=EmbeddingProviderSettings(),
    )

