from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

mcp = QdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
    # Add server settings with complete ASGI configuration
    server_options={
        "host": "0.0.0.0", 
        "port": 8000, 
        "lifespan": "on",
        "workers": 1,
        "debug": True,
        "log_level": "info"
    },
)

