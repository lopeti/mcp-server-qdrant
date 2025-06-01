import json
import logging
from typing import Any, List, Optional

from fastmcp import Context, FastMCP

from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: EmbeddingProviderSettings,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        logging.info("[mcp_server.py] Initializing QdrantMCPServer...")
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.embedding_provider_settings = embedding_provider_settings

        self.embedding_provider = create_embedding_provider(embedding_provider_settings)
        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()
        logging.info("[mcp_server.py] QdrantMCPServer initialized and tools set up.")

    def format_entry(self, entry: Entry) -> str:
        logging.info("[mcp_server.py] Formatting entry for output.")
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        logging.info("[mcp_server.py] Setting up tools for QdrantMCPServer.")
        """
        Register the tools in the server.
        """            
        # --- Memory tools registration ---
        from .memory import memory_query, memory_upsert
        
        async def memory_query_adapter(
            ctx: Context,
            query: str,
            top_k: int = 3,
            collection_name: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> List[str]:
            await ctx.debug(f"Searching for query '{query}' in collection {collection_name or 'default'}")
            if collection_name is None:
                collection_name = "default"
            try:
                result = await memory_query(query, top_k=top_k, collection_name=collection_name, user_id=user_id)
                entries = [
                    Entry(
                        content=entry_dict["content"],
                        metadata=entry_dict.get("metadata"),
                        score=entry_dict.get("score")
                    )
                    for entry_dict in result["result"]
                ]
                if not entries:
                    return [f"No information found for the query '{query}'"]
                return [
                    f"Results for '{query}':"
                ] + [
                    f"• {entry.content} (timestamp: {entry.metadata.get('timestamp', '-')}, collection: {entry.metadata.get('collection_name', '-')})"
                    for entry in entries
                ]
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception("[mcp_server.py] Exception in memory_query_adapter")
                return [f"Error: {str(e)}"]
            # extra safety
            return ["Unknown error!"]

        async def memory_upsert_adapter(
            ctx: Context,
            content: str,
            collection_name: Optional[str] = None,
            metadata: Optional[dict] = None,
            id: Optional[str] = None,
        ) -> List[str]:
            """
            Store information in the memory database.
            :param ctx: The context for the request.
            :param content: The content to store.
            :param collection_name: The collection to store in. Defaults to "default".
            :param metadata: Optional metadata to attach to the content.
            :param id: Optional ID to assign to the content.
            :return: A confirmation message as a list of strings.
            """
            await ctx.debug(f"Storing content in collection {collection_name or 'default'}")
            if collection_name is None:
                collection_name = "default"
            try:
                result = await memory_upsert(content, collection_name=collection_name, metadata=metadata, id=id)
                ts = result["metadata"].get("timestamp", "-")
                return [f"Successfully stored in collection '{collection_name}': '{content}' (timestamp: {ts})"]
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception("[mcp_server.py] Exception in memory_upsert_adapter")
                return [f"An error occurred: {str(e)}"]
            # extra safety
            return ["Unknown error!"]


        # Register memory tools 
        self.add_tool(
            memory_query_adapter,
            name="memory_query",
            description=(
                "Retrieve facts, notes, or memories previously taught to the assistant in conversations or explicit requests. "
                "Not for real-time sensor or device state (use the entity API for that). "
                "Example: Find out when the user last watered the plants, or what birthday message was set last month. "
                "For current sensor or device values, use the Home Assistant entity API/tool, not this memory tool."
            ),
        )
        self.add_tool(
            memory_upsert_adapter,
            name="memory_upsert",
            description=(
                "Store a new fact, event, or personal note for long-term memory. "
                "Use for anything you want the assistant to remember in future conversations. "
                "Not for real-time sensor/device values! "
                "For current sensor or device values, use the Home Assistant entity API/tool, not this memory tool."
            ),
        )
        logging.info("[mcp_server.py] Registered 'memory_query' and 'memory_upsert' tools.")
