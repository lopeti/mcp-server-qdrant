import logging
import datetime
from typing import Any, List, Optional

from fastmcp import Context, FastMCP

from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import Entry, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

# Állítsd be a globális logging-ot a legelején
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
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

        # --- Memory tools registration ---
        from .memory import memory_query, memory_upsert

        async def dummy_adapter(ctx: Context) -> List[str]:
            logger = logging.getLogger(__name__)
            logger.info("[mcp_server.py] DUMMY ADAPTER called")
            await ctx.debug("DUMMY ADAPTER called")
            return [f"Dummy adapter response at {datetime.datetime.utcnow().isoformat()} UTC"]

        async def memory_query_adapter(
            ctx: Context,
            query: str,
            top_k: int = 10,
            collection_name: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> List[str]:
            logger = logging.getLogger(__name__)
            logger.info(f"[mcp_server.py] memory_query_adapter START: query='{query}', top_k={top_k}, collection_name={collection_name}, user_id={user_id}")
            await ctx.debug(f"memory_query_adapter START: query='{query}', top_k={top_k}, collection_name={collection_name}, user_id={user_id}")
            if collection_name is None:
                collection_name = "default"
            try:
                logger.info("[mcp_server.py] About to call memory_query function...")
                result = await memory_query(query, top_k=top_k, collection_name=collection_name, user_id=user_id)
                logger.info(f"[mcp_server.py] memory_query returned successfully: {result}")

                entries = [
                    Entry(
                        content=entry_dict["content"],
                        metadata=entry_dict.get("metadata"),
                        score=entry_dict.get("score")
                    )
                    for entry_dict in result["result"]
                ]

                logger.info(f"[mcp_server.py] Processed {len(entries)} entries from memory_query")

                if not entries:
                    logger.info(f"[mcp_server.py] No entries found for query '{query}'")
                    logger.info("[mcp_server.py] memory_query_adapter END (no entries)")
                    return [f"No information found for the query '{query}'"]

                response = [
                    f"Results for '{query}':"
                ] + [
                    f"• {entry.content} (timestamp: {entry.metadata.get('timestamp', '-')}, collection: {entry.metadata.get('collection_name', '-')})"
                    for entry in entries
                ]
                logger.info(f"[mcp_server.py] Returning response with {len(response)} items")
                logger.info("[mcp_server.py] memory_query_adapter END (success)")
                return  response
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception(f"[mcp_server.py] Exception in memory_query_adapter: {str(e)}")
                logger.info("[mcp_server.py] memory_query_adapter END (exception)")
                return [f"Error searching for '{query}': {str(e)}"]
            # extra safety
            logger.info("[mcp_server.py] memory_query_adapter END (unknown error)")
            return ["Unknown error!"]

        async def memory_upsert_adapter(
            ctx: Context,
            content: str,
            collection_name: Optional[str] = None,
            metadata: Optional[dict] = None,
            id: Optional[str] = None,
        ) -> List[str]:
            logger = logging.getLogger(__name__)
            logger.info(f"[mcp_server.py] memory_upsert_adapter START: content='{content}', collection_name={collection_name}, metadata={metadata}, id={id}")
            await ctx.debug(f"memory_upsert_adapter START: content='{content}', collection_name={collection_name}, metadata={metadata}, id={id}")
            if collection_name is None:
                collection_name = "default"
            try:
                result = await memory_upsert(content, collection_name=collection_name, metadata=metadata, id=id)
                ts = result["metadata"].get("timestamp", "-")
                logger.info(f"[mcp_server.py] memory_upsert_adapter END (success): stored in '{collection_name}' ts={ts}")
                return [f"Successfully stored in collection '{collection_name}': '{content}' (timestamp: {ts})"]
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception("[mcp_server.py] Exception in memory_upsert_adapter")
                logger.info("[mcp_server.py] memory_upsert_adapter END (exception)")
                return [f"An error occurred: {str(e)}"]
            # extra safety
            logger.info("[mcp_server.py] memory_upsert_adapter END (unknown error)")
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
        self.add_tool(
            dummy_adapter,
            name="dummy_tool",
            description="A dummy tool for testing connectivity and latency. Always returns instantly.",
        )
        logging.info("[mcp_server.py] Registered 'memory_query', 'memory_upsert' and 'dummy_tool' tools.")
        # FastAPI router and endpoints removed: not needed in MCP stdio/sse mode

    async def initialize_server(self):
        """
        Perform any asynchronous initialization tasks required for the server.
        """
        logging.info("[mcp_server.py] Performing server initialization...")
        await self.qdrant_connector.initialize()
        logging.info("[mcp_server.py] Qdrant connector initialized.")
        self.initialized = True

    async def setup_tools(self):
        logging.info("[mcp_server.py] Setting up tools for QdrantMCPServer.")
        """
        Register the tools in the server.
        """
        # Ensure server is initialized before registering tools
        if not hasattr(self, 'initialized') or not self.initialized:
            logging.warning("[mcp_server.py] Server not initialized. Initializing now...")
            await self.initialize_server()

        # --- Memory tools registration ---
        from .memory import memory_query, memory_upsert

        async def dummy_adapter(ctx: Context) -> List[str]:
            logger = logging.getLogger(__name__)
            logger.info("[mcp_server.py] DUMMY ADAPTER called")
            await ctx.debug("DUMMY ADAPTER called")
            return [f"Dummy adapter response at {datetime.datetime.utcnow().isoformat()} UTC"]

        async def memory_query_adapter(
            ctx: Context,
            query: str,
            top_k: int = 10,
            collection_name: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> List[str]:
            logger = logging.getLogger(__name__)
            logger.info(f"[mcp_server.py] memory_query_adapter START: query='{query}', top_k={top_k}, collection_name={collection_name}, user_id={user_id}")
            await ctx.debug(f"memory_query_adapter START: query='{query}', top_k={top_k}, collection_name={collection_name}, user_id={user_id}")
            if collection_name is None:
                collection_name = "default"
            try:
                logger.info("[mcp_server.py] About to call memory_query function...")
                result = await memory_query(query, top_k=top_k, collection_name=collection_name, user_id=user_id)
                logger.info(f"[mcp_server.py] memory_query returned successfully: {result}")

                entries = [
                    Entry(
                        content=entry_dict["content"],
                        metadata=entry_dict.get("metadata"),
                        score=entry_dict.get("score")
                    )
                    for entry_dict in result["result"]
                ]

                logger.info(f"[mcp_server.py] Processed {len(entries)} entries from memory_query")

                if not entries:
                    logger.info(f"[mcp_server.py] No entries found for query '{query}'")
                    logger.info("[mcp_server.py] memory_query_adapter END (no entries)")
                    return [f"No information found for the query '{query}'"]

                response = [
                    f"Results for '{query}':"
                ] + [
                    f"• {entry.content} (timestamp: {entry.metadata.get('timestamp', '-')}, collection: {entry.metadata.get('collection_name', '-')})"
                    for entry in entries
                ]
                logger.info(f"[mcp_server.py] Returning response with {len(response)} items")
                logger.info("[mcp_server.py] memory_query_adapter END (success)")
                return  response
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception(f"[mcp_server.py] Exception in memory_query_adapter: {str(e)}")
                logger.info("[mcp_server.py] memory_query_adapter END (exception)")
                return [f"Error searching for '{query}': {str(e)}"]
            # extra safety
            logger.info("[mcp_server.py] memory_query_adapter END (unknown error)")
            return ["Unknown error!"]

        async def memory_upsert_adapter(
            ctx: Context,
            content: str,
            collection_name: Optional[str] = None,
            metadata: Optional[dict] = None,
            id: Optional[str] = None,
        ) -> List[str]:
            logger = logging.getLogger(__name__)
            logger.info(f"[mcp_server.py] memory_upsert_adapter START: content='{content}', collection_name={collection_name}, metadata={metadata}, id={id}")
            await ctx.debug(f"memory_upsert_adapter START: content='{content}', collection_name={collection_name}, metadata={metadata}, id={id}")
            if collection_name is None:
                collection_name = "default"
            try:
                result = await memory_upsert(content, collection_name=collection_name, metadata=metadata, id=id)
                ts = result["metadata"].get("timestamp", "-")
                logger.info(f"[mcp_server.py] memory_upsert_adapter END (success): stored in '{collection_name}' ts={ts}")
                return [f"Successfully stored in collection '{collection_name}': '{content}' (timestamp: {ts})"]
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception("[mcp_server.py] Exception in memory_upsert_adapter")
                logger.info("[mcp_server.py] memory_upsert_adapter END (exception)")
                return [f"An error occurred: {str(e)}"]
            # extra safety
            logger.info("[mcp_server.py] memory_upsert_adapter END (unknown error)")
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
        self.add_tool(
            dummy_adapter,
            name="dummy_tool",
            description="A dummy tool for testing connectivity and latency. Always returns instantly.",
        )
        logging.info("[mcp_server.py] Registered 'memory_query', 'memory_upsert' and 'dummy_tool' tools.")
        # FastAPI router and endpoints removed: not needed in MCP stdio/sse mode
