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

        async def store(
            ctx: Context,
            information: str,
            collection_name: str,
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Optional[Metadata] = None,  # type: ignore
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: str,
            collection_name: str,
        ) -> List[str]:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :return: A list of entries found.
            """
            await ctx.debug(f"Finding results for query {query}")
            if collection_name:
                await ctx.debug(
                    f"Overriding the collection name with {collection_name}"
                )

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
            )
            if not entries:
                return [f"No information found for the query '{query}'"]
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            
        # --- Memory tools registration ---
        from .memory import memory_query, memory_upsert
        
        async def memory_query_adapter(
            ctx: Context,
            query: str,
            top_k: int = 3,
            collection_name: Optional[str] = None,
            user_id: Optional[str] = None,
        ) -> str:
            """
            Query the memory database for relevant entries.
            :param ctx: The context for the request.
            :param query: The query string to search for.
            :param top_k: Maximum number of results to return.
            :param collection_name: The collection to search in. Defaults to "default".
            :param user_id: Optional user ID to filter by.
            :return: A list of formatted entries as strings.
            """
            await ctx.debug(f"Searching for query '{query}' in collection {collection_name or 'default'}")
            if collection_name is None:
                collection_name = "default"
            try:
                result = await memory_query(query, top_k=top_k, collection_name=collection_name, user_id=user_id)
                formatted = []
                for entry_dict in result["result"]:
                    entry = Entry(
                        content=entry_dict["content"],
                        metadata=entry_dict.get("metadata"),
                        score=entry_dict.get("score")
                    )
                    formatted.append(self.format_entry(entry))
                if not formatted:
                    return f"No information found for the query '{query}'"
                return f"Results for the query '{query}':\n" + "\n".join(formatted)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception("[mcp_server.py] Exception in memory_query_adapter")
                return f"Error: {str(e)}"

        async def memory_upsert_adapter(
            ctx: Context,
            content: str,
            collection_name: Optional[str] = None,
            metadata: Optional[dict] = None,
            id: Optional[str] = None,
        ) -> str:
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
                return f"Successfully stored in collection '{collection_name}': <entry><content>{content}</content><metadata>{json.dumps(result['metadata'])}</metadata></entry>"
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.exception("[mcp_server.py] Exception in memory_upsert_adapter")
                return f"Error: {str(e)}"
        # Register regular MCP tools
        self.add_tool(
            find_foo,
            name="find",
            description=self.tool_settings.tool_find_description,
        )
        
        self.add_tool(
            store_foo,
            name="store",
            description=self.tool_settings.tool_store_description,
        )

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
