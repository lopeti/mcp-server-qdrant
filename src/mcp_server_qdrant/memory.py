"""
Fact Memory module for MCP Server Qdrant
Implements memory_query and memory_upsert tools for long-term semantic memory.
"""
import uuid
import datetime
from typing import List, Dict, Optional, Any
import logging
import subprocess

from pydantic import BaseModel, Field

from .embeddings.factory import create_embedding_provider
from .qdrant import QdrantConnector, Entry
from .settings import EmbeddingProviderSettings

# --- Schemas ---

def now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat()

def get_default_embedding_provider():
    # Singleton pattern for embedding provider
    if not hasattr(get_default_embedding_provider, "_provider"):
        settings = EmbeddingProviderSettings()
        get_default_embedding_provider._provider = create_embedding_provider(settings)
    return get_default_embedding_provider._provider

def get_default_qdrant_client():
    logger = logging.getLogger(__name__)
    # Singleton pattern for QdrantConnector
    if not hasattr(get_default_qdrant_client, "_client"):
        from .settings import QdrantSettings
        qdrant_settings = QdrantSettings()
        embedding_provider = get_default_embedding_provider()
        logger.info("[memory.py] Creating new QdrantConnector singleton instance.")
        get_default_qdrant_client._client = QdrantConnector(
            qdrant_url=getattr(qdrant_settings, 'qdrant_url', None),
            qdrant_api_key=getattr(qdrant_settings, 'qdrant_api_key', None),
            collection_name=getattr(qdrant_settings, 'collection_name', None),
            embedding_provider=embedding_provider,
            qdrant_local_path=getattr(qdrant_settings, 'qdrant_local_path', None),
        )
    else:
        logger.info(f"[memory.py] Reusing QdrantConnector singleton instance: {get_default_qdrant_client._client}")
    # Ellenőrizzük, hogy a _client attribútum él-e
    qc = get_default_qdrant_client._client
    logger.info(f"[memory.py] QdrantConnector._client: {getattr(qc, '_client', None)}")
    return qc

# --- memory_query ---
async def memory_query(
    query: str,
    top_k: int = 3,
    collection_name: str = "default",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"[memory.py] memory_query called: query={query}, top_k={top_k}, collection_name={collection_name}, user_id={user_id}")
    try:
        logger.info(f"[memory.py] About to get QdrantConnector client")
        client = get_default_qdrant_client()
        logger.info(f"[memory.py] memory_query using QdrantConnector: {client}, _client: {getattr(client, '_client', None)}")
        logger.info(f"[memory.py] About to call client.search")
        hits = await client.search(
            query=query,
            collection_name=collection_name,
            limit=top_k
        )
        logger.info(f"[memory.py] client.search returned successfully")
    except Exception as e:
        logger.error(f"[memory.py] Error during search: {str(e)}")
        raise
    logger.info(f"[memory.py] memory_query hits: {hits}")
    result = []
    for idx, hit in enumerate(hits):
        content = getattr(hit, "content", "") or ""
        metadata = getattr(hit, "metadata", {}) or {}
        score = hit.score if hit.score is not None else 1.0
        result.append({
            "id": str(idx),
            "content": content,
            "metadata": metadata,
            "score": round(score, 4),
        })
    # Sort by score descending
    result.sort(key=lambda x: x["score"], reverse=True)
    logger.info(f"[memory.py] memory_query returning {len(result)} results.")
    return {"result": result}

# --- memory_upsert ---
async def memory_upsert(
    content: str,
    collection_name: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
    id: Optional[str] = None,
) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    logger.info(f"[memory.py] memory_upsert called: content={content}, collection_name={collection_name}, id={id}")
    client = get_default_qdrant_client()
    logger.info(f"[memory.py] memory_upsert using QdrantConnector: {client}, _client: {getattr(client, '_client', None)}")
    memory_id = id or str(uuid.uuid4())
    meta = dict(metadata or {})
    meta.setdefault("timestamp", now_iso())
    meta["content"] = content
    meta.setdefault("collection_name", collection_name)
    entry = Entry(content=content, metadata=meta)
    await client.store(entry, collection_name=collection_name)
    logger.info(f"[memory.py] memory_upsert stored entry: {entry}")
    return {
        "status": "success",
        "id": memory_id,
        "metadata": meta,
    }

# --- Tool registration (for MCP) ---
# A toolok regisztrációját a szerver setup_tools metódusába kell helyezni, nem itt!
# Ez a szekció csak a Pydantic sémákat tartalmazza, a tényleges regisztrációt a szerver végzi.
class MemoryQueryArgs(BaseModel):
    query: str = Field(..., description="Free-text search query.")
    top_k: int = Field(3, description="Maximum results to return.")
    collection_name: str = Field("default", description="Memory collection name.")
    user_id: Optional[str] = Field(None, description="User ID to filter by.")

class MemoryUpsertArgs(BaseModel):
    content: str = Field(..., description="Fact/memory to store.")
    collection_name: str = Field("default", description="Memory collection name.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata (user_id, timestamp, etc.)")
    id: Optional[str] = Field(None, description="Memory ID (for update).")


# Ready for future extension: memory_delete, memory_list, etc.
