"""Memory retrieval system with semantic search using Qdrant."""

import json
import time
from typing import Dict, List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.core.config import settings
from app.core.logging import logger
from app.core.observability import emit_memory_fallback, emit_memory_retrieve

# Optional imports - only needed if memory retrieval is enabled
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    Distance = None
    VectorParams = None
    logger.warning("qdrant-client not installed. Memory retrieval will be disabled.")


class MemoryRetrieval:
    """Semantic memory retrieval using Qdrant vector database.
    
    Retrieves relevant long-term memories based on semantic similarity
    to the current conversation context.
    """

    def __init__(self):
        """Initialize the memory retrieval system with Qdrant and embeddings."""
        if not settings.FEATURE_MEMORY_RETRIEVAL:
            logger.info("memory_retrieval_disabled")
            self.qdrant_client = None
            self.embeddings = None
            return
        
        if not QDRANT_AVAILABLE:
            logger.warning("qdrant-client package not installed. Memory retrieval disabled.")
            self.qdrant_client = None
            self.embeddings = None
            return
        
        try:
            # Initialize Qdrant client
            api_key = None if settings.QDRANT_API_KEY == "none" else settings.QDRANT_API_KEY
            self.qdrant_client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=api_key,
            )
            
            # Initialize OpenAI embeddings
            self.embeddings = OpenAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                openai_api_key=settings.LLM_API_KEY,
            )
            
            # Collection name
            self.collection_name = "user_memories"
            
            # Initialize collection if it doesn't exist
            self._ensure_collection_exists()
            
            logger.info(
                "memory_retrieval_initialized",
                qdrant_url=settings.QDRANT_URL,
                embedding_model=settings.EMBEDDING_MODEL,
            )
        except Exception as e:
            logger.error("memory_retrieval_init_failed", error=str(e))
            self.qdrant_client = None
            self.embeddings = None

    def _ensure_collection_exists(self):
        """Ensure the Qdrant collection exists, create if it doesn't."""
        if not self.qdrant_client or not QDRANT_AVAILABLE:
            return
        
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                # Create collection with 1536 dimensions (text-embedding-3-small)
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info("memory_collection_created", collection=self.collection_name)
            else:
                logger.debug("memory_collection_exists", collection=self.collection_name)
        except Exception as e:
            logger.error("memory_collection_setup_failed", error=str(e))

    async def store_memory(
        self,
        user_id: int,
        memory_type: str,
        memory_content: Dict[str, any],
        metadata: Optional[Dict[str, any]] = None,
    ) -> bool:
        """Store a memory in both PostgreSQL and Qdrant.
        
        Args:
            user_id: The user ID
            memory_type: Type of memory (vehicle_info, preferences, etc.)
            memory_content: The memory facts as dictionary
            metadata: Optional additional metadata
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not settings.FEATURE_MEMORY_RETRIEVAL or not self.qdrant_client or not self.embeddings:
            return False

        try:
            # Create embedding for the memory content
            memory_text = json.dumps(memory_content)
            embedding = await self.embeddings.aembed_query(memory_text)
            
            # Store in Qdrant with user_id as filter
            payload = {
                "user_id": user_id,
                "memory_type": memory_type,
                "memory_content": memory_content,
                "metadata": metadata or {},
            }
            
            # Use a unique ID combining user_id and timestamp
            point_id = int(f"{user_id}{int(time.time())}") % (2**63 - 1)
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[
                    {
                        "id": point_id,
                        "vector": embedding,
                        "payload": payload,
                    }
                ],
            )
            
            logger.info(
                "memory_stored",
                user_id=user_id,
                memory_type=memory_type,
                point_id=point_id,
            )
            
            return True
            
        except Exception as e:
            logger.error(
                "memory_storage_failed",
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            return False

    async def retrieve_memories(
        self,
        user_id: int,
        query_text: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, any]]:
        """Retrieve relevant memories for a user based on semantic similarity.
        
        Args:
            user_id: The user ID to filter memories
            query_text: The query text to search for
            top_k: Number of memories to retrieve
            min_score: Minimum similarity score
            
        Returns:
            List of relevant memories with scores
        """
        if not settings.FEATURE_MEMORY_RETRIEVAL or not self.qdrant_client or not self.embeddings:
            emit_memory_fallback(user_id=user_id, reason="feature_disabled_or_unavailable")
            return []

        start_time = time.time()
        
        try:
            # Create embedding for query
            query_embedding = await self.embeddings.aembed_query(query_text)
            
            # Retrieve from Qdrant with user filter
            top_k = top_k or settings.MEM_TOP_K
            min_score = min_score or settings.MEM_MIN_SCORE
            
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter={
                    "must": [
                        {
                            "key": "user_id",
                            "match": {"value": user_id}
                        }
                    ]
                },
                limit=top_k,
                score_threshold=min_score,
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Format results
            memories = []
            matched_ids = []
            scores = []
            
            for result in results:
                # Extract safe ID from memory content if available
                memory_id = str(result.payload.get("memory_type", "unknown"))
                matched_ids.append(memory_id)
                scores.append(result.score)
                
                memories.append({
                    "score": result.score,
                    "memory_type": result.payload.get("memory_type"),
                    "memory_content": result.payload.get("memory_content"),
                    "metadata": result.payload.get("metadata", {}),
                })
            
            logger.info(
                "memories_retrieved",
                user_id=user_id,
                count=len(memories),
                top_score=memories[0]["score"] if memories else 0.0,
            )
            
            # Emit Langfuse span for retrieval
            emit_memory_retrieve(
                user_id=user_id,
                matched_ids=matched_ids,
                scores=scores,
                top_k=top_k,
                latency_ms=latency_ms,
            )
            
            return memories
            
        except Exception as e:
            # Emit fallback span for retrieval failure
            emit_memory_fallback(user_id=user_id, reason=f"retrieval_error: {type(e).__name__}")
            
            logger.error(
                "memory_retrieval_failed",
                error=str(e),
                user_id=user_id,
                exc_info=True,
            )
            return []


# Create singleton instance
memory_retrieval = MemoryRetrieval()

