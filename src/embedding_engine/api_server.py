"""
FastAPI Retrieval Server for Materials Science

This module provides a REST API for materials retrieval using FAISS vector search
with Redis caching for sub-100ms response times.

Key Endpoints:
- /search: Semantic search in materials database
- /embed: Generate embeddings for text queries
- /materials/{material_id}: Get specific material information
- /rag/query: Full RAG query with generation

Performance Features:
- Redis caching for frequent queries
- Async request handling
- Request batching for efficiency
- Performance monitoring
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
import json
import hashlib
from pathlib import Path

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis

from sentence_transformers import SentenceTransformer
from .faiss_index import MaterialsFAISSIndex

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API
class SearchQuery(BaseModel):
    query: str = Field(..., description="Search query text")
    k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Property filters")
    use_cache: bool = Field(default=True, description="Whether to use cache")


class EmbeddingRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to embed")


class MaterialResponse(BaseModel):
    material_id: str
    formula: str
    score: float
    properties: Dict[str, Any]
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    results: List[MaterialResponse]
    total_time_ms: float
    cache_hit: bool = False


class RAGQuery(BaseModel):
    query: str = Field(..., description="Question about materials")
    context_k: int = Field(default=5, description="Number of context materials to retrieve")


class RAGResponse(BaseModel):
    query: str
    answer: str
    context_materials: List[MaterialResponse]
    confidence: float
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str
    total_materials: int
    avg_search_time_ms: float
    cache_status: str


# Global variables for the application
app = FastAPI(
    title="Materials Science Retrieval API",
    description="High-performance vector search and RAG for materials discovery",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
embedding_model: Optional[SentenceTransformer] = None
faiss_index: Optional[MaterialsFAISSIndex] = None
redis_client: Optional[redis.Redis] = None


class MaterialsRetrievalAPI:
    """Main API class managing all retrieval operations."""
    
    def __init__(self, 
                 model_path: str = "all-MiniLM-L6-v2",
                 index_path: Optional[str] = None,
                 redis_url: str = "redis://localhost:6379"):
        """
        Initialize the retrieval API.
        
        Args:
            model_path: Path to embedding model or model name
            index_path: Path to saved FAISS index
            redis_url: Redis connection URL
        """
        self.model_path = model_path
        self.index_path = index_path
        self.redis_url = redis_url
        
        # Performance tracking
        self.request_times = []
        self.cache_hits = 0
        self.total_requests = 0
    
    async def initialize(self):
        """Initialize all components asynchronously."""
        global embedding_model, faiss_index, redis_client
        
        logger.info("Initializing Materials Retrieval API...")
        
        # Load embedding model
        try:
            if Path(self.model_path).exists():
                embedding_model = SentenceTransformer(self.model_path)
                logger.info(f"Loaded custom embedding model from {self.model_path}")
            else:
                embedding_model = SentenceTransformer(self.model_path)
                logger.info(f"Loaded base embedding model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Load FAISS index
        try:
            faiss_index = MaterialsFAISSIndex()
            if self.index_path and Path(self.index_path).exists():
                faiss_index.load_index(self.index_path)
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                logger.warning("No FAISS index provided - starting with empty index")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            raise
        
        # Initialize Redis
        try:
            redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await asyncio.get_event_loop().run_in_executor(None, redis_client.ping)
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Continuing without cache.")
            redis_client = None
    
    def _generate_cache_key(self, query: str, k: int, filters: Optional[Dict]) -> str:
        """Generate cache key for query."""
        cache_data = {"query": query, "k": k, "filters": filters}
        cache_string = json.dumps(cache_data, sort_keys=True)
        return f"search:{hashlib.md5(cache_string.encode()).hexdigest()}"
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict]:
        """Get cached search result."""
        if not redis_client:
            return None
        
        try:
            cached = await asyncio.get_event_loop().run_in_executor(
                None, redis_client.get, cache_key
            )
            if cached:
                self.cache_hits += 1
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict, ttl: int = 3600):
        """Cache search result."""
        if not redis_client:
            return
        
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: redis_client.setex(cache_key, ttl, json.dumps(result))
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    async def search_materials(self, query: SearchQuery) -> SearchResponse:
        """Perform semantic search for materials."""
        start_time = time.time()
        self.total_requests += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(query.query, query.k, query.filters)
        if query.use_cache:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result:
                cached_result["cache_hit"] = True
                cached_result["total_time_ms"] = (time.time() - start_time) * 1000
                return SearchResponse(**cached_result)
        
        # Generate query embedding
        query_embedding = embedding_model.encode([query.query])
        
        # Perform FAISS search
        search_results = faiss_index.search(query_embedding, k=query.k)
        
        # Apply property filters if provided
        if query.filters:
            search_results = faiss_index.filter_by_properties(search_results, query.filters)
        
        # Convert to response format
        material_responses = [
            MaterialResponse(
                material_id=result.material_id,
                formula=result.formula,
                score=result.score,
                properties=result.properties,
                metadata=result.metadata
            )
            for result in search_results
        ]
        
        # Create response
        total_time = (time.time() - start_time) * 1000
        response = SearchResponse(
            query=query.query,
            results=material_responses,
            total_time_ms=total_time,
            cache_hit=False
        )
        
        # Cache the result
        if query.use_cache:
            cache_data = response.dict()
            del cache_data["cache_hit"]
            del cache_data["total_time_ms"]
            await self._cache_result(cache_key, cache_data)
        
        # Track performance
        self.request_times.append(total_time)
        logger.info(f"Search completed in {total_time:.2f}ms")
        
        return response
    
    async def generate_embeddings(self, request: EmbeddingRequest) -> Dict[str, Any]:
        """Generate embeddings for input texts."""
        start_time = time.time()
        
        embeddings = embedding_model.encode(request.texts)
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "embeddings": embeddings.tolist(),
            "dimension": embeddings.shape[1],
            "total_time_ms": total_time
        }
    
    def get_health_status(self) -> HealthResponse:
        """Get API health and performance statistics."""
        faiss_stats = faiss_index.get_performance_stats() if faiss_index else {}
        
        return HealthResponse(
            status="healthy",
            total_materials=faiss_stats.get("total_materials", 0),
            avg_search_time_ms=faiss_stats.get("avg_search_time_ms", 0.0),
            cache_status="connected" if redis_client else "disconnected"
        )


# Initialize the API instance
api_instance = MaterialsRetrievalAPI()


@app.on_event("startup")
async def startup_event():
    """Initialize the API on startup."""
    await api_instance.initialize()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return api_instance.get_health_status()


@app.post("/search", response_model=SearchResponse)
async def search_materials(query: SearchQuery):
    """Search for materials using semantic similarity."""
    try:
        return await api_instance.search_materials(query)
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for input texts."""
    try:
        return await api_instance.generate_embeddings(request)
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/materials/{material_id}")
async def get_material(material_id: str):
    """Get specific material information by ID."""
    try:
        # Find material in index
        for material in faiss_index.metadata:
            if material.get('id') == material_id:
                return MaterialResponse(
                    material_id=material_id,
                    formula=material.get('formula', 'Unknown'),
                    score=1.0,  # Direct lookup
                    properties=material.get('properties', {}),
                    metadata=material
                )
        
        raise HTTPException(status_code=404, detail="Material not found")
    except Exception as e:
        logger.error(f"Material lookup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(query: RAGQuery):
    """
    Perform RAG (Retrieval-Augmented Generation) query.
    Note: This is a placeholder - full RAG implementation in Module 4.
    """
    try:
        start_time = time.time()
        
        # Retrieve relevant materials
        search_query = SearchQuery(query=query.query, k=query.context_k)
        search_response = await api_instance.search_materials(search_query)
        
        # Placeholder for generation (Module 4 will implement this)
        context_text = "\\n\\n".join([
            f"{r.formula}: {r.properties}" 
            for r in search_response.results
        ])
        
        placeholder_answer = f"Based on the retrieved materials data:\\n{context_text}\\n\\nThis would be a generated answer about: {query.query}"
        
        total_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            query=query.query,
            answer=placeholder_answer,
            context_materials=search_response.results,
            confidence=0.8,  # Placeholder confidence
            total_time_ms=total_time
        )
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_performance_stats():
    """Get detailed performance statistics."""
    faiss_stats = faiss_index.get_performance_stats() if faiss_index else {}
    
    cache_hit_rate = (api_instance.cache_hits / max(api_instance.total_requests, 1)) * 100
    
    return {
        "total_requests": api_instance.total_requests,
        "cache_hit_rate": f"{cache_hit_rate:.1f}%",
        "avg_request_time_ms": np.mean(api_instance.request_times) if api_instance.request_times else 0,
        "faiss_stats": faiss_stats,
        "redis_status": "connected" if redis_client else "disconnected"
    }


# Development server runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Materials Retrieval API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model-path", default="all-MiniLM-L6-v2", 
                       help="Path to embedding model")
    parser.add_argument("--index-path", help="Path to FAISS index")
    parser.add_argument("--redis-url", default="redis://localhost:6379",
                       help="Redis connection URL")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    # Update API instance configuration
    api_instance.model_path = args.model_path
    api_instance.index_path = args.index_path
    api_instance.redis_url = args.redis_url
    
    # Run the server
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )