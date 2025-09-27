"""
FAISS Vector Similarity Indexing for Materials Science

This module implements high-performance vector similarity search using FAISS
for sub-100ms retrieval of materials science data.

Key Features:
- FAISS indexing with IVF (Inverted File) structure
- GPU acceleration support
- Hierarchical search with coarse-to-fine retrieval
- Index persistence and loading
- Batch processing for large datasets
"""

import logging
import numpy as np
import faiss
import pickle
from pathlib import Path
from typing import List, Dict, Any
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Container for search results."""
    material_id: str
    formula: str
    score: float
    properties: Dict[str, Any]
    metadata: Dict[str, Any]


class MaterialsFAISSIndex:
    """
    High-performance FAISS index for materials science vector search.
    
    This class manages FAISS indexes optimized for materials science queries,
    providing sub-100ms search capabilities for large-scale materials databases.
    """
    
    def __init__(self, 
                 dimension: int = 384,
                 index_type: str = "IVF",
                 nlist: int = 1024,
                 use_gpu: bool = False):
        """
        Initialize FAISS index for materials search.
        
        Args:
            dimension: Embedding dimension (default 384 for MiniLM)
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
            nlist: Number of clusters for IVF index
            use_gpu: Whether to use GPU acceleration
        """
        self.dimension = dimension
        self.index_type = index_type
        self.nlist = nlist
        self.use_gpu = use_gpu
        
        # Initialize index
        self.index = None
        self.metadata = []
        self.id_to_material = {}
        
        # Performance tracking
        self.search_times = []
        
        self._create_index()
    
    def _create_index(self) -> None:
        """Create the appropriate FAISS index based on configuration."""
        if self.index_type == "Flat":
            # Exact search - highest accuracy, slower for large datasets
            self.index = faiss.IndexFlatIP(self.dimension)
            
        elif self.index_type == "IVF":
            # Inverted File index - balanced speed/accuracy
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.nlist)
            
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World - very fast, good accuracy
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Setup GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
            logger.info("Initialized GPU-accelerated FAISS index")
        else:
            logger.info(f"Initialized CPU FAISS index: {self.index_type}")
    
    def add_materials(self, 
                     embeddings: np.ndarray, 
                     materials_data: List[Dict]) -> None:
        """
        Add materials and their embeddings to the index.
        
        Args:
            embeddings: Array of embedding vectors (n_materials, dimension)
            materials_data: List of material metadata dictionaries
        """
        if len(embeddings) != len(materials_data):
            raise ValueError("Number of embeddings must match number of materials")
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Train index if necessary (for IVF)
        if isinstance(self.index, faiss.IndexIVFFlat) and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings_normalized.astype(np.float32))
        
        # Add vectors to index
        start_id = len(self.metadata)
        self.index.add(embeddings_normalized.astype(np.float32))
        
        # Store metadata
        for i, material in enumerate(materials_data):
            material_id = material.get('id', f"mat_{start_id + i}")
            self.metadata.append(material)
            self.id_to_material[start_id + i] = material_id
        
        logger.info(f"Added {len(materials_data)} materials to index. Total: {len(self.metadata)}")
    
    def search(self, 
               query_embedding: np.ndarray,
               k: int = 10,
               nprobe: int = 10) -> List[SearchResult]:
        """
        Search for similar materials using vector similarity.
        
        Args:
            query_embedding: Query vector (1, dimension)
            k: Number of results to return
            nprobe: Number of clusters to search (for IVF index)
            
        Returns:
            List of SearchResult objects
        """
        start_time = time.time()
        
        # Ensure query is 2D and normalized
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        query_normalized = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Set search parameters for IVF
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = nprobe
        
        # Perform search
        scores, indices = self.index.search(query_normalized.astype(np.float32), k)
        
        # Record search time
        search_time = (time.time() - start_time) * 1000  # ms
        self.search_times.append(search_time)
        
        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):  # Valid index
                material = self.metadata[idx]
                result = SearchResult(
                    material_id=material.get('id', f'mat_{idx}'),
                    formula=material.get('formula', 'Unknown'),
                    score=float(score),
                    properties=material.get('properties', {}),
                    metadata=material
                )
                results.append(result)
        
        logger.debug(f"Search completed in {search_time:.2f}ms")
        return results
    
    def batch_search(self, 
                    query_embeddings: np.ndarray,
                    k: int = 10,
                    nprobe: int = 10) -> List[List[SearchResult]]:
        """
        Search for multiple queries in batch for efficiency.
        
        Args:
            query_embeddings: Multiple query vectors (n_queries, dimension)
            k: Number of results per query
            nprobe: Number of clusters to search
            
        Returns:
            List of result lists, one per query
        """
        start_time = time.time()
        
        # Normalize queries
        queries_normalized = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        
        # Set search parameters
        if isinstance(self.index, faiss.IndexIVFFlat):
            self.index.nprobe = nprobe
        
        # Perform batch search
        scores, indices = self.index.search(queries_normalized.astype(np.float32), k)
        
        # Record search time
        search_time = (time.time() - start_time) * 1000  # ms
        self.search_times.append(search_time)
        
        # Convert to SearchResult objects
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            query_results = []
            for score, idx in zip(query_scores, query_indices):
                if idx >= 0 and idx < len(self.metadata):
                    material = self.metadata[idx]
                    result = SearchResult(
                        material_id=material.get('id', f'mat_{idx}'),
                        formula=material.get('formula', 'Unknown'),
                        score=float(score),
                        properties=material.get('properties', {}),
                        metadata=material
                    )
                    query_results.append(result)
            all_results.append(query_results)
        
        logger.info(f"Batch search of {len(query_embeddings)} queries completed in {search_time:.2f}ms")
        return all_results
    
    def filter_by_properties(self, 
                           results: List[SearchResult],
                           property_filters: Dict[str, Any]) -> List[SearchResult]:
        """
        Filter search results by material properties.
        
        Args:
            results: List of search results
            property_filters: Dictionary of property constraints
                            e.g., {'band_gap': {'min': 1.0, 'max': 3.0}}
            
        Returns:
            Filtered list of search results
        """
        filtered_results = []
        
        for result in results:
            properties = result.properties
            include_result = True
            
            for prop_name, constraints in property_filters.items():
                if prop_name not in properties:
                    include_result = False
                    break
                
                prop_value = properties[prop_name]
                if prop_value is None:
                    include_result = False
                    break
                
                # Apply constraints
                if isinstance(constraints, dict):
                    if 'min' in constraints and prop_value < constraints['min']:
                        include_result = False
                        break
                    if 'max' in constraints and prop_value > constraints['max']:
                        include_result = False
                        break
                    if 'equals' in constraints and prop_value != constraints['equals']:
                        include_result = False
                        break
                else:
                    # Direct value match
                    if prop_value != constraints:
                        include_result = False
                        break
            
            if include_result:
                filtered_results.append(result)
        
        return filtered_results
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for the index."""
        if not self.search_times:
            return {'avg_search_time_ms': 0.0, 'total_searches': 0}
        
        return {
            'avg_search_time_ms': np.mean(self.search_times),
            'min_search_time_ms': np.min(self.search_times),
            'max_search_time_ms': np.max(self.search_times),
            'total_searches': len(self.search_times),
            'total_materials': len(self.metadata)
        }
    
    def save_index(self, save_path: str) -> None:
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            save_path: Directory path to save index files
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = save_dir / "faiss_index.bin"
        if self.use_gpu:
            # Move to CPU before saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))
        
        # Save metadata
        metadata_path = save_dir / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'id_to_material': self.id_to_material,
                'config': {
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'nlist': self.nlist,
                    'use_gpu': self.use_gpu
                }
            }, f)
        
        logger.info(f"Index saved to {save_path}")
    
    def load_index(self, load_path: str) -> None:
        """
        Load a previously saved FAISS index.
        
        Args:
            load_path: Directory path containing saved index files
        """
        load_dir = Path(load_path)
        
        # Load FAISS index
        index_path = load_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        
        # Setup GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
        
        # Load metadata
        metadata_path = load_dir / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data['metadata']
            self.id_to_material = data['id_to_material']
            
            # Update config if available
            if 'config' in data:
                config = data['config']
                self.dimension = config['dimension']
                self.index_type = config['index_type']
                self.nlist = config['nlist']
        
        logger.info(f"Index loaded from {load_path}. Total materials: {len(self.metadata)}")


class HierarchicalMaterialsIndex:
    """
    Hierarchical search system with multiple FAISS indexes for different granularities.
    
    This implements a coarse-to-fine search strategy:
    1. Fast coarse search on material types/categories
    2. Fine-grained search within relevant categories
    """
    
    def __init__(self, dimension: int = 384):
        """Initialize hierarchical index system."""
        self.dimension = dimension
        
        # Different indexes for different search levels
        self.category_index = MaterialsFAISSIndex(dimension, "Flat")  # Small, exact
        self.materials_index = MaterialsFAISSIndex(dimension, "IVF")  # Large, approximate
        
        # Category mappings
        self.category_to_materials = {}
        
    def add_materials_hierarchical(self, 
                                 embeddings: np.ndarray,
                                 materials_data: List[Dict],
                                 categories: List[str]) -> None:
        """
        Add materials with hierarchical organization.
        
        Args:
            embeddings: Material embeddings
            materials_data: Material metadata
            categories: Category labels for each material
        """
        # Add to main materials index
        self.materials_index.add_materials(embeddings, materials_data)
        
        # Organize by categories
        for i, (material, category) in enumerate(zip(materials_data, categories)):
            if category not in self.category_to_materials:
                self.category_to_materials[category] = []
            self.category_to_materials[category].append(i)
        
        # Create category-level embeddings (average of materials in category)
        category_embeddings = []
        category_data = []
        
        for category, material_indices in self.category_to_materials.items():
            # Average embeddings for this category
            cat_embedding = np.mean(embeddings[material_indices], axis=0)
            category_embeddings.append(cat_embedding)
            
            # Category metadata
            cat_data = {
                'id': f'cat_{category}',
                'category': category,
                'material_count': len(material_indices),
                'materials': material_indices
            }
            category_data.append(cat_data)
        
        # Add categories to category index
        if category_embeddings:
            self.category_index.add_materials(
                np.array(category_embeddings), 
                category_data
            )
    
    def hierarchical_search(self, 
                          query_embedding: np.ndarray,
                          k: int = 10,
                          category_k: int = 3) -> List[SearchResult]:
        """
        Perform hierarchical search: first find relevant categories, then search within them.
        
        Args:
            query_embedding: Query vector
            k: Total number of results desired
            category_k: Number of categories to search
            
        Returns:
            List of search results
        """
        # Step 1: Find relevant categories
        category_results = self.category_index.search(query_embedding, category_k)
        
        if not category_results:
            # Fallback to direct search
            return self.materials_index.search(query_embedding, k)
        
        # Step 2: Search within relevant categories
        all_results = []
        materials_per_category = max(1, k // len(category_results))
        
        for cat_result in category_results:
            category = cat_result.metadata.get('category', '')
            if category in self.category_to_materials:
                # Search only materials in this category
                # This is a simplified version - in practice, you'd want
                # to implement category-specific sub-indexes
                cat_results = self.materials_index.search(
                    query_embedding, 
                    materials_per_category * 2  # Get extra to account for filtering
                )
                
                # Filter to only materials in this category
                filtered_results = [
                    r for r in cat_results 
                    if any(r.material_id == self.materials_index.metadata[i].get('id') 
                          for i in self.category_to_materials[category])
                ]
                
                all_results.extend(filtered_results[:materials_per_category])
        
        # Sort by score and return top k
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:k]


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample embeddings and materials
    n_materials = 1000
    dimension = 384
    
    embeddings = np.random.randn(n_materials, dimension).astype(np.float32)
    
    materials_data = []
    for i in range(n_materials):
        material = {
            'id': f'mat_{i}',
            'formula': f'Mat{i}O{(i % 5) + 1}',
            'properties': {
                'band_gap': np.random.uniform(0.1, 4.0),
                'formation_energy': np.random.uniform(-3.0, 1.0)
            },
            'source': 'synthetic'
        }
        materials_data.append(material)
    
    # Test basic FAISS index
    print("Testing Materials FAISS Index...")
    index = MaterialsFAISSIndex(dimension=dimension, index_type="IVF")
    
    # Add materials
    index.add_materials(embeddings, materials_data)
    
    # Test search
    query = np.random.randn(1, dimension).astype(np.float32)
    results = index.search(query, k=5)
    
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results):
        print(f"  {i+1}. {result.formula} (score: {result.score:.3f})")
    
    # Test performance
    stats = index.get_performance_stats()
    print(f"Performance: {stats['avg_search_time_ms']:.2f}ms average search time")
    
    # Test saving/loading
    index.save_index("test_index")
    
    new_index = MaterialsFAISSIndex(dimension=dimension)
    new_index.load_index("test_index")
    
    print(f"Loaded index with {len(new_index.metadata)} materials")