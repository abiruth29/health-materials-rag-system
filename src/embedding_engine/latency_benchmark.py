"""
Latency Benchmarking for Materials Retrieval System

This module provides comprehensive benchmarking tools to ensure the retrieval
system meets sub-100ms performance targets.

Key Features:
- Search latency measurement
- Throughput testing
- Index performance comparison
- Cache effectiveness analysis
- Load testing simulation
"""

import asyncio
import time
import statistics
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import requests
import aiohttp

from sentence_transformers import SentenceTransformer
from .faiss_index import MaterialsFAISSIndex
from .embedding_trainer import MaterialsEmbeddingTrainer

import logging
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    num_queries: int
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_qps: float
    success_rate: float
    memory_usage_mb: Optional[float] = None
    cache_hit_rate: Optional[float] = None


class MaterialsLatencyBenchmark:
    """
    Comprehensive benchmarking suite for materials retrieval performance.
    
    Tests various aspects of the system to ensure sub-100ms retrieval goals.
    """
    
    def __init__(self, 
                 model_path: str = "all-MiniLM-L6-v2",
                 index_path: Optional[str] = None):
        """
        Initialize benchmark suite.
        
        Args:
            model_path: Path to embedding model
            index_path: Path to FAISS index
        """
        self.model_path = model_path
        self.index_path = index_path
        
        # Components
        self.embedding_model = None
        self.faiss_index = None
        
        # Test queries for benchmarking
        self.test_queries = [
            "high thermal conductivity materials",
            "low band gap semiconductors",
            "perovskite materials for solar cells",
            "magnetic materials with high Curie temperature",
            "lightweight materials for aerospace",
            "superconducting materials",
            "materials with high dielectric constant",
            "corrosion resistant alloys",
            "materials for battery electrodes",
            "photocatalytic materials for water splitting",
            "materials with negative thermal expansion",
            "transparent conducting oxides",
            "materials for thermoelectric applications",
            "high-strength ceramics",
            "materials with large magnetocaloric effect"
        ]
        
        self.results = []
    
    def setup(self):
        """Initialize components for benchmarking."""
        logger.info("Setting up benchmark components...")
        
        # Load embedding model
        if Path(self.model_path).exists():
            self.embedding_model = SentenceTransformer(self.model_path)
        else:
            self.embedding_model = SentenceTransformer(self.model_path)
        
        # Load or create FAISS index
        self.faiss_index = MaterialsFAISSIndex()
        if self.index_path and Path(self.index_path).exists():
            self.faiss_index.load_index(self.index_path)
            logger.info(f"Loaded index with {len(self.faiss_index.metadata)} materials")
        else:
            # Create synthetic test data
            self._create_synthetic_index()
    
    def _create_synthetic_index(self, n_materials: int = 10000):
        """Create synthetic materials data for benchmarking."""
        logger.info(f"Creating synthetic index with {n_materials} materials...")
        
        # Generate synthetic materials
        materials_data = []
        for i in range(n_materials):
            material = {
                'id': f'synth_mat_{i}',
                'formula': f'Mat{i % 100}O{(i % 5) + 1}',
                'properties': {
                    'band_gap': np.random.uniform(0.1, 4.0),
                    'formation_energy': np.random.uniform(-3.0, 1.0),
                    'thermal_conductivity': np.random.uniform(0.1, 100.0),
                    'density': np.random.uniform(1.0, 20.0)
                },
                'structure': {
                    'crystal_system': np.random.choice([
                        'cubic', 'tetragonal', 'orthorhombic', 'hexagonal'
                    ])
                },
                'source': 'synthetic'
            }
            materials_data.append(material)
        
        # Generate embeddings
        texts = []
        for material in materials_data:
            text = f"{material['formula']} {material['structure']['crystal_system']} "
            text += " ".join([f"{k}: {v}" for k, v in material['properties'].items()])
            texts.append(text)
        
        embeddings = self.embedding_model.encode(texts)
        
        # Add to index
        self.faiss_index.add_materials(embeddings, materials_data)
        logger.info(f"Created synthetic index with {len(materials_data)} materials")
    
    def benchmark_search_latency(self, num_queries: int = 100) -> BenchmarkResult:
        """
        Benchmark search latency for individual queries.
        
        Args:
            num_queries: Number of test queries to run
            
        Returns:
            BenchmarkResult with latency statistics
        """
        logger.info(f"Benchmarking search latency with {num_queries} queries...")
        
        latencies = []
        successful_queries = 0
        
        # Prepare query embeddings
        query_texts = (self.test_queries * (num_queries // len(self.test_queries) + 1))[:num_queries]
        
        for query_text in query_texts:
            try:
                start_time = time.time()
                
                # Generate embedding
                query_embedding = self.embedding_model.encode([query_text])
                
                # Perform search
                results = self.faiss_index.search(query_embedding, k=10)
                
                # Record latency
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
                successful_queries += 1
                
            except Exception as e:
                logger.warning(f"Query failed: {e}")
        
        # Calculate statistics
        if latencies:
            result = BenchmarkResult(
                test_name="search_latency",
                num_queries=num_queries,
                avg_latency_ms=statistics.mean(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                throughput_qps=successful_queries / (sum(latencies) / 1000),
                success_rate=successful_queries / num_queries
            )
        else:
            result = BenchmarkResult(
                test_name="search_latency",
                num_queries=num_queries,
                avg_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_qps=0,
                success_rate=0
            )
        
        self.results.append(result)
        logger.info(f"Average latency: {result.avg_latency_ms:.2f}ms, P95: {result.p95_latency_ms:.2f}ms")
        
        return result
    
    def benchmark_batch_search(self, batch_sizes: List[int] = [1, 5, 10, 20, 50]) -> List[BenchmarkResult]:
        """
        Benchmark batch search performance.
        
        Args:
            batch_sizes: Different batch sizes to test
            
        Returns:
            List of BenchmarkResult for each batch size
        """
        logger.info("Benchmarking batch search performance...")
        
        batch_results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            latencies = []
            num_batches = 20  # Number of batches to test
            
            for _ in range(num_batches):
                # Prepare batch queries
                query_texts = np.random.choice(self.test_queries, batch_size, replace=True)
                
                try:
                    start_time = time.time()
                    
                    # Generate embeddings for batch
                    query_embeddings = self.embedding_model.encode(query_texts)
                    
                    # Perform batch search
                    results = self.faiss_index.batch_search(query_embeddings, k=10)
                    
                    # Record latency per query in batch
                    total_latency_ms = (time.time() - start_time) * 1000
                    latency_per_query = total_latency_ms / batch_size
                    latencies.append(latency_per_query)
                    
                except Exception as e:
                    logger.warning(f"Batch search failed: {e}")
            
            # Calculate statistics
            if latencies:
                result = BenchmarkResult(
                    test_name=f"batch_search_size_{batch_size}",
                    num_queries=num_batches * batch_size,
                    avg_latency_ms=statistics.mean(latencies),
                    min_latency_ms=min(latencies),
                    max_latency_ms=max(latencies),
                    p95_latency_ms=np.percentile(latencies, 95),
                    p99_latency_ms=np.percentile(latencies, 99),
                    throughput_qps=batch_size / (statistics.mean(latencies) / 1000),
                    success_rate=1.0
                )
                batch_results.append(result)
                self.results.append(result)
        
        return batch_results
    
    def benchmark_concurrent_search(self, 
                                  concurrent_users: List[int] = [1, 5, 10, 20, 50]) -> List[BenchmarkResult]:
        """
        Benchmark concurrent search performance.
        
        Args:
            concurrent_users: Different concurrency levels to test
            
        Returns:
            List of BenchmarkResult for each concurrency level
        """
        logger.info("Benchmarking concurrent search performance...")
        
        concurrent_results = []
        
        for num_users in concurrent_users:
            logger.info(f"Testing {num_users} concurrent users")
            
            def single_user_search():
                """Single user search simulation."""
                query_text = np.random.choice(self.test_queries)
                
                start_time = time.time()
                try:
                    query_embedding = self.embedding_model.encode([query_text])
                    results = self.faiss_index.search(query_embedding, k=10)
                    return (time.time() - start_time) * 1000  # Return latency in ms
                except Exception:
                    return None
            
            # Run concurrent searches
            with ThreadPoolExecutor(max_workers=num_users) as executor:
                futures = [executor.submit(single_user_search) for _ in range(num_users * 10)]
                latencies = [f.result() for f in futures if f.result() is not None]
            
            # Calculate statistics
            if latencies:
                result = BenchmarkResult(
                    test_name=f"concurrent_search_{num_users}_users",
                    num_queries=len(latencies),
                    avg_latency_ms=statistics.mean(latencies),
                    min_latency_ms=min(latencies),
                    max_latency_ms=max(latencies),
                    p95_latency_ms=np.percentile(latencies, 95),
                    p99_latency_ms=np.percentile(latencies, 99),
                    throughput_qps=len(latencies) / (sum(latencies) / 1000),
                    success_rate=len(latencies) / (num_users * 10)
                )
                concurrent_results.append(result)
                self.results.append(result)
        
        return concurrent_results
    
    async def benchmark_api_server(self, 
                                 server_url: str = "http://localhost:8000",
                                 num_requests: int = 100) -> BenchmarkResult:
        """
        Benchmark the API server performance.
        
        Args:
            server_url: URL of the running API server
            num_requests: Number of requests to send
            
        Returns:
            BenchmarkResult for API performance
        """
        logger.info(f"Benchmarking API server at {server_url}")
        
        latencies = []
        successful_requests = 0
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for i in range(num_requests):
                query_text = self.test_queries[i % len(self.test_queries)]
                
                async def make_request(query):
                    try:
                        start_time = time.time()
                        
                        payload = {
                            "query": query,
                            "k": 10,
                            "use_cache": False  # Test without cache for pure performance
                        }
                        
                        async with session.post(f"{server_url}/search", json=payload) as response:
                            if response.status == 200:
                                await response.json()
                                latency_ms = (time.time() - start_time) * 1000
                                return latency_ms
                    except Exception as e:
                        logger.warning(f"API request failed: {e}")
                    return None
                
                tasks.append(make_request(query_text))
            
            # Execute all requests
            results = await asyncio.gather(*tasks)
            latencies = [r for r in results if r is not None]
            successful_requests = len(latencies)
        
        # Calculate statistics
        if latencies:
            result = BenchmarkResult(
                test_name="api_server",
                num_queries=num_requests,
                avg_latency_ms=statistics.mean(latencies),
                min_latency_ms=min(latencies),
                max_latency_ms=max(latencies),
                p95_latency_ms=np.percentile(latencies, 95),
                p99_latency_ms=np.percentile(latencies, 99),
                throughput_qps=successful_requests / (sum(latencies) / 1000),
                success_rate=successful_requests / num_requests
            )
        else:
            result = BenchmarkResult(
                test_name="api_server",
                num_queries=num_requests,
                avg_latency_ms=0,
                min_latency_ms=0,
                max_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
                throughput_qps=0,
                success_rate=0
            )
        
        self.results.append(result)
        return result
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Returns:
            Dictionary with all benchmark results
        """
        logger.info("Running full benchmark suite...")
        
        # Setup components
        self.setup()
        
        # Run individual benchmarks
        latency_result = self.benchmark_search_latency(num_queries=100)
        batch_results = self.benchmark_batch_search()
        concurrent_results = self.benchmark_concurrent_search()
        
        # Compile results
        results_summary = {
            "single_search": asdict(latency_result),
            "batch_search": [asdict(r) for r in batch_results],
            "concurrent_search": [asdict(r) for r in concurrent_results],
            "summary": {
                "meets_100ms_target": latency_result.p95_latency_ms < 100,
                "avg_latency_ms": latency_result.avg_latency_ms,
                "p95_latency_ms": latency_result.p95_latency_ms,
                "max_throughput_qps": max([r.throughput_qps for r in self.results])
            }
        }
        
        return results_summary
    
    def generate_performance_report(self, output_path: str = "benchmark_report.json"):
        """
        Generate a comprehensive performance report.
        
        Args:
            output_path: Path to save the report
        """
        logger.info("Generating performance report...")
        
        # Run benchmarks if not already done
        if not self.results:
            self.run_full_benchmark_suite()
        
        # Create visualizations
        self._create_performance_plots()
        
        # Save detailed results
        report = {
            "timestamp": time.time(),
            "system_info": {
                "model_path": self.model_path,
                "index_path": self.index_path,
                "total_materials": len(self.faiss_index.metadata) if self.faiss_index else 0
            },
            "results": [asdict(r) for r in self.results],
            "summary": self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Performance report saved to {output_path}")
    
    def _create_performance_plots(self):
        """Create performance visualization plots."""
        try:
            # Latency distribution plot
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Latency comparison
            plt.subplot(2, 2, 1)
            test_names = [r.test_name for r in self.results if 'search' in r.test_name]
            avg_latencies = [r.avg_latency_ms for r in self.results if 'search' in r.test_name]
            p95_latencies = [r.p95_latency_ms for r in self.results if 'search' in r.test_name]
            
            x = range(len(test_names))
            plt.bar([i - 0.2 for i in x], avg_latencies, 0.4, label='Average', alpha=0.7)
            plt.bar([i + 0.2 for i in x], p95_latencies, 0.4, label='P95', alpha=0.7)
            plt.axhline(y=100, color='r', linestyle='--', label='100ms Target')
            plt.xticks(x, test_names, rotation=45)
            plt.ylabel('Latency (ms)')
            plt.title('Search Latency Comparison')
            plt.legend()
            
            # Plot 2: Throughput comparison
            plt.subplot(2, 2, 2)
            throughputs = [r.throughput_qps for r in self.results if 'search' in r.test_name]
            plt.bar(test_names, throughputs, alpha=0.7)
            plt.xticks(rotation=45)
            plt.ylabel('Throughput (QPS)')
            plt.title('Search Throughput Comparison')
            
            # Plot 3: Batch size performance
            batch_results = [r for r in self.results if 'batch_search' in r.test_name]
            if batch_results:
                plt.subplot(2, 2, 3)
                batch_sizes = [int(r.test_name.split('_')[-1]) for r in batch_results]
                batch_latencies = [r.avg_latency_ms for r in batch_results]
                plt.plot(batch_sizes, batch_latencies, 'o-')
                plt.xlabel('Batch Size')
                plt.ylabel('Latency per Query (ms)')
                plt.title('Batch Size vs Latency')
            
            # Plot 4: Concurrent users performance
            concurrent_results = [r for r in self.results if 'concurrent' in r.test_name]
            if concurrent_results:
                plt.subplot(2, 2, 4)
                user_counts = [int(r.test_name.split('_')[2]) for r in concurrent_results]
                concurrent_latencies = [r.avg_latency_ms for r in concurrent_results]
                plt.plot(user_counts, concurrent_latencies, 'o-')
                plt.xlabel('Concurrent Users')
                plt.ylabel('Average Latency (ms)')
                plt.title('Concurrency vs Latency')
            
            plt.tight_layout()
            plt.savefig('performance_benchmark.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Performance plots saved to performance_benchmark.png")
            
        except Exception as e:
            logger.warning(f"Could not create plots: {e}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
        
        search_results = [r for r in self.results if 'search' in r.test_name]
        
        return {
            "total_tests": len(self.results),
            "avg_latency_ms": statistics.mean([r.avg_latency_ms for r in search_results]),
            "best_p95_latency_ms": min([r.p95_latency_ms for r in search_results]),
            "max_throughput_qps": max([r.throughput_qps for r in search_results]),
            "meets_100ms_target": all([r.p95_latency_ms < 100 for r in search_results]),
            "success_rate": statistics.mean([r.success_rate for r in self.results])
        }


# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Materials Retrieval System")
    parser.add_argument("--model-path", default="all-MiniLM-L6-v2",
                       help="Path to embedding model")
    parser.add_argument("--index-path", help="Path to FAISS index")
    parser.add_argument("--output", default="benchmark_report.json",
                       help="Output file for benchmark report")
    parser.add_argument("--api-url", help="API server URL for API benchmarking")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark with fewer queries")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize benchmark
    benchmark = MaterialsLatencyBenchmark(
        model_path=args.model_path,
        index_path=args.index_path
    )
    
    # Run benchmarks
    if args.quick:
        benchmark.setup()
        result = benchmark.benchmark_search_latency(num_queries=20)
        print(f"Quick benchmark: {result.avg_latency_ms:.2f}ms average, {result.p95_latency_ms:.2f}ms P95")
    else:
        # Full benchmark suite
        results = benchmark.run_full_benchmark_suite()
        
        # Test API if URL provided
        if args.api_url:
            api_result = asyncio.run(benchmark.benchmark_api_server(args.api_url))
            print(f"API benchmark: {api_result.avg_latency_ms:.2f}ms average")
        
        # Generate report
        benchmark.generate_performance_report(args.output)
        
        # Print summary
        summary = benchmark._generate_summary()
        print("\\n=== Benchmark Summary ===")
        print(f"Average latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"Best P95 latency: {summary['best_p95_latency_ms']:.2f}ms")
        print(f"Max throughput: {summary['max_throughput_qps']:.2f} QPS")
        print(f"Meets 100ms target: {'✓' if summary['meets_100ms_target'] else '✗'}")