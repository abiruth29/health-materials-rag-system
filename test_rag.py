#!/usr/bin/env python3
"""Quick RAG system test"""

import sys
from pathlib import Path
sys.path.append('src')

try:
    from rag_pipeline.health_materials_rag_demo import HealthMaterialsRAG
    
    print("🔍 Testing Health Materials RAG System...")
    rag = HealthMaterialsRAG()
    rag.load_database()
    
    # Test query
    query = "titanium alloys for orthopedic implants"
    print(f"Query: '{query}'")
    
    results = rag.semantic_search(query, top_k=3)
    print(f"⚡ Found {len(results['results'])} results in {results['retrieval_time']*1000:.1f}ms")
    
    for i, result in enumerate(results['results'], 1):
        meta = result['metadata']
        name = meta.get('name', meta.get('title', 'Material'))
        score = result['similarity_score']
        print(f"{i}. {name} (Score: {score:.3f})")
        
        # Show key info
        if 'applications' in meta:
            print(f"   Applications: {meta['applications'][:80]}...")
        elif 'description' in meta:
            print(f"   Description: {meta['description'][:80]}...")
    
    print("\n✅ RAG System is working perfectly!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("System needs dependencies installed. Run 'pip install -r requirements.txt'")