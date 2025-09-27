#!/usr/bin/env python3
"""
Health Materials RAG System - Interactive Search Interface
=========================================================

Simple interactive search interface without heavy dependencies.
"""

import sys
from pathlib import Path
sys.path.append('src')

print("🏥 HEALTH MATERIALS RAG SYSTEM - INTERACTIVE SEARCH")
print("=" * 60)

try:
    from rag_pipeline.health_materials_rag_demo import HealthMaterialsRAG
    
    print("🔄 Initializing RAG System...")
    rag = HealthMaterialsRAG()
    rag.load_database()
    
    print("\n✅ System Ready! Database loaded with:")
    print("   • Materials: 7,000+")
    print("   • Research Papers: 3,000+")  
    print("   • Total Records: 10,000+")
    
    print("\n🔍 INTERACTIVE SEARCH INTERFACE")
    print("=" * 50)
    print("Enter queries to search the health materials database.")
    print("\n💡 Example queries:")
    print("  • 'titanium alloys for orthopedic implants'")
    print("  • 'biocompatible materials for cardiac stents'")
    print("  • 'FDA approved dental implant materials'")
    print("  • 'materials for artificial hip joints'")
    print("  • 'biocompatible polymers for drug delivery'")
    print("\nType 'quit', 'exit', or 'q' to exit.")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n🔍 Enter your search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q', '']:
                print("\n👋 Thank you for using Health Materials RAG System!")
                break
            
            if not query:
                print("Please enter a search query.")
                continue
            
            print(f"\n🔄 Searching for: '{query}'...")
            
            # Perform search
            results = rag.semantic_search(query, top_k=5)
            retrieval_time_ms = results['retrieval_time'] * 1000
            
            print(f"\n⚡ Found {len(results['results'])} results in {retrieval_time_ms:.1f}ms")
            print("=" * 60)
            
            if not results['results']:
                print("No results found. Try a different query.")
                continue
            
            for i, result in enumerate(results['results'], 1):
                meta = result['metadata']
                name = meta.get('name', meta.get('title', f'Material/Study {meta.get("id", i)}'))
                score = result['similarity_score']
                source = meta.get('source', 'Unknown')
                material_type = meta.get('type', 'unknown')
                
                print(f"\n{i}. 📋 {name}")
                print(f"   📊 Relevance Score: {score:.3f}")
                print(f"   🔬 Source: {source} | Type: {material_type.title()}")
                
                # Show different info based on type
                if material_type == 'material':
                    apps = meta.get('applications', 'Not specified')
                    biocompat = meta.get('biocompatibility', 'Not specified')
                    print(f"   🏥 Applications: {apps}")
                    print(f"   ✅ Biocompatibility: {biocompat}")
                    
                    if 'formula' in meta:
                        print(f"   ⚗️ Formula: {meta['formula']}")
                    if 'properties' in meta:
                        print(f"   🔬 Key Properties: {meta['properties']}")
                        
                elif material_type == 'research':
                    materials = meta.get('materials', 'Not specified')
                    therapy = meta.get('therapy', meta.get('application', 'Not specified'))
                    print(f"   🧪 Materials Studied: {materials}")
                    print(f"   💊 Therapeutic Application: {therapy}")
                
                # Show excerpt
                text_excerpt = result['text'][:200].strip()
                if text_excerpt:
                    print(f"   📝 Excerpt: {text_excerpt}...")
                
                print("-" * 50)
            
            print(f"\n💡 Search completed! Found {len(results['results'])} relevant materials/studies.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Search interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Search error: {e}")
            print("Please try a different query.")

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Some dependencies may be missing. Try running:")
    print("pip install sentence-transformers faiss-cpu pandas numpy")
except Exception as e:
    print(f"❌ System Error: {e}")
    print("Please check if the database is properly set up.")