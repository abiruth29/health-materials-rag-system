# Health Materials RAG System - Main Application
"""
Health Materials RAG System - Production Implementation
=====================================================

This is the main application entry point for the Health Materials RAG System.
The system provides a complete biomedical materials discovery platform with:

1. Data Acquisition Pipeline (src/data_acquisition/)
2. Knowledge Graph Construction (src/knowledge_graph/) 
3. Vector Embedding Engine (src/embedding_engine/)
4. RAG Pipeline Implementation (src/rag_pipeline/)

Usage:
    python health_materials_rag_system.py [command]
    
Commands:
    setup    - Initialize database and embeddings
    search   - Interactive search interface
    api      - Start REST API server
    demo     - Run demonstration
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.health_materials_rag_setup import main as setup_database
from src.rag_pipeline.health_materials_rag_demo import HealthMaterialsRAG
from src.embedding_engine.api_server import start_api_server

print("🏥 HEALTH MATERIALS RAG SYSTEM - PRODUCTION VERSION")
print("="*60)

class HealthMaterialsSystem:
    """Main Health Materials RAG System Controller"""
    
    def __init__(self):
        self.rag_system = None
        self.initialized = False
    
    def setup_system(self):
        """Complete system setup - database, embeddings, index"""
        print("🔧 Setting up Health Materials RAG System...")
        print("This will:")
        print("  1. Process 10,000+ materials from BIOMATDB, NIST, PubMed")
        print("  2. Generate vector embeddings") 
        print("  3. Build FAISS search index")
        print("  4. Create knowledge graph")
        
        confirm = input("\nProceed with setup? (y/N): ")
        if confirm.lower() != 'y':
            print("Setup cancelled.")
            return
            
        # Run the complete setup
        setup_database()
        print("✅ System setup complete!")
    
    def initialize_rag_system(self):
        """Initialize the RAG system for use"""
        if not self.initialized:
            self.rag_system = HealthMaterialsRAG()
            self.rag_system.load_database()
            self.initialized = True
            print("✅ RAG System initialized and ready!")
        return self.rag_system
    
    def interactive_search(self):
        """Interactive search interface"""
        rag = self.initialize_rag_system()
        
        print("\n🔍 Interactive Search Interface")
        print("="*50)
        print("Enter queries to search the health materials database.")
        print("Examples:")
        print("  • 'titanium alloys for orthopedic implants'")
        print("  • 'biocompatible materials for cardiac stents'") 
        print("  • 'FDA approved dental implant materials'")
        print("Type 'quit' to exit.\n")
        
        while True:
            query = input("🔍 Search query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
                
            if not query:
                continue
                
            print("🔄 Searching...")
            results = rag.search(query, top_k=5)
            
            print(f"⚡ Found {len(results['results'])} results in {results['retrieval_time_ms']:.1f}ms")
            print("-" * 50)
            
            for i, result in enumerate(results['results'], 1):
                meta = result['metadata']
                print(f"\n{i}. {meta.get('name', meta.get('title', 'Unknown'))}")
                print(f"   Source: {meta['source']} | Score: {result['similarity_score']:.3f}")
                if meta['type'] == 'material':
                    print(f"   Applications: {meta.get('applications', 'N/A')}")
                    print(f"   Biocompatibility: {meta.get('biocompatibility', 'N/A')}")
                else:
                    print(f"   Materials: {meta.get('materials', 'N/A')}")
            print()
    
    def start_api_server(self):
        """Start the REST API server"""
        print("🚀 Starting Health Materials RAG API Server...")
        rag = self.initialize_rag_system()
        start_api_server(rag)
    
    def run_demo(self):
        """Run the system demonstration"""
        from main_demo import main as demo_main
        demo_main()
    
    def show_system_status(self):
        """Show current system status"""
        print("📊 SYSTEM STATUS")
        print("="*50)
        
        # Check database files
        data_path = Path("data/rag_optimized")
        if data_path.exists():
            files = list(data_path.glob("*"))
            print(f"✅ Database: {len(files)} files found")
            
            # Check database summary
            summary_file = data_path / "database_summary.json"
            if summary_file.exists():
                import json
                with open(summary_file) as f:
                    summary = json.load(f)
                print(f"   • Materials: {summary['total_materials']:,}")
                print(f"   • Research: {summary['total_research']:,}")
                print(f"   • Embeddings: {summary['total_embeddings']:,}")
            else:
                print("   • Database summary not found")
        else:
            print("❌ Database: Not set up (run 'python health_materials_rag_system.py setup')")
        
        # Check modules
        print("\n🔧 MODULES STATUS")
        modules = [
            ("Data Acquisition", "src/data_acquisition"),
            ("Knowledge Graph", "src/knowledge_graph"), 
            ("Embedding Engine", "src/embedding_engine"),
            ("RAG Pipeline", "src/rag_pipeline")
        ]
        
        for name, path in modules:
            if Path(path).exists():
                py_files = len(list(Path(path).glob("*.py")))
                print(f"   ✅ {name}: {py_files} Python files")
            else:
                print(f"   ❌ {name}: Not found")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Health Materials RAG System")
    parser.add_argument('command', 
                       choices=['setup', 'search', 'api', 'demo', 'status'],
                       help='Command to execute')
    
    if len(sys.argv) == 1:
        print("🏥 Health Materials RAG System - Production Implementation")
        print("\nAvailable commands:")
        print("  setup    - Initialize database and embeddings")  
        print("  search   - Interactive search interface")
        print("  api      - Start REST API server")
        print("  demo     - Run demonstration")
        print("  status   - Show system status")
        print(f"\nUsage: python {sys.argv[0]} [command]")
        return
    
    args = parser.parse_args()
    system = HealthMaterialsSystem()
    
    try:
        if args.command == 'setup':
            system.setup_system()
        elif args.command == 'search':
            system.interactive_search()
        elif args.command == 'api':
            system.start_api_server()
        elif args.command == 'demo':
            system.run_demo()
        elif args.command == 'status':
            system.show_system_status()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Run 'python health_materials_rag_system.py status' to check system state")

if __name__ == "__main__":
    main()