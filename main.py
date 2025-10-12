"""
Health Materials RAG System - Interactive Main Interface

Complete RAG system with LLM integration for biomedical materials discovery.

Features:
- Semantic search across 10,000+ materials
- LLM-powered natural language answers
- Smart routing (materials queries → RAG, general questions → direct LLM)
- Interactive chat interface
- Conversation history tracking

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag_pipeline.health_materials_rag_demo import HealthMaterialsRAG, LLM_AVAILABLE  # noqa: E402

def print_banner():
    """Print system banner."""
    print("\n" + "="*70)
    print("🏥  HEALTH MATERIALS RAG SYSTEM WITH LLM INTEGRATION")
    print("="*70)
    print("💎 10,000+ Biomedical Materials | 3,000+ Research Papers")
    print("🤖 LLM-Powered Answer Generation | Smart Query Routing")
    print("⚡ Sub-10ms Retrieval | Natural Language Responses")
    print("="*70 + "\n")

def print_help():
    """Print help information."""
    print("\n📖 AVAILABLE COMMANDS:")
    print("="*60)
    print("  /help          - Show this help message")
    print("  /examples      - Show example queries")
    print("  /history       - Show conversation history")
    print("  /stats         - Show system statistics")
    print("  /mode          - Show current mode (RAG/retrieval-only)")
    print("  /clear         - Clear conversation history")
    print("  /quit or exit  - Exit the system")
    print("\n💬 USAGE:")
    print("  Just type your question and press Enter!")
    print("  • Materials questions → Uses RAG (retrieval + LLM generation)")
    print("  • General questions → Direct LLM chat")
    print("="*60 + "\n")

def print_examples():
    """Print example queries."""
    print("\n💡 EXAMPLE QUERIES:")
    print("="*60)
    print("\n🔬 Materials Queries (Uses RAG):")
    print("  • What materials are used for orthopedic implants?")
    print("  • Tell me about titanium alloys for medical devices")
    print("  • Which materials have excellent biocompatibility?")
    print("  • What are the properties of hydroxyapatite?")
    print("  • Find materials for cardiovascular stents")
    print("  • What are FDA approved polymers for implants?")
    
    print("\n💭 General Questions (Direct LLM):")
    print("  • How does biocompatibility testing work?")
    print("  • What is the difference between alloys and composites?")
    print("  • Explain the importance of corrosion resistance")
    print("  • Hello, how are you?")
    print("="*60 + "\n")

def print_stats(rag_system):
    """Print system statistics."""
    print("\n📊 SYSTEM STATISTICS:")
    print("="*60)
    print(f"Database Status: {'✅ Loaded' if rag_system.is_loaded else '❌ Not Loaded'}")
    print(f"LLM Status: {'✅ Loaded' if rag_system.llm_loaded else '❌ Not Loaded'}")
    
    if rag_system.is_loaded:
        print(f"Materials: {len(rag_system.materials_db):,}")
        print(f"Research Papers: {len(rag_system.research_db):,}")
        print(f"Total Documents: {len(rag_system.texts):,}")
    
    print(f"Conversation History: {len(rag_system.conversation_history)} exchanges")
    
    if rag_system.llm_loaded:
        print("Mode: Full RAG with LLM Generation")
    else:
        print("Mode: Retrieval-Only (no LLM)")
    
    print("="*60 + "\n")

def print_history(rag_system):
    """Print conversation history."""
    if not rag_system.conversation_history:
        print("\n📝 No conversation history yet.\n")
        return
    
    print("\n📝 CONVERSATION HISTORY:")
    print("="*60)
    
    for i, conv in enumerate(rag_system.conversation_history, 1):
        print(f"\n[{i}] User: {conv['query']}")
        print(f"    Assistant ({conv['mode']}): {conv['answer'][:150]}...")
    
    print("="*60 + "\n")

def run_interactive_session(rag_system):
    """Run interactive chat session."""
    print("\n🚀 Starting interactive session...")
    print("   Type /help for commands, /quit to exit\n")
    
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() in ['/quit', 'exit', '/exit']:
                print("\n👋 Thank you for using Health Materials RAG System!")
                print("   Exiting...\n")
                break
            
            elif user_input.lower() == '/help':
                print_help()
                continue
            
            elif user_input.lower() == '/examples':
                print_examples()
                continue
            
            elif user_input.lower() == '/history':
                print_history(rag_system)
                continue
            
            elif user_input.lower() == '/stats':
                print_stats(rag_system)
                continue
            
            elif user_input.lower() == '/mode':
                if rag_system.llm_loaded:
                    print("\n✅ Mode: Full RAG with LLM Generation")
                    print("   • Material queries → Retrieval + Answer Generation")
                    print("   • General questions → Direct LLM Chat\n")
                else:
                    print("\n⚠️  Mode: Retrieval-Only (no LLM)")
                    print("   • Only semantic search available")
                    print("   • Install transformers for LLM: pip install transformers torch\n")
                continue
            
            elif user_input.lower() == '/clear':
                rag_system.conversation_history = []
                print("\n✅ Conversation history cleared!\n")
                continue
            
            # Process query
            print("\n🔄 Processing...", end='', flush=True)
            
            result = rag_system.chat(user_input)
            
            print("\r" + " "*20 + "\r", end='')  # Clear processing message
            
            # Display answer
            print(f"Assistant ({result['routing']}): {result['answer']}\n")
            
            # Show sources if available
            if 'sources' in result and result['sources']:
                print(f"📚 Sources ({len(result['sources'])}):")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"   {i}. {source['name']} (Score: {source['score']:.3f})")
                print()
            
            # Show processing time
            if 'processing_time_ms' in result:
                print(f"⚡ Processing time: {result['processing_time_ms']:.1f}ms\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Type /quit to exit properly.\n")
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")

def main():
    """Main function."""
    print_banner()
    
    # Initialize RAG system
    print("🔄 Initializing Health Materials RAG System...")
    rag_system = HealthMaterialsRAG()
    
    # Load database
    print("\n📥 Loading database...")
    rag_system.load_database()
    
    # Offer to load LLM
    if LLM_AVAILABLE:
        print("\n🤖 LLM Integration Available!")
        print("   Would you like to load the LLM for answer generation?")
        print("   (This takes 1-2 minutes but enables natural language answers)")
        
        choice = input("\n   Load LLM? [Y/n]: ").strip().lower()
        
        if choice in ['', 'y', 'yes']:
            # Ask for model choice
            print("\n   Select LLM model:")
            print("   1. Phi-3-mini (3.8B) - Best quality, recommended")
            print("   2. Flan-T5-large (770M) - Faster, good quality")
            print("   3. Custom model name")
            
            model_choice = input("\n   Choice [1]: ").strip()
            
            if model_choice == '2':
                model_name = "google/flan-t5-large"
            elif model_choice == '3':
                model_name = input("   Enter model name: ").strip()
            else:
                model_name = "microsoft/Phi-3-mini-4k-instruct"
            
            rag_system.load_llm(model_name)
        else:
            print("\n   ⚠️  Running in retrieval-only mode (semantic search only)")
    else:
        print("\n⚠️  Transformers not installed. Running in retrieval-only mode.")
        print("   Install with: pip install transformers torch")
    
    # Show system status
    print_stats(rag_system)
    
    # Print usage instructions
    print("💬 You can now chat with the system!")
    print("   • Ask about biomedical materials")
    print("   • Search the database")
    print("   • Get LLM-powered answers")
    print("   • Type /help for commands\n")
    
    # Run interactive session
    run_interactive_session(rag_system)

if __name__ == "__main__":
    main()
