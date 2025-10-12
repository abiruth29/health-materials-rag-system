"""
NER Integration Demo for Health Materials RAG Pipeline

Demonstrates seamless NER entity extraction and validation
integrated into the RAG query flow.
"""

import asyncio
import logging
import json
from pathlib import Path

# Setup path
import sys
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline.rag_pipeline import MaterialsRAGPipeline
from rag_pipeline.ner_validator import create_test_cases

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(title: str = None):
    """Print a visual separator."""
    if title:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}\n")
    else:
        print(f"\n{'-'*60}\n")


def display_entities(entities, title="Entities"):
    """Display extracted entities in a formatted way."""
    if not entities:
        print(f"  {title}: None extracted")
        return
    
    print(f"  {title} ({len(entities)} found):")
    
    # Group by label
    by_label = {}
    for entity in entities:
        label = entity.label
        if label not in by_label:
            by_label[label] = []
        by_label[label].append(entity)
    
    # Display grouped
    for label, ents in sorted(by_label.items()):
        print(f"\n    [{label}]")
        for e in ents:
            confidence_bar = '█' * int(e.confidence * 10)
            print(f"      • {e.text:<30} (confidence: {e.confidence:.2f} {confidence_bar})")


def display_rag_result(result, show_entities=True):
    """Display RAG result with entity information."""
    print(f"Query: {result.query}")
    print(f"\nAnswer (Confidence: {result.confidence:.2f}):")
    print(f"  {result.answer}")
    
    if show_entities and result.query_entities is not None:
        print_separator()
        display_entities(result.query_entities, "Query Entities")
        print()
        display_entities(result.answer_entities, "Answer Entities")
        
        # Show entity overlap
        if result.query_entities and result.answer_entities:
            query_texts = {e.text.lower() for e in result.query_entities}
            answer_texts = {e.text.lower() for e in result.answer_entities}
            overlap = query_texts.intersection(answer_texts)
            
            if overlap:
                print(f"\n  Entity Overlap (mentioned in both query and answer):")
                for text in sorted(overlap):
                    print(f"    • {text}")
        
        # Show validation metrics
        if result.entity_validation:
            validation = result.entity_validation
            print(f"\n  Validation Metrics:")
            print(f"    Total entities: {validation.get('total_entities', 0)}")
            print(f"    Avg confidence: {validation.get('avg_confidence', 0):.2f}")
            
            if validation.get('low_confidence_entities'):
                print(f"    ⚠ Low confidence entities: {len(validation['low_confidence_entities'])}")
            
            if validation.get('warnings'):
                print(f"    ⚠ Warnings:")
                for warning in validation['warnings']:
                    print(f"      - {warning}")
    
    print(f"\n  Processing time: {result.processing_time_ms:.2f}ms")
    print(f"  Context materials: {len(result.context_materials)}")


async def demo_basic_ner_extraction():
    """Demo 1: Basic NER extraction in queries."""
    print_separator("Demo 1: Basic NER Extraction")
    
    # Initialize pipeline with NER enabled
    rag = MaterialsRAGPipeline(
        llm_model_name="gpt2",
        embedding_model_name="all-MiniLM-L6-v2",
        use_gpu=False,
        enable_ner=True  # Enable NER
    )
    
    await rag.initialize()
    
    # Test queries with different entity types
    test_queries = [
        "What are the properties of Ti-6Al-4V titanium alloy for orthopedic implants?",
        "How does 316L stainless steel perform in cardiac stents?",
        "Compare PEEK polymer and titanium for spinal fusion cages",
        "What materials are FDA approved for dental implants with Young's modulus above 100 GPa?"
    ]
    
    for query in test_queries:
        print(f"\n{'─'*60}")
        result = await rag.query(query, context_k=3)
        display_rag_result(result, show_entities=True)
    
    return rag


async def demo_entity_validation():
    """Demo 2: Entity validation with gold standard."""
    print_separator("Demo 2: Entity Validation with Gold Standard")
    
    from rag_pipeline.ner_validator import NERExtractor, NERValidator
    
    # Create NER components
    extractor = NERExtractor(use_transformer=False)
    validator = NERValidator()
    
    # Get test cases with gold standard annotations
    test_cases = create_test_cases()
    
    print(f"Running validation on {len(test_cases)} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Text: {test_case['text'][:80]}...")
        
        # Extract entities
        predicted = extractor.extract(test_case['text'])
        gold = test_case['gold_entities']
        
        # Validate
        validation = validator.compare_with_gold_standard(
            predicted,
            gold,
            match_type='partial'  # Allow partial matches
        )
        
        print(f"\n  Results:")
        print(f"    Precision: {validation.precision:.2%}")
        print(f"    Recall:    {validation.recall:.2%}")
        print(f"    F1 Score:  {validation.f1:.2%}")
        print(f"    True Positives:  {validation.true_positives}")
        print(f"    False Positives: {validation.false_positives}")
        print(f"    False Negatives: {validation.false_negatives}")
        
        if validation.false_positives > 0:
            print(f"\n    False Positives:")
            for fp in validation.entity_errors.get('false_positives', [])[:3]:
                print(f"      • {fp.text} ({fp.label})")
        
        if validation.false_negatives > 0:
            print(f"\n    False Negatives:")
            for fn in validation.entity_errors.get('false_negatives', [])[:3]:
                print(f"      • {fn.text} ({fn.label})")
        
        print()


async def demo_entity_statistics():
    """Demo 3: Entity statistics and tracking."""
    print_separator("Demo 3: Entity Statistics and Tracking")
    
    # Initialize pipeline
    rag = MaterialsRAGPipeline(
        llm_model_name="gpt2",
        use_gpu=False,
        enable_ner=True
    )
    
    await rag.initialize()
    
    # Run multiple queries
    queries = [
        "Properties of Ti-6Al-4V for orthopedic implants",
        "316L stainless steel biocompatibility with ISO 10993",
        "PEEK polymer mechanical properties for spinal cages",
        "Cobalt-chromium alloy for hip replacements",
        "Hydroxyapatite coating for dental implants"
    ]
    
    print("Processing queries to gather statistics...\n")
    results = []
    
    for query in queries:
        result = await rag.query(query, context_k=2)
        results.append(result)
        print(f"✓ Processed: {query[:50]}...")
    
    # Get overall statistics
    print_separator()
    stats = rag.get_performance_stats()
    
    print("Overall Performance Statistics:")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Avg processing time: {stats['avg_processing_time_ms']:.2f}ms")
    print(f"  Avg confidence: {stats['avg_confidence']:.2%}")
    
    if 'ner_stats' in stats:
        ner_stats = stats['ner_stats']
        print(f"\nNER Statistics:")
        print(f"  Avg entities per query: {ner_stats['avg_entities_per_query']:.1f}")
        print(f"  Avg entities per answer: {ner_stats['avg_entities_per_answer']:.1f}")
        print(f"  Total entities extracted: {ner_stats['total_entities_extracted']}")
        
        if 'validator_stats' in ner_stats:
            val_stats = ner_stats['validator_stats']
            print(f"\nValidator Statistics:")
            print(f"  Total validations: {val_stats['total_validations']}")
            
            if val_stats['entity_type_stats']:
                print(f"\n  Entity Type Distribution:")
                for etype, estats in val_stats['entity_type_stats'].items():
                    print(f"    {etype}: {estats['count']} (avg confidence: {estats['avg_confidence']:.2f})")
    
    # Show detailed entity summary for one result
    print_separator()
    print("Detailed Entity Summary for Last Query:")
    summary = rag.get_entity_summary(results[-1])
    print(json.dumps(summary, indent=2, default=str))


async def demo_entity_enhanced_search():
    """Demo 4: Using entities to enhance search."""
    print_separator("Demo 4: Entity-Enhanced Search")
    
    from rag_pipeline.ner_validator import NERExtractor
    
    extractor = NERExtractor(use_transformer=False)
    
    # Complex query with multiple entities
    query = "Compare Ti-6Al-4V and 316L stainless steel for FDA-approved orthopedic implants with Young's modulus above 100 GPa"
    
    print(f"Query: {query}\n")
    
    # Extract entities with context
    entity_context = extractor.extract_with_context(query, query_type='material_search')
    
    print(f"Entity Analysis:")
    print(f"  Total entities: {entity_context['entity_count']}")
    print(f"  Entity types: {', '.join(entity_context['entity_types'])}")
    print(f"  Coverage score: {entity_context['coverage_score']:.2%}")
    
    print(f"\n  Key Entities:")
    for entity in entity_context['key_entities']:
        print(f"    • {entity.text} ({entity.label}) - confidence: {entity.confidence:.2f}")
    
    print(f"\n  Entities by Type:")
    for etype, entities in entity_context['by_type'].items():
        print(f"    [{etype}]: {', '.join(e.text for e in entities)}")
    
    print(f"\n  Search Strategy:")
    materials = [e.text for e in entity_context['by_type'].get('MATERIAL', [])]
    properties = [e.text for e in entity_context['by_type'].get('PROPERTY', []) + 
                  entity_context['by_type'].get('MEASUREMENT', [])]
    
    if materials:
        print(f"    → Focus search on materials: {', '.join(materials)}")
    if properties:
        print(f"    → Filter by properties: {', '.join(properties)}")


async def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("  Health Materials RAG - NER Integration Demo")
    print("  Seamless Entity Extraction & Validation Pipeline")
    print("="*60)
    
    try:
        # Run demos
        await demo_basic_ner_extraction()
        await demo_entity_validation()
        await demo_entity_statistics()
        await demo_entity_enhanced_search()
        
        print_separator("All Demos Complete")
        print("✓ NER successfully integrated into RAG pipeline")
        print("✓ Entity extraction works seamlessly during queries")
        print("✓ Validation metrics are tracked automatically")
        print("✓ Entity-enhanced search strategies demonstrated")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
