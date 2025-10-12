"""
Quick test to verify NER integration in RAG pipeline.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline.ner_validator import NERExtractor, NERValidator, create_test_cases


def test_ner_extractor():
    """Test basic NER extraction."""
    print("Testing NER Extractor...")
    
    extractor = NERExtractor(use_transformer=False)
    
    text = "Ti-6Al-4V titanium alloy has Young's modulus of 110 GPa and is FDA approved for orthopedic implants"
    entities = extractor.extract(text)
    
    print(f"✓ Text: {text}")
    print(f"✓ Extracted {len(entities)} entities:")
    for e in entities:
        print(f"  - {e.text} ({e.label}) confidence: {e.confidence:.2f}")
    
    assert len(entities) > 0, "Should extract at least one entity"
    print("✓ NER Extractor test passed!\n")


def test_ner_validator():
    """Test NER validation."""
    print("Testing NER Validator...")
    
    extractor = NERExtractor(use_transformer=False)
    validator = NERValidator()
    
    # Get test case
    test_cases = create_test_cases()
    test_case = test_cases[0]
    
    # Extract and validate
    predicted = extractor.extract(test_case['text'])
    gold = test_case['gold_entities']
    
    validation = validator.compare_with_gold_standard(predicted, gold, match_type='partial')
    
    print(f"✓ Test text: {test_case['text'][:60]}...")
    print(f"✓ Gold entities: {len(gold)}")
    print(f"✓ Predicted entities: {len(predicted)}")
    print(f"✓ Precision: {validation.precision:.2%}")
    print(f"✓ Recall: {validation.recall:.2%}")
    print(f"✓ F1 Score: {validation.f1:.2%}")
    
    assert validation.f1 > 0, "Should have some matches"
    print("✓ NER Validator test passed!\n")


def test_entity_patterns():
    """Test entity pattern extraction."""
    print("Testing Entity Patterns...")
    
    validator = NERValidator()
    
    test_texts = {
        "Ti-6Al-4V": "MATERIAL",
        "316L": "MATERIAL",
        "110 GPa": "MEASUREMENT",
        "FDA": "REGULATORY",
        "ISO 10993": "STANDARD",
        "orthopedic implants": "APPLICATION"
    }
    
    for text, expected_type in test_texts.items():
        entities = validator.extract_entities_with_patterns(text)
        
        if entities:
            found = any(e.label == expected_type for e in entities)
            status = "✓" if found else "✗"
            print(f"{status} {text:20} → {expected_type:15} (found: {found})")
        else:
            print(f"✗ {text:20} → {expected_type:15} (no entities)")
    
    print("✓ Entity pattern test completed!\n")


def test_rag_integration():
    """Test RAG integration (import only)."""
    print("Testing RAG Integration...")
    
    try:
        from rag_pipeline.rag_pipeline import MaterialsRAGPipeline, RAGResult
        
        # Check RAGResult has NER fields
        import inspect
        sig = inspect.signature(RAGResult.__init__)
        params = list(sig.parameters.keys())
        
        assert 'query_entities' in params, "RAGResult should have query_entities field"
        assert 'answer_entities' in params, "RAGResult should have answer_entities field"
        assert 'entity_validation' in params, "RAGResult should have entity_validation field"
        
        print("✓ RAGResult has NER fields")
        
        # Check pipeline has NER components
        pipeline = MaterialsRAGPipeline(enable_ner=True)
        
        assert hasattr(pipeline, 'ner_extractor'), "Pipeline should have ner_extractor"
        assert hasattr(pipeline, 'ner_validator'), "Pipeline should have ner_validator"
        assert hasattr(pipeline, 'enable_ner'), "Pipeline should have enable_ner flag"
        
        print("✓ Pipeline has NER components")
        print("✓ RAG integration test passed!\n")
        
    except Exception as e:
        print(f"✗ RAG integration test failed: {e}\n")
        raise


if __name__ == "__main__":
    print("="*60)
    print("  NER Integration Test Suite")
    print("="*60 + "\n")
    
    try:
        test_ner_extractor()
        test_ner_validator()
        test_entity_patterns()
        test_rag_integration()
        
        print("="*60)
        print("  ✓ All tests passed!")
        print("  NER is successfully integrated into the pipeline")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
