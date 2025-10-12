"""
NER Validation Module for Health Materials RAG Pipeline

Integrated NER validation with seamless data flow through the RAG pipeline.
Validates entity extraction quality and provides metrics for monitoring.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NEREntity:
    """Represents a named entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    method: str = "unknown"  # 'transformer', 'regex', 'pattern'


@dataclass
class NERValidationResult:
    """Results from NER validation."""
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int
    entity_errors: Dict[str, List[Dict]] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)


class NERValidator:
    """
    Validates Named Entity Recognition in RAG pipeline.
    
    Integrates seamlessly with MaterialsRAGPipeline to validate
    entity extraction quality during query processing.
    """
    
    def __init__(self):
        """Initialize NER validator."""
        self.validation_history = []
        self.entity_type_stats = defaultdict(lambda: {
            'count': 0,
            'avg_confidence': 0.0,
            'errors': []
        })
        
        # Known entity patterns for health materials domain
        self.entity_patterns = {
            'MATERIAL': [
                r'\b[A-Z][a-z]*-?\d+[A-Z]*[a-z]*-?\d*[A-Z]*[a-z]*\b',  # Ti-6Al-4V
                r'\b\d{3}[A-Z]?\b',  # 316L
            ],
            'MEASUREMENT': [
                r'\b\d+\.?\d*\s*(?:GPa|MPa|mm|°C|kPa|MPa|N/m)\b',
            ],
            'REGULATORY': [
                r'\b(?:FDA|CE|ISO)\b',
            ],
            'STANDARD': [
                r'\b(?:ISO|ASTM|ASME)\s*[A-Z]?\d+(?:-\d+)?\b',
            ],
            'APPLICATION': [
                r'\b(?:orthopedic|cardiac|dental|spinal|hip|knee)\s+(?:implants?|devices?|stents?)\b',
            ]
        }
    
    def extract_entities_with_patterns(self, text: str) -> List[NEREntity]:
        """
        Extract entities using pattern matching (fast baseline).
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Avoid duplicates at same position
                    duplicate = any(
                        e.start == match.start() and e.end == match.end()
                        for e in entities
                    )
                    if not duplicate:
                        entities.append(NEREntity(
                            text=match.group(),
                            label=entity_type,
                            start=match.start(),
                            end=match.end(),
                            confidence=0.85,
                            method='regex'
                        ))
        
        return entities
    
    def validate_entities(self,
                         extracted_entities: List[NEREntity],
                         expected_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate extracted entities against expected patterns.
        
        Args:
            extracted_entities: List of extracted entities
            expected_types: Expected entity types (if known)
            
        Returns:
            Validation report
        """
        report = {
            'total_entities': len(extracted_entities),
            'entity_distribution': Counter(e.label for e in extracted_entities),
            'avg_confidence': np.mean([e.confidence for e in extracted_entities]) if extracted_entities else 0.0,
            'low_confidence_entities': [
                {'text': e.text, 'label': e.label, 'confidence': e.confidence}
                for e in extracted_entities if e.confidence < 0.5
            ],
            'entity_coverage': {},
            'warnings': []
        }
        
        # Check if expected types are present
        if expected_types:
            found_types = set(e.label for e in extracted_entities)
            missing_types = set(expected_types) - found_types
            if missing_types:
                report['warnings'].append(f"Missing expected entity types: {missing_types}")
        
        # Update statistics
        for entity in extracted_entities:
            stats = self.entity_type_stats[entity.label]
            stats['count'] += 1
            # Running average
            n = stats['count']
            stats['avg_confidence'] = (stats['avg_confidence'] * (n-1) + entity.confidence) / n
        
        return report
    
    def compare_with_gold_standard(self,
                                   predicted: List[NEREntity],
                                   gold: List[NEREntity],
                                   match_type: str = 'exact') -> NERValidationResult:
        """
        Compare predicted entities with gold standard.
        
        Args:
            predicted: Predicted entities
            gold: Gold standard entities
            match_type: 'exact', 'partial', or 'label'
            
        Returns:
            Validation result with metrics
        """
        true_positives = 0
        matched_gold = set()
        matched_pred = set()
        
        confusion = defaultdict(lambda: defaultdict(int))
        
        # Find matches
        for i, pred in enumerate(predicted):
            for j, g in enumerate(gold):
                if j in matched_gold:
                    continue
                
                is_match = False
                if match_type == 'exact':
                    is_match = (pred.start == g.start and 
                              pred.end == g.end and 
                              pred.label == g.label)
                elif match_type == 'partial':
                    is_match = self._partial_match(pred, g)
                else:  # label
                    is_match = (pred.label == g.label and self._spans_overlap(pred, g))
                
                if is_match:
                    true_positives += 1
                    matched_gold.add(j)
                    matched_pred.add(i)
                    confusion[g.label][pred.label] += 1
                    break
        
        false_positives = len(predicted) - true_positives
        false_negatives = len(gold) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Error analysis
        errors = {
            'false_positives': [predicted[i] for i in range(len(predicted)) if i not in matched_pred],
            'false_negatives': [gold[j] for j in range(len(gold)) if j not in matched_gold]
        }
        
        return NERValidationResult(
            precision=precision,
            recall=recall,
            f1=f1,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            entity_errors=errors,
            confusion_matrix=dict(confusion)
        )
    
    def _partial_match(self, pred: NEREntity, gold: NEREntity, threshold: float = 0.5) -> bool:
        """Check partial match using IoU."""
        if pred.label != gold.label:
            return False
        
        intersection_start = max(pred.start, gold.start)
        intersection_end = min(pred.end, gold.end)
        intersection = max(0, intersection_end - intersection_start)
        
        union_start = min(pred.start, gold.start)
        union_end = max(pred.end, gold.end)
        union = union_end - union_start
        
        iou = intersection / union if union > 0 else 0
        return iou >= threshold
    
    def _spans_overlap(self, pred: NEREntity, gold: NEREntity) -> bool:
        """Check if entity spans overlap."""
        return not (pred.end <= gold.start or gold.end <= pred.start)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': len(self.validation_history),
            'entity_type_stats': dict(self.entity_type_stats),
            'validation_history': self.validation_history[-10:]  # Last 10
        }


class NERExtractor:
    """
    Extracts named entities from health materials queries.
    
    Integrates with RAG pipeline for enhanced query understanding.
    """
    
    def __init__(self, use_transformer: bool = False):
        """
        Initialize NER extractor.
        
        Args:
            use_transformer: Whether to use transformer-based NER (slower but more accurate)
        """
        self.use_transformer = use_transformer
        self.ner_pipeline = None
        self.pattern_extractor = NERValidator()
        
        if use_transformer:
            try:
                from transformers import pipeline
                self.ner_pipeline = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded transformer-based NER model")
            except Exception as e:
                logger.warning(f"Could not load transformer NER: {e}")
                self.use_transformer = False
    
    def extract(self, text: str, confidence_threshold: float = 0.5) -> List[NEREntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            confidence_threshold: Minimum confidence for entities
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Method 1: Transformer-based (if available)
        if self.use_transformer and self.ner_pipeline:
            try:
                transformer_results = self.ner_pipeline(text)
                for result in transformer_results:
                    entities.append(NEREntity(
                        text=result['word'],
                        label=self._map_label(result['entity_group']),
                        start=result['start'],
                        end=result['end'],
                        confidence=result['score'],
                        method='transformer'
                    ))
            except Exception as e:
                logger.warning(f"Transformer NER failed: {e}")
        
        # Method 2: Pattern-based (always run as backup/supplement)
        pattern_entities = self.pattern_extractor.extract_entities_with_patterns(text)
        
        # Merge results (avoid duplicates)
        for pe in pattern_entities:
            duplicate = any(
                abs(e.start - pe.start) < 3 and abs(e.end - pe.end) < 3
                for e in entities
            )
            if not duplicate:
                entities.append(pe)
        
        # Filter by confidence
        entities = [e for e in entities if e.confidence >= confidence_threshold]
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        return entities
    
    def _map_label(self, label: str) -> str:
        """Map generic NER labels to domain-specific labels."""
        mapping = {
            'ORG': 'REGULATORY',
            'MISC': 'MATERIAL',
            'LOC': 'APPLICATION',
            'CHEMICAL': 'MATERIAL',
            'DISEASE': 'APPLICATION'
        }
        return mapping.get(label, label)
    
    def extract_with_context(self, 
                            text: str,
                            query_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract entities with additional context.
        
        Args:
            text: Input text
            query_type: Type of query (helps prioritize entity types)
            
        Returns:
            Dictionary with entities and metadata
        """
        entities = self.extract(text)
        
        # Organize by type
        by_type = defaultdict(list)
        for entity in entities:
            by_type[entity.label].append(entity)
        
        # Extract key entities based on query type
        key_entities = []
        if query_type == 'material_search':
            key_entities = by_type.get('MATERIAL', []) + by_type.get('MATERIAL_CLASS', [])
        elif query_type == 'property_query':
            key_entities = by_type.get('PROPERTY', []) + by_type.get('MEASUREMENT', [])
        elif query_type == 'application_search':
            key_entities = by_type.get('APPLICATION', [])
        
        return {
            'entities': entities,
            'by_type': dict(by_type),
            'key_entities': key_entities,
            'entity_count': len(entities),
            'entity_types': list(by_type.keys()),
            'coverage_score': self._calculate_coverage(entities, text)
        }
    
    def _calculate_coverage(self, entities: List[NEREntity], text: str) -> float:
        """Calculate what percentage of text is covered by entities."""
        if not text:
            return 0.0
        
        total_entity_chars = sum(e.end - e.start for e in entities)
        return min(1.0, total_entity_chars / len(text))


def create_test_cases() -> List[Dict[str, Any]]:
    """
    Create standard test cases for NER validation.
    
    Returns:
        List of test cases with gold standard annotations
    """
    return [
        {
            "text": "Ti-6Al-4V titanium alloy has Young's modulus of 110 GPa and is FDA approved for orthopedic implants",
            "gold_entities": [
                NEREntity("Ti-6Al-4V", "MATERIAL", 0, 9, 1.0, "gold"),
                NEREntity("titanium alloy", "MATERIAL_CLASS", 10, 24, 1.0, "gold"),
                NEREntity("Young's modulus", "PROPERTY", 29, 44, 1.0, "gold"),
                NEREntity("110 GPa", "MEASUREMENT", 48, 55, 1.0, "gold"),
                NEREntity("FDA", "REGULATORY", 63, 66, 1.0, "gold"),
                NEREntity("orthopedic implants", "APPLICATION", 80, 99, 1.0, "gold")
            ]
        },
        {
            "text": "316L stainless steel shows excellent biocompatibility with ISO 10993 certification",
            "gold_entities": [
                NEREntity("316L", "MATERIAL", 0, 4, 1.0, "gold"),
                NEREntity("stainless steel", "MATERIAL_CLASS", 5, 20, 1.0, "gold"),
                NEREntity("biocompatibility", "PROPERTY", 37, 53, 1.0, "gold"),
                NEREntity("ISO 10993", "STANDARD", 59, 68, 1.0, "gold")
            ]
        },
        {
            "text": "PEEK polymer for spinal fusion cages",
            "gold_entities": [
                NEREntity("PEEK", "MATERIAL", 0, 4, 1.0, "gold"),
                NEREntity("polymer", "MATERIAL_CLASS", 5, 12, 1.0, "gold"),
                NEREntity("spinal fusion cages", "APPLICATION", 17, 36, 1.0, "gold")
            ]
        }
    ]
