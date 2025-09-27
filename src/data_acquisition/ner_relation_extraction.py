"""
Data Acquisition Module - NER and Relation Extraction

This module provides functionality for Named Entity Recognition (NER) and
relation extraction specific to materials science literature using pre-trained
and fine-tuned models like SciBERT.

Owner: Member 1
"""

import logging
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import spacy
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    AutoModelForSequenceClassification, pipeline
)
import torch
import pandas as pd
from pathlib import Path


@dataclass
class MaterialsEntity:
    """Data structure for materials science entities."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: Optional[str] = None


@dataclass
class MaterialsRelation:
    """Data structure for relations between entities."""
    subject: MaterialsEntity
    predicate: str
    object: MaterialsEntity
    confidence: float
    sentence: str


class MaterialsNERExtractor:
    """Named Entity Recognition for materials science texts."""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        """
        Initialize NER extractor.
        
        Args:
            model_name: Pre-trained model name for materials NER
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
        except Exception as e:
            self.logger.warning(f"Could not load {model_name}, using spaCy instead: {e}")
            self.ner_pipeline = None
            self._load_spacy_model()
    
    def _load_spacy_model(self):
        """Load spaCy model as fallback."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.logger.error("spaCy model not found. Please install: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def extract_entities(self, text: str) -> List[MaterialsEntity]:
        """
        Extract materials science entities from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of MaterialsEntity objects
        """
        entities = []
        
        if self.ner_pipeline:
            entities.extend(self._extract_with_transformers(text))
        elif self.nlp:
            entities.extend(self._extract_with_spacy(text))
        
        # Add rule-based entity extraction
        entities.extend(self._extract_materials_patterns(text))
        
        # Remove duplicates and sort by position
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _extract_with_transformers(self, text: str) -> List[MaterialsEntity]:
        """Extract entities using transformer model."""
        entities = []
        
        try:
            results = self.ner_pipeline(text)
            
            for result in results:
                entity = MaterialsEntity(
                    text=result['word'],
                    label=result['entity_group'],
                    start=result['start'],
                    end=result['end'],
                    confidence=result['score']
                )
                entities.append(entity)
                
        except Exception as e:
            self.logger.error(f"Error in transformer NER: {e}")
        
        return entities
    
    def _extract_with_spacy(self, text: str) -> List[MaterialsEntity]:
        """Extract entities using spaCy."""
        entities = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity = MaterialsEntity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8  # Default confidence for spaCy
                )
                entities.append(entity)
                
        except Exception as e:
            self.logger.error(f"Error in spaCy NER: {e}")
        
        return entities
    
    def _extract_materials_patterns(self, text: str) -> List[MaterialsEntity]:
        """Extract materials using rule-based patterns."""
        entities = []
        
        # Chemical formula patterns
        chemical_formula_pattern = r'\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b'
        
        # Crystal structure patterns
        crystal_patterns = [
            r'\b(?:cubic|tetragonal|orthorhombic|hexagonal|trigonal|monoclinic|triclinic)\b',
            r'\b(?:fcc|bcc|hcp)\b',
            r'\b(?:perovskite|spinel|fluorite|rutile|anatase|brookite)\b'
        ]
        
        # Property patterns
        property_patterns = [
            r'\b(?:band\s+gap|bandgap)\b',
            r'\b(?:thermal\s+conductivity|electrical\s+conductivity)\b',
            r'\b(?:young\'s\s+modulus|bulk\s+modulus|shear\s+modulus)\b',
            r'\b(?:melting\s+point|boiling\s+point)\b'
        ]
        
        # Extract chemical formulas
        for match in re.finditer(chemical_formula_pattern, text):
            if len(match.group()) > 1 and any(c.isdigit() for c in match.group()):
                entity = MaterialsEntity(
                    text=match.group(),
                    label="CHEMICAL_FORMULA",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7
                )
                entities.append(entity)
        
        # Extract crystal structures
        for pattern in crystal_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = MaterialsEntity(
                    text=match.group(),
                    label="CRYSTAL_STRUCTURE",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                )
                entities.append(entity)
        
        # Extract properties
        for pattern in property_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = MaterialsEntity(
                    text=match.group(),
                    label="MATERIAL_PROPERTY",
                    start=match.start(),
                    end=match.end(),
                    confidence=0.8
                )
                entities.append(entity)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[MaterialsEntity]) -> List[MaterialsEntity]:
        """Remove duplicate entities and resolve overlaps."""
        # Sort by start position
        entities.sort(key=lambda x: x.start)
        
        unique_entities = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlap = False
            for existing in unique_entities:
                if (entity.start < existing.end and entity.end > existing.start):
                    # Keep the one with higher confidence
                    if entity.confidence > existing.confidence:
                        unique_entities.remove(existing)
                    else:
                        overlap = True
                        break
            
            if not overlap:
                unique_entities.append(entity)
        
        return unique_entities


class MaterialsRelationExtractor:
    """Extract relations between materials science entities."""
    
    def __init__(self):
        """Initialize relation extractor."""
        self.logger = logging.getLogger(__name__)
        
        # Define relation patterns
        self.relation_patterns = {
            "HAS_PROPERTY": [
                r"(\w+)\s+(?:has|exhibits|shows|displays)\s+(?:a|an|the)?\s*(\w+)",
                r"(\w+)\s+with\s+(?:a|an|the)?\s*(\w+)",
                r"the\s+(\w+)\s+of\s+(\w+)"
            ],
            "IS_TYPE_OF": [
                r"(\w+)\s+is\s+(?:a|an)\s+(\w+)",
                r"(\w+)\s+belongs\s+to\s+the\s+(\w+)\s+family"
            ],
            "HAS_STRUCTURE": [
                r"(\w+)\s+has\s+(?:a|an|the)?\s*(\w+)\s+structure",
                r"(\w+)\s+crystallizes?\s+in\s+(?:a|an|the)?\s*(\w+)"
            ],
            "SYNTHESIZED_BY": [
                r"(\w+)\s+(?:synthesized|prepared|grown)\s+by\s+(\w+)",
                r"(\w+)\s+obtained\s+through\s+(\w+)"
            ]
        }
    
    def extract_relations(self, 
                         text: str, 
                         entities: List[MaterialsEntity]) -> List[MaterialsRelation]:
        """
        Extract relations between entities in text.
        
        Args:
            text: Input text
            entities: List of entities found in the text
            
        Returns:
            List of MaterialsRelation objects
        """
        relations = []
        
        # Split text into sentences
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            # Find entities in this sentence
            sentence_entities = self._get_entities_in_sentence(sentence, entities)
            
            if len(sentence_entities) >= 2:
                # Extract relations within the sentence
                sentence_relations = self._extract_sentence_relations(
                    sentence, sentence_entities
                )
                relations.extend(sentence_relations)
        
        return relations
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_entities_in_sentence(self, 
                                 sentence: str, 
                                 entities: List[MaterialsEntity]) -> List[MaterialsEntity]:
        """Get entities that appear in the given sentence."""
        sentence_entities = []
        
        for entity in entities:
            if entity.text.lower() in sentence.lower():
                sentence_entities.append(entity)
        
        return sentence_entities
    
    def _extract_sentence_relations(self, 
                                   sentence: str, 
                                   entities: List[MaterialsEntity]) -> List[MaterialsRelation]:
        """Extract relations within a single sentence."""
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                
                for match in matches:
                    subject_text = match.group(1)
                    object_text = match.group(2)
                    
                    # Find corresponding entities
                    subject_entity = self._find_entity_by_text(subject_text, entities)
                    object_entity = self._find_entity_by_text(object_text, entities)
                    
                    if subject_entity and object_entity:
                        relation = MaterialsRelation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.7,
                            sentence=sentence
                        )
                        relations.append(relation)
        
        return relations
    
    def _find_entity_by_text(self, 
                           text: str, 
                           entities: List[MaterialsEntity]) -> Optional[MaterialsEntity]:
        """Find entity by its text content."""
        text_lower = text.lower()
        
        for entity in entities:
            if entity.text.lower() == text_lower:
                return entity
            # Partial match for longer entities
            if text_lower in entity.text.lower() or entity.text.lower() in text_lower:
                return entity
        
        return None


class MaterialsKnowledgeExtractor:
    """Combined NER and relation extraction for materials science."""
    
    def __init__(self, model_name: str = "allenai/scibert_scivocab_uncased"):
        """
        Initialize knowledge extractor.
        
        Args:
            model_name: Pre-trained model name
        """
        self.ner_extractor = MaterialsNERExtractor(model_name)
        self.relation_extractor = MaterialsRelationExtractor()
        self.logger = logging.getLogger(__name__)
    
    def extract_knowledge(self, text: str) -> Dict[str, Any]:
        """
        Extract both entities and relations from text.
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary containing entities and relations
        """
        # Extract entities
        entities = self.ner_extractor.extract_entities(text)
        
        # Extract relations
        relations = self.relation_extractor.extract_relations(text, entities)
        
        # Create knowledge structure
        knowledge = {
            "text": text,
            "entities": [
                {
                    "text": entity.text,
                    "label": entity.label,
                    "start": entity.start,
                    "end": entity.end,
                    "confidence": entity.confidence
                }
                for entity in entities
            ],
            "relations": [
                {
                    "subject": relation.subject.text,
                    "predicate": relation.predicate,
                    "object": relation.object.text,
                    "confidence": relation.confidence,
                    "sentence": relation.sentence
                }
                for relation in relations
            ],
            "entity_count": len(entities),
            "relation_count": len(relations)
        }
        
        return knowledge
    
    def process_papers(self, papers_file: str, output_file: str) -> None:
        """
        Process a collection of papers and extract knowledge.
        
        Args:
            papers_file: Path to JSON file containing papers
            output_file: Path to save extracted knowledge
        """
        # Load papers
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        all_knowledge = []
        
        for i, paper in enumerate(papers):
            self.logger.info(f"Processing paper {i+1}/{len(papers)}: {paper.get('title', 'Unknown')[:50]}...")
            
            # Extract from title and abstract
            title_knowledge = self.extract_knowledge(paper.get('title', ''))
            abstract_knowledge = self.extract_knowledge(paper.get('abstract', ''))
            
            paper_knowledge = {
                "paper_id": paper.get('doi', paper.get('arxiv_id', f"paper_{i}")),
                "title": paper.get('title', ''),
                "title_knowledge": title_knowledge,
                "abstract_knowledge": abstract_knowledge,
                "metadata": {
                    "authors": paper.get('authors', []),
                    "year": paper.get('year'),
                    "journal": paper.get('journal'),
                    "source": paper.get('source')
                }
            }
            
            all_knowledge.append(paper_knowledge)
        
        # Save extracted knowledge
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_knowledge, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved knowledge extraction results to {output_file}")
        
        # Create summary statistics
        total_entities = sum(
            k['title_knowledge']['entity_count'] + k['abstract_knowledge']['entity_count'] 
            for k in all_knowledge
        )
        total_relations = sum(
            k['title_knowledge']['relation_count'] + k['abstract_knowledge']['relation_count']
            for k in all_knowledge
        )
        
        self.logger.info(f"Extracted {total_entities} entities and {total_relations} relations from {len(papers)} papers")


# Example usage and CLI interface
def main():
    """Main function for running NER and relation extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract entities and relations from materials science literature")
    parser.add_argument("--input", required=True, help="Input papers JSON file")
    parser.add_argument("--output", default="data/processed/extracted_knowledge.json",
                       help="Output file for extracted knowledge")
    parser.add_argument("--model", default="allenai/scibert_scivocab_uncased",
                       help="Pre-trained model name")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize extractor and process papers
    extractor = MaterialsKnowledgeExtractor(args.model)
    extractor.process_papers(args.input, args.output)


if __name__ == "__main__":
    main()
