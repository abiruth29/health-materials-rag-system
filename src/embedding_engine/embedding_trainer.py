"""
Domain-Specific Embedding Trainer for Materials Science

This module trains and manages specialized embedding models that understand
materials science terminology, crystal structures, and property relationships.

Key Features:
- Materials-specific vocabulary enhancement
- Crystal structure encoding
- Property-aware embeddings
- Transfer learning from scientific literature
"""

import logging
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import re
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle

logger = logging.getLogger(__name__)


class MaterialsEmbeddingTrainer:
    """
    Trainer for domain-specific embeddings optimized for materials science.
    
    This class fine-tunes pre-trained sentence transformers on materials science
    data to create embeddings that better understand domain concepts.
    """
    
    def __init__(self, 
                 base_model: str = "all-MiniLM-L6-v2",
                 model_save_path: str = "models/materials_embeddings"):
        """
        Initialize the embedding trainer.
        
        Args:
            base_model: Base sentence transformer model name
            model_save_path: Path to save the fine-tuned model
        """
        self.base_model = base_model
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Load base model
        self.model = SentenceTransformer(base_model)
        
        # Materials science vocabulary
        self.materials_vocab = self._load_materials_vocabulary()
        
    def _load_materials_vocabulary(self) -> Dict[str, List[str]]:
        """Load materials science specific vocabulary and synonyms."""
        return {
            "crystal_systems": [
                "cubic", "tetragonal", "orthorhombic", "hexagonal", 
                "trigonal", "monoclinic", "triclinic"
            ],
            "properties": [
                "band gap", "thermal conductivity", "electrical conductivity",
                "elastic modulus", "density", "melting point", "hardness",
                "magnetic permeability", "dielectric constant", "refractive index"
            ],
            "synthesis_methods": [
                "chemical vapor deposition", "molecular beam epitaxy", "sputtering",
                "sol-gel", "hydrothermal synthesis", "ball milling", "sintering"
            ],
            "material_types": [
                "semiconductor", "superconductor", "ceramic", "polymer",
                "composite", "alloy", "oxide", "carbide", "nitride"
            ],
            "applications": [
                "photovoltaic", "battery", "catalyst", "sensor", "actuator",
                "optical device", "electronic device", "structural material"
            ]
        }
    
    def create_training_data(self, 
                           kg_data_path: str,
                           papers_data_path: Optional[str] = None) -> List[InputExample]:
        """
        Create training examples from knowledge graph and literature data.
        
        Args:
            kg_data_path: Path to knowledge graph data
            papers_data_path: Path to scientific papers data
            
        Returns:
            List of InputExample objects for training
        """
        examples = []
        
        # Load knowledge graph data
        if Path(kg_data_path).exists():
            with open(kg_data_path, 'r') as f:
                kg_data = json.load(f)
            
            # Create positive examples from material-property relationships
            examples.extend(self._create_kg_examples(kg_data))
        
        # Load papers data if available
        if papers_data_path and Path(papers_data_path).exists():
            papers_df = pd.read_csv(papers_data_path)
            examples.extend(self._create_literature_examples(papers_df))
        
        # Add synthetic examples for materials vocabulary
        examples.extend(self._create_synthetic_examples())
        
        logger.info(f"Created {len(examples)} training examples")
        return examples
    
    def _create_kg_examples(self, kg_data: List[Dict]) -> List[InputExample]:
        """Create training examples from knowledge graph relationships."""
        examples = []
        
        for item in kg_data:
            if 'formula' in item and 'properties' in item:
                formula = item['formula']
                properties = item.get('properties', {})
                
                # Create positive examples: material + property description
                for prop_name, prop_value in properties.items():
                    if prop_value is not None:
                        text1 = f"{formula} {prop_name}"
                        text2 = f"Material {formula} has {prop_name} of {prop_value}"
                        examples.append(InputExample(texts=[text1, text2], label=0.9))
                
                # Create structure-property relationships
                if 'structure' in item:
                    structure_info = item['structure']
                    if isinstance(structure_info, dict):
                        lattice_system = structure_info.get('lattice', {}).get('crystal_system', '')
                        if lattice_system:
                            text1 = f"{formula} crystal structure"
                            text2 = f"{formula} has {lattice_system} crystal system"
                            examples.append(InputExample(texts=[text1, text2], label=0.85))
        
        return examples
    
    def _create_literature_examples(self, papers_df: pd.DataFrame) -> List[InputExample]:
        """Create training examples from scientific literature."""
        examples = []
        
        for _, paper in papers_df.iterrows():
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            
            if title and abstract:
                # Title-abstract similarity
                examples.append(InputExample(texts=[title, abstract], label=0.7))
                
                # Extract material mentions and create relationships
                materials_mentioned = self._extract_materials_from_text(title + " " + abstract)
                for material in materials_mentioned:
                    # Material-context relationship
                    context = self._extract_material_context(abstract, material)
                    if context:
                        examples.append(InputExample(texts=[material, context], label=0.8))
        
        return examples
    
    def _create_synthetic_examples(self) -> List[InputExample]:
        """Create synthetic training examples using materials vocabulary."""
        examples = []
        
        # Property-description relationships
        property_descriptions = {
            "band gap": ["electronic band gap", "energy gap", "forbidden gap"],
            "thermal conductivity": ["heat conduction", "thermal transport", "heat transfer"],
            "electrical conductivity": ["electrical transport", "electronic conduction", "current flow"],
            "elastic modulus": ["stiffness", "mechanical strength", "Young's modulus"]
        }
        
        for prop, descriptions in property_descriptions.items():
            for desc in descriptions:
                examples.append(InputExample(texts=[prop, desc], label=0.9))
        
        # Crystal system relationships
        for crystal_system in self.materials_vocab["crystal_systems"]:
            for variation in [f"{crystal_system} structure", f"{crystal_system} lattice"]:
                examples.append(InputExample(texts=[crystal_system, variation], label=0.95))
        
        return examples
    
    def _extract_materials_from_text(self, text: str) -> List[str]:
        """Extract material formulas and names from text."""
        
        # Simple regex for chemical formulas
        formula_pattern = r'\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b'
        formulas = re.findall(formula_pattern, text)
        
        # Filter out common non-materials words
        common_words = {'In', 'As', 'The', 'We', 'It', 'An', 'On', 'At', 'To', 'By'}
        materials = [f for f in formulas if f not in common_words and len(f) > 1]
        
        return materials
    
    def _extract_material_context(self, text: str, material: str) -> str:
        """Extract context around material mentions."""
        
        # Find sentences containing the material
        sentences = text.split('.')
        for sentence in sentences:
            if material in sentence:
                return sentence.strip()
        
        return ""
    
    def train(self, 
              training_data: List[InputExample],
              validation_split: float = 0.1,
              batch_size: int = 16,
              epochs: int = 4,
              warmup_steps: int = 1000) -> None:
        """
        Train the materials-specific embedding model.
        
        Args:
            training_data: List of training examples
            validation_split: Fraction of data for validation
            batch_size: Training batch size
            epochs: Number of training epochs
            warmup_steps: Number of warmup steps
        """
        logger.info(f"Starting training with {len(training_data)} examples")
        
        # Split training and validation data
        train_examples, val_examples = train_test_split(
            training_data, test_size=validation_split, random_state=42
        )
        
        # Create data loaders
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        
        # Define loss function
        train_loss = losses.CosineSimilarityLoss(model=self.model)
        
        # Create evaluator if validation data exists
        evaluator = None
        if val_examples:
            val_sentences1 = [ex.texts[0] for ex in val_examples]
            val_sentences2 = [ex.texts[1] for ex in val_examples]
            val_scores = [ex.label for ex in val_examples]
            
            evaluator = EmbeddingSimilarityEvaluator(
                val_sentences1, val_sentences2, val_scores,
                name="materials_eval"
            )
        
        # Train the model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(self.model_save_path),
            save_best_model=True,
            evaluation_steps=500
        )
        
        logger.info(f"Training completed. Model saved to {self.model_save_path}")
    
    def encode_materials_batch(self, materials_data: List[Dict]) -> np.ndarray:
        """
        Encode a batch of materials data into embeddings.
        
        Args:
            materials_data: List of material dictionaries
            
        Returns:
            Array of embeddings
        """
        texts = []
        for material in materials_data:
            # Create comprehensive text representation
            text_parts = []
            
            # Add formula
            if 'formula' in material:
                text_parts.append(f"Material: {material['formula']}")
            
            # Add properties
            if 'properties' in material:
                for prop, value in material['properties'].items():
                    if value is not None:
                        text_parts.append(f"{prop}: {value}")
            
            # Add structure information
            if 'structure' in material and isinstance(material['structure'], dict):
                structure = material['structure']
                if 'lattice' in structure:
                    lattice = structure['lattice']
                    if 'crystal_system' in lattice:
                        text_parts.append(f"crystal system: {lattice['crystal_system']}")
            
            # Add synthesis information
            if 'synthesis' in material:
                synthesis = material['synthesis']
                if isinstance(synthesis, dict):
                    for key, value in synthesis.items():
                        text_parts.append(f"{key}: {value}")
            
            texts.append(" | ".join(text_parts))
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, 
                       metadata: List[Dict], 
                       save_path: str) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            embeddings: Array of embeddings
            metadata: List of material metadata
            save_path: Path to save the data
        """
        save_data = {
            'embeddings': embeddings,
            'metadata': metadata,
            'model_name': self.base_model,
            'vocabulary': self.materials_vocab
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved {len(embeddings)} embeddings to {save_path}")
    
    def load_model(self, model_path: str) -> None:
        """Load a previously trained model."""
        self.model = SentenceTransformer(model_path)
        logger.info(f"Loaded model from {model_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train materials-specific embeddings")
    parser.add_argument("--kg-data", required=True, help="Path to knowledge graph data")
    parser.add_argument("--papers-data", help="Path to papers data")
    parser.add_argument("--output-model", default="models/materials_embeddings",
                       help="Output path for trained model")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize trainer
    trainer = MaterialsEmbeddingTrainer(model_save_path=args.output_model)
    
    # Create training data
    training_data = trainer.create_training_data(args.kg_data, args.papers_data)
    
    # Train model
    trainer.train(training_data, epochs=args.epochs)
    
    print(f"Training completed! Model saved to {args.output_model}")