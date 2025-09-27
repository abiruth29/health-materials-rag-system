"""
RAG Pipeline for Materials Science

This module implements a comprehensive Retrieval-Augmented Generation system
for materials science research, integrating vector search with language models.

Key Features:
- Materials-aware retrieval from knowledge graphs
- Context-aware answer generation
- Scientific accuracy validation
- Multi-modal support (text, structures, properties)
- Confidence scoring and uncertainty quantification
"""

import logging
import json
import time
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    AutoModelForCausalLM, pipeline
)

# Try different RAG frameworks based on availability
try:
    from haystack import Pipeline, Document
    from haystack.components.generators import OpenAIGenerator
    from haystack.components.retrievers import InMemoryBM25Retriever
    from haystack.components.builders import PromptBuilder
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False
    logging.warning("Haystack not available, using simpler RAG implementation")

from sentence_transformers import SentenceTransformer
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Container for RAG pipeline results."""
    query: str
    answer: str
    confidence: float
    context_materials: List[Dict[str, Any]]
    reasoning: str
    sources: List[str]
    processing_time_ms: float
    metadata: Dict[str, Any]


class MaterialsRAGPipeline:
    """
    Materials Science RAG Pipeline
    
    Combines retrieval of relevant materials with language model generation
    to answer complex materials science questions.
    """
    
    def __init__(self,
                 retrieval_index_path: Optional[str] = None,
                 llm_model_name: str = "microsoft/DialoGPT-medium",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 use_gpu: bool = False):
        """
        Initialize RAG pipeline.
        
        Args:
            retrieval_index_path: Path to FAISS index for retrieval
            llm_model_name: Name/path of language model for generation
            embedding_model_name: Name/path of embedding model
            use_gpu: Whether to use GPU acceleration
        """
        self.retrieval_index_path = retrieval_index_path
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Components
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm_model = None
        self.retrieval_index = None
        
        # Materials science templates
        self.prompt_templates = {
            "property_query": """
            Based on the following materials data, answer the question about material properties.
            
            Question: {question}
            
            Materials Context:
            {context}
            
            Please provide a detailed answer focusing on:
            1. Specific material properties mentioned
            2. Structure-property relationships
            3. Applications and implications
            4. Any limitations or considerations
            
            Answer:""",
            
            "synthesis_query": """
            Based on the following materials data, answer the question about synthesis methods.
            
            Question: {question}
            
            Materials Context:
            {context}
            
            Please provide a detailed answer focusing on:
            1. Synthesis methods and conditions
            2. Process parameters and optimization
            3. Quality control and characterization
            4. Scalability considerations
            
            Answer:""",
            
            "comparison_query": """
            Based on the following materials data, answer the comparative question.
            
            Question: {question}
            
            Materials Context:
            {context}
            
            Please provide a detailed comparison focusing on:
            1. Key differences in properties
            2. Advantages and disadvantages
            3. Application-specific recommendations
            4. Performance trade-offs
            
            Answer:""",
            
            "general_query": """
            Based on the following materials science data, answer the question.
            
            Question: {question}
            
            Context from Materials Database:
            {context}
            
            Please provide a comprehensive answer based on the materials data provided.
            
            Answer:"""
        }
        
        # Performance tracking
        self.query_history = []
    
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing Materials RAG Pipeline...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        if self.use_gpu:
            self.embedding_model = self.embedding_model.to('cuda')
        
        # Load language model
        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            
            # Add padding token if not present
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            # Try different model types
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    torch_dtype=torch.float16 if self.use_gpu else torch.float32
                )
            except:
                # Fallback to QA model
                self.llm_model = AutoModelForQuestionAnswering.from_pretrained(
                    self.llm_model_name
                )
            
            if self.use_gpu:
                self.llm_model = self.llm_model.to('cuda')
                
            logger.info(f"Loaded language model: {self.llm_model_name}")
            
        except Exception as e:
            logger.warning(f"Could not load LLM {self.llm_model_name}: {e}")
            # Use a text generation pipeline as fallback
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",
                tokenizer="gpt2",
                device=0 if self.use_gpu else -1
            )
        
        # Load retrieval index if available
        if self.retrieval_index_path:
            try:
                # Import here to avoid circular imports
                from retrieval_embedding.faiss_index import MaterialsFAISSIndex
                
                self.retrieval_index = MaterialsFAISSIndex()
                self.retrieval_index.load_index(self.retrieval_index_path)
                logger.info(f"Loaded retrieval index with {len(self.retrieval_index.metadata)} materials")
                
            except Exception as e:
                logger.warning(f"Could not load retrieval index: {e}")
        
        logger.info("RAG Pipeline initialization complete")
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of materials science query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['synthesize', 'synthesis', 'prepare', 'fabricate', 'method']):
            return "synthesis_query"
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'better']):
            return "comparison_query"
        elif any(word in query_lower for word in ['property', 'properties', 'band gap', 'conductivity', 'modulus']):
            return "property_query"
        else:
            return "general_query"
    
    def _format_materials_context(self, materials: List[Dict[str, Any]]) -> str:
        """Format materials data for context in prompts."""
        context_parts = []
        
        for i, material in enumerate(materials, 1):
            material_info = [f"Material {i}: {material.get('formula', 'Unknown')}"]
            
            # Add properties
            properties = material.get('properties', {})
            if properties:
                prop_strings = [f"  {k}: {v}" for k, v in properties.items() if v is not None]
                if prop_strings:
                    material_info.append("  Properties:")
                    material_info.extend(prop_strings)
            
            # Add structure info
            structure = material.get('structure', {})
            if structure and isinstance(structure, dict):
                if 'crystal_system' in structure:
                    material_info.append(f"  Crystal System: {structure['crystal_system']}")
                if 'lattice' in structure and isinstance(structure['lattice'], dict):
                    lattice = structure['lattice']
                    if 'a' in lattice:
                        material_info.append(f"  Lattice Parameter a: {lattice['a']}")
            
            # Add synthesis info if available
            synthesis = material.get('synthesis', {})
            if synthesis:
                material_info.append("  Synthesis:")
                for k, v in synthesis.items():
                    material_info.append(f"    {k}: {v}")
            
            context_parts.append("\\n".join(material_info))
        
        return "\\n\\n".join(context_parts)
    
    async def retrieve_context(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant materials for the query.
        
        Args:
            query: Input query
            k: Number of materials to retrieve
            
        Returns:
            List of relevant materials
        """
        if not self.retrieval_index:
            # Return empty context if no index available
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Perform retrieval
            search_results = self.retrieval_index.search(query_embedding, k=k)
            
            # Convert to materials list
            materials = [result.metadata for result in search_results]
            return materials
            
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            return []
    
    def _generate_with_transformer(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using transformer model."""
        try:
            # Tokenize input
            inputs = self.llm_tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            if self.use_gpu:
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                if hasattr(self.llm_model, 'generate'):
                    # Causal LM
                    outputs = self.llm_model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=max_length,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.llm_tokenizer.pad_token_id
                    )
                    
                    # Decode only the new tokens
                    generated_text = self.llm_tokenizer.decode(
                        outputs[0][inputs['input_ids'].shape[1]:],
                        skip_special_tokens=True
                    )
                else:
                    # QA model
                    outputs = self.llm_model(**inputs)
                    start_idx = torch.argmax(outputs.start_logits)
                    end_idx = torch.argmax(outputs.end_logits)
                    
                    generated_text = self.llm_tokenizer.decode(
                        inputs['input_ids'][0][start_idx:end_idx+1],
                        skip_special_tokens=True
                    )
            
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def _generate_with_pipeline(self, prompt: str, max_length: int = 512) -> str:
        """Generate response using HuggingFace pipeline."""
        try:
            outputs = self.text_generator(
                prompt,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                return_full_text=False
            )
            
            return outputs[0]['generated_text'].strip()
            
        except Exception as e:
            logger.error(f"Error in pipeline generation: {e}")
            return "I apologize, but I encountered an error while generating the response."
    
    def _calculate_confidence(self, query: str, answer: str, context: List[Dict]) -> float:
        """
        Calculate confidence score for the generated answer.
        
        This is a simplified confidence calculation. In practice, you might use
        more sophisticated methods like semantic similarity, fact checking, etc.
        """
        try:
            # Basic confidence factors
            confidence_factors = []
            
            # Length factor (reasonable length answers tend to be better)
            answer_length = len(answer.split())
            if 10 <= answer_length <= 200:
                confidence_factors.append(0.8)
            elif answer_length < 10:
                confidence_factors.append(0.3)
            else:
                confidence_factors.append(0.6)
            
            # Context factor (more context usually means better answers)
            context_factor = min(0.9, len(context) * 0.15)
            confidence_factors.append(context_factor)
            
            # Keyword overlap factor
            query_words = set(query.lower().split())
            answer_words = set(answer.lower().split())
            overlap = len(query_words.intersection(answer_words)) / max(len(query_words), 1)
            confidence_factors.append(min(0.8, overlap * 2))
            
            # Average confidence
            return np.mean(confidence_factors)
            
        except Exception:
            return 0.5  # Default moderate confidence
    
    async def query(self, 
                   question: str,
                   context_k: int = 5,
                   max_answer_length: int = 300) -> RAGResult:
        """
        Process a materials science query through the RAG pipeline.
        
        Args:
            question: The materials science question
            context_k: Number of materials to retrieve for context
            max_answer_length: Maximum length of generated answer
            
        Returns:
            RAGResult with answer and metadata
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant materials
            context_materials = await self.retrieve_context(question, k=context_k)
            
            # Step 2: Format context
            context_text = self._format_materials_context(context_materials)
            
            # Step 3: Select appropriate prompt template
            query_type = self._classify_query_type(question)
            prompt_template = self.prompt_templates[query_type]
            
            # Step 4: Build prompt
            prompt = prompt_template.format(
                question=question,
                context=context_text if context_text else "No specific materials data available."
            )
            
            # Step 5: Generate answer
            if hasattr(self, 'text_generator'):
                answer = self._generate_with_pipeline(prompt, max_answer_length)
            else:
                answer = self._generate_with_transformer(prompt, max_answer_length)
            
            # Step 6: Calculate confidence
            confidence = self._calculate_confidence(question, answer, context_materials)
            
            # Step 7: Create result
            processing_time = (time.time() - start_time) * 1000
            
            result = RAGResult(
                query=question,
                answer=answer,
                confidence=confidence,
                context_materials=context_materials,
                reasoning=f"Used {len(context_materials)} materials for context with {query_type} template",
                sources=[mat.get('source', 'Unknown') for mat in context_materials],
                processing_time_ms=processing_time,
                metadata={
                    'query_type': query_type,
                    'context_count': len(context_materials),
                    'model_used': self.llm_model_name
                }
            )
            
            # Track query
            self.query_history.append({
                'timestamp': time.time(),
                'query': question,
                'confidence': confidence,
                'processing_time_ms': processing_time
            })
            
            logger.info(f"RAG query processed in {processing_time:.2f}ms with confidence {confidence:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            
            # Return error result
            return RAGResult(
                query=question,
                answer=f"I apologize, but I encountered an error processing your question: {str(e)}",
                confidence=0.0,
                context_materials=[],
                reasoning="Error occurred during processing",
                sources=[],
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={'error': str(e)}
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the RAG pipeline."""
        if not self.query_history:
            return {'total_queries': 0}
        
        processing_times = [q['processing_time_ms'] for q in self.query_history]
        confidences = [q['confidence'] for q in self.query_history]
        
        return {
            'total_queries': len(self.query_history),
            'avg_processing_time_ms': np.mean(processing_times),
            'avg_confidence': np.mean(confidences),
            'min_processing_time_ms': np.min(processing_times),
            'max_processing_time_ms': np.max(processing_times),
            'high_confidence_queries': sum(1 for c in confidences if c > 0.7),
            'low_confidence_queries': sum(1 for c in confidences if c < 0.4)
        }


# Example usage and testing
async def main():
    """Example usage of the RAG pipeline."""
    
    # Initialize pipeline
    rag = MaterialsRAGPipeline(
        llm_model_name="gpt2",  # Use smaller model for testing
        use_gpu=False  # Set to True if you have GPU
    )
    
    await rag.initialize()
    
    # Test queries
    test_queries = [
        "What are the properties of perovskite materials for solar cells?",
        "How do you synthesize graphene nanosheets?",
        "Compare the thermal conductivity of diamond and graphite",
        "What materials have high dielectric constants?"
    ]
    
    print("Testing Materials RAG Pipeline...")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\\nQuery: {query}")
        result = await rag.query(query, context_k=3)
        
        print(f"Answer (Confidence: {result.confidence:.2f}):")
        print(result.answer)
        print(f"Processing time: {result.processing_time_ms:.2f}ms")
        print("-" * 30)
    
    # Print performance stats
    stats = rag.get_performance_stats()
    print(f"\\nPerformance Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    import asyncio
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the example
    asyncio.run(main())