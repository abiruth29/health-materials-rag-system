"""
Health Materials Knowledge Graph Builder

Constructs a comprehensive knowledge graph from materials database,
research papers, and domain knowledge for enhanced reasoning and retrieval.
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Any
from pathlib import Path
from collections import defaultdict
import re

import pandas as pd
import networkx as nx
import numpy as np
from sentence_transformers import SentenceTransformer

# NER for entity extraction
try:
    from transformers import pipeline
    NER_AVAILABLE = True
except ImportError:
    NER_AVAILABLE = False
    logging.warning("NER not available, using rule-based extraction")

logger = logging.getLogger(__name__)


class HealthMaterialsKGBuilder:
    """
    Build a comprehensive knowledge graph for health materials domain.
    
    Knowledge Graph Structure:
    - Nodes: Materials, Applications, Properties, Standards, Studies
    - Edges: USED_FOR, HAS_PROPERTY, APPROVED_BY, STUDIED_IN, SIMILAR_TO
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize KG builder."""
        self.embedding_model_name = embedding_model
        self.model = None
        
        # Knowledge graph
        self.graph = nx.MultiDiGraph()
        
        # Entity collections
        self.materials = {}
        self.applications = {}
        self.properties = {}
        self.standards = {}
        self.studies = {}
        
        # Relationships
        self.relationships = []
        
        # NER pipeline for entity extraction
        self.ner_pipeline = None
        
        # Statistics
        self.stats = defaultdict(int)
    
    def initialize(self):
        """Initialize models and components."""
        logger.info("Initializing Knowledge Graph Builder...")
        
        # Load embedding model for entity linking
        self.model = SentenceTransformer(self.embedding_model_name)
        logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        
        # Initialize NER if available
        if NER_AVAILABLE:
            try:
                self.ner_pipeline = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded biomedical NER pipeline")
            except:
                logger.warning("Could not load biomedical NER, using rule-based extraction")
        
        logger.info("KG Builder initialized successfully")
    
    # ========================================================================
    # STEP 1: ENTITY EXTRACTION
    # ========================================================================
    
    def extract_entities_from_materials(self, materials_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Extract entities from materials database.
        
        Args:
            materials_df: DataFrame with materials data
            
        Returns:
            Dictionary of entity lists by type
        """
        logger.info("Extracting entities from materials database...")
        
        entities = {
            'materials': [],
            'applications': [],
            'properties': [],
            'standards': [],
            'classes': []
        }
        
        for idx, row in materials_df.iterrows():
            # Extract material entity
            material_entity = {
                'id': f"MAT_{idx}",
                'name': row.get('material_name', f'Material_{idx}'),
                'class': row.get('material_class', 'Unknown'),
                'type': 'MATERIAL',
                'properties': {}
            }
            
            # Add properties
            for prop in ['youngs_modulus', 'tensile_strength', 'yield_strength', 'density']:
                if prop in row and pd.notna(row[prop]):
                    material_entity['properties'][prop] = row[prop]
            
            entities['materials'].append(material_entity)
            self.materials[material_entity['id']] = material_entity
            
            # Extract application entities
            if 'applications' in row and pd.notna(row['applications']):
                apps = self._parse_applications(row['applications'])
                for app in apps:
                    app_id = self._create_entity_id('APP', app)
                    if app_id not in self.applications:
                        app_entity = {
                            'id': app_id,
                            'name': app,
                            'type': 'APPLICATION'
                        }
                        entities['applications'].append(app_entity)
                        self.applications[app_id] = app_entity
            
            # Extract property entities (biocompatibility, etc.)
            if 'cytotoxicity' in row and pd.notna(row['cytotoxicity']):
                prop_name = f"Biocompatibility: {row['cytotoxicity']}"
                prop_id = self._create_entity_id('PROP', prop_name)
                if prop_id not in self.properties:
                    prop_entity = {
                        'id': prop_id,
                        'name': prop_name,
                        'type': 'PROPERTY',
                        'category': 'biocompatibility'
                    }
                    entities['properties'].append(prop_entity)
                    self.properties[prop_id] = prop_entity
            
            # Extract regulatory standards
            if 'regulatory_status' in row and pd.notna(row['regulatory_status']):
                standards = self._parse_standards(row['regulatory_status'])
                for std in standards:
                    std_id = self._create_entity_id('STD', std)
                    if std_id not in self.standards:
                        std_entity = {
                            'id': std_id,
                            'name': std,
                            'type': 'STANDARD'
                        }
                        entities['standards'].append(std_entity)
                        self.standards[std_id] = std_entity
        
        # Log statistics
        for entity_type, entity_list in entities.items():
            self.stats[f'entities_{entity_type}'] = len(entity_list)
            logger.info(f"Extracted {len(entity_list)} {entity_type} entities")
        
        return entities
    
    def extract_entities_from_research(self, research_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Extract entities from research papers.
        
        Args:
            research_df: DataFrame with research papers
            
        Returns:
            Dictionary of entity lists by type
        """
        logger.info("Extracting entities from research papers...")
        
        entities = {
            'studies': [],
            'materials': [],
            'applications': []
        }
        
        for idx, row in research_df.iterrows():
            # Extract study entity
            study_entity = {
                'id': f"STUDY_{idx}",
                'title': row.get('title', f'Study_{idx}'),
                'pmid': row.get('pmid', 'N/A'),
                'type': 'STUDY',
                'year': row.get('publication_year', 'Unknown'),
                'source': row.get('source', 'PubMed')
            }
            entities['studies'].append(study_entity)
            self.studies[study_entity['id']] = study_entity
            
            # Extract materials mentioned in study
            if 'materials' in row and pd.notna(row['materials']):
                materials = self._parse_materials_from_text(row['materials'])
                for mat in materials:
                    mat_id = self._create_entity_id('MAT', mat)
                    if mat_id not in self.materials:
                        mat_entity = {
                            'id': mat_id,
                            'name': mat,
                            'type': 'MATERIAL',
                            'source': 'research_extraction'
                        }
                        entities['materials'].append(mat_entity)
                        self.materials[mat_id] = mat_entity
        
        # Log statistics
        for entity_type, entity_list in entities.items():
            self.stats[f'research_entities_{entity_type}'] = len(entity_list)
            logger.info(f"Extracted {len(entity_list)} {entity_type} entities from research")
        
        return entities
    
    # ========================================================================
    # STEP 2: RELATIONSHIP EXTRACTION
    # ========================================================================
    
    def extract_relationships_from_materials(self, materials_df: pd.DataFrame) -> List[Dict]:
        """
        Extract relationships from materials data.
        
        Relationship Types:
        - USED_FOR: Material → Application
        - HAS_PROPERTY: Material → Property
        - APPROVED_BY: Material → Standard
        - BELONGS_TO: Material → Class
        """
        logger.info("Extracting relationships from materials...")
        
        relationships = []
        
        for idx, row in materials_df.iterrows():
            material_id = f"MAT_{idx}"
            
            # Material → Application relationships
            if 'applications' in row and pd.notna(row['applications']):
                apps = self._parse_applications(row['applications'])
                for app in apps:
                    app_id = self._create_entity_id('APP', app)
                    relationships.append({
                        'source': material_id,
                        'target': app_id,
                        'type': 'USED_FOR',
                        'confidence': 0.9
                    })
            
            # Material → Property relationships
            if 'cytotoxicity' in row and pd.notna(row['cytotoxicity']):
                prop_name = f"Biocompatibility: {row['cytotoxicity']}"
                prop_id = self._create_entity_id('PROP', prop_name)
                relationships.append({
                    'source': material_id,
                    'target': prop_id,
                    'type': 'HAS_PROPERTY',
                    'confidence': 1.0
                })
            
            # Material → Standard relationships
            if 'regulatory_status' in row and pd.notna(row['regulatory_status']):
                standards = self._parse_standards(row['regulatory_status'])
                for std in standards:
                    std_id = self._create_entity_id('STD', std)
                    relationships.append({
                        'source': material_id,
                        'target': std_id,
                        'type': 'APPROVED_BY',
                        'confidence': 0.95
                    })
        
        self.stats['relationships_materials'] = len(relationships)
        logger.info(f"Extracted {len(relationships)} relationships from materials")
        
        return relationships
    
    def extract_relationships_from_research(self, research_df: pd.DataFrame) -> List[Dict]:
        """
        Extract relationships from research papers.
        
        Relationship Types:
        - STUDIED_IN: Material → Study
        - INVESTIGATES: Study → Application
        """
        logger.info("Extracting relationships from research...")
        
        relationships = []
        
        for idx, row in research_df.iterrows():
            study_id = f"STUDY_{idx}"
            
            # Study → Material relationships
            if 'materials' in row and pd.notna(row['materials']):
                materials = self._parse_materials_from_text(row['materials'])
                for mat in materials:
                    mat_id = self._create_entity_id('MAT', mat)
                    relationships.append({
                        'source': mat_id,
                        'target': study_id,
                        'type': 'STUDIED_IN',
                        'confidence': 0.8
                    })
        
        self.stats['relationships_research'] = len(relationships)
        logger.info(f"Extracted {len(relationships)} relationships from research")
        
        return relationships
    
    def infer_similarity_relationships(self, threshold: float = 0.75) -> List[Dict]:
        """
        Infer SIMILAR_TO relationships between materials based on embeddings.
        
        Args:
            threshold: Similarity threshold for creating edges
            
        Returns:
            List of similarity relationships
        """
        logger.info("Computing material similarity relationships...")
        
        relationships = []
        
        # Get material names and embeddings
        material_ids = list(self.materials.keys())
        material_names = [self.materials[mid]['name'] for mid in material_ids]
        
        if len(material_names) < 2:
            return relationships
        
        # Generate embeddings
        embeddings = self.model.encode(material_names, show_progress_bar=True)
        
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(embeddings)
        
        # Create relationships for similar materials
        for i in range(len(material_ids)):
            for j in range(i + 1, len(material_ids)):
                similarity = similarities[i][j]
                if similarity >= threshold:
                    relationships.append({
                        'source': material_ids[i],
                        'target': material_ids[j],
                        'type': 'SIMILAR_TO',
                        'confidence': float(similarity)
                    })
        
        self.stats['relationships_similarity'] = len(relationships)
        logger.info(f"Inferred {len(relationships)} similarity relationships")
        
        return relationships
    
    # ========================================================================
    # STEP 3: GRAPH CONSTRUCTION
    # ========================================================================
    
    def build_graph(self) -> nx.MultiDiGraph:
        """
        Build NetworkX graph from entities and relationships.
        
        Returns:
            NetworkX MultiDiGraph
        """
        logger.info("Building knowledge graph...")
        
        # Add nodes
        for material_id, material in self.materials.items():
            self.graph.add_node(material_id, **material)
        
        for app_id, app in self.applications.items():
            self.graph.add_node(app_id, **app)
        
        for prop_id, prop in self.properties.items():
            self.graph.add_node(prop_id, **prop)
        
        for std_id, std in self.standards.items():
            self.graph.add_node(std_id, **std)
        
        for study_id, study in self.studies.items():
            self.graph.add_node(study_id, **study)
        
        # Add edges
        for rel in self.relationships:
            self.graph.add_edge(
                rel['source'],
                rel['target'],
                type=rel['type'],
                confidence=rel.get('confidence', 1.0)
            )
        
        self.stats['total_nodes'] = self.graph.number_of_nodes()
        self.stats['total_edges'] = self.graph.number_of_edges()
        
        logger.info(f"Knowledge graph built: {self.stats['total_nodes']} nodes, {self.stats['total_edges']} edges")
        
        return self.graph
    
    # ========================================================================
    # STEP 4: EXPORT & SAVE
    # ========================================================================
    
    def export_to_json(self, output_path: str):
        """
        Export knowledge graph to JSON format.
        
        Args:
            output_path: Path to save JSON file
        """
        logger.info(f"Exporting knowledge graph to {output_path}...")
        
        kg_data = {
            'metadata': {
                'created': pd.Timestamp.now().isoformat(),
                'total_nodes': self.graph.number_of_nodes(),
                'total_edges': self.graph.number_of_edges(),
                'embedding_model': self.embedding_model_name
            },
            'nodes': {
                'materials': list(self.materials.values()),
                'applications': list(self.applications.values()),
                'properties': list(self.properties.values()),
                'standards': list(self.standards.values()),
                'studies': list(self.studies.values())
            },
            'relationships': self.relationships,
            'statistics': dict(self.stats)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Knowledge graph exported successfully to {output_path}")
    
    def export_to_neo4j_cypher(self, output_path: str):
        """
        Export knowledge graph as Neo4j Cypher commands.
        
        Args:
            output_path: Path to save Cypher file
        """
        logger.info(f"Exporting to Neo4j Cypher format: {output_path}...")
        
        cypher_commands = []
        
        # Create nodes
        for material_id, material in self.materials.items():
            props = ', '.join([f"{k}: '{v}'" for k, v in material.items() if k != 'id'])
            cypher_commands.append(
                f"CREATE (:{material['type']} {{id: '{material_id}', {props}}})"
            )
        
        # Create relationships
        for rel in self.relationships:
            cypher_commands.append(
                f"MATCH (a {{id: '{rel['source']}'}}), (b {{id: '{rel['target']}'}}) "
                f"CREATE (a)-[:{rel['type']} {{confidence: {rel['confidence']}}}]->(b)"
            )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cypher_commands))
        
        logger.info(f"Neo4j Cypher export complete: {len(cypher_commands)} commands")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _create_entity_id(self, prefix: str, name: str) -> str:
        """Create consistent entity ID from name."""
        clean_name = re.sub(r'[^a-zA-Z0-9]', '_', name)
        return f"{prefix}_{clean_name.upper()[:50]}"
    
    def _parse_applications(self, applications_text: str) -> List[str]:
        """Parse applications from text."""
        if pd.isna(applications_text):
            return []
        
        # Split by common delimiters
        apps = re.split(r'[;,|]', str(applications_text))
        return [app.strip() for app in apps if app.strip()]
    
    def _parse_standards(self, regulatory_text: str) -> List[str]:
        """Parse regulatory standards from text."""
        if pd.isna(regulatory_text):
            return []
        
        standards = []
        text = str(regulatory_text).upper()
        
        # Extract ISO standards
        iso_matches = re.findall(r'ISO\s*\d+(?:-\d+)?', text)
        standards.extend(iso_matches)
        
        # Extract ASTM standards
        astm_matches = re.findall(r'ASTM\s*[A-Z]\d+', text)
        standards.extend(astm_matches)
        
        # Extract FDA
        if 'FDA' in text:
            standards.append('FDA')
        
        # Extract CE
        if 'CE' in text:
            standards.append('CE')
        
        return list(set(standards))
    
    def _parse_materials_from_text(self, text: str) -> List[str]:
        """Extract material names from text using NER or patterns."""
        if pd.isna(text):
            return []
        
        materials = []
        
        # Use NER if available
        if self.ner_pipeline:
            try:
                entities = self.ner_pipeline(str(text))
                for entity in entities:
                    if entity['entity_group'] in ['MATERIAL', 'CHEMICAL']:
                        materials.append(entity['word'])
            except:
                pass
        
        # Fallback to pattern matching
        # Common material patterns
        patterns = [
            r'\b[A-Z][a-z]*-?\d+[A-Z]*[a-z]*-?\d*[A-Z]*[a-z]*\b',  # Ti-6Al-4V
            r'\b\d{3}[A-Z]?\b',  # 316L
            r'\b(?:titanium|steel|ceramic|polymer|hydroxyapatite|zirconia|peek)\b'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, str(text), re.IGNORECASE)
            materials.extend(matches)
        
        return list(set([m.strip() for m in materials if len(m.strip()) > 2]))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        return {
            **dict(self.stats),
            'graph_density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0,
            'avg_degree': sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1)
        }


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Build knowledge graph from materials database."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*60)
    print("🕸️  HEALTH MATERIALS KNOWLEDGE GRAPH BUILDER")
    print("="*60)
    
    # Initialize builder
    kg_builder = HealthMaterialsKGBuilder()
    kg_builder.initialize()
    
    # Load data
    print("\n📥 Loading materials database...")
    materials_df = pd.read_csv('data/processed/final_materials_database.csv')
    print(f"   Loaded {len(materials_df)} materials")
    
    print("\n📥 Loading research database...")
    research_df = pd.read_csv('data/processed/final_research_database.csv')
    print(f"   Loaded {len(research_df)} research papers")
    
    # Extract entities
    print("\n🔍 Extracting entities...")
    material_entities = kg_builder.extract_entities_from_materials(materials_df)
    research_entities = kg_builder.extract_entities_from_research(research_df)
    
    # Extract relationships
    print("\n🔗 Extracting relationships...")
    material_rels = kg_builder.extract_relationships_from_materials(materials_df)
    research_rels = kg_builder.extract_relationships_from_research(research_df)
    similarity_rels = kg_builder.infer_similarity_relationships(threshold=0.75)
    
    # Combine relationships
    kg_builder.relationships = material_rels + research_rels + similarity_rels
    
    # Build graph
    print("\n🕸️  Building knowledge graph...")
    graph = kg_builder.build_graph()
    
    # Export
    print("\n💾 Exporting knowledge graph...")
    kg_builder.export_to_json('data/processed/biomedical_knowledge_graph.json')
    kg_builder.export_to_neo4j_cypher('data/processed/knowledge_graph.cypher')
    
    # Print statistics
    print("\n📊 Knowledge Graph Statistics:")
    stats = kg_builder.get_statistics()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    print("\n✅ Knowledge graph built successfully!")
    print("="*60)


if __name__ == "__main__":
    main()