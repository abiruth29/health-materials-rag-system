"""
Knowledge Graph Schema and Fusion Module - Schema Definition

This module defines the extensible knowledge graph schema with nodes, edges,
and relation types specific to materials science.

Owner: Member 2
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime


class NodeType(Enum):
    """Enumeration of node types in the materials knowledge graph."""
    MATERIAL = "Material"
    ELEMENT = "Element"
    PROPERTY = "Property"
    STRUCTURE = "Structure"
    SYNTHESIS_METHOD = "SynthesisMethod"
    APPLICATION = "Application"
    MEASUREMENT = "Measurement"
    PAPER = "Paper"
    AUTHOR = "Author"
    INSTITUTION = "Institution"


class RelationType(Enum):
    """Enumeration of relationship types in the knowledge graph."""
    # Material relationships
    HAS_PROPERTY = "HAS_PROPERTY"
    HAS_STRUCTURE = "HAS_STRUCTURE"
    CONTAINS_ELEMENT = "CONTAINS_ELEMENT"
    SYNTHESIZED_BY = "SYNTHESIZED_BY"
    USED_IN = "USED_IN"
    
    # Property relationships
    MEASURED_BY = "MEASURED_BY"
    HAS_VALUE = "HAS_VALUE"
    DEPENDS_ON = "DEPENDS_ON"
    
    # Structure relationships
    HAS_SPACE_GROUP = "HAS_SPACE_GROUP"
    HAS_LATTICE_PARAMETER = "HAS_LATTICE_PARAMETER"
    HAS_COORDINATION = "HAS_COORDINATION"
    
    # Synthesis relationships
    REQUIRES_TEMPERATURE = "REQUIRES_TEMPERATURE"
    REQUIRES_PRESSURE = "REQUIRES_PRESSURE"
    USES_PRECURSOR = "USES_PRECURSOR"
    PRODUCES = "PRODUCES"
    
    # Research relationships
    STUDIED_IN = "STUDIED_IN"
    AUTHORED_BY = "AUTHORED_BY"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    CITES = "CITES"
    
    # Similarity relationships
    SIMILAR_TO = "SIMILAR_TO"
    DERIVED_FROM = "DERIVED_FROM"
    COMPETES_WITH = "COMPETES_WITH"


@dataclass
class NodeSchema:
    """Schema definition for a knowledge graph node."""
    node_type: NodeType
    required_properties: List[str]
    optional_properties: List[str] = field(default_factory=list)
    property_types: Dict[str, str] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def validate_node(self, node_data: Dict[str, Any]) -> bool:
        """Validate if node data conforms to this schema."""
        # Check required properties
        for prop in self.required_properties:
            if prop not in node_data:
                return False
        
        # Check property types
        for prop, expected_type in self.property_types.items():
            if prop in node_data:
                if expected_type == "string" and not isinstance(node_data[prop], str):
                    return False
                elif expected_type == "number" and not isinstance(node_data[prop], (int, float)):
                    return False
                elif expected_type == "list" and not isinstance(node_data[prop], list):
                    return False
                elif expected_type == "dict" and not isinstance(node_data[prop], dict):
                    return False
        
        return True


@dataclass
class RelationSchema:
    """Schema definition for a knowledge graph relationship."""
    relation_type: RelationType
    source_node_types: List[NodeType]
    target_node_types: List[NodeType]
    required_properties: List[str] = field(default_factory=list)
    optional_properties: List[str] = field(default_factory=list)
    property_types: Dict[str, str] = field(default_factory=dict)
    directional: bool = True
    
    def validate_relation(self, source_type: NodeType, target_type: NodeType, 
                         relation_data: Dict[str, Any]) -> bool:
        """Validate if relation data conforms to this schema."""
        # Check node type compatibility
        if source_type not in self.source_node_types:
            return False
        if target_type not in self.target_node_types:
            return False
        
        # Check required properties
        for prop in self.required_properties:
            if prop not in relation_data:
                return False
        
        return True


class MaterialsKnowledgeGraphSchema:
    """Complete schema definition for materials science knowledge graph."""
    
    def __init__(self):
        """Initialize the schema with predefined node and relation types."""
        self.node_schemas = self._define_node_schemas()
        self.relation_schemas = self._define_relation_schemas()
    
    def _define_node_schemas(self) -> Dict[NodeType, NodeSchema]:
        """Define schemas for all node types."""
        schemas = {}
        
        # Material node schema
        schemas[NodeType.MATERIAL] = NodeSchema(
            node_type=NodeType.MATERIAL,
            required_properties=["formula", "id"],
            optional_properties=[
                "name", "aliases", "description", "phase", "space_group",
                "lattice_parameters", "density", "formation_energy",
                "band_gap", "magnetic_moment", "bulk_modulus", "shear_modulus"
            ],
            property_types={
                "formula": "string",
                "id": "string",
                "name": "string",
                "aliases": "list",
                "description": "string",
                "phase": "string",
                "space_group": "string",
                "lattice_parameters": "dict",
                "density": "number",
                "formation_energy": "number",
                "band_gap": "number",
                "magnetic_moment": "number",
                "bulk_modulus": "number",
                "shear_modulus": "number"
            }
        )
        
        # Element node schema
        schemas[NodeType.ELEMENT] = NodeSchema(
            node_type=NodeType.ELEMENT,
            required_properties=["symbol", "atomic_number"],
            optional_properties=[
                "name", "atomic_mass", "electronegativity", "ionization_energy",
                "atomic_radius", "block", "group", "period"
            ],
            property_types={
                "symbol": "string",
                "atomic_number": "number",
                "name": "string",
                "atomic_mass": "number",
                "electronegativity": "number",
                "ionization_energy": "number",
                "atomic_radius": "number",
                "block": "string",
                "group": "number",
                "period": "number"
            }
        )
        
        # Property node schema
        schemas[NodeType.PROPERTY] = NodeSchema(
            node_type=NodeType.PROPERTY,
            required_properties=["name", "property_type"],
            optional_properties=[
                "description", "units", "measurement_method", "typical_range",
                "temperature_dependent", "pressure_dependent"
            ],
            property_types={
                "name": "string",
                "property_type": "string",
                "description": "string",
                "units": "string",
                "measurement_method": "string",
                "typical_range": "dict",
                "temperature_dependent": "bool",
                "pressure_dependent": "bool"
            }
        )
        
        # Structure node schema
        schemas[NodeType.STRUCTURE] = NodeSchema(
            node_type=NodeType.STRUCTURE,
            required_properties=["structure_type"],
            optional_properties=[
                "space_group", "crystal_system", "lattice_parameters",
                "atomic_positions", "coordination_numbers", "bond_lengths",
                "bond_angles", "symmetry_operations"
            ],
            property_types={
                "structure_type": "string",
                "space_group": "string",
                "crystal_system": "string",
                "lattice_parameters": "dict",
                "atomic_positions": "list",
                "coordination_numbers": "dict",
                "bond_lengths": "dict",
                "bond_angles": "dict",
                "symmetry_operations": "list"
            }
        )
        
        # Synthesis Method node schema
        schemas[NodeType.SYNTHESIS_METHOD] = NodeSchema(
            node_type=NodeType.SYNTHESIS_METHOD,
            required_properties=["method_name"],
            optional_properties=[
                "description", "temperature_range", "pressure_range",
                "duration", "atmosphere", "precursors", "equipment",
                "yield", "purity", "scalability"
            ],
            property_types={
                "method_name": "string",
                "description": "string",
                "temperature_range": "dict",
                "pressure_range": "dict",
                "duration": "string",
                "atmosphere": "string",
                "precursors": "list",
                "equipment": "list",
                "yield": "number",
                "purity": "number",
                "scalability": "string"
            }
        )
        
        # Application node schema
        schemas[NodeType.APPLICATION] = NodeSchema(
            node_type=NodeType.APPLICATION,
            required_properties=["application_name"],
            optional_properties=[
                "description", "industry", "requirements", "performance_metrics",
                "market_size", "competitors", "advantages", "limitations"
            ],
            property_types={
                "application_name": "string",
                "description": "string",
                "industry": "string",
                "requirements": "list",
                "performance_metrics": "dict",
                "market_size": "string",
                "competitors": "list",
                "advantages": "list",
                "limitations": "list"
            }
        )
        
        # Measurement node schema
        schemas[NodeType.MEASUREMENT] = NodeSchema(
            node_type=NodeType.MEASUREMENT,
            required_properties=["technique", "measured_property"],
            optional_properties=[
                "value", "uncertainty", "units", "conditions", "equipment",
                "operator", "date", "sample_id", "notes"
            ],
            property_types={
                "technique": "string",
                "measured_property": "string",
                "value": "number",
                "uncertainty": "number",
                "units": "string",
                "conditions": "dict",
                "equipment": "string",
                "operator": "string",
                "date": "string",
                "sample_id": "string",
                "notes": "string"
            }
        )
        
        # Paper node schema
        schemas[NodeType.PAPER] = NodeSchema(
            node_type=NodeType.PAPER,
            required_properties=["title", "authors"],
            optional_properties=[
                "abstract", "doi", "arxiv_id", "pubmed_id", "journal",
                "year", "volume", "pages", "keywords", "citations"
            ],
            property_types={
                "title": "string",
                "authors": "list",
                "abstract": "string",
                "doi": "string",
                "arxiv_id": "string",
                "pubmed_id": "string",
                "journal": "string",
                "year": "number",
                "volume": "string",
                "pages": "string",
                "keywords": "list",
                "citations": "list"
            }
        )
        
        # Author node schema
        schemas[NodeType.AUTHOR] = NodeSchema(
            node_type=NodeType.AUTHOR,
            required_properties=["name"],
            optional_properties=[
                "orcid", "affiliation", "email", "research_interests",
                "h_index", "citation_count", "notable_works"
            ],
            property_types={
                "name": "string",
                "orcid": "string",
                "affiliation": "string",
                "email": "string",
                "research_interests": "list",
                "h_index": "number",
                "citation_count": "number",
                "notable_works": "list"
            }
        )
        
        # Institution node schema
        schemas[NodeType.INSTITUTION] = NodeSchema(
            node_type=NodeType.INSTITUTION,
            required_properties=["name"],
            optional_properties=[
                "country", "city", "type", "ranking", "research_areas",
                "funding", "collaborations"
            ],
            property_types={
                "name": "string",
                "country": "string",
                "city": "string",
                "type": "string",
                "ranking": "number",
                "research_areas": "list",
                "funding": "string",
                "collaborations": "list"
            }
        )
        
        return schemas
    
    def _define_relation_schemas(self) -> Dict[RelationType, RelationSchema]:
        """Define schemas for all relation types."""
        schemas = {}
        
        # Material relationships
        schemas[RelationType.HAS_PROPERTY] = RelationSchema(
            relation_type=RelationType.HAS_PROPERTY,
            source_node_types=[NodeType.MATERIAL],
            target_node_types=[NodeType.PROPERTY],
            required_properties=["value"],
            optional_properties=["uncertainty", "units", "conditions", "source"],
            property_types={
                "value": "number",
                "uncertainty": "number",
                "units": "string",
                "conditions": "dict",
                "source": "string"
            }
        )
        
        schemas[RelationType.HAS_STRUCTURE] = RelationSchema(
            relation_type=RelationType.HAS_STRUCTURE,
            source_node_types=[NodeType.MATERIAL],
            target_node_types=[NodeType.STRUCTURE],
            optional_properties=["confidence", "source", "experimental"]
        )
        
        schemas[RelationType.CONTAINS_ELEMENT] = RelationSchema(
            relation_type=RelationType.CONTAINS_ELEMENT,
            source_node_types=[NodeType.MATERIAL],
            target_node_types=[NodeType.ELEMENT],
            required_properties=["stoichiometry"],
            optional_properties=["oxidation_state", "coordination"],
            property_types={
                "stoichiometry": "number",
                "oxidation_state": "number",
                "coordination": "string"
            }
        )
        
        schemas[RelationType.SYNTHESIZED_BY] = RelationSchema(
            relation_type=RelationType.SYNTHESIZED_BY,
            source_node_types=[NodeType.MATERIAL],
            target_node_types=[NodeType.SYNTHESIS_METHOD],
            optional_properties=["success_rate", "yield", "purity", "conditions"]
        )
        
        schemas[RelationType.USED_IN] = RelationSchema(
            relation_type=RelationType.USED_IN,
            source_node_types=[NodeType.MATERIAL],
            target_node_types=[NodeType.APPLICATION],
            optional_properties=["performance", "market_share", "advantages"]
        )
        
        # Property relationships
        schemas[RelationType.MEASURED_BY] = RelationSchema(
            relation_type=RelationType.MEASURED_BY,
            source_node_types=[NodeType.PROPERTY],
            target_node_types=[NodeType.MEASUREMENT],
            optional_properties=["accuracy", "precision", "repeatability"]
        )
        
        # Research relationships
        schemas[RelationType.STUDIED_IN] = RelationSchema(
            relation_type=RelationType.STUDIED_IN,
            source_node_types=[NodeType.MATERIAL, NodeType.PROPERTY, NodeType.STRUCTURE],
            target_node_types=[NodeType.PAPER],
            optional_properties=["focus", "methodology", "findings"]
        )
        
        schemas[RelationType.AUTHORED_BY] = RelationSchema(
            relation_type=RelationType.AUTHORED_BY,
            source_node_types=[NodeType.PAPER],
            target_node_types=[NodeType.AUTHOR],
            optional_properties=["position", "contribution", "corresponding"]
        )
        
        schemas[RelationType.AFFILIATED_WITH] = RelationSchema(
            relation_type=RelationType.AFFILIATED_WITH,
            source_node_types=[NodeType.AUTHOR],
            target_node_types=[NodeType.INSTITUTION],
            optional_properties=["position", "duration", "primary"]
        )
        
        # Similarity relationships
        schemas[RelationType.SIMILAR_TO] = RelationSchema(
            relation_type=RelationType.SIMILAR_TO,
            source_node_types=[NodeType.MATERIAL, NodeType.STRUCTURE, NodeType.PROPERTY],
            target_node_types=[NodeType.MATERIAL, NodeType.STRUCTURE, NodeType.PROPERTY],
            required_properties=["similarity_score"],
            optional_properties=["similarity_type", "method", "basis"],
            property_types={
                "similarity_score": "number",
                "similarity_type": "string",
                "method": "string",
                "basis": "string"
            },
            directional=False
        )
        
        return schemas
    
    def validate_node(self, node_type: NodeType, node_data: Dict[str, Any]) -> bool:
        """Validate a node against its schema."""
        if node_type not in self.node_schemas:
            return False
        return self.node_schemas[node_type].validate_node(node_data)
    
    def validate_relation(self, relation_type: RelationType, 
                         source_type: NodeType, target_type: NodeType,
                         relation_data: Dict[str, Any]) -> bool:
        """Validate a relation against its schema."""
        if relation_type not in self.relation_schemas:
            return False
        return self.relation_schemas[relation_type].validate_relation(
            source_type, target_type, relation_data
        )
    
    def get_node_schema(self, node_type: NodeType) -> Optional[NodeSchema]:
        """Get schema for a specific node type."""
        return self.node_schemas.get(node_type)
    
    def get_relation_schema(self, relation_type: RelationType) -> Optional[RelationSchema]:
        """Get schema for a specific relation type."""
        return self.relation_schemas.get(relation_type)
    
    def export_schema(self, file_path: str) -> None:
        """Export schema to JSON file."""
        schema_dict = {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "node_types": {
                node_type.value: {
                    "required_properties": schema.required_properties,
                    "optional_properties": schema.optional_properties,
                    "property_types": schema.property_types,
                    "constraints": schema.constraints
                }
                for node_type, schema in self.node_schemas.items()
            },
            "relation_types": {
                relation_type.value: {
                    "source_node_types": [nt.value for nt in schema.source_node_types],
                    "target_node_types": [nt.value for nt in schema.target_node_types],
                    "required_properties": schema.required_properties,
                    "optional_properties": schema.optional_properties,
                    "property_types": schema.property_types,
                    "directional": schema.directional
                }
                for relation_type, schema in self.relation_schemas.items()
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(schema_dict, f, indent=2, ensure_ascii=False)
    
    def generate_cypher_constraints(self) -> List[str]:
        """Generate Cypher constraints for Neo4j database."""
        constraints = []
        
        # Node uniqueness constraints
        for node_type in self.node_schemas:
            # Create constraint on id property (assuming all nodes have id)
            constraint = f"CREATE CONSTRAINT {node_type.value.lower()}_id IF NOT EXISTS FOR (n:{node_type.value}) REQUIRE n.id IS UNIQUE"
            constraints.append(constraint)
        
        # Index creation for frequently queried properties
        indexes = [
            "CREATE INDEX material_formula IF NOT EXISTS FOR (m:Material) ON (m.formula)",
            "CREATE INDEX element_symbol IF NOT EXISTS FOR (e:Element) ON (e.symbol)",
            "CREATE INDEX property_name IF NOT EXISTS FOR (p:Property) ON (p.name)",
            "CREATE INDEX paper_doi IF NOT EXISTS FOR (p:Paper) ON (p.doi)",
            "CREATE INDEX author_name IF NOT EXISTS FOR (a:Author) ON (a.name)"
        ]
        
        constraints.extend(indexes)
        return constraints


# Example usage
def main():
    """Example usage of the schema system."""
    # Initialize schema
    schema = MaterialsKnowledgeGraphSchema()
    
    # Export schema to file
    schema.export_schema("config/kg_schema.json")
    
    # Generate Neo4j constraints
    constraints = schema.generate_cypher_constraints()
    
    print("Generated Cypher constraints:")
    for constraint in constraints:
        print(f"  {constraint}")
    
    # Example node validation
    material_data = {
        "id": "mp-1234",
        "formula": "LiFePO4",
        "name": "Lithium Iron Phosphate",
        "band_gap": 3.5,
        "formation_energy": -2.8
    }
    
    is_valid = schema.validate_node(NodeType.MATERIAL, material_data)
    print(f"\nMaterial validation result: {is_valid}")


if __name__ == "__main__":
    main()
