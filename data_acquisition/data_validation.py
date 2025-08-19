"""
Data Acquisition Module - Data Validation

This module provides quality validation filters to ensure domain relevance
and appropriate granularity for materials science data.

Owner: Member 1
"""

import logging
import re
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd
from collections import Counter
import numpy as np


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    confidence: float
    issues: List[str]
    recommendations: List[str]


class MaterialsDataValidator:
    """Validator for materials science data quality."""
    
    def __init__(self):
        """Initialize the validator with materials science criteria."""
        self.logger = logging.getLogger(__name__)
        
        # Materials science keywords for relevance checking
        self.materials_keywords = {
            "materials": ["material", "crystal", "alloy", "polymer", "ceramic", "composite", 
                         "metal", "semiconductor", "superconductor", "catalyst"],
            "properties": ["conductivity", "thermal", "mechanical", "electrical", "optical",
                          "magnetic", "elastic", "hardness", "strength", "modulus"],
            "structures": ["crystalline", "amorphous", "cubic", "tetragonal", "hexagonal",
                          "orthorhombic", "monoclinic", "triclinic", "perovskite", "spinel"],
            "synthesis": ["synthesis", "fabrication", "preparation", "growth", "deposition",
                         "annealing", "sintering", "sol-gel", "hydrothermal", "chemical vapor"],
            "characterization": ["xrd", "x-ray", "diffraction", "spectroscopy", "microscopy",
                               "tem", "sem", "afm", "raman", "ftir", "xps"],
            "applications": ["energy", "battery", "solar", "photovoltaic", "catalyst",
                           "sensor", "electronic", "optoelectronic", "magnetic", "structural"]
        }
        
        # Chemical formula patterns
        self.chemical_formula_pattern = re.compile(
            r'\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)*\b'
        )
        
        # Common non-materials terms to filter out
        self.exclusion_terms = {
            "biological", "medical", "pharmaceutical", "biological", "clinical",
            "software", "algorithm", "computer", "database", "internet",
            "social", "economic", "financial", "political", "legal"
        }
    
    def validate_paper_relevance(self, paper: Dict[str, Any]) -> ValidationResult:
        """
        Validate if a paper is relevant to materials science.
        
        Args:
            paper: Dictionary containing paper information
            
        Returns:
            ValidationResult object
        """
        issues = []
        recommendations = []
        
        title = paper.get('title', '').lower()
        abstract = paper.get('abstract', '').lower()
        keywords = paper.get('keywords', [])
        
        # Combine text for analysis
        full_text = f"{title} {abstract} {' '.join(keywords or [])}".lower()
        
        # Check for materials science relevance
        relevance_score = self._calculate_relevance_score(full_text)
        
        # Check for exclusion terms
        exclusion_score = self._calculate_exclusion_score(full_text)
        
        # Check for chemical formulas
        has_chemical_formulas = bool(self.chemical_formula_pattern.findall(full_text))
        
        # Check abstract length and quality
        abstract_quality = self._check_abstract_quality(paper.get('abstract', ''))
        
        # Calculate overall confidence
        confidence = (relevance_score * 0.4 + 
                     (1 - exclusion_score) * 0.2 + 
                     (0.1 if has_chemical_formulas else 0) +
                     abstract_quality * 0.3)
        
        # Determine validity
        is_valid = confidence >= 0.6
        
        # Generate issues and recommendations
        if relevance_score < 0.3:
            issues.append("Low materials science relevance")
            recommendations.append("Verify paper discusses materials, properties, or synthesis")
        
        if exclusion_score > 0.3:
            issues.append("Contains terms suggesting non-materials focus")
            recommendations.append("Review for biological, software, or social science content")
        
        if not has_chemical_formulas and relevance_score < 0.5:
            issues.append("No chemical formulas found")
            recommendations.append("Check if paper discusses specific materials")
        
        if abstract_quality < 0.3:
            issues.append("Poor abstract quality or missing abstract")
            recommendations.append("Verify abstract contains sufficient technical detail")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations
        )
    
    def _calculate_relevance_score(self, text: str) -> float:
        """Calculate materials science relevance score."""
        total_keywords = 0
        found_keywords = 0
        
        for category, keywords in self.materials_keywords.items():
            for keyword in keywords:
                total_keywords += 1
                if keyword in text:
                    found_keywords += 1
        
        return found_keywords / total_keywords if total_keywords > 0 else 0.0
    
    def _calculate_exclusion_score(self, text: str) -> float:
        """Calculate score for non-materials terms."""
        found_exclusions = sum(1 for term in self.exclusion_terms if term in text)
        return min(found_exclusions / len(self.exclusion_terms), 1.0)
    
    def _check_abstract_quality(self, abstract: str) -> float:
        """Check abstract quality and completeness."""
        if not abstract or len(abstract.strip()) < 50:
            return 0.0
        
        # Check for key components
        has_objective = any(word in abstract.lower() for word in 
                           ["study", "investigate", "explore", "analyze", "examine"])
        has_methods = any(word in abstract.lower() for word in 
                         ["method", "technique", "approach", "synthesis", "preparation"])
        has_results = any(word in abstract.lower() for word in 
                         ["result", "show", "demonstrate", "find", "observe"])
        
        quality_score = (
            (0.2 if len(abstract) > 100 else 0) +
            (0.3 if has_objective else 0) +
            (0.3 if has_methods else 0) +
            (0.2 if has_results else 0)
        )
        
        return min(quality_score, 1.0)
    
    def validate_materials_data(self, material_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate materials database entry.
        
        Args:
            material_data: Dictionary containing material information
            
        Returns:
            ValidationResult object
        """
        issues = []
        recommendations = []
        
        # Check required fields
        required_fields = ['id', 'formula']
        missing_fields = [field for field in required_fields 
                         if not material_data.get(field)]
        
        # Check formula validity
        formula = material_data.get('formula', '')
        formula_valid = bool(self.chemical_formula_pattern.match(formula))
        
        # Check properties completeness
        properties = material_data.get('properties', {})
        has_properties = bool(properties and isinstance(properties, dict))
        
        # Check structure information
        structure = material_data.get('structure', {})
        has_structure = bool(structure and isinstance(structure, dict))
        
        # Calculate confidence
        confidence = (
            (0.4 if not missing_fields else 0) +
            (0.3 if formula_valid else 0) +
            (0.2 if has_properties else 0) +
            (0.1 if has_structure else 0)
        )
        
        # Determine validity
        is_valid = confidence >= 0.7 and not missing_fields
        
        # Generate issues and recommendations
        if missing_fields:
            issues.append(f"Missing required fields: {', '.join(missing_fields)}")
            recommendations.append("Ensure all required fields are present")
        
        if not formula_valid:
            issues.append("Invalid or missing chemical formula")
            recommendations.append("Verify chemical formula follows standard notation")
        
        if not has_properties:
            issues.append("No properties data available")
            recommendations.append("Add material properties information")
        
        if not has_structure:
            issues.append("No structure data available")
            recommendations.append("Include crystallographic structure information")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations
        )
    
    def validate_extracted_entities(self, entities: List[Dict[str, Any]]) -> ValidationResult:
        """
        Validate extracted entities for quality and relevance.
        
        Args:
            entities: List of extracted entities
            
        Returns:
            ValidationResult object
        """
        issues = []
        recommendations = []
        
        if not entities:
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                issues=["No entities extracted"],
                recommendations=["Review text for materials science content"]
            )
        
        # Analyze entity types
        entity_types = Counter(entity.get('label', 'UNKNOWN') for entity in entities)
        
        # Check for diverse entity types
        expected_types = {'CHEMICAL_FORMULA', 'MATERIAL_PROPERTY', 'CRYSTAL_STRUCTURE'}
        found_types = set(entity_types.keys())
        type_coverage = len(expected_types.intersection(found_types)) / len(expected_types)
        
        # Check entity quality
        high_confidence_entities = [e for e in entities if e.get('confidence', 0) > 0.7]
        confidence_ratio = len(high_confidence_entities) / len(entities)
        
        # Check for chemical formulas
        formula_entities = [e for e in entities if e.get('label') == 'CHEMICAL_FORMULA']
        has_formulas = len(formula_entities) > 0
        
        # Calculate overall confidence
        confidence = (
            type_coverage * 0.4 +
            confidence_ratio * 0.4 +
            (0.2 if has_formulas else 0)
        )
        
        # Determine validity
        is_valid = confidence >= 0.6
        
        # Generate issues and recommendations
        if type_coverage < 0.5:
            issues.append("Limited diversity in entity types")
            recommendations.append("Verify text contains various materials science concepts")
        
        if confidence_ratio < 0.5:
            issues.append("Many low-confidence entities")
            recommendations.append("Review entity extraction parameters or text quality")
        
        if not has_formulas:
            issues.append("No chemical formulas detected")
            recommendations.append("Ensure text discusses specific chemical compounds")
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=confidence,
            issues=issues,
            recommendations=recommendations
        )


class DatasetValidator:
    """Validator for entire datasets."""
    
    def __init__(self):
        """Initialize dataset validator."""
        self.validator = MaterialsDataValidator()
        self.logger = logging.getLogger(__name__)
    
    def validate_paper_dataset(self, papers_file: str) -> Dict[str, Any]:
        """
        Validate a dataset of papers.
        
        Args:
            papers_file: Path to JSON file containing papers
            
        Returns:
            Dictionary with validation results
        """
        with open(papers_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)
        
        results = []
        valid_papers = []
        
        for i, paper in enumerate(papers):
            result = self.validator.validate_paper_relevance(paper)
            results.append({
                'paper_index': i,
                'title': paper.get('title', 'Unknown')[:100],
                'is_valid': result.is_valid,
                'confidence': result.confidence,
                'issues': result.issues,
                'recommendations': result.recommendations
            })
            
            if result.is_valid:
                valid_papers.append(paper)
        
        # Calculate statistics
        total_papers = len(papers)
        valid_count = len(valid_papers)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        validation_summary = {
            'total_papers': total_papers,
            'valid_papers': valid_count,
            'validity_rate': valid_count / total_papers if total_papers > 0 else 0,
            'average_confidence': avg_confidence,
            'validation_results': results
        }
        
        self.logger.info(f"Validated {total_papers} papers: {valid_count} valid ({valid_count/total_papers*100:.1f}%)")
        
        return validation_summary
    
    def validate_materials_dataset(self, materials_file: str) -> Dict[str, Any]:
        """
        Validate a dataset of materials.
        
        Args:
            materials_file: Path to JSON file containing materials data
            
        Returns:
            Dictionary with validation results
        """
        with open(materials_file, 'r', encoding='utf-8') as f:
            materials = json.load(f)
        
        results = []
        valid_materials = []
        
        for i, material in enumerate(materials):
            result = self.validator.validate_materials_data(material)
            results.append({
                'material_index': i,
                'formula': material.get('formula', 'Unknown'),
                'is_valid': result.is_valid,
                'confidence': result.confidence,
                'issues': result.issues,
                'recommendations': result.recommendations
            })
            
            if result.is_valid:
                valid_materials.append(material)
        
        # Calculate statistics
        total_materials = len(materials)
        valid_count = len(valid_materials)
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        validation_summary = {
            'total_materials': total_materials,
            'valid_materials': valid_count,
            'validity_rate': valid_count / total_materials if total_materials > 0 else 0,
            'average_confidence': avg_confidence,
            'validation_results': results
        }
        
        self.logger.info(f"Validated {total_materials} materials: {valid_count} valid ({valid_count/total_materials*100:.1f}%)")
        
        return validation_summary
    
    def create_filtered_dataset(self, 
                               input_file: str, 
                               output_file: str,
                               min_confidence: float = 0.6) -> None:
        """
        Create filtered dataset with only valid entries.
        
        Args:
            input_file: Path to input JSON file
            output_file: Path to output filtered JSON file
            min_confidence: Minimum confidence threshold
        """
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Determine data type
        if isinstance(data, list) and len(data) > 0:
            if 'title' in data[0] or 'abstract' in data[0]:
                # Papers dataset
                validation_func = self.validator.validate_paper_relevance
            else:
                # Materials dataset
                validation_func = self.validator.validate_materials_data
        else:
            self.logger.error("Cannot determine dataset type")
            return
        
        filtered_data = []
        
        for item in data:
            result = validation_func(item)
            if result.is_valid and result.confidence >= min_confidence:
                # Add validation metadata
                item['_validation'] = {
                    'confidence': result.confidence,
                    'validated_at': pd.Timestamp.now().isoformat()
                }
                filtered_data.append(item)
        
        # Save filtered dataset
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Filtered dataset: {len(data)} -> {len(filtered_data)} items saved to {output_file}")


# Example usage and CLI interface
def main():
    """Main function for running data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate materials science data quality")
    parser.add_argument("--input", required=True, help="Input data file (JSON)")
    parser.add_argument("--output", help="Output file for validation results")
    parser.add_argument("--filtered-output", help="Output file for filtered valid data")
    parser.add_argument("--min-confidence", type=float, default=0.6,
                       help="Minimum confidence threshold for filtering")
    parser.add_argument("--data-type", choices=['papers', 'materials'], 
                       help="Type of data to validate")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize validator
    validator = DatasetValidator()
    
    # Determine validation type
    if args.data_type == 'papers':
        results = validator.validate_paper_dataset(args.input)
    elif args.data_type == 'materials':
        results = validator.validate_materials_dataset(args.input)
    else:
        # Auto-detect
        with open(args.input, 'r', encoding='utf-8') as f:
            sample = json.load(f)
        
        if isinstance(sample, list) and len(sample) > 0:
            if 'title' in sample[0] or 'abstract' in sample[0]:
                results = validator.validate_paper_dataset(args.input)
            else:
                results = validator.validate_materials_dataset(args.input)
        else:
            print("Cannot determine data type. Please specify --data-type")
            return
    
    # Save validation results
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Validation results saved to {args.output}")
    
    # Create filtered dataset
    if args.filtered_output:
        validator.create_filtered_dataset(
            args.input, 
            args.filtered_output, 
            args.min_confidence
        )
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Total items: {results.get('total_papers', results.get('total_materials', 0))}")
    print(f"Valid items: {results.get('valid_papers', results.get('valid_materials', 0))}")
    print(f"Validity rate: {results.get('validity_rate', 0)*100:.1f}%")
    print(f"Average confidence: {results.get('average_confidence', 0):.3f}")


if __name__ == "__main__":
    main()
