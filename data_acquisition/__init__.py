"""
Data Acquisition Module - Initialization

This module initializes the data acquisition package and provides
common utilities for data fetching and processing.

Owner: Member 1
"""

from .api_connectors import MaterialsProjectConnector, OpenMaterialsDBConnector, DatabaseAggregator
from .corpus_scraper import ArXivScraper, PubMedScraper, MaterialsLiteratureScraper, ScientificPaper
from .ner_relation_extraction import MaterialsNERExtractor, MaterialsRelationExtractor, MaterialsKnowledgeExtractor
from .data_validation import MaterialsDataValidator, DatasetValidator, ValidationResult

__version__ = "0.1.0"
__author__ = "Materials Discovery RAG Team"

__all__ = [
    "MaterialsProjectConnector",
    "OpenMaterialsDBConnector", 
    "DatabaseAggregator",
    "ArXivScraper",
    "PubMedScraper",
    "MaterialsLiteratureScraper",
    "ScientificPaper",
    "MaterialsNERExtractor",
    "MaterialsRelationExtractor", 
    "MaterialsKnowledgeExtractor",
    "MaterialsDataValidator",
    "DatasetValidator",
    "ValidationResult"
]
