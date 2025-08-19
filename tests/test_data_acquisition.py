"""
Unit tests for data acquisition module.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from data_acquisition.api_connectors import MaterialsProjectConnector, DatabaseAggregator
from data_acquisition.corpus_scraper import ArXivScraper, MaterialsLiteratureScraper
from data_acquisition.data_validation import MaterialsDataValidator, DatasetValidator


class TestMaterialsProjectConnector:
    """Test cases for Materials Project connector."""
    
    def test_init(self):
        """Test connector initialization."""
        with patch('data_acquisition.api_connectors.MPRester'):
            connector = MaterialsProjectConnector("test_api_key")
            assert connector.api_key == "test_api_key"
    
    @pytest.mark.asyncio
    async def test_fetch_materials_by_formula(self):
        """Test fetching materials by formula."""
        with patch('data_acquisition.api_connectors.MPRester') as mock_rester:
            # Mock the client
            mock_client = MagicMock()
            mock_rester.return_value = mock_client
            
            # Mock material data
            mock_material = MagicMock()
            mock_material.material_id = "mp-1234"
            mock_material.formula_pretty = "LiFePO4"
            mock_material.structure = None
            mock_material.energy_above_hull = 0.05
            
            mock_client.materials.search.return_value = [mock_material]
            
            connector = MaterialsProjectConnector("test_api_key")
            results = await connector.fetch_materials_by_formula("LiFePO4")
            
            assert len(results) > 0
            assert results[0].material_id == "mp-1234"
            assert results[0].formula == "LiFePO4"


class TestArXivScraper:
    """Test cases for ArXiv scraper."""
    
    def test_init(self):
        """Test scraper initialization."""
        scraper = ArXivScraper()
        assert scraper.base_url == "http://export.arxiv.org/api/query"
    
    @pytest.mark.asyncio
    async def test_search_papers(self):
        """Test searching papers."""
        scraper = ArXivScraper()
        
        # Mock response XML
        mock_xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Test Paper Title</title>
                <author><name>Test Author</name></author>
                <summary>Test abstract content</summary>
                <id>http://arxiv.org/abs/2301.00001v1</id>
                <published>2023-01-01T00:00:00Z</published>
            </entry>
        </feed>'''
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = Mock(return_value=mock_xml)
            mock_get.return_value.__aenter__.return_value = mock_response
            
            results = await scraper.search_papers("test query", max_results=1)
            
            assert len(results) > 0
            assert "Test Paper Title" in results[0].title


class TestMaterialsDataValidator:
    """Test cases for data validator."""
    
    def test_init(self):
        """Test validator initialization."""
        validator = MaterialsDataValidator()
        assert len(validator.materials_keywords) > 0
    
    def test_validate_paper_relevance(self, sample_paper_data):
        """Test paper relevance validation."""
        validator = MaterialsDataValidator()
        result = validator.validate_paper_relevance(sample_paper_data)
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1
    
    def test_validate_materials_data(self, sample_material_data):
        """Test materials data validation."""
        validator = MaterialsDataValidator()
        result = validator.validate_materials_data(sample_material_data)
        
        assert isinstance(result.is_valid, bool)
        assert isinstance(result.confidence, float)


class TestDatasetValidator:
    """Test cases for dataset validator."""
    
    def test_init(self):
        """Test dataset validator initialization."""
        validator = DatasetValidator()
        assert validator.validator is not None
    
    def test_validate_paper_dataset(self, sample_paper_data, test_output_dir):
        """Test paper dataset validation."""
        # Create test dataset file
        test_file = test_output_dir / "test_papers.json"
        with open(test_file, 'w') as f:
            json.dump([sample_paper_data], f)
        
        validator = DatasetValidator()
        results = validator.validate_paper_dataset(str(test_file))
        
        assert "total_papers" in results
        assert "valid_papers" in results
        assert "validity_rate" in results
        assert results["total_papers"] == 1


if __name__ == "__main__":
    pytest.main([__file__])
