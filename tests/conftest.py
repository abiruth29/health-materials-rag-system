"""
Test configuration for materials discovery RAG project.
"""

import pytest
import os
from pathlib import Path

# Test data directories
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_OUTPUT_DIR = Path(__file__).parent / "test_output"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

@pytest.fixture
def test_data_dir():
    """Fixture providing test data directory."""
    return TEST_DATA_DIR

@pytest.fixture
def test_output_dir():
    """Fixture providing test output directory."""
    return TEST_OUTPUT_DIR

@pytest.fixture
def sample_material_data():
    """Fixture providing sample material data for testing."""
    return {
        "id": "test-material-1",
        "formula": "LiFePO4",
        "name": "Lithium Iron Phosphate",
        "properties": {
            "band_gap": 3.5,
            "formation_energy": -2.8,
            "density": 3.6
        },
        "structure": {
            "space_group": "Pnma",
            "crystal_system": "orthorhombic"
        }
    }

@pytest.fixture
def sample_paper_data():
    """Fixture providing sample paper data for testing."""
    return {
        "title": "High-performance lithium iron phosphate cathode materials",
        "authors": ["John Doe", "Jane Smith"],
        "abstract": "This paper investigates the synthesis and characterization of LiFePO4 cathode materials for lithium-ion batteries. The materials were prepared using sol-gel method and characterized using XRD and electrochemical testing.",
        "doi": "10.1000/test.doi",
        "year": 2023,
        "journal": "Journal of Materials Science",
        "keywords": ["lithium-ion battery", "cathode materials", "LiFePO4"]
    }

# Test environment variables
os.environ["TEST_MODE"] = "true"
os.environ["NEO4J_URI"] = "bolt://localhost:7687"
os.environ["NEO4J_USER"] = "neo4j"
os.environ["NEO4J_PASSWORD"] = "test_password"
