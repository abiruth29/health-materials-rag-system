"""
Data Acquisition Module - API Connectors

This module provides connectors for fetching data from various materials science
databases including the Materials Project, Open Materials Database, and others.

Owner: Member 1
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
import aiohttp
import requests
from dataclasses import dataclass
from pathlib import Path
import json
import time

# Materials Project API
from mp_api.client import MPRester
import pandas as pd


@dataclass
class MaterialsData:
    """Data structure for materials information."""
    material_id: str
    formula: str
    structure: Optional[Dict] = None
    properties: Optional[Dict] = None
    synthesis: Optional[Dict] = None
    metadata: Optional[Dict] = None


class MaterialsProjectConnector:
    """Connector for Materials Project API."""
    
    def __init__(self, api_key: str):
        """
        Initialize Materials Project connector.
        
        Args:
            api_key: Materials Project API key
        """
        self.api_key = api_key
        self.client = MPRester(api_key)
        self.logger = logging.getLogger(__name__)
    
    async def fetch_materials_by_formula(self, formula: str, limit: int = 100) -> List[MaterialsData]:
        """
        Fetch materials data by chemical formula.
        
        Args:
            formula: Chemical formula (e.g., "Li2O")
            limit: Maximum number of results
            
        Returns:
            List of MaterialsData objects
        """
        try:
            # Get material IDs for the formula
            materials = self.client.materials.search(
                formula=formula,
                fields=["material_id", "formula_pretty", "structure", "energy_above_hull"]
            )
            
            results = []
            for material in materials[:limit]:
                material_data = MaterialsData(
                    material_id=material.material_id,
                    formula=material.formula_pretty,
                    structure=material.structure.as_dict() if material.structure else None,
                    properties={"energy_above_hull": material.energy_above_hull},
                    metadata={"source": "Materials Project"}
                )
                results.append(material_data)
            
            self.logger.info(f"Fetched {len(results)} materials for formula {formula}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error fetching materials for formula {formula}: {e}")
            return []
    
    async def fetch_material_properties(self, material_id: str) -> Optional[Dict]:
        """
        Fetch detailed properties for a specific material.
        
        Args:
            material_id: Material ID from Materials Project
            
        Returns:
            Dictionary of material properties or None if error
        """
        try:
            # Fetch comprehensive material data
            material = self.client.materials.get_data_by_id(
                material_id,
                fields=[
                    "structure", "energy_above_hull", "formation_energy_per_atom",
                    "band_gap", "density", "volume", "nsites", "elements"
                ]
            )
            
            if material:
                return {
                    "structure": material[0].structure.as_dict() if material[0].structure else None,
                    "energy_above_hull": material[0].energy_above_hull,
                    "formation_energy_per_atom": material[0].formation_energy_per_atom,
                    "band_gap": material[0].band_gap,
                    "density": material[0].density,
                    "volume": material[0].volume,
                    "nsites": material[0].nsites,
                    "elements": [str(el) for el in material[0].elements]
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching properties for material {material_id}: {e}")
            return None


class OpenMaterialsDBConnector:
    """Connector for Open Materials Database."""
    
    def __init__(self, base_url: str = "https://nomad-lab.eu/prod/rae/api/v1"):
        """
        Initialize Open Materials DB connector.
        
        Args:
            base_url: Base URL for the NOMAD API
        """
        self.base_url = base_url
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
    
    async def search_materials(self, query: Dict[str, Any], limit: int = 100) -> List[Dict]:
        """
        Search materials in NOMAD database.
        
        Args:
            query: Search query parameters
            limit: Maximum number of results
            
        Returns:
            List of material dictionaries
        """
        try:
            search_url = f"{self.base_url}/entries"
            params = {
                "limit": limit,
                **query
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("data", [])
                    else:
                        self.logger.error(f"Error searching NOMAD: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error searching NOMAD database: {e}")
            return []


class DatabaseAggregator:
    """Aggregates data from multiple materials databases."""
    
    def __init__(self, mp_api_key: str):
        """
        Initialize database aggregator.
        
        Args:
            mp_api_key: Materials Project API key
        """
        self.mp_connector = MaterialsProjectConnector(mp_api_key)
        self.nomad_connector = OpenMaterialsDBConnector()
        self.logger = logging.getLogger(__name__)
    
    async def fetch_comprehensive_data(self, 
                                     formulas: List[str], 
                                     output_path: str = "data/raw/materials_data.json") -> None:
        """
        Fetch comprehensive materials data from all sources.
        
        Args:
            formulas: List of chemical formulas to search
            output_path: Path to save the aggregated data
        """
        all_materials = []
        
        for formula in formulas:
            self.logger.info(f"Fetching data for formula: {formula}")
            
            # Fetch from Materials Project
            mp_data = await self.mp_connector.fetch_materials_by_formula(formula)
            
            # Fetch from NOMAD
            nomad_query = {"chemical_formula_hill": formula}
            nomad_data = await self.nomad_connector.search_materials(nomad_query)
            
            # Combine and structure data
            for material in mp_data:
                material_dict = {
                    "id": material.material_id,
                    "formula": material.formula,
                    "structure": material.structure,
                    "properties": material.properties,
                    "source": "Materials Project",
                    "timestamp": time.time()
                }
                all_materials.append(material_dict)
            
            # Add NOMAD data
            for entry in nomad_data:
                material_dict = {
                    "id": entry.get("entry_id", ""),
                    "formula": entry.get("chemical_formula_hill", ""),
                    "structure": entry.get("structure", {}),
                    "properties": entry.get("properties", {}),
                    "source": "NOMAD",
                    "timestamp": time.time()
                }
                all_materials.append(material_dict)
        
        # Save aggregated data
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_materials, f, indent=2)
        
        self.logger.info(f"Saved {len(all_materials)} materials to {output_path}")
    
    def save_to_dataframe(self, data_path: str, output_path: str = "data/processed/materials_df.parquet") -> None:
        """
        Convert JSON data to pandas DataFrame and save.
        
        Args:
            data_path: Path to JSON data file
            output_path: Path to save the DataFrame
        """
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Saved DataFrame with {len(df)} records to {output_path}")


# Example usage and CLI interface
async def main():
    """Main function for running data acquisition."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch materials data from various databases")
    parser.add_argument("--mp-api-key", required=True, help="Materials Project API key")
    parser.add_argument("--formulas", nargs="+", default=["Li2O", "Fe2O3", "TiO2"], 
                       help="Chemical formulas to search")
    parser.add_argument("--output", default="data/raw/materials_data.json", 
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize aggregator and fetch data
    aggregator = DatabaseAggregator(args.mp_api_key)
    await aggregator.fetch_comprehensive_data(args.formulas, args.output)
    
    # Convert to DataFrame
    aggregator.save_to_dataframe(args.output)


if __name__ == "__main__":
    asyncio.run(main())
