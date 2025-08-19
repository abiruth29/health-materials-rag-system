"""
Data Acquisition Module - Corpus Scraper

This module provides functionality for scraping and preprocessing scientific
literature from various sources including arXiv, PubMed, and journal websites.

Owner: Member 1
"""

import asyncio
import logging
import re
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json
import time
import requests
from bs4 import BeautifulSoup
import aiohttp
import pandas as pd
from urllib.parse import urljoin, urlparse
import xml.etree.ElementTree as ET


@dataclass
class ScientificPaper:
    """Data structure for scientific paper information."""
    title: str
    authors: List[str]
    abstract: str
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    pubmed_id: Optional[str] = None
    journal: Optional[str] = None
    year: Optional[int] = None
    keywords: Optional[List[str]] = None
    full_text: Optional[str] = None
    url: Optional[str] = None
    source: Optional[str] = None


class ArXivScraper:
    """Scraper for arXiv preprints."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.logger = logging.getLogger(__name__)
    
    async def search_papers(self, 
                          query: str, 
                          max_results: int = 100,
                          start: int = 0) -> List[ScientificPaper]:
        """
        Search arXiv for papers matching the query.
        
        Args:
            query: Search query (e.g., "materials science perovskite")
            max_results: Maximum number of results
            start: Starting index for pagination
            
        Returns:
            List of ScientificPaper objects
        """
        try:
            params = {
                'search_query': query,
                'start': start,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_arxiv_response(content)
                    else:
                        self.logger.error(f"ArXiv API error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _parse_arxiv_response(self, xml_content: str) -> List[ScientificPaper]:
        """Parse arXiv API XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            namespace = {'atom': 'http://www.w3.org/2005/Atom',
                        'arxiv': 'http://arxiv.org/schemas/atom'}
            
            for entry in root.findall('atom:entry', namespace):
                # Extract basic information
                title = entry.find('atom:title', namespace)
                title = title.text.strip().replace('\n', ' ') if title is not None else ""
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', namespace):
                    name = author.find('atom:name', namespace)
                    if name is not None:
                        authors.append(name.text)
                
                # Abstract
                summary = entry.find('atom:summary', namespace)
                abstract = summary.text.strip().replace('\n', ' ') if summary is not None else ""
                
                # ArXiv ID
                id_elem = entry.find('atom:id', namespace)
                arxiv_id = id_elem.text.split('/')[-1] if id_elem is not None else None
                
                # Published year
                published = entry.find('atom:published', namespace)
                year = None
                if published is not None:
                    year = int(published.text[:4])
                
                # DOI (if available)
                doi = None
                for link in entry.findall('atom:link', namespace):
                    if link.get('title') == 'doi':
                        doi = link.get('href')
                        break
                
                # Categories (as keywords)
                keywords = []
                for category in entry.findall('arxiv:category', namespace):
                    keywords.append(category.get('term'))
                
                paper = ScientificPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    year=year,
                    keywords=keywords,
                    source="arXiv"
                )
                papers.append(paper)
                
        except ET.ParseError as e:
            self.logger.error(f"Error parsing arXiv XML: {e}")
        
        return papers


class PubMedScraper:
    """Scraper for PubMed database."""
    
    def __init__(self, email: str):
        """
        Initialize PubMed scraper.
        
        Args:
            email: Email for NCBI API (required for courtesy)
        """
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.email = email
        self.logger = logging.getLogger(__name__)
    
    async def search_papers(self, 
                          query: str, 
                          max_results: int = 100) -> List[ScientificPaper]:
        """
        Search PubMed for papers.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of ScientificPaper objects
        """
        try:
            # First, get PMIDs
            pmids = await self._search_pmids(query, max_results)
            
            # Then, fetch detailed information
            papers = await self._fetch_paper_details(pmids)
            
            return papers
            
        except Exception as e:
            self.logger.error(f"Error searching PubMed: {e}")
            return []
    
    async def _search_pmids(self, query: str, max_results: int) -> List[str]:
        """Search for PMIDs matching the query."""
        search_url = f"{self.base_url}/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'email': self.email
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    root = ET.fromstring(content)
                    return [id_elem.text for id_elem in root.findall('.//Id')]
                return []
    
    async def _fetch_paper_details(self, pmids: List[str]) -> List[ScientificPaper]:
        """Fetch detailed information for given PMIDs."""
        if not pmids:
            return []
        
        fetch_url = f"{self.base_url}/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'email': self.email
        }
        
        papers = []
        
        async with aiohttp.ClientSession() as session:
            async with session.get(fetch_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    papers = self._parse_pubmed_response(content)
        
        return papers
    
    def _parse_pubmed_response(self, xml_content: str) -> List[ScientificPaper]:
        """Parse PubMed XML response."""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article in root.findall('.//PubmedArticle'):
                # Title
                title_elem = article.find('.//ArticleTitle')
                title = title_elem.text if title_elem is not None else ""
                
                # Authors
                authors = []
                for author in article.findall('.//Author'):
                    first_name = author.find('ForeName')
                    last_name = author.find('LastName')
                    if first_name is not None and last_name is not None:
                        authors.append(f"{first_name.text} {last_name.text}")
                
                # Abstract
                abstract_elem = article.find('.//AbstractText')
                abstract = abstract_elem.text if abstract_elem is not None else ""
                
                # PMID
                pmid_elem = article.find('.//PMID')
                pubmed_id = pmid_elem.text if pmid_elem is not None else None
                
                # DOI
                doi = None
                for id_elem in article.findall('.//ArticleId'):
                    if id_elem.get('IdType') == 'doi':
                        doi = id_elem.text
                        break
                
                # Journal
                journal_elem = article.find('.//Journal/Title')
                journal = journal_elem.text if journal_elem is not None else None
                
                # Year
                year_elem = article.find('.//PubDate/Year')
                year = int(year_elem.text) if year_elem is not None else None
                
                # Keywords
                keywords = []
                for keyword in article.findall('.//Keyword'):
                    if keyword.text:
                        keywords.append(keyword.text)
                
                paper = ScientificPaper(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    pubmed_id=pubmed_id,
                    doi=doi,
                    journal=journal,
                    year=year,
                    keywords=keywords,
                    source="PubMed"
                )
                papers.append(paper)
                
        except ET.ParseError as e:
            self.logger.error(f"Error parsing PubMed XML: {e}")
        
        return papers


class MaterialsLiteratureScraper:
    """Comprehensive scraper for materials science literature."""
    
    def __init__(self, email: str):
        """
        Initialize materials literature scraper.
        
        Args:
            email: Email for PubMed API
        """
        self.arxiv_scraper = ArXivScraper()
        self.pubmed_scraper = PubMedScraper(email)
        self.logger = logging.getLogger(__name__)
    
    async def search_materials_papers(self, 
                                    materials_query: str,
                                    max_results_per_source: int = 100) -> List[ScientificPaper]:
        """
        Search for materials science papers across multiple sources.
        
        Args:
            materials_query: Query related to materials (e.g., "perovskite solar cells")
            max_results_per_source: Maximum results per source
            
        Returns:
            Combined list of papers from all sources
        """
        all_papers = []
        
        # Search arXiv
        self.logger.info(f"Searching arXiv for: {materials_query}")
        arxiv_papers = await self.arxiv_scraper.search_papers(
            f"cat:cond-mat.mtrl-sci AND {materials_query}",
            max_results_per_source
        )
        all_papers.extend(arxiv_papers)
        
        # Search PubMed
        self.logger.info(f"Searching PubMed for: {materials_query}")
        pubmed_papers = await self.pubmed_scraper.search_papers(
            f"{materials_query} AND materials",
            max_results_per_source
        )
        all_papers.extend(pubmed_papers)
        
        # Remove duplicates based on DOI or title similarity
        unique_papers = self._remove_duplicates(all_papers)
        
        self.logger.info(f"Found {len(unique_papers)} unique papers")
        return unique_papers
    
    def _remove_duplicates(self, papers: List[ScientificPaper]) -> List[ScientificPaper]:
        """Remove duplicate papers based on DOI or title similarity."""
        seen_dois = set()
        seen_titles = set()
        unique_papers = []
        
        for paper in papers:
            # Check DOI first
            if paper.doi and paper.doi in seen_dois:
                continue
            if paper.doi:
                seen_dois.add(paper.doi)
            
            # Check title similarity (simple approach)
            title_normalized = re.sub(r'[^\w\s]', '', paper.title.lower())
            if title_normalized in seen_titles:
                continue
            
            seen_titles.add(title_normalized)
            unique_papers.append(paper)
        
        return unique_papers
    
    def save_papers(self, papers: List[ScientificPaper], output_path: str) -> None:
        """
        Save papers to JSON file.
        
        Args:
            papers: List of ScientificPaper objects
            output_path: Path to save the papers
        """
        # Convert to dictionaries
        papers_data = []
        for paper in papers:
            paper_dict = {
                'title': paper.title,
                'authors': paper.authors,
                'abstract': paper.abstract,
                'doi': paper.doi,
                'arxiv_id': paper.arxiv_id,
                'pubmed_id': paper.pubmed_id,
                'journal': paper.journal,
                'year': paper.year,
                'keywords': paper.keywords,
                'source': paper.source,
                'timestamp': time.time()
            }
            papers_data.append(paper_dict)
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(papers_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(papers_data)} papers to {output_path}")
    
    def papers_to_dataframe(self, papers: List[ScientificPaper]) -> pd.DataFrame:
        """Convert papers to pandas DataFrame."""
        data = []
        for paper in papers:
            data.append({
                'title': paper.title,
                'authors': '; '.join(paper.authors) if paper.authors else '',
                'abstract': paper.abstract,
                'doi': paper.doi,
                'arxiv_id': paper.arxiv_id,
                'pubmed_id': paper.pubmed_id,
                'journal': paper.journal,
                'year': paper.year,
                'keywords': '; '.join(paper.keywords) if paper.keywords else '',
                'source': paper.source
            })
        
        return pd.DataFrame(data)


# Example usage and CLI interface
async def main():
    """Main function for running literature scraping."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape materials science literature")
    parser.add_argument("--query", required=True, help="Search query for materials")
    parser.add_argument("--email", required=True, help="Email for PubMed API")
    parser.add_argument("--max-results", type=int, default=100, 
                       help="Maximum results per source")
    parser.add_argument("--output", default="data/raw/literature.json",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize scraper and search
    scraper = MaterialsLiteratureScraper(args.email)
    papers = await scraper.search_materials_papers(args.query, args.max_results)
    
    # Save results
    scraper.save_papers(papers, args.output)
    
    # Also save as CSV for easier analysis
    df = scraper.papers_to_dataframe(papers)
    csv_path = args.output.replace('.json', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(papers)} papers to {args.output} and {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
