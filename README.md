# Accelerating Materials Discovery: Multi-Scale Knowledge Graph Construction and Low-Latency Retrieval-Augmented Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project develops a domain-optimized Retrieval-Augmented Generation (RAG) system customized for materials science. The system integrates multi-scale knowledge from atomic structures, synthesis, properties, and performance data into a unified knowledge graph built from scientific literature and databases such as the Materials Project. 

The architecture includes hierarchical low-latency retrieval supported by domain-specific embeddings, optimized for sub-100ms response times, and contextual AI-driven answer generation. The project delivers a scalable knowledge graph pipeline, retrieval APIs, an integrated RAG system, and benchmarking tools, culminating in a publishable research pipeline.

## Abstract

Develop a domain-optimized Retrieval-Augmented Generation (RAG) system customized for materials science. The system will integrate multi-scale knowledge from atomic structures, synthesis, properties, and performance data into a unified knowledge graph built from scientific literature and databases such as the Materials Project. The architecture will include hierarchical low-latency retrieval supported by domain-specific embeddings, optimized for sub-100ms response times, and contextual AI-driven answer generation.

## Technology Stack

- **Python 3.8+** - Core development language
- **requests, BeautifulSoup4** - Web scraping and API access
- **spaCy, transformers (Hugging Face)** - NLP and language models
- **pandas, SQLite** - Data processing and storage
- **Neo4j Community Edition** - Knowledge graph database
- **NetworkX, BERTopic** - Graph algorithms and topic modeling
- **FAISS, sentence-transformers, pymatgen** - Vector search and materials science
- **FastAPI, Redis** - API server and caching
- **Haystack** - RAG framework
- **scikit-learn, matplotlib, seaborn** - Machine learning and visualization
- **GitHub, Overleaf** - Version control and collaborative writing

## Project Structure

```
accelerating-materials-discovery-rag/
├── data_acquisition/          # Module 1: Data fetching and preprocessing
│   ├── api_connectors.py     # Materials Project and database APIs
│   ├── corpus_scraper.py     # Scientific literature scraping
│   ├── ner_relation_extraction.py  # NER and relation extraction
│   └── data_validation.py    # Quality validation filters
├── kg_schema_fusion/         # Module 2: Knowledge graph construction
│   ├── kg_schema.py         # Graph schema definition
│   ├── fusion_algorithms.py # Entity merging and conflict resolution
│   └── kg_loader.py         # Neo4j database operations
├── retrieval_embedding/      # Module 3: Low-latency retrieval system
│   ├── embedding_trainer.py # Domain-specific embeddings
│   ├── faiss_index.py       # Vector similarity indexing
│   ├── api_server.py        # FastAPI retrieval endpoints
│   └── latency_benchmark.py # Performance benchmarking
├── rag_evaluation/          # Module 4: RAG generation and evaluation
│   ├── rag_pipeline.py      # Integrated RAG system
│   ├── evaluation_metrics.py # Accuracy and performance metrics
│   ├── visualization_tools.py # Results visualization
│   └── paper_drafts/        # Research publication drafts
├── tests/                   # Unit and integration tests
├── config/                  # Configuration files
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
├── setup.py               # Package setup
└── README.md              # This file
```

## Module Breakdown

### Module 1: Data Acquisition and Knowledge Extraction
**Owner: Member 1**

**Objectives:**
- Implement scripts for fetching data from Materials Project API and other open-source databases
- Develop pipeline for scraping and preprocessing scientific literature
- Adapt/train LLMs and NER models for materials-specific entity and relation extraction
- Implement quality validation filters to ensure domain relevance and granularity

**Key Deliverables:**
- API connectors for materials databases
- Literature scraping and preprocessing pipeline
- Trained SciBERT models for materials NER
- Clean, validated datasets

### Module 2: Knowledge Graph Schema and Fusion Engine
**Owner: Member 2**

**Objectives:**
- Design extensible KG schema with nodes, edges, and relation types for materials science
- Build knowledge fusion pipeline handling entity merging and conflict resolution
- Setup prototype KG in Neo4j Community Edition with version control

**Key Deliverables:**
- Graph schema specification
- Fusion algorithms and conflict resolution
- Neo4j database with imported data
- NetworkX prototypes and BERTopic clustering

### Module 3: Low-Latency Retrieval & Material-Specific Embedding
**Owner: Member 3**

**Objectives:**
- Develop hierarchical retrieval architecture with multi-tier cache
- Generate domain-specific embeddings encoding crystal structures and properties
- Optimize query routing and ensure sub-100ms latency responses

**Key Deliverables:**
- FAISS vector similarity indexes
- Domain-specific embedding models
- FastAPI retrieval API with Redis caching
- Latency benchmarking suite

### Module 4: RAG Generation, Evaluation, and Publication Pipeline
**Owner: Member 4**

**Objectives:**
- Integrate KG retrieval with LLM-based generative models using Haystack
- Implement evaluation pipelines for accuracy, latency, and semantic similarity
- Oversee paper writing with version control and collaborative tools

**Key Deliverables:**
- Complete RAG pipeline implementation
- Evaluation metrics and benchmarking datasets
- Scientific accuracy validation
- Research publication drafts

## Installation

### Prerequisites
- Python 3.8 or higher
- Git
- Neo4j Community Edition (optional for full KG functionality)
- Redis (optional for caching)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/accelerating-materials-discovery-rag.git
   cd accelerating-materials-discovery-rag
   git checkout abiruth
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables:**
   ```bash
   cp config/.env.example config/.env
   # Edit config/.env with your API keys and configuration
   ```

5. **Initialize databases (optional):**
   ```bash
   # Start Neo4j (if installed locally)
   neo4j start
   
   # Start Redis (if using caching)
   redis-server
   ```

## Quick Start

### Running Individual Modules

1. **Data Acquisition:**
   ```bash
   python -m data_acquisition.api_connectors --config config/data_config.yaml
   ```

2. **Knowledge Graph Construction:**
   ```bash
   python -m kg_schema_fusion.kg_loader --input data/processed/ --output neo4j://localhost:7687
   ```

3. **Retrieval API:**
   ```bash
   python -m retrieval_embedding.api_server --port 8000
   ```

4. **RAG Pipeline:**
   ```bash
   python -m rag_evaluation.rag_pipeline --query "What are the properties of perovskite materials?"
   ```

### Running Benchmarks

```bash
# Latency benchmarks
python -m retrieval_embedding.latency_benchmark

# Evaluation metrics
python -m rag_evaluation.evaluation_metrics --dataset MaterialsQA
```

## API Endpoints

The retrieval API provides the following endpoints:

- `GET /health` - Health check
- `POST /search` - Semantic search in knowledge graph
- `POST /embed` - Generate embeddings for materials data
- `GET /materials/{material_id}` - Get specific material information
- `POST /rag/query` - Full RAG query with generation

## Testing

Run the test suite:

```bash
# All tests
pytest

# Specific module tests
pytest tests/test_data_acquisition.py
pytest tests/test_kg_schema_fusion.py
pytest tests/test_retrieval_embedding.py
pytest tests/test_rag_evaluation.py

# With coverage
pytest --cov=. --cov-report=html
```

## Contributing

### Development Workflow

1. Create a feature branch from `abiruth`:
   ```bash
   git checkout abiruth
   git pull origin abiruth
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests
3. Run tests and linting:
   ```bash
   pytest
   flake8 .
   black .
   ```

4. Commit and push:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

5. Create a pull request to `abiruth` branch

### Code Style

- Follow PEP 8 style guidelines
- Use Black for code formatting
- Add type hints where appropriate
- Write docstrings for all functions and classes
- Include unit tests for new functionality

## Performance Targets

- **Retrieval Latency:** Sub-100ms response times
- **Knowledge Graph:** Support for 1M+ entities and 10M+ relations
- **Embedding Quality:** >0.85 similarity for related materials
- **RAG Accuracy:** >90% factual correctness on MaterialsQA benchmark

## Publications

This project aims to produce high-quality research publications. Draft papers are managed in the `rag_evaluation/paper_drafts/` directory with collaborative editing via Overleaf.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Materials Project for providing open materials data
- Hugging Face for transformer models and tools
- Neo4j for graph database technology
- FastAPI and related ecosystem for API development

## Contact

For questions and collaboration opportunities, please reach out to the project maintainers or create an issue in the repository.

---

**Project Status:** Initial Development Phase  
**Branch:** abiruth  
**Last Updated:** August 19, 2025
