# Accelerating Materials Discovery RAG - Project Instructions

This repository contains a comprehensive implementation of a domain-optimized Retrieval-Augmented Generation (RAG) system for materials science research.

## Project Overview

The system integrates multi-scale knowledge from atomic structures, synthesis data, properties, and performance relationships from scientific literature and structured databases. It includes hierarchical low-latency retrieval, domain-specific embeddings, and AI-driven contextual answer generation.

## Module Organization

### Module 1: Data Acquisition (`data_acquisition/`)
- **Owner**: Member 1
- **Purpose**: Fetch and preprocess data from Materials Project, scientific literature, and other sources
- **Key Components**:
  - `api_connectors.py`: Materials Project and database API interfaces
  - `corpus_scraper.py`: Scientific literature scraping (arXiv, PubMed)
  - `ner_relation_extraction.py`: Named entity recognition and relation extraction
  - `data_validation.py`: Quality validation and filtering

### Module 2: Knowledge Graph Schema (`kg_schema_fusion/`)
- **Owner**: Member 2
- **Purpose**: Design and implement knowledge graph schema and fusion algorithms
- **Key Components**:
  - `kg_schema.py`: Extensible graph schema definition
  - `fusion_algorithms.py`: Entity merging and conflict resolution
  - `kg_loader.py`: Neo4j database operations and data import

### Module 3: Retrieval & Embedding (`retrieval_embedding/`)
- **Owner**: Member 3
- **Purpose**: Low-latency retrieval system with domain-specific embeddings
- **Key Components**:
  - `embedding_trainer.py`: Materials-specific embedding models
  - `faiss_index.py`: Vector similarity indexing
  - `api_server.py`: FastAPI retrieval endpoints
  - `latency_benchmark.py`: Performance benchmarking

### Module 4: RAG & Evaluation (`rag_evaluation/`)
- **Owner**: Member 4
- **Purpose**: Integrated RAG system with evaluation metrics
- **Key Components**:
  - `rag_pipeline.py`: Complete RAG implementation
  - `evaluation_metrics.py`: Accuracy and performance evaluation
  - `visualization_tools.py`: Results analysis and visualization
  - `paper_drafts/`: Research publication materials

## Development Workflow

### Branch Strategy
- `main`: Production-ready code
- `abiruth`: Primary development branch
- `feature/*`: Individual feature development
- `module/*`: Module-specific development

### Coding Standards
- Follow PEP 8 style guidelines
- Use type hints for all function signatures
- Write comprehensive docstrings
- Include unit tests for new functionality
- Maintain >90% test coverage

### Code Review Process
1. Create feature branch from `abiruth`
2. Implement changes with tests
3. Run linting and testing locally
4. Submit pull request to `abiruth`
5. Address review feedback
6. Merge after approval

## Getting Started

### Prerequisites
- Python 3.8+
- Git
- Neo4j Community Edition (optional)
- Redis (optional)

### Quick Setup
```bash
# Clone repository
git clone <repository-url>
cd accelerating-materials-discovery-rag

# Run setup script
python setup_project.py

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp config/.env.example config/.env
# Edit config/.env with your settings

# Run tests
pytest tests/ -v
```

### Configuration
Edit `config/.env` with your API keys and database settings:
- Materials Project API key
- NCBI email for PubMed access
- Neo4j database credentials
- Redis connection details

## API Endpoints

The retrieval system provides these endpoints:

- `GET /health` - Health check
- `POST /search` - Semantic search
- `POST /embed` - Generate embeddings
- `GET /materials/{id}` - Material information
- `POST /rag/query` - RAG query processing

## Testing

### Running Tests
```bash
# All tests
pytest

# Specific modules
pytest tests/test_data_acquisition.py
pytest tests/test_kg_schema_fusion.py
pytest tests/test_retrieval_embedding.py
pytest tests/test_rag_evaluation.py

# With coverage
pytest --cov=. --cov-report=html
```

### Test Data
Test data is stored in `tests/test_data/` and automatically created during test runs.

## Performance Targets

- **Retrieval Latency**: Sub-100ms response times
- **Knowledge Graph**: 1M+ entities, 10M+ relations
- **Embedding Quality**: >0.85 similarity for related materials
- **RAG Accuracy**: >90% factual correctness

## Contributing

### Adding New Features
1. Create feature branch: `git checkout -b feature/feature-name`
2. Implement with tests and documentation
3. Follow coding standards and conventions
4. Submit pull request with clear description

### Module Integration
- Use well-defined interfaces between modules
- Follow data schema specifications
- Implement proper error handling
- Add integration tests

### Documentation
- Update README.md for significant changes
- Add docstrings for all public functions
- Include examples in module documentation
- Update API documentation for endpoint changes

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **API Failures**: Check API keys in config/.env
3. **Database Errors**: Verify Neo4j/Redis are running
4. **Memory Issues**: Reduce batch sizes in config
5. **Performance**: Check vector index optimization

### Getting Help
- Check module documentation
- Review test cases for usage examples
- Open issues for bugs or feature requests
- Contact module owners for specific questions

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Materials Project for open materials data
- Hugging Face for transformer models
- Neo4j for graph database technology
- Scientific community for research foundations

---

**Note**: This project is in active development. Features and APIs may change as we progress toward the research publication goals.
