"""Setup configuration for Accelerating Materials Discovery RAG project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="accelerating-materials-discovery-rag",
    version="0.1.0",
    author="Materials Discovery RAG Team",
    author_email="team@materials-rag.org",
    description="Multi-Scale Knowledge Graph Construction and Low-Latency RAG for Materials Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/accelerating-materials-discovery-rag",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/accelerating-materials-discovery-rag/issues",
        "Documentation": "https://accelerating-materials-discovery-rag.readthedocs.io",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=4.0.0",
            "black>=22.8.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
            "torch-geometric>=2.1.0",
            "dgl>=0.9.0",
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "materials-rag-server=retrieval_embedding.api_server:main",
            "materials-rag-pipeline=rag_evaluation.rag_pipeline:main",
            "materials-kg-loader=kg_schema_fusion.kg_loader:main",
            "materials-data-fetch=data_acquisition.api_connectors:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.json", "*.txt", "*.md"],
    },
)
