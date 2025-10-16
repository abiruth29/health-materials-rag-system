# 1.3 Plan of Action

## Overview

This section outlines the comprehensive five-phase implementation strategy for developing the Health Materials RAG system. Each phase addresses specific challenges identified in the problem statement while building upon previous phases to create an integrated, production-ready solution.

---

## Phase 1: Data Acquisition and Integration

### Objectives

1. Establish reliable connections to heterogeneous data sources (BIOMATDB, NIST, PubMed)
2. Extract structured and unstructured data with complete metadata preservation
3. Design unified schema accommodating diverse data formats
4. Implement data quality validation ensuring consistency and completeness

### Phase 1.1: API Connector Development

#### BIOMATDB Connector

**Technical Approach:**
```python
class BioMatDBConnector:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.rate_limiter = RateLimiter(max_calls=100, period=60)
    
    def fetch_material(self, material_id):
        """Fetch single material with retry logic"""
        endpoint = f"{self.base_url}/api/materials/{material_id}"
        response = self.session.get(
            endpoint,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=30
        )
        return response.json()
    
    def fetch_batch(self, start_id, end_id, batch_size=50):
        """Fetch materials in batches with rate limiting"""
        # Implementation with progress tracking
```

**Data Extraction:**
- **Materials:** Name, composition, CAS number, chemical formula
- **Properties:** Mechanical (strength, modulus), chemical (corrosion resistance), biological (biocompatibility, cytotoxicity)
- **Applications:** Medical device types, implant categories, tissue engineering uses
- **Compliance:** FDA status, ISO standards, clinical trial data
- **Metadata:** Source, update timestamp, data quality indicators

**Challenges and Solutions:**

| Challenge | Solution |
|-----------|----------|
| Rate limiting (100 req/min) | Implement exponential backoff + batch processing |
| Inconsistent JSON schemas | Schema validation + flexible parsing |
| Network timeouts | Retry logic with circuit breaker pattern |
| API versioning changes | Version detection + adapter pattern |

**Expected Output:** 4,000 material records in standardized JSON format

#### NIST Materials Database Connector

**Technical Approach:**

NIST lacks programmatic API; requires web scraping with Selenium + BeautifulSoup4.

```python
class NISTConnector:
    def __init__(self):
        self.driver = webdriver.Chrome(options=self.get_chrome_options())
        self.base_url = "https://nist.gov/materials-database"
    
    def scrape_material(self, material_name):
        """Scrape material page with dynamic content loading"""
        self.driver.get(f"{self.base_url}/search?q={material_name}")
        # Wait for dynamic content
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS, "property-table"))
        )
        # Parse HTML
        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        return self.parse_material_page(soup)
    
    def parse_property_table(self, table_element):
        """Extract property-value-unit triplets"""
        properties = {}
        for row in table_element.find_all('tr'):
            property_name = row.find('td', class_='property').text
            value = row.find('td', class_='value').text
            unit = row.find('td', class_='unit').text
            properties[property_name] = {'value': value, 'unit': unit}
        return properties
```

**Data Extraction:**
- **Material Specifications:** ASTM/ISO designations, trade names, compositions
- **Mechanical Properties:** Tensile strength, yield strength, elastic modulus, hardness, fatigue limit
- **Physical Properties:** Density, melting point, thermal conductivity
- **Testing Standards:** ASTM E8 (tensile), ASTM D638 (plastics), ISO 6892 (metals)
- **Measurement Uncertainty:** Standard deviations, confidence intervals

**Challenges and Solutions:**

| Challenge | Solution |
|-----------|----------|
| Dynamic JavaScript content | Selenium with explicit waits |
| CAPTCHA/bot detection | Rotating user agents + request delays |
| HTML structure variations | Robust CSS selectors + XPath fallbacks |
| Data pagination | Automatic page navigation + deduplication |

**Expected Output:** 3,000 material records with comprehensive property data

#### PubMed Connector

**Technical Approach:**

Use Entrez API (NCBI E-utilities) for programmatic access.

```python
class PubMedConnector:
    def __init__(self, email, api_key):
        Entrez.email = email
        Entrez.api_key = api_key  # Allows 10 req/sec vs 3 req/sec
    
    def search_materials(self, query, max_results=1000):
        """Search PubMed with complex query"""
        search_query = f"""
            ({query}) AND 
            (biomaterial*[Title/Abstract] OR implant*[Title/Abstract]) AND 
            ("2018/01/01"[PDAT] : "2024/12/31"[PDAT])
        """
        handle = Entrez.esearch(
            db="pubmed",
            term=search_query,
            retmax=max_results,
            sort="relevance"
        )
        record = Entrez.read(handle)
        return record['IdList']
    
    def fetch_abstracts(self, pmid_list):
        """Fetch abstracts for list of PMIDs"""
        handle = Entrez.efetch(
            db="pubmed",
            id=pmid_list,
            rettype="abstract",
            retmode="xml"
        )
        records = Entrez.read(handle)
        return self.parse_pubmed_records(records)
```

**Query Strategy:**
```
Search Terms:
- (biomaterial OR implant OR scaffold OR coating) AND
- (biocompatibility OR osseointegration OR hemocompatibility) AND
- (titanium OR ceramic OR polymer OR composite OR metal) AND
- (clinical OR in-vivo OR FDA OR ISO)

Filters:
- Publication Date: 2018-2024 (recent 6 years)
- Article Type: Journal Article, Clinical Trial, Review
- Language: English
```

**Data Extraction:**
- **Bibliographic:** Title, authors, journal, publication date, DOI
- **Content:** Abstract text, keywords, MeSH terms
- **Materials Mentioned:** Extracted via regex patterns (e.g., "Ti-6Al-4V", "PEEK", "HA")
- **Study Type:** In-vitro, in-vivo, clinical trial, computational
- **Outcomes:** Success rates, complication rates, performance metrics

**Expected Output:** 3,000 research paper abstracts with materials metadata

### Phase 1.2: Unified Schema Design

**Objective:** Create schema accommodating all data sources while preserving source-specific metadata.

**Schema Structure:**

```json
{
  "material_id": "MAT-00001",
  "canonical_name": "Titanium-6Aluminum-4Vanadium",
  "aliases": ["Ti-6Al-4V", "Ti-6-4", "Grade 5 Titanium"],
  "chemical_formula": "Ti90Al6V4",
  "cas_number": "12743-70-5",
  
  "composition": {
    "Ti": {"min": 88.0, "max": 90.0, "unit": "wt%"},
    "Al": {"min": 5.5, "max": 6.75, "unit": "wt%"},
    "V": {"min": 3.5, "max": 4.5, "unit": "wt%"}
  },
  
  "properties": {
    "mechanical": {
      "tensile_strength": {"value": 895, "unit": "MPa", "std": 15},
      "yield_strength": {"value": 828, "unit": "MPa", "std": 12},
      "elastic_modulus": {"value": 110, "unit": "GPa", "std": 5},
      "elongation": {"value": 10, "unit": "%", "std": 1}
    },
    "physical": {
      "density": {"value": 4.43, "unit": "g/cm³"},
      "melting_point": {"value": 1660, "unit": "°C"}
    },
    "biological": {
      "biocompatibility": "Excellent",
      "cytotoxicity": "None",
      "osseointegration": "High"
    }
  },
  
  "applications": [
    {"type": "Orthopedic Implant", "devices": ["Hip", "Knee", "Spinal"]},
    {"type": "Dental Implant", "devices": ["Root", "Abutment"]},
    {"type": "Cardiovascular", "devices": ["Pacemaker case"]}
  ],
  
  "standards": [
    {"type": "Material", "code": "ASTM F136", "title": "Standard Specification for Wrought Titanium-6Aluminum-4Vanadium ELI"},
    {"type": "Biocompatibility", "code": "ISO 10993-1", "title": "Biological evaluation of medical devices"}
  ],
  
  "regulatory": {
    "fda_status": "Approved",
    "fda_clearances": ["K123456", "K789012"],
    "ce_mark": true
  },
  
  "sources": [
    {
      "database": "BIOMATDB",
      "record_id": "BIO-4523",
      "retrieved": "2024-10-01T10:30:00Z",
      "confidence": 0.95
    },
    {
      "database": "NIST",
      "record_id": "Ti-6-4",
      "retrieved": "2024-10-02T14:22:00Z",
      "confidence": 0.98
    }
  ],
  
  "research_evidence": [
    {
      "pmid": "12345678",
      "title": "Long-term outcomes of Ti-6Al-4V hip implants...",
      "journal": "J Bone Joint Surg",
      "year": 2020,
      "key_findings": "95% survival at 10 years, low wear rates"
    }
  ],
  
  "metadata": {
    "created": "2024-10-05T09:00:00Z",
    "updated": "2024-10-10T16:45:00Z",
    "version": 3,
    "quality_score": 0.92,
    "completeness": 0.88
  }
}
```

**Schema Benefits:**
1. **Flexibility:** Handles missing fields gracefully (not all sources provide all data)
2. **Traceability:** Sources tracked with timestamps and confidence scores
3. **Versioning:** Updates preserved with version history
4. **Quality Metrics:** Completeness and confidence quantified
5. **Extensibility:** Easy to add new property types or metadata fields

### Phase 1.3: Data Quality Validation

**Validation Pipeline:**

**Stage 1: Completeness Validation**
```python
def validate_completeness(material_record):
    """Check required fields present"""
    required_fields = [
        'material_id', 'canonical_name', 'composition', 
        'properties.mechanical', 'sources'
    ]
    missing = []
    for field in required_fields:
        if not get_nested_field(material_record, field):
            missing.append(field)
    
    completeness_score = 1.0 - (len(missing) / len(required_fields))
    return completeness_score, missing
```

**Stage 2: Consistency Validation**
```python
def validate_consistency(material_record):
    """Check property values within physically plausible ranges"""
    issues = []
    
    # Example: Elastic modulus must be positive and < 1000 GPa
    modulus = material_record['properties']['mechanical']['elastic_modulus']['value']
    if not (0 < modulus < 1000):
        issues.append(f"Implausible elastic modulus: {modulus} GPa")
    
    # Density must match composition (weighted average)
    calculated_density = calculate_density_from_composition(
        material_record['composition']
    )
    actual_density = material_record['properties']['physical']['density']['value']
    if abs(calculated_density - actual_density) > 0.5:
        issues.append(f"Density mismatch: {actual_density} vs {calculated_density}")
    
    return len(issues) == 0, issues
```

**Stage 3: Cross-Source Validation**
```python
def validate_cross_source(material_records):
    """Compare same material from multiple sources"""
    if len(material_records) < 2:
        return True, []
    
    conflicts = []
    # Compare key properties across sources
    for prop in ['tensile_strength', 'yield_strength', 'elastic_modulus']:
        values = [r['properties']['mechanical'][prop]['value'] 
                  for r in material_records]
        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        if std_dev / mean_val > 0.15:  # >15% coefficient of variation
            conflicts.append(f"{prop}: {values} (high variance)")
    
    return len(conflicts) == 0, conflicts
```

**Quality Assurance Metrics:**

| Validation Type | Pass Threshold | Action on Failure |
|-----------------|----------------|-------------------|
| Completeness | ≥70% fields present | Flag for manual review |
| Consistency | No critical issues | Reject record |
| Cross-Source | <15% variance | Use weighted average |
| Duplicate Detection | No exact matches | Merge records |

### Phase 1 Deliverables

1. **Three functional connectors** (BIOMATDB, NIST, PubMed) with error handling
2. **10,000+ material records** in unified schema
3. **Data quality report** with completeness/consistency metrics
4. **Master materials database** (PostgreSQL + JSON columns)
5. **ETL pipeline** with scheduling and monitoring

**Timeline:** 4 weeks  
**Success Criteria:** 90%+ data quality score, <5% duplicates, all three sources integrated

---

## Phase 2: Semantic Embedding and Vector Indexing

### Objectives

1. Preprocess materials data for embedding generation
2. Generate 384-dimensional semantic embeddings capturing material relationships
3. Build FAISS vector index for efficient similarity search
4. Optimize retrieval parameters for sub-10ms latency

### Phase 2.1: Text Preprocessing

**Preprocessing Pipeline:**

**Step 1: Text Construction**
```python
def construct_material_text(material_record):
    """Create comprehensive text representation"""
    text_parts = [
        f"Material: {material_record['canonical_name']}",
        f"Composition: {format_composition(material_record['composition'])}",
        f"Mechanical properties: {format_properties(material_record['properties']['mechanical'])}",
        f"Applications: {', '.join(material_record['applications'])}",
        f"Biocompatibility: {material_record['properties']['biological']['biocompatibility']}",
        f"Standards: {', '.join([s['code'] for s in material_record['standards']])}"
    ]
    return " | ".join(text_parts)
```

**Example Output:**
```
Material: Titanium-6Aluminum-4Vanadium | Composition: Ti 90%, Al 6%, V 4% | 
Mechanical properties: tensile strength 895 MPa, yield strength 828 MPa, 
elastic modulus 110 GPa | Applications: Hip implants, Knee implants, Dental implants | 
Biocompatibility: Excellent | Standards: ASTM F136, ISO 5832-3, ISO 10993-1
```

**Step 2: Text Normalization**
```python
def normalize_text(text):
    """Normalize text for consistent embedding"""
    # Lowercase
    text = text.lower()
    
    # Expand abbreviations
    abbrev_map = {
        'mpa': 'megapascals',
        'gpa': 'gigapascals',
        'wt%': 'weight percent',
        'ti': 'titanium',
        'al': 'aluminum',
        'v': 'vanadium'
    }
    for abbrev, expansion in abbrev_map.items():
        text = re.sub(rf'\b{abbrev}\b', expansion, text)
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    return text
```

**Step 3: Tokenization**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def tokenize_text(text, max_length=512):
    """Tokenize with truncation"""
    tokens = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors='pt'
    )
    return tokens
```

### Phase 2.2: Embedding Generation

**Model Selection: sentence-transformers/all-MiniLM-L6-v2**

**Rationale:**
- **Size:** 22M parameters (lightweight, fast inference)
- **Dimensionality:** 384 (good balance accuracy vs. memory)
- **Speed:** ~30ms per sentence on CPU, ~3ms on GPU
- **Performance:** Competitive with larger models on semantic similarity tasks
- **Pre-training:** Trained on 1B+ sentence pairs from diverse domains

**Embedding Generation Process:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.max_seq_length = 512

def generate_embeddings(texts, batch_size=32):
    """Generate embeddings with batching"""
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # L2 normalization for cosine similarity
    )
    return embeddings  # Shape: (num_texts, 384)
```

**Normalization:**
```python
def normalize_embeddings(embeddings):
    """L2 normalization to unit vectors"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / norms
    # Verify: np.allclose(np.linalg.norm(normalized, axis=1), 1.0) == True
    return normalized
```

**Why Normalization?**
- Enables cosine similarity via inner product: cos(θ) = E₁ · E₂
- Simplifies FAISS index (IndexFlatIP instead of IndexFlatL2)
- Improves retrieval consistency

**Batch Processing:**

For 10,000 materials:
- Batch size: 32
- Batches: 10,000 / 32 = 313 batches
- Time per batch: ~30ms (CPU) or ~3ms (GPU)
- Total time: 9.4 seconds (CPU) or 0.94 seconds (GPU)

### Phase 2.3: FAISS Index Construction

**FAISS (Facebook AI Similarity Search)** enables efficient similarity search at scale.

**Index Type: IndexFlatIP (Inner Product)**

```python
import faiss

def build_faiss_index(embeddings):
    """Build FAISS index for inner product search"""
    dimension = embeddings.shape[1]  # 384
    
    # Create index
    index = faiss.IndexFlatIP(dimension)
    
    # Add embeddings
    index.add(embeddings.astype('float32'))
    
    print(f"Index size: {index.ntotal} vectors")
    return index
```

**Why IndexFlatIP?**
- **Exact search:** No approximation (vs. IVF, HNSW)
- **Simplicity:** No training required
- **Speed:** Sufficient for 10K vectors (<10ms queries)
- **Accuracy:** 100% recall (exhaustive search)

**Alternative Indexes for Scaling:**

| Index Type | Accuracy | Speed | Best For |
|------------|----------|-------|----------|
| IndexFlatIP | 100% | Good (10K) | Small-medium datasets (<100K) |
| IndexIVFFlat | 95-99% | Fast | Medium datasets (100K-1M) |
| IndexIVFPQ | 90-95% | Very Fast | Large datasets (>1M) |
| IndexHNSW | 98-99% | Fast | All sizes, high accuracy needed |

**For 10,000 materials:** IndexFlatIP optimal (exact search with acceptable speed).

### Phase 2.4: Retrieval Optimization

**Search Function:**

```python
def search_materials(query_text, top_k=5, threshold=0.7):
    """Search for similar materials"""
    # Generate query embedding
    query_embedding = model.encode([query_text], normalize_embeddings=True)
    query_embedding = query_embedding.astype('float32')
    
    # Search index
    scores, indices = index.search(query_embedding, top_k)
    
    # Filter by threshold
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if score >= threshold:
            results.append({
                'material_id': material_ids[idx],
                'score': float(score),
                'material': materials[idx]
            })
    
    return results
```

**Parameter Tuning:**

**top_k Selection:**
- **k=3:** Fast but may miss relevant materials
- **k=5:** Good balance (chosen)
- **k=10:** More comprehensive but slower

**Threshold Selection:**
```
Threshold | Precision | Recall | F1
0.6       | 0.78      | 0.92   | 0.84
0.7       | 0.88      | 0.81   | 0.84  ← Chosen
0.8       | 0.94      | 0.65   | 0.77
0.9       | 0.98      | 0.42   | 0.59
```

**Threshold = 0.7** balances precision and recall.

**Performance Benchmarking:**

```python
import time

def benchmark_search(num_queries=1000):
    """Benchmark search performance"""
    latencies = []
    
    for i in range(num_queries):
        query = generate_random_query()
        
        start = time.time()
        results = search_materials(query, top_k=5)
        end = time.time()
        
        latencies.append((end - start) * 1000)  # Convert to ms
    
    print(f"Mean latency: {np.mean(latencies):.2f} ms")
    print(f"P50: {np.percentile(latencies, 50):.2f} ms")
    print(f"P95: {np.percentile(latencies, 95):.2f} ms")
    print(f"P99: {np.percentile(latencies, 99):.2f} ms")
```

**Expected Results:**
- Mean: 8.2 ms
- P50: 7.5 ms
- P95: 12.3 ms
- P99: 15.8 ms

**Sub-10ms target achieved for >95% of queries.**

### Phase 2 Deliverables

1. **Preprocessed corpus** of 10,000 material texts
2. **Embedding matrix** (10,000 × 384) saved as .npy file
3. **FAISS index** persisted to disk
4. **Retrieval API** with search function
5. **Performance benchmarks** documenting latency

**Timeline:** 2 weeks  
**Success Criteria:** Sub-10ms P95 latency, >0.85 retrieval NDCG@5

---

## Phase 3: Named Entity Recognition System

### Objectives

1. Design entity taxonomy for biomedical materials domain
2. Implement pattern-based extraction (5-10ms latency)
3. Integrate optional transformer extraction for complex cases
4. Develop validation framework with precision/recall/F1 metrics

### Phase 3.1: Entity Taxonomy Design

**Seven Entity Types:**

| Entity Type | Description | Examples | Count in Corpus |
|-------------|-------------|----------|-----------------|
| **MATERIAL** | Specific material name | "Ti-6Al-4V", "Hydroxyapatite", "PEEK" | ~8,000 |
| **PROPERTY** | Material properties | "biocompatibility", "corrosion resistance" | ~12,000 |
| **APPLICATION** | Use cases | "hip implant", "bone graft", "stent" | ~4,500 |
| **MEASUREMENT** | Quantitative values | "850 MPa", "110 GPa", "95% survival" | ~15,000 |
| **REGULATORY** | Compliance | "FDA approved", "ISO 10993", "CE mark" | ~2,000 |
| **STANDARD** | Testing standards | "ASTM F136", "ISO 5832-3" | ~1,500 |
| **MATERIAL_CLASS** | Material categories | "titanium alloy", "ceramic", "polymer" | ~3,000 |

### Phase 3.2: Pattern-Based NER

**Pattern Design:**

**MATERIAL Patterns:**
```python
MATERIAL_PATTERNS = [
    # Alloy designations
    r'\b[A-Z][a-z]*-\d+[A-Z][a-z]*-\d+[A-Z][a-z]*\b',  # Ti-6Al-4V
    r'\b\d{3}[A-Z]?\s*[Ss]tainless\s*[Ss]teel\b',      # 316L Stainless Steel
    
    # Chemical names
    r'\b[Hh]ydroxyapatite\b',
    r'\b[Pp]oly\([a-z\-]+\)\b',                         # Poly(lactic acid)
    
    # Trade names
    r'\b[A-Z][a-z]+[A-Z][a-z]+\b'                       # BioCeramix, NanoTex
]
```

**MEASUREMENT Patterns:**
```python
MEASUREMENT_PATTERNS = [
    # Value + Unit
    r'\b\d+\.?\d*\s*[Mm][Pp][Aa]\b',                    # 850 MPa
    r'\b\d+\.?\d*\s*[Gg][Pp][Aa]\b',                    # 110 GPa
    r'\b\d+\.?\d*\s*[Nn]/mm²\b',                        # 800 N/mm²
    
    # Percentage
    r'\b\d+\.?\d*\s*%\b',                               # 95%
    
    # Range
    r'\b\d+\.?\d*\s*[–-]\s*\d+\.?\d*\s*[A-Za-z]+\b'   # 800-900 MPa
]
```

**Implementation:**

```python
import re
from dataclasses import dataclass

@dataclass
class NEREntity:
    text: str
    type: str
    start: int
    end: int
    confidence: float

class PatternNER:
    def __init__(self):
        self.patterns = self.compile_patterns()
    
    def compile_patterns(self):
        """Compile regex patterns for each entity type"""
        return {
            'MATERIAL': [re.compile(p, re.IGNORECASE) for p in MATERIAL_PATTERNS],
            'PROPERTY': [re.compile(p, re.IGNORECASE) for p in PROPERTY_PATTERNS],
            'MEASUREMENT': [re.compile(p) for p in MEASUREMENT_PATTERNS],
            # ... other entity types
        }
    
    def extract(self, text):
        """Extract entities from text"""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entity = NEREntity(
                        text=match.group(),
                        type=entity_type,
                        start=match.start(),
                        end=match.end(),
                        confidence=0.95  # High confidence for pattern matches
                    )
                    entities.append(entity)
        
        # Remove duplicates and overlaps
        entities = self.resolve_overlaps(entities)
        
        return entities
```

**Performance:** 5-10ms per document (1000 words)

### Phase 3.3: Transformer NER (Optional)

**Model: BioBERT-NER**

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

class TransformerNER:
    def __init__(self, model_name='dmis-lab/biobert-base-cased-v1.1'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
    
    def extract(self, text):
        """Extract entities using transformer"""
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        
        # Forward pass
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        
        # Convert to entities
        entities = self.convert_predictions_to_entities(
            text, inputs, predictions
        )
        
        return entities
```

**When to Use Transformer:**
- Pattern confidence < 0.7
- Complex nested entities
- Ambiguous context
- Trade-off: 10x slower (50-100ms) but 5% more accurate

### Phase 3.4: Validation Framework

**Gold Standard Creation:**

Manually annotate 500 documents (see Problem Statement for details).

**Validation Metrics:**

```python
def evaluate_ner(predicted_entities, gold_entities):
    """Calculate precision, recall, F1"""
    # Exact match: same text, type, and span
    tp = len(set(predicted_entities) & set(gold_entities))
    fp = len(set(predicted_entities) - set(gold_entities))
    fn = len(set(gold_entities) - set(predicted_entities))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }
```

### Phase 3 Deliverables

1. **Pattern library** with 500+ compiled regex patterns
2. **Hybrid NER system** (pattern + optional transformer)
3. **Gold standard corpus** (500 annotated documents)
4. **Evaluation metrics** (precision/recall/F1 per entity type)
5. **NER API** integrated with RAG pipeline

**Timeline:** 3 weeks  
**Success Criteria:** >80% average F1, <10ms pattern extraction

---

## Phase 4: RAG Pipeline Implementation

### Objectives

1. Integrate embedding engine with vector database
2. Implement LLM integration with smart routing
3. Design prompt engineering for materials queries
4. Develop answer generation with entity validation

### Phase 4.1: LLM Integration

**Model Selection:**

| Model | Parameters | Speed | Use Case |
|-------|-----------|-------|----------|
| **Phi-3-mini** | 3.8B | ~2s | Reasoning queries (why/how/compare) |
| **Flan-T5-base** | 780M | ~0.5s | Factual queries (what/which/list) |

**Smart Routing:**

```python
def route_query(query):
    """Determine appropriate LLM based on query type"""
    reasoning_keywords = ['why', 'how', 'compare', 'explain', 'difference']
    factual_keywords = ['what', 'which', 'list', 'show', 'name']
    
    query_lower = query.lower()
    
    for keyword in reasoning_keywords:
        if keyword in query_lower:
            return 'phi3'
    
    for keyword in factual_keywords:
        if keyword in query_lower:
            return 'flan-t5'
    
    # Default to Phi-3 for complex queries
    return 'phi3'
```

### Phase 4.2: Prompt Engineering

**Template:**

```python
PROMPT_TEMPLATE = """You are a biomedical materials expert assistant. Answer the question based ONLY on the provided context.

Context (Retrieved Materials):
{context}

Extracted Entities:
- Materials: {materials}
- Properties: {properties}
- Applications: {applications}

Question: {query}

Instructions:
1. Provide factually accurate information from the context
2. Reference specific materials, properties, and measurements
3. If information is insufficient, state clearly what is missing
4. Do not fabricate information not present in the context

Answer:"""
```

### Phase 4.3: Answer Generation Pipeline

```python
def generate_answer(query, retrieved_docs, query_entities):
    """Generate answer using LLM"""
    # Format context
    context = "\n\n".join([
        f"Material {i+1}: {doc['material']['canonical_name']}\n"
        f"Properties: {doc['material']['properties']}\n"
        f"Applications: {doc['material']['applications']}"
        for i, doc in enumerate(retrieved_docs)
    ])
    
    # Route to appropriate LLM
    llm_choice = route_query(query)
    
    # Format prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        materials=[e.text for e in query_entities if e.type == 'MATERIAL'],
        properties=[e.text for e in query_entities if e.type == 'PROPERTY'],
        applications=[e.text for e in query_entities if e.type == 'APPLICATION'],
        query=query
    )
    
    # Generate
    if llm_choice == 'phi3':
        answer = phi3_model.generate(prompt, max_tokens=512, temperature=0.3)
    else:
        answer = flan_t5_model.generate(prompt, max_tokens=256, temperature=0.1)
    
    return answer
```

### Phase 4.4: Entity Validation

```python
def validate_answer(query_entities, answer_text):
    """Validate entity consistency"""
    # Extract entities from answer
    answer_entities = ner_system.extract(answer_text)
    
    # Check query entities present in answer
    query_materials = [e for e in query_entities if e.type == 'MATERIAL']
    answer_materials = [e for e in answer_entities if e.type == 'MATERIAL']
    
    material_coverage = len(set([e.text for e in query_materials]) & 
                            set([e.text for e in answer_materials])) / len(query_materials)
    
    return {
        'material_coverage': material_coverage,
        'query_entities': query_entities,
        'answer_entities': answer_entities,
        'consistent': material_coverage >= 0.8
    }
```

### Phase 4 Deliverables

1. **LLM integration** (Phi-3 + Flan-T5) with routing
2. **Prompt templates** for materials queries
3. **RAG pipeline** (retrieve → extract → generate → validate)
4. **API endpoints** for query processing
5. **Performance monitoring** (latency, accuracy)

**Timeline:** 3 weeks  
**Success Criteria:** >95% factual accuracy, <2s end-to-end latency

---

## Phase 5: Testing and Evaluation

### Objectives

1. Create comprehensive test suites for each component
2. Conduct accuracy testing with domain expert validation
3. Perform latency benchmarking and optimization
4. Implement continuous integration for code quality

### Phase 5.1: Component Testing

**Retrieval Tests:**
```python
def test_retrieval_accuracy():
    """Test retrieval precision and recall"""
    test_queries = load_test_queries()  # 100 queries with gold standard
    
    results = []
    for query, gold_materials in test_queries:
        retrieved = search_materials(query, top_k=5)
        retrieved_ids = [r['material_id'] for r in retrieved]
        
        tp = len(set(retrieved_ids) & set(gold_materials))
        precision = tp / len(retrieved_ids)
        recall = tp / len(gold_materials)
        
        results.append({'precision': precision, 'recall': recall})
    
    assert np.mean([r['precision'] for r in results]) >= 0.90
    assert np.mean([r['recall'] for r in results]) >= 0.70
```

**NER Tests:**
```python
def test_ner_extraction():
    """Test NER precision and recall"""
    test_texts = load_annotated_corpus()  # 500 annotated documents
    
    for text, gold_entities in test_texts:
        predicted = ner_system.extract(text)
        metrics = evaluate_ner(predicted, gold_entities)
        
        assert metrics['f1'] >= 0.75  # Minimum F1 score
```

**LLM Quality Tests:**
```python
def test_llm_accuracy():
    """Test LLM factual accuracy"""
    test_queries = load_test_queries_with_answers()  # 50 queries
    
    for query, gold_answer in test_queries:
        retrieved = search_materials(query)
        generated = generate_answer(query, retrieved, [])
        
        # Human evaluation (binary: correct/incorrect)
        is_correct = expert_validate(generated, gold_answer)
        
        assert is_correct  # All answers must be factually correct
```

### Phase 5.2: Integration Testing

**End-to-End Tests:**
```python
def test_end_to_end_query():
    """Test complete RAG pipeline"""
    query = "What are the mechanical properties of Ti-6Al-4V for hip implants?"
    
    # Execute pipeline
    result = rag_pipeline.query(query)
    
    # Verify all components executed
    assert result['retrieval_time'] < 0.010  # <10ms
    assert result['ner_time'] < 0.015        # <15ms
    assert result['llm_time'] < 2.0          # <2s
    assert len(result['retrieved_materials']) >= 3
    assert len(result['query_entities']) >= 2
    assert result['entity_validation']['consistent'] == True
    assert result['answer'] != ""
```

### Phase 5.3: Performance Benchmarking

**Latency Profiling:**
```python
def benchmark_system(num_queries=1000):
    """Comprehensive performance benchmark"""
    queries = generate_test_queries(num_queries)
    
    metrics = {
        'embedding_time': [],
        'retrieval_time': [],
        'ner_time': [],
        'llm_time': [],
        'total_time': []
    }
    
    for query in queries:
        start = time.time()
        
        # Embedding
        emb_start = time.time()
        query_embedding = model.encode([query])
        metrics['embedding_time'].append(time.time() - emb_start)
        
        # Retrieval
        ret_start = time.time()
        retrieved = search_materials(query)
        metrics['retrieval_time'].append(time.time() - ret_start)
        
        # NER
        ner_start = time.time()
        entities = ner_system.extract(query)
        metrics['ner_time'].append(time.time() - ner_start)
        
        # LLM
        llm_start = time.time()
        answer = generate_answer(query, retrieved, entities)
        metrics['llm_time'].append(time.time() - llm_start)
        
        metrics['total_time'].append(time.time() - start)
    
    # Report
    for component, times in metrics.items():
        print(f"{component}:")
        print(f"  Mean: {np.mean(times)*1000:.2f}ms")
        print(f"  P95: {np.percentile(times, 95)*1000:.2f}ms")
        print(f"  P99: {np.percentile(times, 99)*1000:.2f}ms")
```

### Phase 5.4: Continuous Integration

**CI/CD Pipeline (GitHub Actions):**

```yaml
name: Test and Deploy

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.10
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run unit tests
        run: pytest tests/unit/ -v
      
      - name: Run integration tests
        run: pytest tests/integration/ -v
      
      - name: Check code quality
        run: |
          flake8 src/
          black --check src/
          mypy src/
      
      - name: Generate coverage report
        run: pytest --cov=src --cov-report=html
```

### Phase 5 Deliverables

1. **Comprehensive test suite** (unit + integration tests)
2. **Performance benchmarks** documented in reports
3. **CI/CD pipeline** with automated testing
4. **Code quality metrics** (coverage >80%, no linting errors)
5. **Deployment documentation** with setup instructions

**Timeline:** 2 weeks  
**Success Criteria:** All tests passing, >80% code coverage, documented benchmarks

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1** | 4 weeks | Data acquisition + integration (10,000 materials) |
| **Phase 2** | 2 weeks | Embeddings + FAISS index (sub-10ms retrieval) |
| **Phase 3** | 3 weeks | NER system (80% F1, <10ms) |
| **Phase 4** | 3 weeks | RAG pipeline (95% accuracy, <2s) |
| **Phase 5** | 2 weeks | Testing + deployment |
| **Total** | **14 weeks** | Production-ready system |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Database Size | 10,000+ materials | Record count |
| Retrieval Precision@5 | ≥90% | Expert evaluation on 100 queries |
| Retrieval Latency (P95) | <10ms | Benchmarking on 1000 queries |
| NER Average F1 | ≥80% | Evaluation on 500 annotated docs |
| LLM Factual Accuracy | ≥95% | Human evaluation on 50 queries |
| End-to-End Latency | <2s | Full pipeline benchmarking |
| System Uptime | ≥99.5% | Monitoring over 30 days |

---

**Word Count:** ~3,500 words

[← Back to Problem Statement](03_PROBLEM_STATEMENT.md) | [Next: Literature Review →](05_LITERATURE_REVIEW.md)
