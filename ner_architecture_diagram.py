"""
Visual representation of NER integration in RAG pipeline
"""

PIPELINE_FLOW = """
╔══════════════════════════════════════════════════════════════════════════╗
║                     HEALTH MATERIALS RAG PIPELINE                        ║
║                    WITH NER VALIDATION INTEGRATED                        ║
╚══════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│  USER QUERY: "What are properties of Ti-6Al-4V for orthopedic implants?"│
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │    NER EXTRACTION (Query)              │
        │    • Pattern matching                  │
        │    • Optional transformer              │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    QUERY ENTITIES EXTRACTED            │
        │    • Ti-6Al-4V (MATERIAL, 0.95)       │
        │    • orthopedic implants (APP, 0.90)  │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    SEMANTIC SEARCH (FAISS)             │
        │    • Encode query with MiniLM-L6-v2   │
        │    • Search 10,000+ materials         │
        │    • Retrieve top-k matches            │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    CONTEXT MATERIALS                   │
        │    1. Ti-6Al-4V (score: 0.95)         │
        │    2. Ti-6Al-7Nb (score: 0.82)        │
        │    3. Ti-15Mo (score: 0.78)           │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    ANSWER GENERATION (LLM)             │
        │    • Build prompt with context        │
        │    • Generate response                │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    GENERATED ANSWER                    │
        │    "Ti-6Al-4V has Young's modulus of  │
        │     110 GPa, excellent biocompat..."   │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    NER EXTRACTION (Answer)             │
        │    • Extract from generated text      │
        │    • Confidence scoring               │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    ANSWER ENTITIES EXTRACTED           │
        │    • Ti-6Al-4V (MATERIAL, 0.95)       │
        │    • Young's modulus (PROPERTY, 0.85) │
        │    • 110 GPa (MEASUREMENT, 0.92)      │
        │    • biocompatibility (PROPERTY, 0.88)│
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    ENTITY VALIDATION                   │
        │    • Check expected types             │
        │    • Calculate metrics                │
        │    • Error analysis                   │
        └────────────────┬───────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────────────┐
        │    VALIDATION METRICS                  │
        │    • Total entities: 4                │
        │    • Avg confidence: 0.90             │
        │    • Entity overlap: [Ti-6Al-4V]      │
        │    • Distribution: {MATERIAL: 1, ...} │
        └────────────────┬───────────────────────┘
                         │
                         ▼
╔════════════════════════════════════════════════════════════════════════╗
║                            RAG RESULT                                  ║
╠════════════════════════════════════════════════════════════════════════╣
║  query: "What are properties..."                                      ║
║  answer: "Ti-6Al-4V has Young's modulus..."                          ║
║  confidence: 0.85                                                     ║
║  processing_time_ms: 145.3                                           ║
║                                                                        ║
║  query_entities: [                                                    ║
║    NEREntity(text='Ti-6Al-4V', label='MATERIAL', confidence=0.95)   ║
║    NEREntity(text='orthopedic implants', label='APPLICATION', ...)   ║
║  ]                                                                     ║
║                                                                        ║
║  answer_entities: [                                                   ║
║    NEREntity(text='Ti-6Al-4V', label='MATERIAL', confidence=0.95)   ║
║    NEREntity(text='Young's modulus', label='PROPERTY', ...)         ║
║    NEREntity(text='110 GPa', label='MEASUREMENT', ...)              ║
║    NEREntity(text='biocompatibility', label='PROPERTY', ...)        ║
║  ]                                                                     ║
║                                                                        ║
║  entity_validation: {                                                 ║
║    'total_entities': 4,                                              ║
║    'avg_confidence': 0.90,                                           ║
║    'entity_distribution': {'MATERIAL': 1, 'PROPERTY': 2, ...}       ║
║  }                                                                     ║
╚════════════════════════════════════════════════════════════════════════╝
"""

COMPONENT_ARCHITECTURE = """
╔══════════════════════════════════════════════════════════════════════════╗
║                        COMPONENT ARCHITECTURE                            ║
╚══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                      MaterialsRAGPipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Embedding Components:                                                  │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  • embedding_model (all-MiniLM-L6-v2)                   │          │
│  │  • retrieval_index (FAISS IndexFlatIP)                  │          │
│  │  • 10,000+ health materials database                    │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  LLM Components:                                                        │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  • llm_model (GPT-2 / DialoGPT)                         │          │
│  │  • llm_tokenizer                                         │          │
│  │  • Prompt templates for different query types           │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  NEW: NER Components:                                                   │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  • ner_extractor (NERExtractor)                         │          │
│  │    ├─ Pattern-based extraction (5-10ms)                 │          │
│  │    ├─ Transformer-based (optional, 50-100ms)            │          │
│  │    └─ 7 entity types                                    │          │
│  │                                                          │          │
│  │  • ner_validator (NERValidator)                         │          │
│  │    ├─ Gold standard comparison                          │          │
│  │    ├─ Precision/Recall/F1 metrics                       │          │
│  │    ├─ Error analysis                                    │          │
│  │    └─ Entity statistics tracking                        │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  Methods:                                                               │
│  • async query(question) → RAGResult                                   │
│    └─ Now extracts entities from query & answer                       │
│  • get_entity_summary(result) → Dict [NEW]                            │
│  • get_performance_stats() → Dict                                      │
│    └─ Now includes NER statistics                                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                         NER Module (ner_validator.py)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  NERExtractor:                                                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Methods:                                                │          │
│  │  • extract(text) → List[NEREntity]                      │          │
│  │  • extract_with_context(text, query_type) → Dict       │          │
│  │                                                          │          │
│  │  Extraction Methods:                                     │          │
│  │  1. Pattern-based (fast, always active)                 │          │
│  │     - Regex patterns for 7 entity types                 │          │
│  │     - 5-10ms latency                                    │          │
│  │                                                          │          │
│  │  2. Transformer-based (optional)                        │          │
│  │     - d4data/biomedical-ner-all                         │          │
│  │     - 50-100ms latency                                  │          │
│  │     - Higher accuracy                                   │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  NERValidator:                                                          │
│  ┌──────────────────────────────────────────────────────────┐          │
│  │  Methods:                                                │          │
│  │  • validate_entities(extracted, expected) → Dict        │          │
│  │  • compare_with_gold_standard(pred, gold) → Result     │          │
│  │  • get_statistics() → Dict                              │          │
│  │                                                          │          │
│  │  Validation Methods:                                     │          │
│  │  • Exact match (span + label)                           │          │
│  │  • Partial match (IoU ≥ 0.5)                            │          │
│  │  • Label match (overlap + label)                        │          │
│  │                                                          │          │
│  │  Metrics:                                                │          │
│  │  • Precision, Recall, F1                                │          │
│  │  • Confusion matrix                                      │          │
│  │  • Error analysis                                       │          │
│  └──────────────────────────────────────────────────────────┘          │
│                                                                          │
│  Data Structures:                                                       │
│  • NEREntity(text, label, start, end, confidence, method)              │
│  • NERValidationResult(precision, recall, f1, ...)                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
"""

ENTITY_TYPES = """
╔══════════════════════════════════════════════════════════════════════════╗
║                        SUPPORTED ENTITY TYPES                            ║
╚══════════════════════════════════════════════════════════════════════════╝

┌────────────────────────────────────────────────────────────────────────┐
│  1. MATERIAL                                                           │
│     Examples: Ti-6Al-4V, 316L, PEEK, CoCr                            │
│     Pattern: [A-Z][a-z]*-?\d+[A-Z]*...                                │
│     Use: Identify specific material formulations                       │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  2. MATERIAL_CLASS                                                     │
│     Examples: titanium alloy, stainless steel, polymer                │
│     Pattern: (titanium|stainless|polymer|ceramic) ...                 │
│     Use: Identify general material categories                          │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  3. PROPERTY                                                           │
│     Examples: biocompatibility, Young's modulus, corrosion resistance │
│     Pattern: (biocompatibility|modulus|conductivity) ...              │
│     Use: Identify material properties                                  │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  4. MEASUREMENT                                                        │
│     Examples: 110 GPa, 37°C, 5 mm                                     │
│     Pattern: \d+\.?\d*\s*(GPa|MPa|mm|°C) ...                         │
│     Use: Quantify property values                                      │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  5. APPLICATION                                                        │
│     Examples: orthopedic implants, cardiac stents, dental crowns      │
│     Pattern: (orthopedic|cardiac|dental) (implants|stents|devices)    │
│     Use: Identify medical applications                                 │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  6. REGULATORY                                                         │
│     Examples: FDA, CE, ISO                                            │
│     Pattern: (FDA|CE|ISO) ...                                          │
│     Use: Track regulatory approvals                                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│  7. STANDARD                                                           │
│     Examples: ISO 10993, ASTM F136                                    │
│     Pattern: (ISO|ASTM|ASME)\s*[A-Z]?\d+ ...                         │
│     Use: Reference industry standards                                  │
└────────────────────────────────────────────────────────────────────────┘
"""

if __name__ == "__main__":
    print(PIPELINE_FLOW)
    print("\n" + "="*80 + "\n")
    print(COMPONENT_ARCHITECTURE)
    print("\n" + "="*80 + "\n")
    print(ENTITY_TYPES)
