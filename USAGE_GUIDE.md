# Health Materials RAG System - Usage Guide

## 🚀 Quick Start

The Health Materials RAG System now includes **LLM integration** for natural language answer generation!

### Run the Interactive System

```bash
python main.py
```

This starts the full interactive chat interface with:
- 🤖 LLM-powered answer generation
- 🔍 Semantic search across 10,000+ materials
- 💬 Smart query routing (materials → RAG, general → direct LLM)
- 📝 Conversation history tracking

### Run Quick Demo (Retrieval Only)

```bash
python main_demo.py
```

Shows basic retrieval capabilities without LLM.

---

## 📖 How It Works

### 1. **Smart Query Routing**

The system automatically detects the type of query:

**Materials Query** → Uses RAG Pipeline
```
You: What materials are used for orthopedic implants?
→ Retrieves relevant materials from database
→ Generates answer using retrieved context + LLM
```

**General Question** → Direct LLM Chat
```
You: How does biocompatibility testing work?
→ Uses LLM directly (no retrieval needed)
```

### 2. **Two Operating Modes**

#### Full Mode (with LLM)
- Natural language answer generation
- Context-aware responses from retrieved materials
- General conversation capability
- **Requires**: `pip install transformers torch`

#### Retrieval-Only Mode
- Fast semantic search
- Returns ranked material documents
- No answer generation
- **No extra dependencies needed**

---

## 💬 Interactive Commands

Once you run `python main.py`, you can use:

| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/examples` | Show example queries |
| `/history` | View conversation history |
| `/stats` | System statistics and status |
| `/mode` | Check current operating mode |
| `/clear` | Clear conversation history |
| `/quit` | Exit the system |

---

## 🎯 Example Queries

### Materials Queries (Uses RAG)
```
• What materials are used for orthopedic implants?
• Tell me about titanium alloys for medical devices
• Which materials have excellent biocompatibility?
• What are the properties of hydroxyapatite?
• Find materials for cardiovascular stents
• What are FDA approved polymers for implants?
```

### General Questions (Direct LLM)
```
• How does biocompatibility testing work?
• What is the difference between alloys and composites?
• Explain the importance of corrosion resistance
• Hello, how are you?
```

---

## 🔧 Installation

### Basic (Retrieval Only)
```bash
pip install sentence-transformers faiss-cpu pandas numpy
```

### Full (with LLM)
```bash
pip install sentence-transformers faiss-cpu pandas numpy transformers torch
```

**Note**: First LLM load takes 1-2 minutes to download the model.

---

## 🤖 LLM Models

When starting `main.py`, you can choose from:

1. **Phi-3-mini (3.8B)** - Best quality, recommended
   - Model: `microsoft/Phi-3-mini-4k-instruct`
   - Size: ~7GB
   - Speed: Medium

2. **Flan-T5-large (770M)** - Faster, good quality
   - Model: `google/flan-t5-large`
   - Size: ~3GB
   - Speed: Fast

3. **Custom** - Any HuggingFace model
   - Enter custom model name

---

## 📊 System Architecture

```
User Query
    ↓
Smart Router (is_material_query?)
    ↓
    ├─→ YES: Materials Query
    │        ↓
    │   RAG Pipeline:
    │   1. Semantic Search (retrieve top-k materials)
    │   2. Create prompt with retrieved context
    │   3. LLM generates answer
    │   4. Return answer + sources
    │
    └─→ NO: General Question
             ↓
        Direct LLM Chat:
        1. Add conversation history
        2. LLM generates response
        3. Return answer
```

---

## 🎓 Code Structure

```
Project/
├── main.py                           # 🎯 Main interactive interface
├── main_demo.py                      # Quick demo script
├── src/
│   └── rag_pipeline/
│       └── health_materials_rag_demo.py  # Core RAG class with LLM
└── data/
    └── rag_optimized/               # Database files
```

### Key Methods

```python
# Initialize
rag = HealthMaterialsRAG()
rag.load_database()
rag.load_llm()  # Optional

# Semantic search (retrieval only)
results = rag.semantic_search("titanium alloys", top_k=5)

# Generate answer with RAG (retrieval + LLM)
result = rag.generate_answer("What materials for implants?")

# Smart chat (automatic routing)
result = rag.chat("Tell me about biocompatible materials")
```

---

## ⚡ Performance

- **Retrieval Speed**: <10ms average
- **Full RAG (retrieval + generation)**: ~2-5 seconds (depends on LLM model)
- **Database Size**: 10,000+ materials, 3,000+ research papers
- **Memory Usage**: 
  - Retrieval-only: ~500MB
  - With LLM: 3-8GB (depends on model)

---

## 🔍 Features

✅ **Semantic Search**
- Vector similarity search with FAISS
- 384-dimensional embeddings
- Normalized cosine similarity

✅ **LLM Integration**
- Multiple model options
- Context-aware generation
- Conversation history

✅ **Smart Routing**
- Automatic query classification
- Materials queries → RAG
- General questions → Direct LLM

✅ **Interactive Interface**
- Real-time chat
- Command system
- History tracking

✅ **Production Ready**
- Error handling
- Fallback modes
- Performance monitoring

---

## 💡 Tips

1. **First run?** Let the system load the LLM (1-2 min), then it's fast!

2. **No GPU?** System works fine on CPU, just slower generation

3. **Memory issues?** Use Flan-T5-large (smaller model) or retrieval-only mode

4. **Want faster answers?** Skip LLM loading for instant retrieval results

5. **Testing?** Use `main_demo.py` for quick verification

---

## 📝 Example Session

```
$ python main.py

🏥  HEALTH MATERIALS RAG SYSTEM WITH LLM INTEGRATION
======================================================================
💎 10,000+ Biomedical Materials | 3,000+ Research Papers
🤖 LLM-Powered Answer Generation | Smart Query Routing
⚡ Sub-10ms Retrieval | Natural Language Responses
======================================================================

🔄 Initializing Health Materials RAG System...
📥 Loading database...
   ✓ Loading sentence transformer (all-MiniLM-L6-v2)...
   ✓ Loading FAISS search index...
   ✓ Loading corpus and metadata...
✅ Database loaded: 13,245 documents indexed

🤖 LLM Integration Available!
   Would you like to load the LLM for answer generation?
   
   Load LLM? [Y/n]: y
   
   Select LLM model:
   1. Phi-3-mini (3.8B) - Best quality, recommended
   2. Flan-T5-large (770M) - Faster, good quality
   
   Choice [1]: 1
   
🤖 Loading LLM for answer generation (microsoft/Phi-3-mini-4k-instruct)...
   ✓ Using device: CPU
✅ LLM loaded successfully!

💬 You can now chat with the system!

You: What materials are best for orthopedic implants?

🔄 Processing...

Assistant (RAG - materials query detected): Based on the retrieved materials, titanium alloys such as Ti-6Al-4V are excellent choices for orthopedic implants due to their biocompatibility, corrosion resistance, and mechanical properties similar to bone. Stainless steel 316L and cobalt-chromium alloys are also commonly used...

📚 Sources (5):
   1. Ti-6Al-4V Titanium Alloy (Score: 0.876)
   2. 316L Stainless Steel (Score: 0.823)
   3. CoCrMo Alloy for Hip Implants (Score: 0.791)

⚡ Processing time: 3247.5ms

You: /quit

👋 Thank you for using Health Materials RAG System!
```

---

## 🛠️ Troubleshooting

**Import Error**: `ModuleNotFoundError: No module named 'transformers'`
- Solution: `pip install transformers torch`

**Memory Error**: Out of memory when loading LLM
- Solution: Use smaller model (Flan-T5) or retrieval-only mode

**Slow Generation**: LLM takes too long
- Solution: Use GPU if available, or switch to faster model

**Database Not Found**: Can't find rag_optimized directory
- Solution: Run `python src/health_materials_rag_setup.py` first

---

## 📞 Support

- Check `/help` for commands
- Check `/examples` for query ideas
- Check `/stats` for system status

---

**Ready to explore 10,000+ biomedical materials with AI-powered search! 🚀**
