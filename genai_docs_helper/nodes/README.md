# Graph Nodes Module 🔄

This module contains the core processing nodes used in the LangGraph workflow. Each node represents a distinct processing step in the document retrieval and generation pipeline.

## 📋 Overview

The nodes implement the CRAG (Corrective Retrieval-Augmented Generation) pattern, providing intelligent document retrieval with self-correction capabilities.

## 🧩 Components

### 1. **retrieve.py** - Document Retrieval Node

The most complex node in the system, implementing multi-strategy document retrieval.

**Features:**
- **Three Retrieval Strategies:**
  - Fast path: 20 documents, optimized for speed
  - Standard path: 50 documents, balanced approach
  - Comprehensive path: 100 documents with query expansion
- **Parallel Processing**: Retrieves documents for multiple query variations simultaneously
- **Semantic Reranking**: Uses embeddings to reorder documents by relevance
- **Intelligent Caching**: Dual-layer cache with Redis and memory fallback

**Standalone Usage:**
```python
from genai_docs_helper.nodes.retrieve import retrieve
from genai_docs_helper.state import GraphState

state = {
    "question": "What are the best practices for demand forecasting?",
    "original_question": "What are the best practices for demand forecasting?",
    "documents": [],
    "error_log": []
}

result = retrieve(state)
print(f"Retrieved {len(result['documents'])} documents")
```

### 2. **grade_documents.py** - Document Grading Node

Evaluates retrieved documents for relevance to the user's question.

**Features:**
- **Batch Processing**: Grades multiple documents simultaneously for efficiency
- **Confidence Scoring**: Each document gets a relevance score (0.0-1.0)
- **Early Stopping**: Stops grading when sufficient high-quality documents are found
- **Fallback Mechanisms**: Graceful handling of grading failures

**Standalone Usage:**
```python
from genai_docs_helper.nodes.grade_documents import grade_documents

# Assuming you have retrieved documents
state = {
    "question": "What is machine learning?",
    "documents": retrieved_documents,  # From retrieve node
    "performance_metrics": {}
}

result = grade_documents(state)
print(f"Filtered to {len(result['documents'])} relevant documents")
print(f"Overall confidence: {result['confidence_score']:.2f}")
```

### 3. **generate.py** - Answer Generation Node

Generates answers using the filtered documents as context.

**Features:**
- **Context Limiting**: Automatically limits to top 10 documents for faster generation
- **Answer Caching**: Caches generated answers for identical questions
- **Performance Tracking**: Monitors generation time and document usage
- **Error Handling**: Graceful fallback messages on generation failure

**Standalone Usage:**
```python
from genai_docs_helper.nodes.generate import generate

state = {
    "question": "Explain neural networks",
    "documents": relevant_documents,  # From grade_documents node
    "history": [],
    "cache_key": "unique_key"
}

result = generate(state)
print(f"Generated answer: {result['generation'][:200]}...")
```

### 4. **paraphrase.py** - Query Reformulation Node

Reformulates questions when initial retrieval fails to find relevant documents.

**Features:**
- **Intelligent Paraphrasing**: Uses LLM to rephrase while maintaining intent
- **Retry Tracking**: Monitors retry attempts to prevent infinite loops
- **Fallback Strategies**: Simple transformations when LLM paraphrasing fails
- **State Reset**: Clears previous retrieval artifacts for fresh attempt

**Standalone Usage:**
```python
from genai_docs_helper.nodes.paraphrase import paraphrase

state = {
    "question": "ML models in project",
    "original_question": "ML models in project",
    "retry_count": 0,
    "error_log": []
}

result = paraphrase(state)
print(f"Paraphrased to: {result['question']}")
```

## 🔧 Configuration

Node behavior can be configured in `config.py`:

```python
# Retrieval settings
RETRIEVAL_CONFIGS = {
    "fast": {"k": 20},
    "standard": {"k": 50},
    "comprehensive": {"k": 100}
}

# Grading settings
GRADING_CONFIG = {
    "batch_size": 5,
    "min_relevant_docs": 5,
    "confidence_threshold": 0.7,
    "early_stopping_enabled": True
}
```

## 🏃 Running Nodes Independently

Each node can be tested independently for development and debugging:

```bash
# Test retrieval node
python -c "
from genai_docs_helper.nodes.retrieve import retrieve
state = {'question': 'test question', 'documents': []}
print(retrieve(state))
"

# Test with custom configuration
python -c "
from genai_docs_helper.nodes.grade_documents import GRADING_CONFIG
GRADING_CONFIG['batch_size'] = 10
# Run grading with new config
"
```

## 📊 Performance Characteristics

| Node | Typical Execution Time | Memory Usage | API Calls |
|------|----------------------|--------------|-----------|
| Retrieve | 1-3s (fast), 3-5s (comprehensive) | ~50MB | 0 (uses embeddings) |
| Grade Documents | 1-2s per 20 docs | ~20MB | 1 per 5 docs |
| Generate | 2-4s | ~100MB | 1 |
| Paraphrase | 0.5-1s | ~10MB | 1 |

## 🐛 Debugging

Enable detailed logging for any node:

```python
import logging
logging.getLogger("genai_docs_helper.nodes.retrieve").setLevel(logging.DEBUG)
```

## 🔄 Integration with LangGraph

Nodes are designed to work seamlessly with LangGraph's state management:

```python
from langgraph.graph import StateGraph
from genai_docs_helper.nodes import retrieve, grade_documents, generate

workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate)
```
