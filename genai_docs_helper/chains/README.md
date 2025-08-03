# LangChain Components Module 🔗

This module contains specialized LangChain chains and prompts that power the intelligence behind document grading, answer generation, and quality checks.

## 📋 Overview

Each chain is designed as a reusable component that combines prompts, LLMs, and output parsing to perform specific NLP tasks in the RAG pipeline.

## 🧩 Components

### 1. **generation.py** - Answer Generation Chain

Generates comprehensive answers using retrieved documents as context.

**Features:**
- Uses hub-based prompt template for consistency
- Structured output parsing
- Context-aware generation

**Standalone Usage:**
```python
from genai_docs_helper.chains.generation import generation_chain

context = ["Document 1 content...", "Document 2 content..."]
question = "What is machine learning?"

answer = generation_chain.invoke({
    "context": context,
    "question": question
})
print(answer)
```

### 2. **retrieval_grader.py** - Document Relevance Grader

Evaluates if a document is relevant to a given question.

**Features:**
- Binary scoring (yes/no)
- Lenient grading to avoid over-filtering
- Structured output with Pydantic models

**Standalone Usage:**
```python
from genai_docs_helper.chains.retrieval_grader import retrieval_grader

result = retrieval_grader.invoke({
    "question": "What are neural networks?",
    "document": "Neural networks are computing systems inspired by biological neural networks..."
})
print(f"Relevant: {result.binary_score}")  # Output: yes/no
```

### 3. **batch_grader.py** - Batch Document Grader

Grades multiple documents simultaneously for efficiency.

**Features:**
- Processes up to 5 documents in one LLM call
- Confidence scoring (0.0-1.0) for each document
- Structured output with document indices

**Standalone Usage:**
```python
from genai_docs_helper.chains.batch_grader import batch_document_grader

documents = [
    "Document 1 about ML...",
    "Document 2 about data...",
    "Document 3 about models..."
]

result = batch_document_grader.invoke({
    "question": "Machine learning techniques",
    "documents": documents
})

for score in result.scores:
    print(f"Doc {score.document_index}: relevant={score.is_relevant}, confidence={score.confidence}")
```

### 4. **hallucination_grader.py** - Hallucination Detector

Checks if generated answers are grounded in the provided documents.

**Features:**
- Binary scoring (grounded/not grounded)
- Lenient evaluation to reduce false positives
- Semantic understanding of paraphrasing

**Standalone Usage:**
```python
from genai_docs_helper.chains.hallucination_grader import hallucination_grader

result = hallucination_grader.invoke({
    "documents": "Python is a high-level programming language...",
    "generation": "Python is an interpreted language known for its simplicity."
})
print(f"Grounded: {result.binary_score}")  # True/False
```

### 5. **answer_grader.py** - Answer Quality Checker

Evaluates if an answer adequately addresses the user's question.

**Features:**
- Binary assessment (addresses/doesn't address)
- Question-answer alignment checking

**Standalone Usage:**
```python
from genai_docs_helper.chains.answer_grader import answer_grader

result = answer_grader.invoke({
    "question": "What is Python?",
    "generation": "Python is a versatile programming language..."
})
print(f"Addresses question: {result.binary_score}")  # True/False
```

### 6. **paraphraser.py** - Question Paraphraser

Reformulates questions while preserving intent.

**Features:**
- Maintains semantic meaning
- Improves retrieval by varying phrasing
- Clean output without explanations

**Standalone Usage:**
```python
from genai_docs_helper.chains.paraphraser import paraphraser_chain

original = "What ML models did we use?"
paraphrased = paraphraser_chain.invoke({"question": original})
print(f"Original: {original}")
print(f"Paraphrased: {paraphrased}")
```

### 7. **query_expander.py** - Query Expansion Chain

Generates multiple query variations for comprehensive retrieval.

**Features:**
- Creates 3-5 variations per query
- Includes synonyms and related concepts
- Maintains query intent

**Standalone Usage:**
```python
from genai_docs_helper.chains.query_expander import query_expander_chain

variations = query_expander_chain.invoke({
    "question": "best practices for demand forecasting"
})
for i, var in enumerate(variations):
    print(f"{i+1}. {var}")
```

### 8. **confidence_scorer.py** - Document Confidence Scorer

Provides detailed confidence scores for document relevance.

**Features:**
- Granular scoring (0.0-1.0)
- Reasoning explanation
- Multiple evaluation criteria

**Standalone Usage:**
```python
from genai_docs_helper.chains.confidence_scorer import confidence_scorer

result = confidence_scorer.invoke({
    "question": "How to implement caching?",
    "document": "Caching can be implemented using Redis..."
})
print(f"Confidence: {result.confidence_score:.2f}")
print(f"Reasoning: {result.reasoning}")
```

## 🔧 Configuration

All chains use the configured LLM from `config.py`:

```python
from genai_docs_helper.config import llm

# Chains automatically use the configured LLM
# Switch between Ollama and OpenAI in config.py
```

## 🎯 Prompt Engineering

Each chain uses carefully crafted prompts:

```python
# Example from retrieval_grader.py
system = """You are a grader assessing relevance...
Be LENIENT in your grading...
Only mark as 'no' if completely unrelated..."""
```

## 🏃 Testing Chains

Test individual chains:

```bash
# Test generation chain
python genai_docs_helper/chains/generation.py

# Test paraphraser
python genai_docs_helper/chains/paraphraser.py

# Test query expander
python genai_docs_helper/chains/query_expander.py
```

## 📊 Performance Tips

1. **Batching**: Use `batch_grader` for multiple documents to reduce API calls
2. **Caching**: Chain outputs are automatically cached when enabled
3. **Temperature**: Set to 0 for consistent, deterministic outputs
4. **Model Selection**: Use faster models (gpt-3.5-turbo) for grading, better models (gpt-4) for generation

## 🔄 Chain Composition

Chains can be composed for complex workflows:

```python
from langchain.schema.runnable import RunnableSequence

# Create a pipeline
pipeline = (
    query_expander_chain 
    | RunnablePassthrough.assign(
        expanded=lambda x: x
    )
)
```

## 🐛 Debugging

Enable verbose output:

```python
from langchain.globals import set_verbose
set_verbose(True)

# Now chain invocations will show detailed execution
```
