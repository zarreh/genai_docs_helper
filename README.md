# 🧠 GenAI Docs Helper

**GenAI Docs Helper** is an intelligent document assistant designed to help users and developers efficiently retrieve relevant content from large collections of Confluence documents using an agentic Retrieval-Augmented Generation (RAG) approach. This project demonstrates how to build a robust, iterative, and explainable RAG pipeline using LangGraph, Llama 3.2, and structured reasoning.

---

## 🚀 Use Case

Finding accurate and relevant information in large enterprise documentation systems like Confluence can be time-consuming and error-prone. This project addresses that challenge by:

- Automatically retrieving relevant documents based on a user query.
- Iteratively refining answers using hallucination and relevance grading.
- Rephrasing questions when no relevant documents are found.
- (Planned) Answering code-related questions by integrating a code interpreter.

---

## 🧩 Key Features

- ✅ **Agentic RAG with Additive Reasoning**  
  Ensures answers are grounded in retrieved documents and refined through multiple validation steps.

- 🔁 **Iterative Generation with Hallucination Detection**  
  Uses structured graders to assess factual grounding and relevance.

- 🔄 **Question Rephrasing**  
  Automatically rephrases unclear or unsupported questions to improve retrieval.

- 🧠 **Llama 3.2 Integration**  
  Used for both embedding generation and LLM-based reasoning.

- 📦 **Poetry for Dependency Management**  
  Ensures reproducible and clean environment setup.

- 📊 **Graph-based Workflow**  
  Built using LangGraph to visualize and control the reasoning flow.

---

## 🧱 Architecture Overview

![Graph](graph.png)

> 📎 A visual version of this graph is saved as `graph.png`.

---

## 🛠️ Technologies Used

- **LangGraph** – Agentic workflow orchestration
- **LangChain** – Prompting and chaining
- **Llama 3.2** – Embeddings + LLM calls
- **Poetry** – Dependency and environment management
- **Colab** – Used to generate pseudo data for testing

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/zarreh/genai_docs_helper.git
cd genai_docs_helper
```

### 2. Install Dependencies with Poetry

```bash
poetry install
```

### 3. Activate the Environment

```bash
poetry shell
```

### 4. Set Environment Variables

Create a `.env` file in the root directory and add your API keys or configuration values. You can use the provided template in `.env.example`.
Note: If you are using Ollama only you don't need to set up the `.env`.

```env
LLM_API_KEY=your_key_here
```
---

## 🧪 Running the Project

You can test the pipeline by running:

```bash
python graph.py
```

This will invoke the LangGraph workflow and print the final answer based on your input question.

---

## 🔮 Future Enhancements

- 🧑‍💻 **Code Interpreter Integration**  
  To support answering questions directly from codebases.

- 📈 **Evaluation Dashboard**  
  For tracking performance and grading outcomes.

- 🌐 **Web Interface**  
  To make the assistant accessible via a browser.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or ideas.

---

Let me know if you'd like this saved as a file or want to tweak any section!

# TODOs:
- [x] Fix linitng in graph.py
- [x] Add output generation if there were no output.
- [x] Modify the prompt to the older version in the main.py
- [x] Move old graph to a sub-directory
- [ ] Generate more fake documents
- [ ] Create fake codes

notebooks: follow this notebook for the rest of the notebooks and talks 
- https://www.kaggle.com/code/konradb/ts-0-the-basics