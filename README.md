# End-to-End RAG Chatbot

RAG (Retrieval-Augmented Generation) chatbot with hybrid search, multi-query expansion, CRAG (Corrective RAG), and Cohere re-ranking.

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env` and fill in your API keys:
```bash
cp .env.example .env
```

## Usage

```bash
streamlit run app.py
```

Upload a PDF and ask questions about its content.

## Architecture

- **Ingestion**: PDF loading, chunking (800 chars, 100 overlap), hybrid vector store (dense + sparse)
- **Retrieval**: Multi-query expansion → hybrid search → Cohere re-ranking → CRAG grading → final re-rank
- **Generation**: History-aware retriever + QA chain with trimmed chat history