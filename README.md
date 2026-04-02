# RAG Car Driver Assistant (MVP)

This repo implements a minimal Retrieval-Augmented Generation (RAG) pipeline:

1. Embed a small set of documents
2. Store them in a FAISS vector index
3. Retrieve relevant chunks for a user question
4. Send retrieved context to a small LLM to generate an answer

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 1: Build the FAISS index

```bash
python scripts/build_index.py
```

This generates:
- `models/faiss_index`
- `models/docs.txt`

## Step 2: Run the chatbot

```bash
python scripts/chat.py
```

Try asking:
- `What does engine oil warning mean?`
- `What does ABS warning light mean?`

