# NVMe Helper — Design Spec

## Overview
A local web application that generates NVMe CLI commands from plain English queries, powered by a local LLM (Ollama + llama3) and RAG over the official nvme-cli documentation.

## Components

### Backend (Python + FastAPI)
- Clones nvme-cli repo, parses man pages and documentation
- Chunks and embeds docs into ChromaDB using sentence-transformers
- RAG pipeline: query → embed → retrieve relevant docs → LLM generates command
- Ollama running llama3 (8B) as the local LLM
- REST API: POST /generate with { "query": "..." } returns { "command": "...", "explanation": "...", "breakdown": [...] }

### Frontend (HTML/CSS/JS)
- Distinctive, sleek design (not generic AI aesthetic)
- Search bar for natural language input
- Quick action buttons for common NVMe operations
- Command output with syntax highlighting, copy button, breakdown section
- Responsive layout

### Setup Script (setup.sh)
- Installs Ollama if not present
- Pulls llama3 model
- Clones nvme-cli repo and parses docs
- Installs Python dependencies
- Builds vector embeddings
- Starts the server

## Data Flow
```
User query → FastAPI → embed query → search ChromaDB →
top docs + query → Ollama (llama3) → generated command → Web UI
```

## Project Structure
```
~/nvme-helper/
├── setup.sh
├── requirements.txt
├── app/
│   ├── main.py           # FastAPI server
│   ├── rag.py            # RAG pipeline (embed, retrieve, generate)
│   ├── doc_parser.py     # Parse nvme-cli man pages
│   └── static/
│       └── index.html    # Frontend
└── data/
    ├── nvme-cli/         # Cloned repo
    └── chroma/           # Vector store
```

## Constraints
- Show command + explanation only (no execution from UI)
- Works fully offline after setup
- Docs can be refreshed by re-running setup
