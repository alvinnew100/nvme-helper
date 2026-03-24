# NVMe Helper

A local web tool that generates NVMe CLI commands from plain English queries. Powered by a local LLM (Ollama + llama3) and RAG over the official nvme-cli documentation.

![NVMe Helper UI](https://img.shields.io/badge/status-working-brightgreen) ![Offline](https://img.shields.io/badge/offline-ready-blue)

## How It Works

1. Type what you need in plain English (e.g., "show SMART health log for drive 0")
2. The tool searches through 295+ NVMe CLI docs using vector similarity
3. Sends the relevant docs + your query to a local LLM
4. Returns the exact `nvme` command with explanation and breakdown

## Quick Start

```bash
# Clone the repo
git clone https://github.com/alvinnew100/nvme-helper.git
cd nvme-helper

# Run setup (installs Ollama, pulls llama3, builds embeddings)
chmod +x setup.sh
./setup.sh

# Start the server
source venv/bin/activate
python3 -m app.main

# Open in browser
open http://localhost:8080
```

## Requirements

- macOS or Linux
- Python 3.10+
- ~5GB disk space (for llama3 model)
- 8GB+ RAM recommended

## Architecture

```
User query → FastAPI → embed query → search ChromaDB →
top docs + query → Ollama (llama3) → generated command → Web UI
```

## Stack

- **Backend**: Python, FastAPI, ChromaDB, sentence-transformers
- **LLM**: Ollama + llama3 (8B) — runs fully offline
- **Frontend**: Vanilla HTML/CSS/JS — industrial terminal aesthetic
- **Docs**: Parsed from official [nvme-cli](https://github.com/linux-nvme/nvme-cli) repository
