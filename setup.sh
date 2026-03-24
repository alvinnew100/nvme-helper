#!/bin/bash
set -e

echo "========================================"
echo "  NVMe Helper - Setup"
echo "========================================"
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"

# 1. Check/Install Ollama
echo "[1/5] Checking Ollama..."
if ! command -v ollama &> /dev/null; then
    echo "  Ollama not found. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Downloading Ollama for macOS..."
        curl -fsSL https://ollama.com/install.sh | sh
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "  ERROR: Unsupported OS. Please install Ollama manually: https://ollama.com"
        exit 1
    fi
else
    echo "  Ollama already installed."
fi

# 2. Start Ollama and pull model
echo ""
echo "[2/5] Pulling llama3 model (this may take a while on first run)..."

# Start Ollama in background if not running
if ! pgrep -x "ollama" > /dev/null 2>&1; then
    ollama serve &> /dev/null &
    sleep 3
fi

ollama pull llama3

# 3. Clone nvme-cli docs
echo ""
echo "[3/5] Fetching NVMe CLI documentation..."
if [ -d "$DATA_DIR/nvme-cli" ]; then
    echo "  nvme-cli repo already cloned. Pulling latest..."
    cd "$DATA_DIR/nvme-cli" && git pull && cd "$PROJECT_DIR"
else
    git clone --depth 1 https://github.com/linux-nvme/nvme-cli.git "$DATA_DIR/nvme-cli"
fi

# 4. Install Python dependencies
echo ""
echo "[4/5] Installing Python dependencies..."
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

# 5. Verify setup (FAISS index builds at server startup, no pre-build needed)
echo ""
echo "[5/5] Verifying installation..."
python3 -c "
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
print('  LangChain + FAISS + BM25 imports OK.')
from app.doc_parser import parse_nvme_docs
docs = parse_nvme_docs('$DATA_DIR/nvme-cli')
print(f'  Parsed {len(docs)} docs. Setup verified.')
"

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Run: source venv/bin/activate && python3 -m app.main"
echo "  Then open: http://localhost:8080"
echo "========================================"
