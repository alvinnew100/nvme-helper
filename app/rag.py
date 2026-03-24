import json
import chromadb
import requests
from sentence_transformers import SentenceTransformer

from app.doc_parser import doc_to_chunks

OLLAMA_URL = "http://localhost:11434"
MODEL_NAME = "llama3"
COLLECTION_NAME = "nvme_docs"

_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_vector_store(docs: list[dict], chroma_path: str):
    """Build ChromaDB vector store from parsed docs."""
    embedder = get_embedder()
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    for doc in docs:
        all_chunks.extend(doc_to_chunks(doc))

    print(f"  Embedding {len(all_chunks)} chunks...")

    texts = [c["text"] for c in all_chunks]
    ids = [c["id"] for c in all_chunks]
    metadatas = [{"command": c["command"], "section": c["section"]} for c in all_chunks]

    # Embed in batches
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        embeddings = embedder.encode(batch_texts).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )

    print(f"  Stored {len(all_chunks)} chunks in vector store.")


def search_docs(query: str, chroma_path: str, n_results: int = 5) -> list[dict]:
    """Search the vector store for relevant docs."""
    embedder = get_embedder()
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = []
    for i in range(len(results["ids"][0])):
        docs.append({
            "id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "command": results["metadatas"][0][i]["command"],
            "distance": results["distances"][0][i],
        })
    return docs


def generate_command(query: str, chroma_path: str) -> dict:
    """Generate an NVMe CLI command from a natural language query using RAG."""
    relevant_docs = search_docs(query, chroma_path, n_results=3)

    # Keep context concise - only first 500 chars per doc
    context = "\n---\n".join([d["text"][:500] for d in relevant_docs])

    prompt = f"""You are an NVMe CLI expert. Given the docs below, return the exact nvme command for the user's request.

DOCS:
{context}

REQUEST: {query}

Rules:
- Command must start with "nvme" followed by the subcommand and device path like /dev/nvme0 or /dev/nvme0n1
- Include relevant flags if the user asked for specific options
- For "drive 0" use /dev/nvme0, for "namespace 1" use /dev/nvme0n1

Respond in JSON only:
{{"command": "nvme smart-log /dev/nvme0", "explanation": "brief explanation", "breakdown": [{{"flag": "smart-log", "description": "subcommand that retrieves SMART health log"}}, {{"flag": "/dev/nvme0", "description": "target device"}}], "warning": "warning if destructive command, else null"}}"""

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": 300,
                    "num_ctx": 2048,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
        result_text = response.json().get("response", "")

        result = json.loads(result_text)
        result["sources"] = [d["command"] for d in relevant_docs[:3]]
        return result

    except json.JSONDecodeError:
        return {
            "command": "Error: Could not parse LLM response",
            "explanation": f"Raw response: {result_text[:500]}",
            "breakdown": [],
            "warning": None,
            "sources": [],
        }
    except requests.exceptions.ConnectionError:
        return {
            "command": "Error: Cannot connect to Ollama",
            "explanation": "Make sure Ollama is running: ollama serve",
            "breakdown": [],
            "warning": None,
            "sources": [],
        }
    except Exception as e:
        return {
            "command": f"Error: {str(e)}",
            "explanation": "An unexpected error occurred",
            "breakdown": [],
            "warning": None,
            "sources": [],
        }
