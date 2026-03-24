import re
import logging
from typing import Optional
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.doc_parser import doc_to_chunks, chunks_to_langchain_docs

logger = logging.getLogger(__name__)

# ── Global state (initialized at startup, read-only after) ──
_embeddings: Optional[HuggingFaceEmbeddings] = None
_faiss_retriever = None
_bm25_retriever: Optional[BM25Retriever] = None
_doc_db: dict[str, dict] = {}
_response_cache: dict[str, dict] = {}
_rag_chain = None

# ── Constants ──
VENDOR_PREFIXES = [
    "wdc", "seagate", "micron", "intel", "solidigm", "ocp",
    "dapustor", "sndk", "virtium", "toshiba", "shannon",
    "memblaze", "netapp", "ymtc", "inspur", "huawei", "transcend",
]

NO_DEVICE_COMMANDS = {
    "nvme-list", "nvme-list-subsys", "nvme-discover",
    "nvme-connect-all", "nvme-gen-hostnqn", "nvme-show-hostnqn", "nvme-version",
}

KEYWORD_OVERRIDES = {
    "list": "nvme-list",
    "list devices": "nvme-list",
    "list all": "nvme-list",
    "list all devices": "nvme-list",
    "list all nvme devices": "nvme-list",
    "show all drives": "nvme-list",
    "show devices": "nvme-list",
    "list subsystems": "nvme-list-subsys",
    "list namespaces": "nvme-list-ns",
    "list controllers": "nvme-list-ctrl",
    "identify controller": "nvme-id-ctrl",
    "identify namespace": "nvme-id-ns",
    "smart log": "nvme-smart-log",
    "smart health": "nvme-smart-log",
    "health log": "nvme-smart-log",
    "error log": "nvme-error-log",
    "firmware log": "nvme-fw-log",
    "firmware version": "nvme-fw-log",
    "firmware info": "nvme-fw-log",
    "firmware download": "nvme-fw-download",
    "firmware activate": "nvme-fw-commit",
    "format": "nvme-format",
    "format drive": "nvme-format",
    "format namespace": "nvme-format",
    "sanitize": "nvme-sanitize",
    "secure erase": "nvme-sanitize",
    "reset": "nvme-reset",
    "connect": "nvme-connect",
    "disconnect": "nvme-disconnect",
    "read": "nvme-read",
    "write": "nvme-write",
    "write zeroes": "nvme-write-zeroes",
    "get feature": "nvme-get-feature",
    "set feature": "nvme-set-feature",
    "self test": "nvme-device-self-test",
    "self-test": "nvme-device-self-test",
    "telemetry": "nvme-telemetry-log",
    "attach namespace": "nvme-attach-ns",
    "detach namespace": "nvme-detach-ns",
    "create namespace": "nvme-create-ns",
    "delete namespace": "nvme-delete-ns",
}


# ═══════════════════════════════════════════
# INITIALIZATION (called once at startup)
# ═══════════════════════════════════════════

def init_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
        )
    return _embeddings


def init_retrieval_pipeline(docs: list[dict]) -> None:
    """Build FAISS + BM25 indices from parsed docs."""
    global _faiss_retriever, _bm25_retriever

    all_chunks = []
    for doc in docs:
        all_chunks.extend(doc_to_chunks(doc))
    lc_docs = chunks_to_langchain_docs(all_chunks)

    logger.info(f"Building indices for {len(lc_docs)} chunks...")

    embeddings = init_embeddings()

    # FAISS in-memory index
    faiss_store = FAISS.from_documents(lc_docs, embeddings)
    _faiss_retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 7},
    )

    # BM25 keyword retriever
    _bm25_retriever = BM25Retriever.from_documents(lc_docs, k=7)

    logger.info("Retrieval pipeline ready (FAISS + BM25 + RRF).")


def load_doc_db(docs: list[dict]) -> None:
    global _doc_db
    for doc in docs:
        _doc_db[doc["command"]] = doc
    logger.info(f"Loaded {len(_doc_db)} commands into memory.")


# ═══════════════════════════════════════════
# HYBRID SEARCH
# ═══════════════════════════════════════════

@dataclass
class SearchResult:
    command: str
    text: str
    score: float
    section: str


def _reciprocal_rank_fusion(result_lists: list[list[Document]], k: int = 60) -> list[tuple[Document, float]]:
    """Combine multiple ranked lists using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            doc_id = doc.metadata["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank + k)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[did], scores[did]) for did in sorted_ids]


def hybrid_search(query: str, n_results: int = 3) -> list[SearchResult]:
    """Run hybrid BM25+semantic search with Reciprocal Rank Fusion, then re-rank."""
    if _faiss_retriever is None or _bm25_retriever is None:
        raise RuntimeError("Retrieval pipeline not initialized")

    # Get results from both retrievers
    faiss_docs = _faiss_retriever.invoke(query)
    bm25_docs = _bm25_retriever.invoke(query)

    # Fuse with RRF
    fused = _reciprocal_rank_fusion([faiss_docs, bm25_docs])

    query_lower = query.lower()
    query_mentions_vendor = any(v in query_lower for v in VENDOR_PREFIXES)

    scored = []
    for doc, rrf_score in fused:
        cmd = doc.metadata["command"]
        section = doc.metadata["section"]
        # Invert RRF score so lower = better (matching existing convention)
        score = 1.0 - rrf_score

        if any(v in cmd.lower() for v in VENDOR_PREFIXES) and not query_mentions_vendor:
            score += 0.3
        if section == "options":
            score += 0.1

        scored.append(SearchResult(command=cmd, text=doc.page_content, score=score, section=section))

    scored.sort(key=lambda r: r.score)
    seen = set()
    unique = []
    for r in scored:
        if r.command not in seen:
            seen.add(r.command)
            unique.append(r)
        if len(unique) >= n_results:
            break

    return unique


# ═══════════════════════════════════════════
# HELPER FUNCTIONS (preserved from original)
# ═══════════════════════════════════════════

def _check_keyword_override(query: str) -> str | None:
    q = query.lower().strip()
    q_clean = re.sub(r"\b(show|get|display|check|view|me|the|for|on|of|please|can you|i want to|how to)\b", "", q).strip()
    q_clean = re.sub(r"\s+", " ", q_clean).strip()
    q_clean = re.sub(r"(drive|device|nvme)\s*\d+\w*", "", q_clean).strip()
    q_clean = re.sub(r"/dev/nvme\S+", "", q_clean).strip()
    q_clean = re.sub(r"\s+", " ", q_clean).strip()

    if q_clean in KEYWORD_OVERRIDES:
        return KEYWORD_OVERRIDES[q_clean]

    for keyword, cmd in sorted(KEYWORD_OVERRIDES.items(), key=lambda x: -len(x[0])):
        if keyword in q_clean:
            return cmd

    return None


def extract_device(query: str) -> str:
    dev_match = re.search(r"/dev/nvme\d+n?\d*", query)
    if dev_match:
        return dev_match.group()

    drive_match = re.search(r"(?:drive|device|nvme)\s*(\d+)", query, re.IGNORECASE)
    if drive_match:
        num = drive_match.group(1)
        if re.search(r"namespace|ns\s*\d|nvme\d+n", query, re.IGNORECASE):
            ns_match = re.search(r"(?:namespace|ns)\s*(\d+)", query, re.IGNORECASE)
            ns_num = ns_match.group(1) if ns_match else "1"
            return f"/dev/nvme{num}n{ns_num}"
        return f"/dev/nvme{num}"

    if re.search(r"namespace|ns\b", query, re.IGNORECASE):
        ns_match = re.search(r"(?:namespace|ns)\s*(\d+)", query, re.IGNORECASE)
        ns_num = ns_match.group(1) if ns_match else "1"
        return f"/dev/nvme0n{ns_num}"

    return "/dev/nvme0"


def extract_flags_from_query(query: str, doc: dict) -> list[str]:
    flags = []
    q = query.lower()

    if "json" in q:
        flags.append("--output-format=json")
    elif "binary" in q or "raw" in q:
        flags.append("--raw-binary")

    ns_match = re.search(r"(?:namespace|ns|nsid)\s*(\d+)", q)
    if ns_match and "namespace-id" not in " ".join(flags):
        flags.append(f"--namespace-id={ns_match.group(1)}")

    block_match = re.search(r"(\d+)\s*[kK]", q)
    if block_match and "format" in doc.get("command", ""):
        size = int(block_match.group(1))
        lbaf_map = {512: 0, 4096: 1, 4: 1}
        lbaf = lbaf_map.get(size, 1)
        flags.append(f"--lbaf={lbaf}")

    if "verbose" in q or "detail" in q:
        flags.append("--verbose")

    return flags


def build_example_command(doc: dict, device: str) -> str:
    synopsis = doc.get("synopsis", "")
    cmd_match = re.search(r"'(nvme\s+[\w-]+)'", synopsis)
    if cmd_match:
        return cmd_match.group(1)
    return doc["command"].replace("nvme-", "nvme ", 1)


def _clean_name(name: str) -> str:
    parts = name.split(" - ", 1)
    return parts[1].strip() if len(parts) > 1 else name.strip()


def _get_flag_desc(flag: str, doc: dict) -> str:
    options = doc.get("options", "")
    flag_clean = flag.lstrip("-")
    for line in options.split("\n"):
        if flag_clean in line.lower():
            desc = line.split("::")[-1].strip() if "::" in line else line.strip()
            if desc and len(desc) > 5:
                return desc[:100]
    flag_descriptions = {
        "--output-format": "Set output format (normal, json, binary)",
        "--raw-binary": "Output raw binary data",
        "--namespace-id": "Target specific namespace ID",
        "--lbaf": "LBA format index for block size",
        "--verbose": "Increase output detail",
    }
    return flag_descriptions.get(flag, f"Option: {flag}")


# ═══════════════════════════════════════════
# TIER ROUTING
# ═══════════════════════════════════════════

def _is_complex_query(query: str) -> bool:
    q = query.lower()
    indicators = [
        "how do i", "how to", "what is the difference",
        "compare", "explain", "why", "troubleshoot",
        "step by step", "best way", "should i",
        "multiple", "sequence", "workflow",
    ]
    return any(ind in q for ind in indicators)


# ═══════════════════════════════════════════
# TIER 1: INSTANT DETERMINISTIC LOOKUP
# ═══════════════════════════════════════════

def _generate_tier1(query: str) -> dict:
    override_cmd = _check_keyword_override(query)

    if override_cmd and override_cmd in _doc_db:
        results = [SearchResult(command=override_cmd, text="", score=0.0, section="main")]
        hybrid_results = hybrid_search(query, n_results=2)
        results.extend(hybrid_results)
    else:
        results = hybrid_search(query, n_results=3)

    if not results:
        return {
            "command": "No matching command found",
            "explanation": "Try rephrasing your query.",
            "breakdown": [],
            "warning": None,
            "sources": [],
            "tier": 1,
            "tier2_available": False,
        }

    best = results[0]
    cmd_name = best.command
    doc = _doc_db.get(cmd_name, {})

    base_cmd = build_example_command(doc, "")
    extra_flags = extract_flags_from_query(query, doc)

    if cmd_name in NO_DEVICE_COMMANDS:
        full_command = base_cmd
    else:
        device = extract_device(query)
        full_command = f"{base_cmd} {device}"

    if extra_flags:
        full_command += " " + " ".join(extra_flags)

    parts = base_cmd.split()
    subcmd = parts[1] if len(parts) > 1 else parts[0]
    breakdown = [
        {"flag": subcmd, "description": _clean_name(doc.get("name", subcmd))},
    ]
    if cmd_name not in NO_DEVICE_COMMANDS:
        breakdown.append({"flag": extract_device(query), "description": "Target NVMe device"})
    for f in extra_flags:
        flag_name = f.split("=")[0]
        breakdown.append({"flag": f, "description": _get_flag_desc(flag_name, doc)})

    warning = None
    destructive = ["format", "sanitize", "delete-ns", "reset", "write-zeroes", "security-send"]
    if any(d in cmd_name for d in destructive):
        warning = "This is a destructive command. Double-check the device and parameters before running."

    desc = doc.get("description", "")
    explanation = desc.split("\n")[0][:200] if desc else f"Runs the {subcmd} subcommand."

    return {
        "command": full_command,
        "explanation": explanation,
        "breakdown": breakdown,
        "warning": warning,
        "sources": [r.command for r in results[:3]],
        "tier": 1,
        "tier2_available": _is_complex_query(query),
    }


# ═══════════════════════════════════════════
# TIER 2: LANGCHAIN RAG WITH OLLAMA (STREAMING)
# ═══════════════════════════════════════════

_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an NVMe CLI expert assistant. Given the user's question and relevant documentation, provide the exact nvme command(s) needed.

Rules:
- Always provide the complete command with device path
- Explain each flag briefly
- Warn about destructive operations
- If multiple steps are needed, number them
- Be concise but complete

Relevant NVMe CLI documentation:
{context}"""),
    ("human", "{question}"),
])


def _get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        llm = ChatOllama(
            model="llama3",
            temperature=0.1,
            num_predict=512,
        )
        _rag_chain = _RAG_PROMPT | llm | StrOutputParser()
    return _rag_chain


def _build_tier2_context(query: str) -> str:
    results = hybrid_search(query, n_results=5)
    parts = []
    for r in results:
        doc = _doc_db.get(r.command, {})
        part = f"=== {r.command} ===\n"
        part += f"Name: {doc.get('name', '')}\n"
        part += f"Synopsis: {doc.get('synopsis', '')}\n"
        part += f"Description: {doc.get('description', '')[:300]}\n"
        if doc.get("examples"):
            part += f"Examples: {doc.get('examples', '')[:200]}\n"
        parts.append(part)
    return "\n".join(parts)


async def generate_tier2_stream(query: str):
    """Async generator that streams LLM tokens for Tier 2 queries."""
    chain = _get_rag_chain()
    context = _build_tier2_context(query)
    async for chunk in chain.astream({"context": context, "question": query}):
        yield chunk


# ═══════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════

def generate_command(query: str) -> dict:
    """Route query to Tier 1 (instant) or flag Tier 2 availability."""
    cache_key = query.lower().strip()
    if cache_key in _response_cache:
        return _response_cache[cache_key]

    result = _generate_tier1(query)

    if result["command"] == "No matching command found":
        result["tier2_available"] = True

    if len(_response_cache) < 500:
        _response_cache[cache_key] = result

    return result
