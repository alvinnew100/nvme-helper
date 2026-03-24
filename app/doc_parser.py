import os
import re
from pathlib import Path


def parse_nvme_doc(filepath: str) -> dict | None:
    """Parse a single nvme-cli AsciiDoc .txt file into structured sections."""
    try:
        text = Path(filepath).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    filename = os.path.basename(filepath)
    if not filename.startswith("nvme-") or not filename.endswith(".txt"):
        return None

    sections = {}
    current_section = None
    current_lines = []

    for line in text.split("\n"):
        # Section headers in asciidoc: NAME, SYNOPSIS, DESCRIPTION, OPTIONS, EXAMPLES
        if re.match(r"^[A-Z][A-Z ]+$", line.strip()) and len(line.strip()) < 30:
            if current_section:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line.strip()
            current_lines = []
        elif line.strip().startswith("----") or line.strip().startswith("~~~~"):
            continue
        else:
            current_lines.append(line)

    if current_section:
        sections[current_section] = "\n".join(current_lines).strip()

    if not sections:
        return None

    command_name = filename.replace(".txt", "")

    return {
        "command": command_name,
        "name": sections.get("NAME", ""),
        "synopsis": sections.get("SYNOPSIS", ""),
        "description": sections.get("DESCRIPTION", ""),
        "options": sections.get("OPTIONS", ""),
        "examples": sections.get("EXAMPLES", ""),
        "full_text": text,
    }


def parse_nvme_docs(repo_path: str) -> list[dict]:
    """Parse all nvme-cli documentation files from the cloned repo."""
    docs_dir = os.path.join(repo_path, "Documentation")
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Documentation directory not found: {docs_dir}")

    docs = []
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt") and filename.startswith("nvme-"):
            filepath = os.path.join(docs_dir, filename)
            parsed = parse_nvme_doc(filepath)
            if parsed:
                docs.append(parsed)

    print(f"  Parsed {len(docs)} NVMe command docs.")
    return docs


def doc_to_chunks(doc: dict) -> list[dict]:
    """Convert a parsed doc into chunks for embedding.

    Creates one main chunk per command (name + synopsis + description + examples)
    and one chunk for options if they exist.
    """
    chunks = []
    command = doc["command"]

    # Main chunk: what the command does + how to use it
    main_text = f"Command: {command}\n"
    if doc["name"]:
        main_text += f"{doc['name']}\n"
    if doc["synopsis"]:
        main_text += f"\nUsage:\n{doc['synopsis']}\n"
    if doc["description"]:
        main_text += f"\nDescription:\n{doc['description']}\n"
    if doc["examples"]:
        main_text += f"\nExamples:\n{doc['examples']}\n"

    chunks.append({
        "id": f"{command}-main",
        "text": main_text.strip(),
        "command": command,
        "section": "main",
    })

    # Options chunk (separate so it doesn't dilute semantic search)
    if doc["options"] and len(doc["options"]) > 50:
        options_text = f"Command: {command}\nOptions and flags:\n{doc['options']}"
        chunks.append({
            "id": f"{command}-options",
            "text": options_text.strip(),
            "command": command,
            "section": "options",
        })

    return chunks
