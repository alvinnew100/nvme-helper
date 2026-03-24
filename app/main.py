import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.rag import (
    generate_command,
    init_embeddings,
    init_retrieval_pipeline,
    load_doc_db,
    generate_tier2_stream,
)
from app.doc_parser import parse_nvme_docs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
NVME_REPO = os.path.join(DATA_DIR, "nvme-cli")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t0 = time.time()
    logger.info("Initializing NVMe Helper...")
    init_embeddings()
    docs = parse_nvme_docs(NVME_REPO)
    load_doc_db(docs)
    init_retrieval_pipeline(docs)
    logger.info(f"Ready in {time.time() - t0:.1f}s")
    yield


app = FastAPI(title="NVMe Helper", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class CommandResponse(BaseModel):
    command: str
    explanation: str
    breakdown: list[dict]
    warning: str | None = None
    sources: list[str] = []
    tier: int = 1
    tier2_available: bool = False


@app.post("/api/generate", response_model=CommandResponse)
async def api_generate(request: QueryRequest):
    result = generate_command(request.query)
    return CommandResponse(**result)


@app.post("/api/generate/stream")
async def api_generate_stream(request: QueryRequest):
    async def event_stream():
        async for chunk in generate_tier2_stream(request.query):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
