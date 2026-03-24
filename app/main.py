import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag import generate_command

app = FastAPI(title="NVMe Helper")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CHROMA_PATH = os.path.join(DATA_DIR, "chroma")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class QueryRequest(BaseModel):
    query: str


class CommandResponse(BaseModel):
    command: str
    explanation: str
    breakdown: list[dict]
    warning: str | None = None
    sources: list[str] = []


@app.post("/api/generate", response_model=CommandResponse)
async def api_generate(request: QueryRequest):
    result = generate_command(request.query, CHROMA_PATH)
    return CommandResponse(**result)


@app.get("/")
async def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
