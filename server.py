from typing import Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from memory_agents import (
    LongContextGPT4oMini,
    LongContextGPT4o,
    LongContextGPT41Mini,
    LongContextO4Mini,
    LongContextClaudeSonnet,
    LongContextGeminiFlash,
)


# ---------- Request schemas ----------
class InitializeRequest(BaseModel):
    user_id: str
    memory_system: str


class AddRequest(BaseModel):
    user_id: str
    chunk: str
    memory_system: str


class QueryRequest(BaseModel):
    user_id: str
    question: str
    memory_system: str

class ActRequest(BaseModel):
    user_id: str
    prompt: str
    memory_system: str

# ---------- Memory/Agent implementations (unified) ----------
MEMORY_FACTORIES: Dict[str, callable] = {
    "long_context_gpt-4o-mini": LongContextGPT4oMini,
    "long_context_gpt-4o": LongContextGPT4o,
    "long_context_gpt-4.1-mini": LongContextGPT41Mini,
    "long_context_o4-mini": LongContextO4Mini,
    "long_context_claude-3-7-sonnet-20250219": LongContextClaudeSonnet,
    "long_context_gemini-2.0-flash": LongContextGeminiFlash,
}


# ---------- FastAPI wiring ----------
app = FastAPI(title="Memory Agent Server")
MEMORIES: Dict[str, object] = {}
MEMORY_SYSTEMS: Dict[str, str] = {}


def _get_memory(user_id: str, memory_system: str):
    if user_id not in MEMORIES:
        raise HTTPException(status_code=404, detail="User not initialized")
    if MEMORY_SYSTEMS[user_id] != memory_system:
        raise HTTPException(status_code=400, detail="Mismatched memory_system for user")
    return MEMORIES[user_id]


@app.post("/memory/initialize")
def initialize(req: InitializeRequest):
    factory = MEMORY_FACTORIES.get(req.memory_system)
    if factory is None:
        raise HTTPException(status_code=400, detail=f"Unsupported memory_system: {req.memory_system}")
    memory = factory()
    MEMORIES[req.user_id] = memory
    MEMORY_SYSTEMS[req.user_id] = req.memory_system
    return {"status": "ok", "user_id": req.user_id, "memory_system": req.memory_system}


@app.post("/memory/add")
def add(req: AddRequest):
    memory = _get_memory(req.user_id, req.memory_system)
    memory.add_chunk(req.chunk)
    return {"status": "ok", "user_id": req.user_id}


@app.post("/agent/wrap_user_prompt")
def wrap_user_prompt(req: QueryRequest):
    memory = _get_memory(req.user_id, req.memory_system)
    prompt = memory.wrap_user_prompt(req.question)
    return {"status": "ok", "user_id": req.user_id, "prompt": prompt}


@app.post("/agent/act")
def act(req: ActRequest):
    memory = _get_memory(req.user_id, req.memory_system)
    answer = memory.act(req.prompt)
    return {"status": "ok", "user_id": req.user_id, "answer": answer}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
