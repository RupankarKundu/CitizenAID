"""
CitizenAid — Legal Rights & RTI Filing Assistant
FastAPI Backend with RAG Pipeline (Gemini LLM)
"""

import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env from project root (parent of backend/)
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag_pipeline import CitizenAidRAG
from rti_agent import RTIAgent

rag: CitizenAidRAG | None = None
rti_agent: RTIAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag, rti_agent
    print("🚀 Initialising CitizenAid RAG pipeline …")
    rag = CitizenAidRAG()
    rag.ingest_documents()
    rti_agent = RTIAgent(rag)
    print("✅ CitizenAid ready.")
    yield
    print("🛑 Shutting down.")


app = FastAPI(
    title="CitizenAid API",
    description="Legal Rights & RTI Filing Assistant for Indian Citizens",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    language: Optional[str] = "en"
    category: Optional[str] = None


class RTIRequest(BaseModel):
    situation: str
    department: Optional[str] = None
    state: Optional[str] = None
    applicant_name: Optional[str] = "Applicant"
    language: Optional[str] = "en"


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    category: str
    confidence: str
    needs_lawyer: bool
    quick_actions: list[str]


class RTIResponse(BaseModel):
    draft: str
    steps: list[str]
    authority: str
    portal_link: str
    tips: list[str]


@app.get("/health")
async def health():
    return {"status": "ok", "rag_ready": rag is not None}


@app.post("/api/query", response_model=QueryResponse)
async def query_legal(req: QueryRequest):
    if rag is None:
        raise HTTPException(503, "RAG pipeline not initialised yet")
    result = rag.query(req.question, category=req.category, language=req.language)
    return QueryResponse(**result)


@app.post("/api/rti/draft", response_model=RTIResponse)
async def generate_rti(req: RTIRequest):
    if rti_agent is None:
        raise HTTPException(503, "RTI agent not initialised yet")
    result = rti_agent.generate(
        situation=req.situation,
        department=req.department,
        state=req.state,
        applicant_name=req.applicant_name,
        language=req.language or "en",
    )
    return RTIResponse(**result)


@app.get("/api/categories")
async def list_categories():
    return {
        "categories": [
            {"id": "RTI",            "label": "Right to Information",  "emoji": "📄"},
            {"id": "Consumer",       "label": "Consumer Rights",        "emoji": "🛒"},
            {"id": "Constitutional", "label": "Fundamental Rights",     "emoji": "⚖️"},
            {"id": "Criminal",       "label": "FIR & Criminal Law",     "emoji": "🚨"},
            {"id": "Labour",         "label": "Labour & Employment",    "emoji": "👷"},
        ]
    }


FRONTEND = Path(__file__).parent.parent / "frontend"
if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(str(FRONTEND / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)