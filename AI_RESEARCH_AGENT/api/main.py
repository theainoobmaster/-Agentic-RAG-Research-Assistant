# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from typing import Optional, Dict, Any

# # from inference.model_loader import InferenceModel
# # from inference.agent_graph import build_agent_graph
# # from retriever.paper_loader import load_paper
# # from retriever.temp_retriever import build_temp_retriever
# # from agents.state import AgentState

# # app = FastAPI(title="Agentic RAG Research Assistant")

# # model = InferenceModel(
# #     base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
# #     lora_path="models/lora_adapter",
# # )
# # agent_graph = build_agent_graph(model)


# # class ChatRequest(BaseModel):
# #     query: str
# #     paper_url: Optional[str] = None


# # class ChatResponse(BaseModel):
# #     final_answer: str
# #     answer_type: str
# #     grounded: bool
# #     refusal_reason: Optional[str]
# #     metadata: Dict[str, Any]


# # @app.post("/chat", response_model=ChatResponse)
# # def chat(req: ChatRequest):
# #     if not req.query.strip():
# #         raise HTTPException(400, "Empty query")

# #     state: AgentState = {
# #         "user_query": req.query,
# #         "paper_url": req.paper_url,
# #         "paper_retriever": None,
# #         "domain": None,
# #         "intent": None,
# #         "rewritten_query": req.query,
# #         "allow_retrieval": False,
# #         "reasoning_mode": None,
# #         "thoughts": [],
# #         "actions": [],
# #         "observations": [],
# #         "final_answer": None,
# #         "grounded": False,
# #         "answer_type": None,
# #         "refusal_reason": None,
# #         "metadata": {},
# #     }

# #     if req.paper_url:
# #         docs = load_paper(req.paper_url)
# #         state["paper_retriever"] = build_temp_retriever(docs)

# #     final = agent_graph.invoke(state)

# #     return ChatResponse(
# #         final_answer=final["final_answer"],
# #         answer_type=final["answer_type"],
# #         grounded=final["grounded"],
# #         refusal_reason=final.get("refusal_reason"),
# #         metadata=final.get("metadata", {}),
# #     )
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import Optional

# from inference.model_loader import InferenceModel
# from inference.agent_graph import build_agent_graph
# from retriever.paper_loader import load_paper
# from retriever.persistent_store import ingest_documents

# app = FastAPI(title="Agentic RAG Research Assistant")

# # Load model ONCE
# model = InferenceModel(
#     base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     lora_path="models/lora_adapter"
# )

# agent_graph = build_agent_graph(model)

# class ChatRequest(BaseModel):
#     query: str
#     paper_url: Optional[str] = None

# class ChatResponse(BaseModel):
#     answer: str

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/chat", response_model=ChatResponse)
# def chat(req: ChatRequest):
#     if not req.query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")

#     state = {
#         "query": req.query,
#         "rewritten_query": req.query,  # 👈 IMPORTANT
#         "conversation_history": [],
#         "tools": [],
#     }

#     if req.paper_url:
#         docs = load_paper(req.paper_url)
#         ingest_documents(docs)
#         state["paper_docs"] = docs

#     final_state = agent_graph.invoke(state)
#     return ChatResponse(answer=final_state["final_answer"])
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from inference.model_loader import InferenceModel
from inference.agent_graph import build_agent_graph
from retriever.paper_loader import load_paper
from retriever.persistent_store import ingest_documents
from retriever.langchain_retriever import LangChainRetriever

app = FastAPI(title="Agentic RAG Research Assistant")

# Load model ONCE
model = InferenceModel(
    base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    lora_path="models/lora_adapter"
)

agent_graph = build_agent_graph(model)

class ChatRequest(BaseModel):
    query: str
    paper_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str

@app.get("/health")
def health():
    return {"status": "ok"}

# @app.post("/chat", response_model=ChatResponse)
# def chat(req: ChatRequest):
#     if not req.query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")

#     # ✅ Base agent state (MINIMAL + CORRECT)
#     state = {
#         "user_query": req.query,
#         "final_answer": None,
#     }

#     # ✅ If a paper is provided → load, store, and enable RAG
#     if req.paper_url:
#         docs = load_paper(req.paper_url)
#         ingest_documents(docs)

#         state["paper_retriever"] = LangChainRetriever(
#             index_path="data/vectorstore"
#         )

#     # 🔁 Run agent graph
#     final_state = agent_graph.invoke(state)

#     return ChatResponse(
#         answer=final_state["final_answer"]
#     )
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # ✅ SINGLE SOURCE OF TRUTH
    state = {
        "user_query": req.query,
        "final_answer": None,
    }

    if req.paper_url:
        docs = load_paper(req.paper_url)
        ingest_documents(docs)

        state["paper_retriever"] = LangChainRetriever(
            index_path="data/vectorstore"
        )

    final_state = agent_graph.invoke(state)

    return ChatResponse(answer=final_state["final_answer"])