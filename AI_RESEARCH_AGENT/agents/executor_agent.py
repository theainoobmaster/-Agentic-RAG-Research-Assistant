from inference.model_loader import InferenceModel
from retriever.langchain_retriever import LangChainRetriever
from typing import TYPE_CHECKING

# # if TYPE_CHECKING:
# #     from agents.state import AgentState
# # # default_retriever = LangChainRetriever(index_path="data/vectorstore")


# # # def clean_answer(text: str) -> str:
# # #     blacklist = [
# # #         "### System:",
# # #         "### Retrieved Context:",
# # #         "### Instruction:",
# # #         "### Thought:",
# # #         "### Final Answer:",
# # #     ]
# # #     for b in blacklist:
# # #         text = text.replace(b, "")
# # #     return text.strip()


# # # def executor_agent(state: "AgentState", model: InferenceModel) -> "AgentState":
# # #     query = state["rewritten_query"]

# # #     thoughts = []
# # #     actions = []
# # #     observations = []

# # #     # 🔑 Choose retriever
# # #     retriever = state.get("paper_retriever", default_retriever)

# # #     # -----------------------------
# # #     # ALWAYS RETRIEVE IF ALLOWED
# # #     # -----------------------------
# # #     if state.get("allow_retrieval", False):
# # #         actions.append({"tool": "retriever", "query": query})

# # #         docs_scores = retriever.similarity_search_with_score(query, k=12)
# # #         docs = [d for d, _ in docs_scores]

# # #         context = "\n".join(d.page_content for d in docs)
# # #         observations.append(context)

# # #         prompt = f"""
# # # You are a research assistant.
# # # Answer using ONLY the content below.

# # # ### Context:
# # # {context}

# # # ### Question:
# # # {query}

# # # ### Answer:
# # # """
# # #         raw = model.generate(prompt, max_new_tokens=400)

# # #         return {
# # #             **state,
# # #             "final_answer": clean_answer(raw),
# # #             "grounded": True,  # still true: retrieval happened
# # #             "answer_type": "paper_grounded_answer",
# # #             "thoughts": thoughts,
# # #             "actions": actions,
# # #             "observations": observations,
# # #         }

# # #     # -----------------------------
# # #     # GENERAL MODE (NO PAPER)
# # #     # -----------------------------
# # #     raw = model.generate(query, max_new_tokens=256)

# # #     return {
# # #         **state,
# # #         "final_answer": clean_answer(raw),
# # #         "grounded": False,
# # #         "answer_type": "general_agent_answer",
# # #         "thoughts": thoughts,
# # #         "actions": actions,
# # #         "observations": observations,
# # #     }
# # # from agents.profile_qa_agent import profile_qa_agent
# # # from agents.react_paper_agent import react_paper_agent

# # # def executor_agent(state, model):
# # #     route = state["metadata"]["decision"]["route"]

# # #     if route == "profile":
# # #         return profile_qa_agent(state, model)

# # #     if route == "react":
# # #         return react_paper_agent(state, model)

# # #     # General (non-paper) question
# # #     answer = model.generate(state["user_query"], max_new_tokens=300)
# # #     return {
# # #         **state,
# # #         "final_answer": answer.strip()
# # #     }

# # def executor_agent(state: "AgentState", model: "InferenceModel") -> "AgentState":
# #     query = state["rewritten_query"]

# #     # Select retriever
# #     if "paper_retriever" in state:
# #         retriever = state["paper_retriever"]
# #         strict = True
# #     else:
# #         retriever = default_retriever
# #         strict = False

# #     # Retrieve context if needed
# #     context = ""
# #     if retriever:
# #         docs = retriever.retrieve(query, k=4)
# #         context = "\n".join(docs)

# #     # Build prompt
# #     if strict:
# #         prompt = f"""
# # Below is an instruction that describes a task.
# # Write a response that completes the task.

# # ### Instruction:
# # Using ONLY the retrieved context below, answer the question.

# # If the answer is not explicitly present in the retrieved context,
# # respond exactly with:
# # "Not stated in the paper."

# # Retrieved Context:
# # {context}

# # Question:
# # {query}

# # ### Response:
# # """
# #     else:
# #         prompt = f"""
# # You are an AI research assistant.

# # Question:
# # {query}

# # Answer:
# # """

# #     # Generate
# #     answer = model.generate(
# #         prompt,
# #         max_new_tokens=128,
# #         do_sample=False,
# #         temperature=0.2,
# #     )

# #     state["final_answer"] = answer
# #     return state
# from typing import TYPE_CHECKING
# from inference.model_loader import InferenceModel

if TYPE_CHECKING:
    from agents.state import AgentState

# def executor_agent(state: "AgentState", model: InferenceModel) -> "AgentState":
#     query = state["rewritten_query"]

#     retriever = state.get("paper_retriever")
#     strict = retriever is not None

#     context = ""
#     if retriever:
#         docs = retriever.retrieve(query, k=4)
#         context = "\n".join(d.page_content for d in docs)

#     if strict:
#         prompt = f"""
# Using ONLY the retrieved context below, answer the question.

# If the answer is not explicitly present, respond exactly with:
# "Not stated in the paper."

# Context:
# {context}

# Question:
# {query}

# Answer:
# """
#     else:
#         prompt = f"Question:\n{query}\nAnswer:"

#     answer = model.generate(
#         prompt,
#         max_new_tokens=128,
#         do_sample=False,
#         temperature=0.2,
#     )

#     state["final_answer"] = answer.strip()
#     return state
def executor_agent(state, model):
    """
    Final robust executor agent.
    NEVER assumes optional state keys.
    """

    # ✅ SAFE QUERY RESOLUTION (NO KeyError possible)
    query = (
        state.get("rewritten_query")
        or state.get("user_query")
        or state.get("query")
    )

    if not query:
        raise ValueError("ExecutorAgent: No query found in state")

    # ✅ Paper / RAG mode
    retriever = state.get("paper_retriever")
    strict = retriever is not None

    # context = ""
    if retriever:
        docs = retriever.retrieve(query, k=4)
    #     context = "\n".join(docs)
        context = "\n".join(
        d.page_content if hasattr(d, "page_content") else str(d)
        for d in docs
        )   

    # ✅ Prompt
    if strict:
#             prompt = f"""
#             Answer ONLY using exact text from the context.

# If the answer text does NOT appear exactly in the context,
# output this exact line and nothing else:
# Not stated in the paper.

# Rules:
# - Copy text directly from context.
# - Do not explain.
# - Do not repeat the context.
# - Do not repeat the question.
# - Do not add words.

# Context:
# {context}

# Question:
# {query}

# Answer in one paragraph only If .
#             """
        # prompt = f"""
        # You are a research assistant.

        # Answer the question using ONLY the information present in the context below.
        # If the answer is not present or cannot be inferred from the context, reply exactly:
        # Not stated in the paper.

        # Context:
        # {context}

        # Question:
        # {query}

        # ### Response:
        # Answer in one sentence only, using information stated or clearly implied in the context.
        # """
        # prompt = f"""
        # You are a research assistant.

        # Answer the question using ONLY the information explicitly stated in the context below.
        # Do NOT use prior knowledge.
        # Do NOT infer or explain.

        # If the answer is not directly stated word-for-word in the context,
        # reply exactly with:
        # Not stated in the paper.

        # Context:
        # {context}

        # Question:
        # {query}

        # ### Response:
        # Provide ONE clear sentence copied or lightly rephrased from the context.
        # """
        prompt = f"""
    You are a scientific research assistant.

    Your task is to answer the question using ONLY the information explicitly stated
    in the provided context.

    Rules:
    - Do NOT use prior knowledge.
    - Do NOT infer or guess.
    - Do NOT paraphrase beyond what is stated.
    - If the answer is not explicitly present as a clear factual statement,
    respond EXACTLY with:
    Not stated in the paper.

    Context:
    {context}

    Question:
    {query}

    ### Response:
    - Answer in ONE complete sentence.
    - The answer must be copied or minimally rephrased from the context.
    """

    answer = model.generate(
        prompt,
        max_new_tokens=200
    )

    state["final_answer"] = answer.strip()
    return state
# from inference.model_loader import InferenceModel
# from typing import Dict, Any, List


# def executor_agent(state: Dict[str, Any], model: InferenceModel) -> Dict[str, Any]:
#     """
#     Hybrid RAG executor:
#     - Answers when context supports it
#     - Adds confidence
#     - Refuses cleanly when unsupported
#     """

#     query = (
#         state.get("rewritten_query")
#         or state.get("query")
#         or state.get("user_query")
#     )

#     retrieved_chunks: List[str] = state.get("retrieved_context", [])

#     context = "\n".join(retrieved_chunks).strip()

#     # -------------------------
#     # STRICT REFUSAL (NO CONTEXT)
#     # -------------------------
#     if not context:
#         return {
#             **state,
#             "final_answer": "Not stated in the paper.",
#             "confidence": 0.0,
#             "grounded": False,
#         }

#     # -------------------------
#     # HYBRID PROMPT
#     # -------------------------
#     prompt = f"""
# You are an academic research assistant.

# Rules:
# - Use ONLY the provided context.
# - If the answer is not explicitly stated, say "Not stated in the paper."
# - Do NOT invent equations, datasets, or claims.
# - Be concise and factual.

# Context:
# {context}

# Question:
# {query}

# Answer:
# """

#     raw = model.generate(prompt, max_new_tokens=180).strip()

#     # -------------------------
#     # POST-GENERATION SAFETY
#     # -------------------------
#     refusal_phrases = [
#         "not stated",
#         "not mentioned",
#         "does not specify",
#         "not provided",
#     ]
#     print(raw)
#     grounded = not any(p in raw.lower() for p in refusal_phrases)

#     confidence = 0.85 if grounded else 0.0

#     final_answer = raw if grounded else "Not stated in the paper."
#     print("====== CONTEXT PASSED TO MODEL ======")
#     print(context)
#     print("====== END CONTEXT ======")
#     return {
#         **state,
#         "final_answer": raw,
#         "confidence": confidence,
#         "grounded": grounded,
#     }