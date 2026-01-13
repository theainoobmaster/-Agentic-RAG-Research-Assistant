# def planner_agent(state: "AgentState") -> "AgentState":
#     query = state["user_query"].lower()

#     if any(k in query for k in ["transformer", "neural network", "deep learning", "llm"]):
#         domain = "machine_learning"
#     else:
#         domain = "general"

#     if query.startswith(("what", "how", "explain")):
#         intent = "explanation"
#     elif "compare" in query:
#         intent = "comparison"
#     elif "code" in query or "implement" in query:
#         intent = "coding"
#     else:
#         intent = "general"

#     return {
#         **state,
#         "domain": domain,
#         "intent": intent,
#         "rewritten_query": state["user_query"],
#         "metadata": {
#             **state.get("metadata", {}),
#             "planner": {
#                 "domain": domain,
#                 "intent": intent,
#             },
#         },
#     }
def planner_agent(state):
    query = state["user_query"].lower()

    paper_indicators = [
        "paper", "this work", "authors", "model", "dataset",
        "experiment", "architecture", "equation", "proof",
        "derive", "loss", "theorem"
    ]

    is_paper_question = any(k in query for k in paper_indicators)

    if any(k in query for k in ["summarize", "summary", "overview"]):
        task = "summary"
    elif is_paper_question:
        task = "paper_question"
    else:
        task = "general_question"

    return {
        **state,
        "metadata": {
            **state.get("metadata", {}),
            "planner": {
                "is_paper_question": is_paper_question,
                "task": task
            }
        }
    }
