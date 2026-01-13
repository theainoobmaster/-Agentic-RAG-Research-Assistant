# def decision_agent(state: "AgentState") -> "AgentState":
#     allow = bool(state.get("paper_url"))

#     return {
#         **state,
#         "allow_retrieval": allow,
#         "reasoning_mode": "paper_grounded" if allow else "general",
#         "metadata": {
#             **state.get("metadata", {}),
#             "decision": {
#                 "allow_retrieval": allow,
#                 "forced_by": "paper_url" if allow else None,
#             },
#         },
#     }
def decision_agent(state):
    planner = state["metadata"]["planner"]

    if not planner["is_paper_question"]:
        route = "general"
    elif planner["task"] == "summary":
        route = "profile"
    else:
        route = "react"

    return {
        **state,
        "metadata": {
            **state["metadata"],
            "decision": {"route": route}
        }
    }
