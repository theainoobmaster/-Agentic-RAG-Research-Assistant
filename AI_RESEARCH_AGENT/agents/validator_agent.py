def validator_agent(state: "AgentState") -> "AgentState":
    if not state.get("grounded", False):
        return {
            **state,
            "answer_type": "out_of_scope_for_paper",
            "refusal_reason": "Insufficient evidence in retrieved context.",
        }

    return {
        **state,
        "answer_type": "paper_grounded_answer",
        "refusal_reason": None,
    }
