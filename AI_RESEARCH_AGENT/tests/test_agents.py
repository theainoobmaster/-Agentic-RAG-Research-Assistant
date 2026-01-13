from agents.graph import build_agent_graph
from agents.executor_agent import executor_agent
from inference.model_loader import InferenceModel


def get_initial_state(query: str):
    return {
        "user_query": query,
        "domain": None,
        "intent": None,
        "rewritten_query": None,
        "reasoning_mode": None,
        "tools": [],
        "thoughts": [],
        "actions": [],
        "observations": [],
        "final_answer": None,
        "conversation_history": [],
        "long_term_memory": [],
        "human_feedback": [],
        "metadata": {},
    }


def test_planner_semantic_control():
    """
    Planner should disambiguate 'transformers'
    and rewrite to 'transformer neural networks'.
    """
    graph = build_agent_graph()

    state = get_initial_state("Explain how transformers work")
    state = graph.invoke(state)

    assert state["domain"] == "machine_learning"
    assert "transformer neural networks" in state["rewritten_query"]


def test_executor_uses_rewritten_query():
    """
    Executor must either:
    - produce a correct neural-network answer
    OR
    - take a valid ReAct action (no silent failure)
    """
    graph = build_agent_graph()

    model = InferenceModel(
        base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_path="models/lora_adapter",
    )

    state = get_initial_state("Explain how transformers work")
    state = graph.invoke(state)
    state = executor_agent(state, model)

    if state["final_answer"] is not None:
        answer = state["final_answer"].lower()
        assert "neural" in answer
        assert "electric" not in answer
    else:
        # ReAct path taken
        assert len(state["actions"]) > 0


def test_human_feedback_override():
    """
    Human feedback must override ambiguity.
    Agent must respect correction or take a valid action.
    """
    graph = build_agent_graph()

    model = InferenceModel(
        base_model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        lora_path="models/lora_adapter",
    )

    state = get_initial_state("Explain transformers")

    state["human_feedback"].append({
        "feedback": "When I say transformers, I always mean neural networks",
        "priority": "high",
    })

    state = graph.invoke(state)
    state = executor_agent(state, model)

    if state["final_answer"] is not None:
        answer = state["final_answer"].lower()
        assert "neural" in answer
        assert "electric" not in answer
    else:
        # ReAct path taken
        assert len(state["actions"]) > 0


# Optional: allows manual execution, pytest will ignore this
if __name__ == "__main__":
    test_planner_semantic_control()
    test_executor_uses_rewritten_query()
    test_human_feedback_override()
    print("All tests ran")
