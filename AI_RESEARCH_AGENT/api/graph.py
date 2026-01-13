from langgraph.graph import StateGraph, END
from agents import (
    planner_agent,
    decision_agent,
    executor_agent,
    validator_agent,
    AgentState,
)
from inference.model_loader import InferenceModel


def build_agent_graph(model: InferenceModel):
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("executor", lambda s: executor_agent(s, model))
    graph.add_node("validator", validator_agent)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "decision")
    graph.add_edge("decision", "executor")
    graph.add_edge("executor", "validator")
    graph.add_edge("validator", END)

    return graph.compile()
