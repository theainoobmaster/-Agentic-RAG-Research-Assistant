# # # from langgraph.graph import StateGraph, END
# # # from agents.planner_agent import planner_agent
# # # from agents.decision_agent import decision_agent
# # # from agents.executor_agent import executor_agent
# # # from agents.validator_agent import validator_agent
# # # from agents.state import AgentState
# # # from inference.model_loader import InferenceModel


# # # def build_agent_graph(model: InferenceModel):
# # #     graph = StateGraph(AgentState)

# # #     graph.add_node("planner", planner_agent)
# # #     graph.add_node("decision", decision_agent)
# # #     graph.add_node("executor", lambda s: executor_agent(s, model))
# # #     graph.add_node("validator", validator_agent)

# # #     graph.set_entry_point("planner")
# # #     graph.add_edge("planner", "decision")
# # #     graph.add_edge("decision", "executor")
# # #     graph.add_edge("executor", "validator")
# # #     graph.add_edge("validator", END)

# # #     return graph.compile()
# # from langgraph.graph import StateGraph, END
# # from agents.paper_profile_agent import paper_profile_agent
# # from agents.planner_agent import planner_agent
# # from agents.decision_agent import decision_agent
# # from agents.executor_agent import executor_agent

# # def build_agent_graph(model):
# #     graph = StateGraph(dict)

# #     graph.add_node("profile_extract", lambda s: paper_profile_agent(s, model))
# #     graph.add_node("planner", planner_agent)
# #     graph.add_node("decision", decision_agent)
# #     graph.add_node("executor", lambda s: executor_agent(s, model))

# #     graph.set_entry_point("profile_extract")
# #     graph.add_edge("profile_extract", "planner")
# #     graph.add_edge("planner", "decision")
# #     graph.add_edge("decision", "executor")
# #     graph.add_edge("executor", END)

# #     return graph.compile()
# from langgraph.graph import StateGraph, END

# from agents.planner_agent import planner_agent
# from agents.decision_agent import decision_agent
# from agents.executor_agent import executor_agent
# from agents.paper_profile_agent import paper_profile_agent


# def build_agent_graph(model):
#     graph = StateGraph(dict)

#     graph.add_node("profile", lambda s: paper_profile_agent(s, model))
#     graph.add_node("planner", planner_agent)
#     graph.add_node("decision", decision_agent)
#     graph.add_node("executor", lambda s: executor_agent(s, model))

#     # Conditional entry
#     graph.set_entry_point("profile")

#     graph.add_edge("profile", "planner")
#     graph.add_edge("planner", "decision")
#     graph.add_edge("decision", "executor")
#     graph.add_edge("executor", END)

#     return graph.compile()
from langgraph.graph import StateGraph, END

from agents.paper_profile_agent import paper_profile_agent
from agents.planner_agent import planner_agent
from agents.decision_agent import decision_agent
from agents.executor_agent import executor_agent


def build_agent_graph(model):
    graph = StateGraph(dict)

    # -------------------------
    # Nodes
    # -------------------------
    graph.add_node("profile", lambda s: paper_profile_agent(s, model))
    graph.add_node("planner", planner_agent)
    graph.add_node("decision", decision_agent)
    graph.add_node("executor", lambda s: executor_agent(s, model))

    # -------------------------
    # Entry point
    # -------------------------
    graph.set_entry_point("profile")

    # -------------------------
    # Flow
    # -------------------------
    graph.add_edge("profile", "planner")
    graph.add_edge("planner", "decision")
    graph.add_edge("decision", "executor")
    graph.add_edge("executor", END)

    return graph.compile()
    
