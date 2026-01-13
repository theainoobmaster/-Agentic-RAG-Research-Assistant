# from typing import TypedDict, Optional, List, Dict, Any


# class AgentState(TypedDict):
#     # -------- Input --------
#     user_query: str
#     paper_url: Optional[str]
#     paper_retriever: Optional[Any]  # 🔥 MUST EXIST

#     # -------- Planner --------
#     domain: Optional[str]
#     intent: Optional[str]
#     rewritten_query: Optional[str]

#     # -------- Decision --------
#     allow_retrieval: bool
#     reasoning_mode: Optional[str]

#     # -------- Executor --------
#     thoughts: List[str]
#     actions: List[Dict[str, Any]]
#     observations: List[str]

#     # -------- Output --------
#     final_answer: Optional[str]
#     grounded: bool

#     # -------- Validation --------
#     answer_type: Optional[str]
#     refusal_reason: Optional[str]

#     # -------- Debug --------
#     metadata: Dict[str, Any]
from typing import TypedDict, Optional, Dict, Any, List

class AgentState(TypedDict):
    user_query: str
    paper_url: Optional[str]

    paper_docs: Optional[List[Any]]
    paper_profile: Optional[str]

    final_answer: Optional[str]

    metadata: Dict[str, Any]
