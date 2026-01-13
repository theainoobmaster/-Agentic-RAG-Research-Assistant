from pydantic import BaseModel
from typing import List, Dict, Optional


class ChatRequest(BaseModel):
    query: str
    paper_url: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    thoughts: Optional[List[str]] = None
    actions: Optional[List[Dict]] = None
