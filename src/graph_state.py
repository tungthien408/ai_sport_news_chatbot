from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    current_question: str
    time_preference_hours: int
    docs_found: list[Document]
    recency_status: Literal['RECENT', 'OLD', 'NOT_FOUND']
    crawl_history: list[str]
    crawl_count: int
    user_feedback: str | None
