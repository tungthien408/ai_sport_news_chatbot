from typing import Annotated, Literal, TypedDict

from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict):
    """State schema for the sports news chatbot LangGraph workflow.

    Attributes:
        messages: Conversation message history (auto-merged by LangGraph).
        current_question: The user's raw question text.
        question_type: Classification result from the gatekeeper node.
            - 'sports_news': recent sports news → route to vector DB.
            - 'sports_wiki': general sports knowledge → route to Wikipedia.
            - 'off_topic': not sports-related → politely decline.
        time_preference_hours: How far back to search (parsed from question).
        docs_found: Documents retrieved from vector DB or Wikipedia.
        recency_status: Whether retrieved docs are recent, old, or missing.
        wiki_content: Raw Wikipedia summary text (if Wikipedia was used).
        crawl_history: List of URLs already crawled in this session.
        crawl_count: Number of crawl attempts made.
        final_answer: The generated response to present to the user.
        user_feedback: Optional user feedback signal.
    """

    messages: Annotated[list, add_messages]
    current_question: str
    question_type: Literal["sports_news", "sports_wiki", "off_topic"]
    time_preference_hours: int
    docs_found: list[Document]
    recency_status: Literal["RECENT", "OLD", "NOT_FOUND"]
    wiki_content: str | None
    crawl_history: list[str]
    crawl_count: int
    final_answer: str
    user_feedback: str | None
