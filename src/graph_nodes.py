from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
from util.tools import build_agent_tools, retrieve_context, current_time
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from graph_state import GraphState
import re
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def process_input(state: GraphState) -> GraphState:
    """Node 1: Parse user input, extract time preference, initialize state."""
    question = state["current_question"]
    
    # Parse time preference from question
    # "mới nhất" → 12h, "24h" → 24h, "tuần" → 168h, etc.
    time_hours: int = extract_time_preference(question)
    
    state["time_preference_hours"] = time_hours
    state["crawl_count"] = 0
    state["crawl_history"] = []
    state["user_feedback"] = None
    state["recency_status"] = "NOT_FOUND"
    state["docs_found"] = []
    
    logger.info(f"[process_input] Question: {question}, Time pref: {time_hours}h")
    return state


def retrieve_initial(state: GraphState) -> GraphState:
    """Node 2: Call retrieve_context(question) and populate docs_found."""
    question: str = state["current_question"]
    
    # Gọi retrieve_context từ tools
    result: str = retrieve_context(question)
    
    # Parse result thành documents
    docs: list = parse_retrieve_result(result)
    state["docs_found"] = docs
    
    logger.info(f"[retrieve_initial] Found {len(docs)} docs")
    return state


def check_recency(state: GraphState) -> GraphState:
    """Node 3: Check nếu docs recent hay old."""
    # Implement recency check logic
    time_now: datetime = datetime.fromisoformat(current_time())
    time_allow: datetime = time_now - timedelta(hours=state["time_preference_hours"])
    docs: list[Document] = state["docs_found"]
    if not docs:
        state["recency_status"] = "NOT_FOUND"
        return state
    for doc in docs:
        doc_time = datetime.fromisoformat(doc.metadata["date"])
        if doc_time > time_allow:
            state["recency_status"] = "RECENT"
            return state
    state["recency_status"] = "OLD"
    return state


def decide_crawl(state: GraphState) -> GraphState:
    """Node 4: Decide crawl source (user URL → VnExpress → BBC)."""
    # Implement crawl source decision
    pass


def crawl_and_process(state: GraphState) -> GraphState:
    """Node 5: Execute crawl, update crawl_history."""
    # Implement crawl logic
    pass


def retrieve_after_crawl(state: GraphState) -> GraphState:
    """Node 6: Re-retrieve after crawl."""
    # Implement post-crawl retrieve
    pass


def generate_answer(state: GraphState) -> GraphState:
    """Node 7: LLM generates final answer with citations."""
    # Implement answer generation
    pass


# Helper functions
def extract_time_preference(question: str) -> int:
    """Extract time preference from question text."""
    # Regex patterns để detect "mới nhất", "24h", "tuần", etc.
    if re.search(r"mới nhất|gần nhất", question, re.IGNORECASE):
        return 12
    match = re.search(r"(\d+)\s*giờ", question)
    if match:
        return int(match.group(1))
    # ... more patterns
    return 12  # default


def parse_retrieve_result(result: str) -> list:
    """Convert retrieve_context result string to Document list."""
    # Parse result string and convert to Document objects

    if not result or "Không tìm thấy thông tin liên quan" in result or "Lỗi khi query database" in result:
        return []

    blocks = [b.strip() for b in result.split("\n\n") if b.strip()]
    docs: list[Document] = []

    for blk in blocks:
        title_m = re.search(r"Nguồn:\s*(.*)", blk)
        url_m = re.search(r"URL:\s*(.*)", blk)
        date_m = re.search(r"Ngày:\s*(.*)", blk)
        content_m = re.search(r"Nội dung:\s*(.*)", blk, re.DOTALL)

        title = title_m.group(1).strip() if title_m else ""
        url = url_m.group(1).strip() if url_m else ""
        date = date_m.group(1).strip() if date_m else ""
        content = content_m.group(1).strip() if content_m else ""

        doc = Document(
            page_content=content,
            metadata={
                "title": title,
                "url": url,
                "date": date,
            },
        )
        docs.append(doc)

    return docs
