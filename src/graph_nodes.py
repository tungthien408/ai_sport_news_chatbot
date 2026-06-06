"""
graph_nodes.py
==============
Mỗi function là 1 node trong LangGraph.  Node nhận GraphState, trả về dict
chứa các key cần update (LangGraph tự merge vào state, không cần trả toàn bộ state).

Flow:
  process_input → retrieve_initial → check_recency
      ├─ RECENT  → generate_answer → END
      ├─ OLD     → generate_answer (with recency warning) → END
      └─ NOT_FOUND → decide_crawl → crawl_and_process → retrieve_after_crawl
                          └─ check_recency (loop, tối đa 2 lần)
                                └─ (nếu vẫn NOT_FOUND sau 2 crawl) → generate_answer → END
"""

import logging
import re
from datetime import datetime, timedelta, timezone

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from graph_state import GraphState
from util.tools import crawl_news_feed, current_time, retrieve_context

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# LLM instance — dùng chung, khởi tạo 1 lần
# ──────────────────────────────────────────────────────────────────────────────
_llm = ChatNVIDIA(
    model="nvidia/nemotron-3-super-120b-a12b",
    temperature=0.2,
)

GENERATE_SYSTEM = """Bạn là trợ lý tin tức thể thao. Trả lời ngắn gọn, chính xác, chỉ dựa trên context được cung cấp.
Không được bịa đặt hay suy diễn. Luôn kèm citation: tiêu đề bài + URL + ngày.
Nếu context không đủ, nói thẳng không có đủ thông tin."""

# URLs crawl theo thứ tự mặc định
DEFAULT_CRAWL_URLS = [
    "https://vnexpress.net/rss/the-thao.rss",
    "https://feeds.bbci.co.uk/sport/rss.xml",
]


# ──────────────────────────────────────────────────────────────────────────────
# Node 1: process_input
# ──────────────────────────────────────────────────────────────────────────────
def process_input(state: GraphState) -> dict:
    """Parse user input, extract time preference, init crawl tracking."""
    question = state["current_question"]
    time_hours = _extract_time_preference(question)

    logger.info("[process_input] question=%r time_pref=%dh", question, time_hours)

    return {
        "time_preference_hours": time_hours,
        "crawl_count": 0,
        "crawl_history": [],
        "next_crawl_url": None,
        "user_feedback": None,
        "recency_status": "NOT_FOUND",
        "docs_found": [],
        "final_answer": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 2: retrieve_initial
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_initial(state: GraphState) -> dict:
    """Query vector DB và populate docs_found."""
    question = state["current_question"]
    result_str = retrieve_context(question)
    docs = _parse_retrieve_result(result_str)

    logger.info("[retrieve_initial] found %d docs", len(docs))
    return {"docs_found": docs}


# ──────────────────────────────────────────────────────────────────────────────
# Node 3: check_recency
# ──────────────────────────────────────────────────────────────────────────────
def check_recency(state: GraphState) -> dict:
    """Kiểm tra docs có nằm trong khung thời gian không."""
    docs: list[Document] = state["docs_found"]
    hours: int = state["time_preference_hours"]

    if not docs:
        logger.info("[check_recency] no docs → NOT_FOUND")
        return {"recency_status": "NOT_FOUND"}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=hours)

    for doc in docs:
        pub_str = doc.metadata.get("published_at") or doc.metadata.get("date", "")
        if not pub_str:
            continue
        try:
            pub_dt = datetime.fromisoformat(pub_str)
            if pub_dt.tzinfo is None:
                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
            if pub_dt >= cutoff:
                logger.info(
                    "[check_recency] RECENT doc found: %s", doc.metadata.get("title")
                )
                return {"recency_status": "RECENT"}
        except (ValueError, TypeError):
            continue

    logger.info("[check_recency] all docs are OLD (cutoff=%s)", cutoff.isoformat())
    return {"recency_status": "OLD"}


# ──────────────────────────────────────────────────────────────────────────────
# Node 4: decide_crawl
# ──────────────────────────────────────────────────────────────────────────────
def decide_crawl(state: GraphState) -> dict:
    """
    Chọn URL tiếp theo để crawl.
    Thứ tự: user-provided URL (nếu có trong câu hỏi) → VnExpress → BBC.
    Nếu đã crawl hết hoặc đủ 2 lần → next_crawl_url = None (signal để stop).
    """
    crawl_history: list[str] = state.get("crawl_history", [])
    question = state["current_question"]
    crawl_count: int = state.get("crawl_count", 0)

    if crawl_count >= 2:
        logger.info("[decide_crawl] đã crawl %d lần, dừng.", crawl_count)
        return {"next_crawl_url": None}

    # Kiểm tra user có cung cấp URL RSS không
    user_url = _extract_rss_url_from_question(question)
    if user_url and user_url not in crawl_history:
        logger.info("[decide_crawl] dùng user URL: %s", user_url)
        return {"next_crawl_url": user_url}

    # Dùng thứ tự mặc định
    for url in DEFAULT_CRAWL_URLS:
        if url not in crawl_history:
            logger.info("[decide_crawl] next default URL: %s", url)
            return {"next_crawl_url": url}

    logger.info("[decide_crawl] không còn URL nào để crawl")
    return {"next_crawl_url": None}


# ──────────────────────────────────────────────────────────────────────────────
# Node 5: crawl_and_process
# ──────────────────────────────────────────────────────────────────────────────
def crawl_and_process(state: GraphState) -> dict:
    """Thực hiện crawl URL đã được chọn, cập nhật lịch sử."""
    url: str | None = state.get("next_crawl_url")

    if not url:
        logger.warning("[crawl_and_process] không có URL để crawl")
        return {}

    crawl_history: list[str] = list(state.get("crawl_history", []))
    crawl_count: int = state.get("crawl_count", 0)

    logger.info("[crawl_and_process] crawling %s", url)
    result = crawl_news_feed(url)
    logger.info("[crawl_and_process] result: %s", result)

    crawl_history.append(url)

    return {
        "crawl_history": crawl_history,
        "crawl_count": crawl_count + 1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Node 6: retrieve_after_crawl
# ──────────────────────────────────────────────────────────────────────────────
def retrieve_after_crawl(state: GraphState) -> dict:
    """Re-query vector DB sau khi crawl xong."""
    question = state["current_question"]
    result_str = retrieve_context(question)
    docs = _parse_retrieve_result(result_str)

    logger.info("[retrieve_after_crawl] found %d docs after crawl", len(docs))
    return {"docs_found": docs}


# ──────────────────────────────────────────────────────────────────────────────
# Node 7: generate_answer
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer(state: GraphState) -> dict:
    """
    LLM tổng hợp câu trả lời từ docs_found.
    - Nếu RECENT: trả lời bình thường + citation.
    - Nếu OLD: thêm cảnh báo "thông tin lịch sử".
    - Nếu NOT_FOUND: thông báo không tìm thấy.
    """
    question = state["current_question"]
    docs: list[Document] = state.get("docs_found", [])
    recency_status = state.get("recency_status", "NOT_FOUND")
    crawl_count: int = state.get("crawl_count", 0)
    crawl_history: list[str] = state.get("crawl_history", [])

    # ── Build context string ───────────────────────────────────────────────────
    if not docs:
        if crawl_count >= 2:
            answer = (
                "Không có thông tin cập nhật từ các nguồn hợp lệ trong khung thời gian yêu cầu.\n"
                f"Đã thử crawl: {', '.join(crawl_history)}\n"
                "Vui lòng cung cấp nguồn RSS cụ thể hoặc mở rộng khung thời gian."
            )
        else:
            answer = "Không tìm thấy thông tin liên quan trong cơ sở dữ liệu."

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    context_parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        recency_warn = meta.get("recency_warning", "")
        warn_text = f"\n    ⚠️  {recency_warn}" if recency_warn else ""
        context_parts.append(
            f"[{i}] Tiêu đề: {meta.get('title', 'N/A')}\n"
            f"    URL: {meta.get('url', 'N/A')}\n"
            f"    Ngày: {meta.get('date', 'N/A')}{warn_text}\n"
            f"    Nội dung: {doc.page_content[:600]}"
        )
    context_str = "\n\n".join(context_parts)

    # ── Build prompt ───────────────────────────────────────────────────────────
    recency_note = ""
    if recency_status == "OLD":
        recency_note = (
            "\n⚠️  LƯU Ý: Các tài liệu tìm được đều cũ hơn khung thời gian yêu cầu. "
            "Hãy nêu rõ đây là thông tin lịch sử và ghi ngày của bài."
        )

    user_prompt = (
        f"Câu hỏi: {question}\n\n"
        f"Context từ database:{recency_note}\n\n"
        f"{context_str}\n\n"
        "Hãy trả lời câu hỏi dựa trên context trên. "
        "Tóm tắt 1-2 câu, sau đó list nguồn (tiêu đề + URL + ngày)."
    )

    messages = [
        SystemMessage(content=GENERATE_SYSTEM),
        HumanMessage(content=user_prompt),
    ]

    logger.info("[generate_answer] calling LLM, recency_status=%s", recency_status)
    response = _llm.invoke(messages)
    answer_text = response.content

    return {
        "final_answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Router functions (dùng trong conditional_edges)
# ──────────────────────────────────────────────────────────────────────────────
def route_after_recency_check(state: GraphState) -> str:
    """
    Sau check_recency:
    - RECENT / OLD → generate_answer
    - NOT_FOUND    → decide_crawl (nếu còn slot) hoặc generate_answer (nếu hết)
    """
    status = state.get("recency_status", "NOT_FOUND")
    crawl_count = state.get("crawl_count", 0)

    if status in ("RECENT", "OLD"):
        return "generate_answer"

    # NOT_FOUND
    if crawl_count < 2:
        return "decide_crawl"

    # Đã crawl 2 lần vẫn NOT_FOUND
    return "generate_answer"


def route_after_decide_crawl(state: GraphState) -> str:
    """
    Sau decide_crawl:
    - Có URL → crawl_and_process
    - Không có URL (đã hết / limit) → generate_answer
    """
    next_url = state.get("next_crawl_url")
    if next_url:
        return "crawl_and_process"
    return "generate_answer"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _extract_time_preference(question: str) -> int:
    """Trích xuất time preference từ câu hỏi. Default = 12h."""
    q = question.lower()

    if re.search(r"tháng này|trong tháng", q):
        return 720   # ~30 ngày

    if re.search(r"tuần này|trong tuần|7 ngày", q):
        return 168

    if re.search(r"hôm nay|trong ngày", q):
        return 24

    # "trong X giờ" hoặc "Xh"
    match = re.search(r"trong\s*(\d+)\s*giờ", q) or re.search(r"(\d+)\s*h\b", q)
    if match:
        return int(match.group(1))

    # "mới nhất", "gần nhất" → default 12h
    return 12


def _extract_rss_url_from_question(question: str) -> str | None:
    """Trích xuất RSS URL do user cung cấp trong câu hỏi."""
    match = re.search(r"https?://\S+\.rss\b", question, re.IGNORECASE)
    if match:
        return match.group(0)
    return None


def _parse_retrieve_result(result: str) -> list[Document]:
    """Chuyển chuỗi trả về của retrieve_context() thành list Document."""
    if not result or "Không tìm thấy" in result or "Lỗi khi query" in result:
        return []

    blocks = [b.strip() for b in result.split("\n\n") if b.strip()]
    docs: list[Document] = []

    for blk in blocks:
        title_m = re.search(r"Nguồn:\s*(.*)", blk)
        url_m = re.search(r"URL:\s*(.*)", blk)
        date_m = re.search(r"Ngày:\s*(.*)", blk)
        content_m = re.search(r"Nội dung:\s*(.*)", blk, re.DOTALL)

        docs.append(
            Document(
                page_content=content_m.group(1).strip() if content_m else "",
                metadata={
                    "title": title_m.group(1).strip() if title_m else "",
                    "url": url_m.group(1).strip() if url_m else "",
                    "date": date_m.group(1).strip() if date_m else "",
                },
            )
        )

    return docs