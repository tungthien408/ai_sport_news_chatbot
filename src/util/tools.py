from langchain_core.tools import StructuredTool
from rag.news_pipeline.vector_store import VectorStore
from rag.rag_update_news import rag_update_news
from datetime import datetime, timezone

rss_feed_link: list[str] = [
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://vnexpress.net/rss/the-thao.rss",
    ""
]

def current_time():
    """Return current UTC time."""
    return datetime.now(timezone.utc).isoformat()

def retrieve_context(question: str, hours_ago: int) -> str:
    """
    Query vector database để lấy thông tin liên quan đến câu hỏi.
    Dùng tool này SAU KHI crawl_news_feed, hoặc bất cứ lúc nào cần
    lấy context từ dữ liệu đã được lưu trữ.
    """
    try:
        docs = VectorStore().query(question, k=5, hours_ago=hours_ago)
        if not docs:
            return "Không tìm thấy thông tin liên quan trong database."

        results = []
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            results.append(
                f"[{i}] Nguồn: {meta.get('title', 'N/A')}\n"
                f"    URL: {meta.get('url', 'N/A')}\n"
                f"    Ngày: {meta.get('date', 'N/A')}\n"
                f"    Nội dung: {doc.page_content[:500]}"
            )
        return "\n\n".join(results)
    except Exception as e:
        return f"Lỗi khi query database: {e}"

def build_agent_tools():
    return [
        StructuredTool.from_function(
            func=retrieve_context,
            name="retrieve_context",
            description=(
                "Query vector database để lấy thông tin liên quan đến một câu hỏi. "
                "Dùng tool này TRƯỚC khi trả lời để kiểm tra dữ liệu có sẵn. "
            ),
        ),
        StructuredTool.from_function(
            func=current_time,
            name="current_time",
            description=(
                "Return current UTC time in ISO format."
                "Sử dụng nếu muốn kiểm tra xem thông tin có được nó có thật sự là thông tin cập nhật mới nhất hay không"
                "Sử dụng nếu muốn kiểm tra xem thông tin có được nó có thỏa mãn với query của người dùng về mặt thời gian hay không"
            ),
        )
    ]