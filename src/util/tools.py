from langchain_core.tools import StructuredTool
from rag.rag_update_news import rag_update_news

rss_feed_link: list[str] = [
    "https://feeds.bbci.co.uk/sport/rss.xml",
    "https://vnexpress.net/rss/the-thao.rss",
    ""
]

def crawl_news_feed(url: str, output_file: str = "logs/raw_output.json") -> str:
    """
    Crawl RSS feed và tự động insert dữ liệu vào vector database.
    Trả về số lượng documents đã được insert.
    """
    num_inserted = rag_update_news(url, output_file)
    if num_inserted > 0:
        return f"✓ Crawled {url} và đã insert {num_inserted} documents vào vector database."
    else:
        return f"⚠ Crawled {url} nhưng không có documents để insert (có thể feed trống hoặc lỗi xử lý)."

def build_agent_tools():
    return [
        StructuredTool.from_function(
            func=crawl_news_feed,
            name="crawl_news_feed",
            description=(
                "Crawl dữ liệu từ một feed RSS cụ thể và tự động insert vào vector database. "
                "Chỉ dùng khi bạn cần thêm dữ liệu mới để trả lời câu hỏi của user. "
                "Chỉ dùng với RSS feed hợp lệ hoặc RSS mặc định "
                "https://vnexpress.net/rss/the-thao.rss nếu user không cung cấp nguồn khác. "
                "Không tự tạo RSS feed, URL bài báo, hoặc tìm trang web khác."
            ),
        )
    ]