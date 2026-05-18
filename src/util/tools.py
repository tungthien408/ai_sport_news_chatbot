from langchain.tools import StructuredTool
from pydantic import BaseModel
from rag import rag_update_news


def build_agent_tools():
    return [
        StructuredTool.from_function(
            func=rag_update_news,
            name="news_crawler",
            description="Crawl dữ liệu bài báo từ một feed RSS và lưu vào file JSON. Đầu vào là URL của feed RSS và đường dẫn file JSON để lưu dữ liệu.",
        )
    ]
