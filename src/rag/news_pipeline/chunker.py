from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

VIETNAMESE_SEPARATORS = [
    "\n\n",   # Ngăn cách đoạn — tốt nhất
    "\n",     # Xuống dòng đơn
    ".",      # Kết thúc câu
    "!",
    "?",
    " ",      # Từ — fallback cuối cùng
    "",       # Ký tự — fallback tuyệt đối
]

logger = logging.getLogger(__name__)
class Chunker:
    def __init__(self):
        self.splitter= RecursiveCharacterTextSplitter(
            separators=VIETNAMESE_SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,  # Đếm ký tự, không phải tokens
        )
    
    def chunk_article(self, article: dict) -> list[Document]:
        chunks: list[str] = self.splitter.split_text(article["content"])

        documents: list = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "date": article.get("date", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        return documents
    
    def chunk_all(self, articles: list[dict]) -> list[Document]:
        all_docs = []
        for article in articles:
            if not article.get('content'):
                continue
            all_docs.extend(self.chunk_article(article))
        return all_docs