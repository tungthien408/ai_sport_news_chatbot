from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import logging
import re

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

VIETNAMESE_SEPARATORS = [
    "\n\n",  # Ngăn cách đoạn — tốt nhất
    "\n",  # Xuống dòng đơn
    ".",  # Kết thúc câu
    "!",
    "?",
    " ",  # Từ — fallback cuối cùng
    "",  # Ký tự — fallback tuyệt đối
]

logger = logging.getLogger(__name__)


def parse_vietnamese_date(date_str: str) -> str:
    """
    Parse 'Thứ sáu, 22/5/2026, 18:45 (GMT+7)' -> ISO 8601 string.
    Trả về chuỗi rỗng nếu không parse được.
    """
    if not date_str:
        return ""
    match = re.search(r"(\d{1,2})/(\d{1,2})/(\d{4}),\s*(\d{1,2}):(\d{2})", date_str)
    if not match:
        return ""
    d, m, y, hr, mn = match.groups()
    try:
        tz_match = re.search(r"\(GMT([+-]\d+)\)", date_str)
        offset = int(tz_match.group(1)) if tz_match else 7
        tz = timezone(timedelta(hours=offset))
        dt = datetime(int(y), int(m), int(d), int(hr), int(mn), tzinfo=tz)
        return dt.isoformat()
    except Exception:
        return ""


class Chunker:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=VIETNAMESE_SEPARATORS,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,  # Đếm ký tự, không phải tokens
        )

    def chunk_article(self, article: dict) -> list[Document]:
        chunks: list[str] = self.splitter.split_text(article["content"])
        published_at = parse_vietnamese_date(article.get("date", ""))

        documents: list = []
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "title": article.get("title", ""),
                    "url": article.get("url", ""),
                    "date": article.get("date", ""),
                    "published_at": published_at,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            documents.append(doc)

        return documents

    def chunk_all(self, articles: list[dict]) -> list[Document]:
        all_docs = []
        for article in articles:
            if not article.get("content"):
                continue
            all_docs.extend(self.chunk_article(article))
        return all_docs
