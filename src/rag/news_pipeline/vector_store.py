from langchain_postgres import PGVector
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import logging
import psycopg2
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

COLLECTION_NAME = "news_articles"


class VectorStore:
    def __init__(self):
        load_dotenv()
        self.db_url = os.getenv("DB_URL")
        self.embeddings: NVIDIAEmbeddings = NVIDIAEmbeddings(
            model="nvidia/nv-embed-v1",
            batch_size=16,
        )
        self.vector_store: PGVector = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=self.db_url,
            use_jsonb=True,
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=True,
        )

    def _get_existing_urls(self) -> set[str]:
        try:
            dsn = self.db_url.replace("postgresql+psycopg://", "postgresql://")
            conn = psycopg2.connect(dsn)
            cur = conn.cursor()
            cur.execute("SELECT DISTINCT cmetadata->>'url' FROM langchain_pg_embedding")
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return {row[0] for row in rows if row[0]}
        except Exception as e:
            logger.warning("Không kiểm tra được duplicate URL: %s", e)
            return set()

    def insert(self, documents: list[Document]) -> int:
        if not documents:
            logger.info("insert() nhận danh sách rỗng, skip.")
            return 0

        existing_urls = self._get_existing_urls()
        new_docs = [d for d in documents if d.metadata.get("url") not in existing_urls]

        if not new_docs:
            logger.info(
                "Tất cả %i docs đã tồn tại trong DB (theo URL), skip insert.",
                len(documents),
            )
            return 0

        print(
            f"[DEBUG] insert() called with {len(new_docs)} new documents "
            f"(lọc từ tổng {len(documents)})"
        )
        try:
            self.vector_store.add_documents(new_docs)
            print("[DEBUG] add_documents completed")
            logger.info("Đã lưu %i chunks mới vào PGVector.", len(new_docs))
            return len(new_docs)
        except Exception as e:
            print(f"[DEBUG] EXCEPTION in insert: {type(e).__name__}: {e}")
            raise

    def query(
        self, question: str, k: int = 20, hours_ago: int | None = None
    ) -> list[Document]:
        """
        Query vector store.
        - Luôn lấy pool lớn (mặc định 20) để có đủ candidate.
        - Lọc unique theo URL.
        - Nếu hours_ago được truyền, chỉ trả về bài trong khoảng thời gian đó,
          sắp xếp mới nhất lên đầu.
        - Nếu không có bài nào trong khung thời gian, trả về bài mới nhất có
          liên quan và gắn flag 'recency_warning' vào metadata.
        """
        results = self.vector_store.similarity_search_with_score(question, k=k)
        filtered: list[Document] = []

        for doc, score in results:
            if score >= 0.5:
                doc.metadata["similarity_score"] = score
                filtered.append(doc)
        if not filtered:
            logger.info("Query '%s': không tìm thấy doc nào có score >= 0.6.", question[:50])
            return []

        # Lọc unique theo URL
        seen_urls = set()
        unique_docs: list[Document] = []
        for doc in filtered:
            url = doc.metadata.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_docs.append(doc)
            elif not url:
                unique_docs.append(doc)

        if not unique_docs:
            return []

        # Không lọc thời gian -> trả về top 5 unique
        if hours_ago is None:
            logger.info(
                "Query '%s': tìm được %d docs (trả về %d unique).",
                question[:50],
                len(docs),
                len(unique_docs[:5]),
            )
            return unique_docs[:5]

        # Lọc theo recency
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        recent_docs: list[Document] = []

        for doc in unique_docs:
            pub_str = doc.metadata.get("published_at")
            if not pub_str:
                continue
            try:
                pub_dt = datetime.fromisoformat(pub_str)
                if pub_dt.tzinfo is None:
                    pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                if pub_dt >= cutoff:
                    recent_docs.append(doc)
            except Exception:
                continue

        if recent_docs:
            recent_docs.sort(
                key=lambda d: (
                    datetime.fromisoformat(d.metadata["published_at"])
                    if d.metadata.get("published_at")
                    else datetime.min.replace(tzinfo=timezone.utc)
                ),
                reverse=True,
            )
            logger.info(
                "Query '%s': %d recent docs trong %dh.",
                question[:50],
                len(recent_docs),
                hours_ago,
            )
            return recent_docs[:5]

        # Không có tin mới — trả về bài mới nhất có liên quan + cảnh báo
        dated_docs = []
        for doc in unique_docs:
            pub_str = doc.metadata.get("published_at")
            if pub_str:
                try:
                    dt = datetime.fromisoformat(pub_str)
                    dated_docs.append((dt, doc))
                except Exception:
                    pass

        if dated_docs:
            dated_docs.sort(key=lambda x: x[0], reverse=True)
            best_doc = dated_docs[0][1]
            best_doc.metadata["recency_warning"] = (
                f"Không có tin trong {hours_ago} giờ qua. "
                f"Tin gần nhất: {best_doc.metadata.get('date', 'N/A')}"
            )
            logger.info(
                "Query '%s': không có tin trong %dh, trả về tin gần nhất (%s).",
                question[:50],
                hours_ago,
                best_doc.metadata.get("date", "N/A"),
            )
            return [best_doc]

        return unique_docs[:5]
