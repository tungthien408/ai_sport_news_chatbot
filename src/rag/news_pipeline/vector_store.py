from langchain_postgres import PGVector
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
import logging
import uuid  # <-- Added for generating unique IDs
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

    # REMOVED _get_existing_urls() completely!

    def insert(self, documents: list[Document]) -> int:
        if not documents:
            logger.info("insert() nhận danh sách rỗng, skip.")
            return 0

        # 1. Generate a Deterministic UUID for every chunk
        doc_ids = []
        for doc in documents:
            # Grab the URL and chunk index from the metadata you set in chunker.py
            url = doc.metadata.get("url", "unknown_url")
            chunk_idx = doc.metadata.get("chunk_index", 0)
            
            # Create a unique string signature for this exact piece of text
            unique_signature = f"{url}_chunk_{chunk_idx}"
            
            # Hash it into a standard UUID format
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, unique_signature))
            doc_ids.append(chunk_id)

        print(f"[DEBUG] insert() called with {len(documents)} documents")
        
        try:
            # 2. Pass the IDs to add_documents. 
            # PGVector will now UPSERT automatically. No duplicates will be created.
            self.vector_store.add_documents(documents, ids=doc_ids)
            
            print("[DEBUG] add_documents completed")
            logger.info("Đã lưu/cập nhật %i chunks vào PGVector.", len(documents))
            return len(documents)
            
        except Exception as e:
            print(f"[DEBUG] EXCEPTION in insert: {type(e).__name__}: {e}")
            raise

    def query(
        self, question: str, k: int = 20, hours_ago: int | None = None
    ) -> list[Document]:
        # ... (Keep your query function exactly the same as before)
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
                len(filtered),
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