from rag.crawler import news_crawler_v2
from rag.news_pipeline.cleaner import TextCleaner
from rag.news_pipeline.chunker import Chunker
from rag.news_pipeline.vector_store import VectorStore
from langchain_core.documents import Document
import json, logging

logger = logging.getLogger(__name__)


def rag_update_news(
    feed_url: str = "https://vnexpress.net/rss/the-thao.rss",
    output_raw: str = "logs/raw_output.json",
) -> int:
    output_clean: str = "logs/cleaned_output.json"

    news_crawler_v2.operation(feed_url, output_raw)

    news_data: list[dict] = TextCleaner(output_raw).operation()
    with open(output_clean, "w", encoding="utf-8") as f:
        json.dump(news_data, f, ensure_ascii=False, indent=4)

    documents: list[Document] = Chunker().chunk_all(news_data)

    # If no documents were produced by the pipeline, skip DB insert to avoid
    # unnecessary operations and crashes when downstream DB/schema is not ready.
    if not documents:
        logger.info(
            "rag_update_news: no documents to insert, skipping VectorStore.insert()"
        )
        return 0

    try:
        print(f"[DEBUG] About to insert {len(documents)} documents")
        num_inserted = VectorStore().insert(documents)
        logger.info(
            "rag_update_news: VectorStore.insert(): inserted %i sucessfully",
            num_inserted,
        )
        return num_inserted
    except Exception as e:
        logger.exception("rag_update_news: VectorStore.insert failed: %s", e)
        return 0

