from langchain_postgres import PGVector
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os, logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "news_articles"

class VectorStore:
    def __init__(self):
        load_dotenv()
        DB_URL = os.getenv("DB_URL")
        self.embeddings: NVIDIAEmbeddings = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
        self.vector_store: PGVector = PGVector(
            embeddings=self.embeddings,
            collection_name=COLLECTION_NAME,
            connection=DB_URL,
            use_jsonb=True,
        )
    
    def insert(self, documents: list[Document]) -> int:
        """Insert documents vào vector store và trả về số lượng documents đã insert."""
        self.vector_store.add_documents(documents)
        logger.info("Đã lưu %i chunks vào PGVector.", len(documents))
        return len(documents)
    
    def query(self):
        ...