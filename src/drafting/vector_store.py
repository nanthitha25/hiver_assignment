"""ChromaDB Vector Store for indexing historical @AppleSupport resolution pairs."""

import logging
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from src.drafting.historical_data import HISTORICAL_APPLE_RESOLUTIONS

logger = logging.getLogger(__name__)


class HistoricalVectorStore:
    """Manages the embedded ChromaDB vector collection for customer support RAG."""

    def __init__(self, persist_dir: Optional[str] = None, collection_name: str = "apple_support_resolutions"):
        self.persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.encoder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._ensure_seed_data()

    def _ensure_seed_data(self):
        """Populates seed data if collection is empty."""
        if self.collection.count() == 0:
            logger.info("Initializing vector store with seed Apple Support resolutions...")
            self.index_records(HISTORICAL_APPLE_RESOLUTIONS)

    def index_records(self, records: List[Dict[str, str]]):
        """Indexes a list of resolution records into ChromaDB."""
        ids = []
        documents = []
        metadatas = []
        embeddings = []

        texts_to_embed = [r["customer_text"] for r in records]
        encoded_vecs = self.encoder.encode(texts_to_embed, convert_to_numpy=True, normalize_embeddings=True)

        for record, vec in zip(records, encoded_vecs):
            ids.append(record["tweet_id"])
            documents.append(record["customer_text"])
            metadatas.append({
                "agent_reply": record["agent_reply"],
                "intent": record.get("intent", "OS_SOFTWARE_TROUBLESHOOTING"),
            })
            embeddings.append(vec.tolist())

        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(self, query_text: str, intent: Optional[str] = None, top_k: int = 3) -> Dict:
        """Queries the collection for semantically similar historical customer tweets."""
        query_vec = self.encoder.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

        where_filter = {"intent": intent} if intent and intent != "OUT_OF_SCOPE_AMBIGUOUS" else None

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        return results
