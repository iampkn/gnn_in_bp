"""
Phase 2: Vector Database integration using Qdrant.
Stores event/resource/device feature vectors for similarity search.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np


class VectorStore:
    """
    Wrapper for Qdrant vector database operations.
    Falls back to in-memory numpy-based search if Qdrant is unavailable.
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        self.host = host
        self.port = port
        self.client = None
        self._memory_store: Dict[str, List[dict]] = {}
        self._try_connect()

    def _try_connect(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=self.host, port=self.port)
            self.client.get_collections()
            print(f"Connected to Qdrant at {self.host}:{self.port}")
        except Exception:
            self.client = None
            print("Qdrant unavailable. Using in-memory vector store.")

    def create_collection(self, name: str, vector_size: int):
        if self.client:
            from qdrant_client.models import Distance, VectorParams
            self.client.recreate_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=vector_size, distance=Distance.COSINE
                ),
            )
        else:
            self._memory_store[name] = []

    def upsert(self, collection: str, points: List[dict]):
        """
        Insert vectors. Each point: {'id': int, 'vector': list/np, 'payload': dict}
        """
        if self.client:
            from qdrant_client.models import PointStruct
            qdrant_points = [
                PointStruct(
                    id=p["id"],
                    vector=p["vector"].tolist() if hasattr(p["vector"], "tolist") else p["vector"],
                    payload=p.get("payload", {}),
                )
                for p in points
            ]
            self.client.upsert(collection_name=collection, points=qdrant_points)
        else:
            if collection not in self._memory_store:
                self._memory_store[collection] = []
            self._memory_store[collection].extend(points)

    def search(
        self, collection: str, query_vector, top_k: int = 5, filters: Optional[dict] = None
    ) -> List[dict]:
        """Search for similar vectors."""
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        if self.client:
            results = self.client.search(
                collection_name=collection,
                query_vector=query_vector,
                limit=top_k,
            )
            return [
                {"id": r.id, "score": r.score, "payload": r.payload}
                for r in results
            ]
        else:
            return self._memory_search(collection, query_vector, top_k, filters)

    def _memory_search(
        self, collection: str, query_vector: list, top_k: int, filters: Optional[dict]
    ) -> List[dict]:
        points = self._memory_store.get(collection, [])
        if not points:
            return []

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        scored = []
        for p in points:
            v = np.array(p["vector"], dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0:
                continue
            cos_sim = np.dot(q, v) / (q_norm * v_norm)

            if filters:
                payload = p.get("payload", {})
                match = all(payload.get(k) == v for k, v in filters.items())
                if not match:
                    continue

            scored.append({"id": p["id"], "score": float(cos_sim), "payload": p.get("payload", {})})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
