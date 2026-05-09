# app/rag/indexer.py
# Purpose: embed all bus documents and store vectors in Qdrant
# This runs once (or when data changes). Agents query Qdrant at runtime.

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct
)
import ollama

# --- Configuration ---
COLLECTION_NAME = "bus_schedules"
EMBEDDING_MODEL = "nomic-embed-text"   # free, runs via Ollama locally
VECTOR_SIZE = 768                       # nomic-embed-text output dimension


def get_embedding(text: str) -> list[float]:
    """
    Call Ollama local API to embed a single text.
    Interview point: in production you'd batch these for speed.
    Ollama runs the model locally — zero cost, zero API key.
    """
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def create_collection(client: QdrantClient):
    """
    Create Qdrant collection if it doesn't exist.
    Interview point: cosine distance is standard for text similarity.
    Dot product is faster but requires normalized vectors.
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping create.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    print(f"Created collection '{COLLECTION_NAME}'")


def index_documents(client: QdrantClient, documents: list[dict]):
    """
    Embed each document and upsert into Qdrant.
    Upsert = insert or update — safe to re-run without duplicates.
    """
    points = []

    for i, doc in enumerate(documents):
        if i % 50 == 0:
            print(f"  Embedding document {i}/{len(documents)}...")

        vector = get_embedding(doc["text"])

        # PointStruct = one record in Qdrant
        # id: unique integer (we convert SC00001 -> 1)
        # vector: the embedding
        # payload: metadata dict — queryable, filterable
        point = PointStruct(
            id=i,
            vector=vector,
            payload={
                "text": doc["text"],   # store original text for display
                **doc["metadata"]       # spread all metadata fields
            }
        )
        points.append(point)

    # upsert in batches of 100 — avoids memory issues with large datasets
    batch_size = 100
    for start in range(0, len(points), batch_size):
        batch = points[start : start + batch_size]
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=batch
        )

    print(f"Indexed {len(points)} documents into Qdrant.")


if __name__ == "__main__":
    # load documents built in previous step
    docs_path = Path("data/raw/documents.json")
    documents = json.loads(docs_path.read_text())

    # connect to local Qdrant (no auth needed for local)
    client = QdrantClient(url="http://localhost:6333")

    create_collection(client)
    index_documents(client, documents)

    # verify
    info = client.get_collection(COLLECTION_NAME)
    print(f"\nCollection stats:")
    print(f"  Points indexed: {info.points_count}")
    print(f"  Status: {info.status}")