# app/rag/retriever.py
# Hybrid retrieval: BM25 sparse + dense vector search combined
# via Reciprocal Rank Fusion (RRF)
#
# Interview point: why hybrid?
# Dense search understands meaning — "evening" matches "18:30, 20:00, 21:30"
# BM25 understands keywords — "TSRTC" or "SC00012" matches exactly
# Neither alone is best. RRF combines both ranked lists without
# needing to tune score scales (BM25 scores and cosine scores
# are on completely different scales — you can't just add them)

import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
from rank_bm25 import BM25Okapi
import ollama

from app.config import COLLECTION_NAME, QDRANT_URL, EMBEDDING_MODEL

# --- BM25 index built once at module load time ---
# We load the same documents.json used for indexing
# Interview point: BM25 is an in-memory index — fast to query,
# no extra infrastructure needed. At 309 documents this is trivial.
# At 1M documents you'd use Elasticsearch instead.

def _build_bm25_index():
    docs_path = Path("data/raw/documents.json")
    documents = json.loads(docs_path.read_text())
    
    # tokenise by splitting on whitespace — BM25Okapi expects list of tokens
    corpus = [doc["text"].lower().split() for doc in documents]
    index = BM25Okapi(corpus)
    
    return index, documents

BM25_INDEX, ALL_DOCUMENTS = _build_bm25_index()


def get_embedding(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]


def build_filter(
    origin: str = None,
    destination: str = None,
    seat_type: str = None,
    max_price: float = None,
    min_available: int = 1
) -> Filter | None:

    conditions = []   # this line was accidentally deleted

    if origin:
        conditions.append(
            FieldCondition(key="origin", match=MatchValue(value=origin))
        )
    if destination:
        conditions.append(
            FieldCondition(key="destination", match=MatchValue(value=destination))
        )
    if seat_type:
        conditions.append(
            FieldCondition(key="seat_type", match=MatchValue(value=seat_type))
        )
    if max_price:
        conditions.append(
            FieldCondition(key="price", range=Range(lte=max_price))
        )

    conditions.append(
        FieldCondition(key="available", range=Range(gte=min_available))
    )

    if not conditions:
        return None

    return Filter(must=conditions)


def reciprocal_rank_fusion(
    dense_results: list,
    bm25_ids: list[str],
    k: int = 60
) -> list[str]:
    """
    Combine dense and BM25 rankings using RRF.
    
    RRF score = sum of 1/(k + rank) across all result lists.
    k=60 is the standard default — dampens the effect of very high ranks.
    
    Interview point: RRF is scale-independent. It only uses rank position,
    not raw scores. So you can safely combine BM25 scores (unbounded)
    with cosine similarity (0 to 1) without normalisation.
    """
    scores = {}

    # dense ranking contribution
    for rank, hit in enumerate(dense_results):
        doc_id = hit.payload["schedule_id"]
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # BM25 ranking contribution
    for rank, doc_id in enumerate(bm25_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)

    # sort by combined RRF score descending
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def retrieve(
    query: str,
    origin: str = None,
    destination: str = None,
    seat_type: str = None,
    max_price: float = None,
    time_of_day: str = None,
    top_k: int = 5
) -> list[dict]:
    """
    Hybrid retrieval combining dense vector search and BM25.
    
    Steps:
    1. Dense search in Qdrant with metadata pre-filter
    2. BM25 search over all document texts
    3. Reciprocal Rank Fusion to merge rankings
    4. Return top_k results with full metadata
    """
    client = QdrantClient(url=QDRANT_URL, check_compatibility=False)

    # step 1 — dense search with metadata filter
    query_vector = get_embedding(query)
    search_filter = build_filter(
        origin=origin,
        destination=destination,
        seat_type=seat_type,
        max_price=max_price
    )

    dense_results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=search_filter,
        limit=top_k * 5,      # was top_k * 3 — fetch more candidates
        with_payload=True
    ).points

    # step 2 — BM25 search over all document texts
    tokenised_query = query.lower().split()
    bm25_scores = BM25_INDEX.get_scores(tokenised_query)

    # get top BM25 results by score, map back to schedule_id
    import numpy as np
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:top_k * 3]
    bm25_ids = [ALL_DOCUMENTS[i]["id"] for i in top_bm25_indices
                if bm25_scores[i] > 0]   # only include docs with non-zero score

    # step 3 — RRF fusion
    fused_ids = reciprocal_rank_fusion(dense_results, bm25_ids)

    # step 4 — build result list from fused order
    # create a lookup from schedule_id to payload
    payload_lookup = {hit.payload["schedule_id"]: hit.payload
                      for hit in dense_results}

    # also add BM25-only results not in dense results
    for doc in ALL_DOCUMENTS:
        if doc["id"] in fused_ids and doc["id"] not in payload_lookup:
            payload_lookup[doc["id"]] = doc["metadata"]

    results = []
    for doc_id in fused_ids[:top_k]:
        if doc_id not in payload_lookup:
            continue
        payload = payload_lookup[doc_id]

        # apply availability filter to BM25-sourced results
        # (dense results already filtered by Qdrant, BM25 has no filter)
        if payload.get("available", 0) < 1:
            continue
        if origin and payload.get("origin") != origin:
            continue
        if destination and payload.get("destination") != destination:
            continue

        results.append({
            "schedule_id": payload.get("schedule_id", payload.get("id", "")),
            "route_id":    payload.get("route_id", ""),
            "origin":      payload.get("origin", ""),
            "destination": payload.get("destination", ""),
            "operator":    payload.get("operator", ""),
            "seat_type":   payload.get("seat_type", ""),
            "departure":   payload.get("departure", ""),
            "arrival":     payload.get("arrival", ""),
            "price":       payload.get("price", 0),
            "available":   payload.get("available", 0),
            "days":        payload.get("days", ""),
            "distance_km": payload.get("distance_km", 0)
        })

    if time_of_day and time_of_day != "all" and results:
        hour_ranges = {
            "morning":   ("05:00", "11:59"),
            "afternoon": ("12:00", "16:59"),
            "evening":   ("17:00", "20:59"),
            "night":     ("21:00", "23:59"),
        }
        if time_of_day.lower() in hour_ranges:
            start, end = hour_ranges[time_of_day.lower()]
            results = [
                r for r in results
                if start <= r["departure"] <= end
            ]


    return results


if __name__ == "__main__":
    print("=== Hybrid retrieval test ===")
    results = retrieve(
        query="evening AC bus Hyderabad Bangalore",
        origin="Hyderabad",
        destination="Bangalore",
        time_of_day="evening"
    )
    for r in results:
        print(f"  {r['operator']} | {r['seat_type']} | "
              f"Rs {r['price']} | {r['departure']} | {r['available']} seats")