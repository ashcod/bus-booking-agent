# data/build_documents.py
# Purpose: convert SQLite rows into text chunks for embedding
# Interview point: the quality of your documents directly determines
# retrieval quality. Garbage in = garbage out, no matter how good your model.

import sqlite3
import json
from pathlib import Path


def row_to_document(route: tuple, schedule: tuple) -> dict:
    """
    Convert a route+schedule pair into a natural language document.
    
    Why natural language instead of raw JSON?
    Embedding models are trained on natural language text.
    A sentence like 'AC Sleeper bus from Hyderabad to Bangalore'
    embeds much better than '{"seat_type": "AC Sleeper"}' because
    the model understands the words, not the JSON structure.
    """
    route_id, origin, destination, distance, operator = route
    (schedule_id, _, departure, arrival,
     seat_type, total, available, price, days) = schedule

    # human-readable availability signal
    if available == 0:
        availability = "fully booked"
    elif available <= 5:
        availability = f"almost full — only {available} seats left"
    else:
        availability = f"{available} seats available"

    # the document text — written so a user's question overlaps naturally
    text = (
        f"{operator} runs a {seat_type} bus from {origin} to {destination}. "
        f"Departure at {departure}, arrives around {arrival}. "
        f"Distance is {distance} km. "
        f"Price is Rs {price:.0f} per seat. "
        f"Operates on {days}. "
        f"Seat availability: {availability}. "
        f"Total capacity: {total} seats."
    )

    return {
        "id": schedule_id,
        "text": text,
        # metadata stored alongside vector — used for filtering
        # Interview point: metadata filters let you do pre-filtering
        # before vector search, which is faster and more precise
        "metadata": {
            "schedule_id": schedule_id,
            "route_id": route_id,
            "origin": origin,
            "destination": destination,
            "operator": operator,
            "seat_type": seat_type,
            "departure": departure,
            "arrival": arrival,
            "price": price,
            "available": available,
            "days": days,
            "distance_km": distance
        }
    }


def build_documents() -> list[dict]:
    conn = sqlite3.connect("data/db/bus_booking.db")

    query = """
        SELECT r.route_id, r.origin, r.destination, r.distance_km, r.operator,
               s.schedule_id, s.route_id, s.departure, s.arrival,
               s.seat_type, s.total_seats, s.available, s.price, s.days
        FROM routes r
        JOIN schedules s ON r.route_id = s.route_id
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    documents = []
    for row in rows:
        route = row[:5]
        schedule = row[5:]
        documents.append(row_to_document(route, schedule))

    return documents


if __name__ == "__main__":
    docs = build_documents()
    
    # save as JSON so you can inspect before embedding
    out = Path("data/raw/documents.json")
    out.write_text(json.dumps(docs, indent=2))
    
    print(f"Built {len(docs)} documents")
    print("\nSample document text:")
    print(docs[0]["text"])
    print("\nSample metadata:")
    print(json.dumps(docs[0]["metadata"], indent=2))
