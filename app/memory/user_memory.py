# app/memory/user_memory.py
# Purpose: persist user preferences across sessions
#
# Interview point: we use a JSON file here for simplicity.
# In production this would be Redis (already running in your Docker)
# with a TTL of 30 days. The interface is identical — swap the
# read/write calls without touching agent code.
#
# What we store:
# - preferred_origin: city they usually travel from
# - preferred_seat_type: AC Sleeper, Seater etc.
# - recent_routes: last 5 routes searched
# - booking_count: total bookings made
# - last_booking_id: most recent booking reference

import json
from pathlib import Path
from datetime import datetime

MEMORY_FILE = Path("data/db/user_memory.json")


def _load_all() -> dict:
    """Load the full memory store. Returns empty dict if file missing."""
    if not MEMORY_FILE.exists():
        return {}
    return json.loads(MEMORY_FILE.read_text())


def _save_all(data: dict):
    """Write full memory store to disk."""
    MEMORY_FILE.write_text(json.dumps(data, indent=2))


def get_user_memory(user_id: str) -> dict:
    """
    Retrieve memory for a specific user.
    Returns default profile if user is new.
    """
    all_memory = _load_all()
    return all_memory.get(user_id, {
        "user_id":            user_id,
        "preferred_origin":   None,
        "preferred_seat_type":None,
        "recent_routes":      [],
        "booking_count":      0,
        "last_booking_id":    None,
        "first_seen":         datetime.now().isoformat(),
        "last_seen":          datetime.now().isoformat(),
    })


def update_user_memory(user_id: str, updates: dict):
    """
    Merge updates into existing user memory.
    Interview point: we merge, not overwrite — preserves fields
    not included in the update dict.
    """
    all_memory = _load_all()
    current = get_user_memory(user_id)
    current.update(updates)
    current["last_seen"] = datetime.now().isoformat()
    all_memory[user_id] = current
    _save_all(all_memory)


def record_search(user_id: str, origin: str, destination: str):
    """
    Called after every search. Updates preferred origin and recent routes.
    Interview point: we infer preferences from behaviour, not explicit
    settings. If a user searches Hyderabad→Bangalore 5 times,
    we set that as their preferred route automatically.
    """
    memory = get_user_memory(user_id)

    # update preferred origin if this origin appears frequently
    recent = memory["recent_routes"]
    recent.append({
        "origin": origin,
        "destination": destination,
        "searched_at": datetime.now().isoformat()
    })
    # keep only last 10 searches
    memory["recent_routes"] = recent[-10:]

    # infer preferred origin from most common departure city
    origins = [r["origin"] for r in memory["recent_routes"] if r.get("origin")]
    if origins:
        memory["preferred_origin"] = max(set(origins), key=origins.count)

    update_user_memory(user_id, memory)


def record_booking(user_id: str, booking_id: str, seat_type: str):
    """Called after every successful booking."""
    memory = get_user_memory(user_id)
    update_user_memory(user_id, {
        "last_booking_id":    booking_id,
        "booking_count":      memory["booking_count"] + 1,
        "preferred_seat_type":seat_type,
    })


def get_memory_context(user_id: str) -> str:
    """
    Format user memory as a string for injection into agent prompts.
    Interview point: this is context injection — we prepend user
    preferences to every agent prompt so the LLM personalises responses.
    A returning user gets 'Welcome back! Travelling from Hyderabad again?'
    instead of starting from scratch every time.
    """
    memory = get_user_memory(user_id)

    parts = []
    if memory.get("preferred_origin"):
        parts.append(f"Preferred departure city: {memory['preferred_origin']}")
    if memory.get("preferred_seat_type"):
        parts.append(f"Usually books: {memory['preferred_seat_type']}")
    if memory.get("booking_count", 0) > 0:
        parts.append(f"Total bookings: {memory['booking_count']}")
    if memory.get("last_booking_id"):
        parts.append(f"Last booking: {memory['last_booking_id']}")
    if memory.get("recent_routes"):
        last = memory["recent_routes"][-1]
        parts.append(
            f"Last searched: {last.get('origin')} to {last.get('destination')}"
        )

    if not parts:
        return "New user — no preferences recorded yet."

    return "\n".join(parts)