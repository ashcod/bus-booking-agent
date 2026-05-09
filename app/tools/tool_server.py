# app/tools/tool_server.py
# Purpose: MCP-style tool server — exposes tools as HTTP endpoints
# Any agent calls these via HTTP rather than importing functions directly
#
# Interview point: this is the MCP pattern — tool discovery + execution
# via a standard interface. Benefits:
# 1. Tools are independently deployable and scalable
# 2. Any agent (Python, JS, another service) can call them
# 3. You can version tools without touching agent code
# 4. Easy to add auth, rate limiting, logging at the tool layer

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

app = FastAPI(title="Bus Booking MCP Tool Server")
DB_PATH = "data/db/bus_booking.db"


# --- Tool input schemas ---
# Pydantic models define exactly what each tool accepts
# Interview point: explicit schemas = self-documenting API
# Agents can discover available tools by calling GET /tools

class SearchBusesInput(BaseModel):
    origin: str
    destination: str
    seat_type: str | None = None
    max_price: float | None = None
    time_of_day: str | None = None


class BookTicketInput(BaseModel):
    schedule_id: str
    user_name: str
    user_email: str
    seats: int = 1


class CancelBookingInput(BaseModel):
    booking_id: str


class GetPriceInput(BaseModel):
    schedule_id: str
    seats: int = 1


# --- Tool discovery endpoint ---
# Interview point: this is the MCP "list tools" capability
# Agents call this first to know what tools are available
# without needing hardcoded knowledge of the tool server

@app.get("/tools")
def list_tools():
    """Returns all available tools with their schemas."""
    return {
        "tools": [
            {
                "name": "search_buses",
                "description": "Search for available buses between two cities",
                "endpoint": "/tools/search_buses",
                "method": "POST",
                "input_schema": SearchBusesInput.model_json_schema()
            },
            {
                "name": "book_ticket",
                "description": "Book a bus ticket for a specific schedule",
                "endpoint": "/tools/book_ticket",
                "method": "POST",
                "input_schema": BookTicketInput.model_json_schema()
            },
            {
                "name": "cancel_booking",
                "description": "Cancel an existing booking by booking ID",
                "endpoint": "/tools/cancel_booking",
                "method": "POST",
                "input_schema": CancelBookingInput.model_json_schema()
            },
            {
                "name": "get_price",
                "description": "Get current price for a schedule including surge",
                "endpoint": "/tools/get_price",
                "method": "POST",
                "input_schema": GetPriceInput.model_json_schema()
            }
        ]
    }


# --- Tool implementations ---

@app.post("/tools/search_buses")
def search_buses(params: SearchBusesInput):
    """
    Search SQLite directly for matching buses.
    Interview point: this tool bypasses RAG intentionally —
    it's a precise structured query, not a semantic search.
    RAG is for natural language. SQL is for exact filters.
    Both have their place in the same system.
    """
    conn = sqlite3.connect(DB_PATH)

    query = """
        SELECT s.schedule_id, r.origin, r.destination, r.operator,
               s.seat_type, s.departure, s.arrival, s.price, s.available, s.days
        FROM schedules s
        JOIN routes r ON s.route_id = r.route_id
        WHERE r.origin = ? AND r.destination = ? AND s.available > 0
    """
    args = [params.origin, params.destination]

    if params.seat_type:
        query += " AND s.seat_type = ?"
        args.append(params.seat_type)

    if params.max_price:
        query += " AND s.price <= ?"
        args.append(params.max_price)

    rows = conn.execute(query, args).fetchall()
    conn.close()

    results = []
    for row in rows:
        schedule_id, origin, destination, operator, seat_type, \
        departure, arrival, price, available, days = row

        # apply time of day filter in Python
        if params.time_of_day:
            hour_ranges = {
                "morning":   ("05:00", "11:59"),
                "afternoon": ("12:00", "16:59"),
                "evening":   ("17:00", "20:59"),
                "night":     ("21:00", "23:59"),
            }
            if params.time_of_day in hour_ranges:
                start, end = hour_ranges[params.time_of_day]
                if not (start <= departure <= end):
                    continue

        results.append({
            "schedule_id": schedule_id,
            "origin":      origin,
            "destination": destination,
            "operator":    operator,
            "seat_type":   seat_type,
            "departure":   departure,
            "arrival":     arrival,
            "price":       price,
            "available":   available,
            "days":        days
        })

    return {"results": results, "count": len(results)}


class BookTicketInput(BaseModel):
    schedule_id: str
    user_name:   str
    user_email:  str
    gender:      str = "M"
    seat_number: int = None
    seats:       int = 1


@app.post("/tools/book_ticket")
def book_ticket(params: BookTicketInput):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT available, price, departure, arrival, seat_type "
        "FROM schedules WHERE schedule_id = ?",
        (params.schedule_id,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Schedule not found")

    available, price, departure, arrival, seat_type = row

    if available < params.seats:
        conn.close()
        raise HTTPException(
            status_code=400,
            detail=f"Only {available} seats available"
        )

    # book specific seat if provided
    if params.seat_number:
        seat_row = conn.execute(
            """SELECT status FROM seat_inventory
               WHERE schedule_id = ? AND seat_number = ?""",
            (params.schedule_id, params.seat_number)
        ).fetchone()

        if not seat_row:
            conn.close()
            raise HTTPException(status_code=404, detail="Seat not found")

        if seat_row[0] == "booked":
            conn.close()
            raise HTTPException(status_code=400, detail="Seat already booked")

        conn.execute(
            """UPDATE seat_inventory
               SET status='booked', gender=?, booked_by=?
               WHERE schedule_id=? AND seat_number=?""",
            (params.gender, params.user_name,
             params.schedule_id, params.seat_number)
        )

    # deduct from available count
    conn.execute(
        "UPDATE schedules SET available = available - ? WHERE schedule_id = ?",
        (params.seats, params.schedule_id)
    )

    booking_id  = f"BK{uuid.uuid4().hex[:8].upper()}"
    total_price = round(price * params.seats, 2)

    conn.execute(
        "INSERT INTO bookings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (booking_id, params.schedule_id, params.user_name,
         params.user_email, params.gender, params.seats,
         total_price, datetime.now().isoformat(), "confirmed")
    )
    conn.commit()
    conn.close()

    return {
        "success":      True,
        "booking_id":   booking_id,
        "schedule_id":  params.schedule_id,
        "seat_number":  params.seat_number,
        "seat_type":    seat_type,
        "departure":    departure,
        "arrival":      arrival,
        "seats_booked": params.seats,
        "total_price":  total_price,
        "gender":       params.gender
    }


@app.post("/tools/cancel_booking")
def cancel_booking(params: CancelBookingInput):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT schedule_id, seats_booked, status FROM bookings "
        "WHERE booking_id = ?",
        (params.booking_id,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Booking not found")

    schedule_id, seats_booked, status = row

    if status == "cancelled":
        conn.close()
        raise HTTPException(status_code=400, detail="Booking already cancelled")

    # restore seats
    conn.execute(
        "UPDATE schedules SET available = available + ? WHERE schedule_id = ?",
        (seats_booked, schedule_id)
    )
    conn.execute(
        "UPDATE bookings SET status = 'cancelled' WHERE booking_id = ?",
        (params.booking_id,)
    )
    conn.commit()
    conn.close()

    return {
        "success":    True,
        "booking_id": params.booking_id,
        "status":     "cancelled",
        "message":    "Booking cancelled. Refund processed per policy."
    }


@app.post("/tools/get_price")
def get_price(params: GetPriceInput):
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT price, seat_type, available, total_seats FROM schedules "
        "WHERE schedule_id = ?",
        (params.schedule_id,)
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Schedule not found")

    base_price, seat_type, available, total_seats = row
    conn.close()

    # dynamic surge pricing based on remaining availability
    # Interview point: this is a simple demand signal.
    # In production you'd factor in day of week, holiday calendar,
    # competitor prices, and historical booking velocity.
    occupancy = 1 - (available / total_seats)
    if occupancy > 0.8:
        surge = 1.5
        surge_reason = "high demand"
    elif occupancy > 0.6:
        surge = 1.2
        surge_reason = "moderate demand"
    else:
        surge = 1.0
        surge_reason = "normal pricing"

    final_price = round(base_price * surge * params.seats, 2)

    return {
        "schedule_id":  params.schedule_id,
        "seat_type":    seat_type,
        "base_price":   base_price,
        "surge_multiplier": surge,
        "surge_reason": surge_reason,
        "seats":        params.seats,
        "total_price":  final_price
    }

class SeatMapInput(BaseModel):
    schedule_id: str
    deck: str = "lower"


@app.post("/tools/get_seat_map")
def get_seat_map(params: SeatMapInput):
    """
    Returns seat map for a specific schedule and deck.
    Interview point: this is a read-only endpoint — no side effects.
    The booking endpoint handles the actual reservation.
    """
    conn = sqlite3.connect(DB_PATH)

    seats = conn.execute(
        """SELECT seat_number, deck, is_window, status, gender
           FROM seat_inventory
           WHERE schedule_id = ? AND deck = ?
           ORDER BY seat_number""",
        (params.schedule_id, params.deck)
    ).fetchall()

    schedule = conn.execute(
        """SELECT s.total_seats, s.seat_type, s.has_upper,
                  r.origin, r.destination
           FROM schedules s JOIN routes r ON s.route_id = r.route_id
           WHERE s.schedule_id = ?""",
        (params.schedule_id,)
    ).fetchone()

    conn.close()

    if not schedule:
        raise HTTPException(status_code=404, detail="Schedule not found")

    total_seats, seat_type, has_upper, origin, destination = schedule

    return {
        "schedule_id": params.schedule_id,
        "seat_type":   seat_type,
        "has_upper":   bool(has_upper),
        "origin":      origin,
        "destination": destination,
        "deck":        params.deck,
        "seats": [
            {
                "seat_number": row[0],
                "deck":        row[1],
                "is_window":   bool(row[2]),
                "status":      row[3],
                "gender":      row[4]
            }
            for row in seats
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)