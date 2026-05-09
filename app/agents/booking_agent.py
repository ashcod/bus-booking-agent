# app/agents/booking_agent.py
import sqlite3
import uuid
import re
from datetime import datetime
from langchain_core.messages import AIMessage
from app.agents.state import BookingState
from app.memory.user_memory import record_booking

DB_PATH = "data/db/bus_booking.db"


def book_ticket(schedule_id: str, user_name: str, user_email: str,
                seats: int = 1) -> dict:
    """
    Write booking to SQLite and return confirmation dict.
    Interview point: we check availability inside the same connection
    to avoid race conditions. In production this would be a
    database transaction with SELECT FOR UPDATE row locking.
    """
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        "SELECT available, price, departure, arrival, seat_type "
        "FROM schedules WHERE schedule_id = ?",
        (schedule_id,)
    ).fetchone()

    if not row:
        conn.close()
        return {"success": False, "error": "Schedule not found"}

    available, price, departure, arrival, seat_type = row

    if available < seats:
        conn.close()
        return {
            "success": False,
            "error": f"Only {available} seats available, you requested {seats}"
        }

    conn.execute(
        "UPDATE schedules SET available = available - ? WHERE schedule_id = ?",
        (seats, schedule_id)
    )

    booking_id = f"BK{uuid.uuid4().hex[:8].upper()}"
    total_price = round(price * seats, 2)
    booked_at = datetime.now().isoformat()

    conn.execute(
        "INSERT INTO bookings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (booking_id, schedule_id, user_name, user_email,
         seats, total_price, booked_at, "confirmed")
    )
    conn.commit()
    conn.close()

    return {
        "success":      True,
        "booking_id":   booking_id,
        "schedule_id":  schedule_id,
        "seats_booked": seats,
        "total_price":  total_price,
        "departure":    departure,
        "arrival":      arrival,
        "seat_type":    seat_type,
        "booked_at":    booked_at
    }


def booking_agent_node(state: BookingState) -> dict:
    last_message = state["messages"][-1].content
    schedule_id = state.get("selected_schedule_id")

    # try to find schedule ID mentioned in the message (format: SC00000)
    if not schedule_id:
        match = re.search(r'SC\d{5}', last_message.upper())
        if match:
            schedule_id = match.group()

    if not schedule_id:
        return {
            "responding_agent": "booking",
            "messages": [AIMessage(content=(
                "Which bus would you like to book? "
                "Please mention the schedule ID — for example: 'book SC00000'. "
                "You can find the schedule ID in the search results."
            ))]
        }

    confirmation = book_ticket(
        schedule_id=schedule_id,
        user_name="Guest User",
        user_email="guest@example.com",
        seats=1
    )

    if not confirmation["success"]:
        return {
            "responding_agent": "booking",
            "messages": [AIMessage(content=(
                f"Sorry, booking failed: {confirmation['error']}. "
                f"Please search again for available options."
            ))]
        }

    # record booking in long-term memory
    record_booking(
        user_id="default_user",
        booking_id=confirmation["booking_id"],
        seat_type=confirmation["seat_type"]
    )

    response = (
        f"Booking confirmed!\n\n"
        f"  Booking ID:  {confirmation['booking_id']}\n"
        f"  Schedule:    {confirmation['schedule_id']}\n"
        f"  Seat type:   {confirmation['seat_type']}\n"
        f"  Departure:   {confirmation['departure']}\n"
        f"  Arrival:     {confirmation['arrival']}\n"
        f"  Seats:       {confirmation['seats_booked']}\n"
        f"  Total price: Rs {confirmation['total_price']}\n\n"
        f"Please save your Booking ID: {confirmation['booking_id']}"
    )

    return {
        "booking_confirmation": confirmation,
        "selected_schedule_id": schedule_id,
        "responding_agent":     "booking",
        "messages":             [AIMessage(content=response)]
    }