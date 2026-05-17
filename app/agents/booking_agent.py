# app/agents/booking_agent.py
import sqlite3
import uuid
import re
from datetime import datetime
from langchain_core.messages import AIMessage
from app.agents.state import BookingState
from app.memory.user_memory import record_booking
from app.tools.mcp_client import call_tool

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
        "INSERT INTO bookings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (booking_id, params.schedule_id, params.user_name,
         params.user_email, params.gender, params.seats,
         total_price, datetime.now().isoformat(),
         None,   # travel_date — set by UI via tool_server
         "confirmed")
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


from app.tools.mcp_client import call_tool

def booking_agent_node(state: BookingState) -> dict:
    last_message = state["messages"][-1].content
    schedule_id = state.get("selected_schedule_id")

    if not schedule_id:
        import re
        match = re.search(r'SC\d{5}', last_message.upper())
        if match:
            schedule_id = match.group()

    if not schedule_id:
        return {
            "responding_agent": "booking",
            "messages": [AIMessage(content=(
                "Which bus would you like to book? "
                "Please mention the schedule ID — for example: 'book SC00000'."
            ))]
        }

    # call MCP tool server for booking
    from datetime import date
    result = call_tool("book_ticket", {
        "schedule_id": schedule_id,
        "user_name":   "Guest User",
        "user_email":  "guest@example.com",
        "gender":      "M",
        "seats":       1,
        "travel_date": date.today().isoformat()
    })

    if "error" in result:
        return {
            "responding_agent": "booking",
            "messages": [AIMessage(content=(
                f"Booking failed: {result.get('detail', result.get('error', 'Please try again.'))}. "
            ))]
        }

    from app.memory.user_memory import record_booking
    record_booking(
        user_id="default_user",
        booking_id=result["booking_id"],
        seat_type=result["seat_type"]
    )

    response = (
        f"Booking confirmed!\n\n"
        f"  Booking ID:  {result['booking_id']}\n"
        f"  Schedule:    {schedule_id}\n"
        f"  Seat type:   {result['seat_type']}\n"
        f"  Departure:   {result['departure']}\n"
        f"  Arrival:     {result['arrival']}\n"
        f"  Seats:       {result['seats_booked']}\n"
        f"  Total price: Rs {result['total_price']}\n\n"
        f"Please save your Booking ID: {result['booking_id']}"
    )

    return {
        "booking_confirmation": result,
        "selected_schedule_id": schedule_id,
        "responding_agent":     "booking",
        "messages":             [AIMessage(content=response)]
    }
