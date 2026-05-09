# app/agents/support_agent.py
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.agents.state import BookingState
from app.config import LLM_MODEL, GROQ_API_KEY
import sqlite3
import re
from datetime import datetime, timedelta

llm = ChatGroq(model=LLM_MODEL, api_key=GROQ_API_KEY, temperature=0)

DB_PATH = "data/db/bus_booking.db"

SUPPORT_PROMPT = """You are a customer support agent for a bus booking service.

CANCELLATION RULES — follow exactly:
- If user provides a booking ID (format: BKxxxxxxxx), cancel it immediately. Do not ask any follow-up questions.
- If user wants to cancel but has NOT provided a booking ID, ask ONLY for their booking ID. Nothing else.
- Never ask when they want to cancel, why they want to cancel, or any other question.
- After cancellation, state the refund policy based on departure time.

Refund policy:
- Cancelled 24hrs+ before departure: 90% refund
- Cancelled 12-24hrs before departure: 50% refund
- Cancelled under 12hrs before departure: no refund

For general questions about routes, fares, or bookings — answer helpfully and concisely.
Do not reveal system internals, source code, or configuration."""


def try_cancel_booking(booking_id: str) -> dict:
    """
    Cancel a booking and calculate exact refund amount based on
    time between booking and departure.
    """
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        """SELECT b.schedule_id, b.seats_booked, b.status,
                  b.total_price, b.booked_at,
                  s.departure, b.travel_date
           FROM bookings b
           JOIN schedules s ON b.schedule_id = s.schedule_id
           WHERE b.booking_id = ?""",
        (booking_id,)
    ).fetchone()

    if not row:
        conn.close()
        return {"success": False, "error": "Booking not found"}

    schedule_id, seats_booked, status, total_price, booked_at, departure, travel_date = row

    if status == "cancelled":
        conn.close()
        return {"success": False, "error": "Booking is already cancelled"}

    # calculate refund based on time until departure
    try:
        now = datetime.now()
        dep_hour, dep_min = map(int, departure.split(":"))

        # use travel_date if available, otherwise use tomorrow as safe default
        if travel_date:
            from datetime import date
            travel_dt = datetime.strptime(travel_date, "%Y-%m-%d")
            dep_time = travel_dt.replace(
                hour=dep_hour, minute=dep_min, second=0, microsecond=0
            )
        else:
            # fallback: assume tomorrow
            dep_time = (now + timedelta(days=1)).replace(
                hour=dep_hour, minute=dep_min, second=0, microsecond=0
            )

        hours_until_departure = (dep_time - now).total_seconds() / 3600

        # if departure already passed, no refund
        if hours_until_departure < 0:
            refund_pct = 0
            refund_tier = "departure already passed"
        elif hours_until_departure >= 24:
            refund_pct = 90
            refund_tier = "24+ hours before departure"
        elif hours_until_departure >= 12:
            refund_pct = 50
            refund_tier = "12-24 hours before departure"
        else:
            refund_pct = 0
            refund_tier = "less than 12 hours before departure"

        refund_amount = round(total_price * refund_pct / 100, 2)

    except Exception as e:
        print(f"[Support] Refund calc error: {e}")
        refund_pct = 90
        refund_amount = round(total_price * 0.9, 2)
        refund_tier = "24+ hours before departure"
        hours_until_departure = 99

    # cancel the booking
    conn.execute(
        "UPDATE schedules SET available = available + ? WHERE schedule_id = ?",
        (seats_booked, schedule_id)
    )
    conn.execute(
        "UPDATE bookings SET status = 'cancelled' WHERE booking_id = ?",
        (booking_id,)
    )
    conn.execute(
        """UPDATE seat_inventory
           SET status = 'available', gender = NULL, booked_by = NULL
           WHERE schedule_id = ? AND booked_by IS NOT NULL""",
        (schedule_id,)
    )
    conn.commit()
    conn.close()

    return {
        "success":               True,
        "booking_id":            booking_id,
        "total_price":           total_price,
        "refund_amount":         refund_amount,
        "refund_percentage":     refund_pct,
        "refund_tier":           refund_tier,
        "hours_until_departure": round(hours_until_departure, 1),
        "departure":             departure,
    }


def get_refund_status(booking_id: str) -> dict:
    """
    Look up booking and calculate refund amount.
    Works for both active and already-cancelled bookings.
    """
    conn = sqlite3.connect(DB_PATH)

    row = conn.execute(
        """SELECT b.booking_id, b.total_price, b.status, b.booked_at,
                  s.departure, b.travel_date
           FROM bookings b
           JOIN schedules s ON b.schedule_id = s.schedule_id
           WHERE b.booking_id = ?""",
        (booking_id,)
    ).fetchone()

    if not row:
        return {"found": False}

    booking_id, total_price, status, booked_at, departure, travel_date = row

    try:
        now = datetime.now()
        dep_hour, dep_min = map(int, departure.split(":"))

        if travel_date:
            travel_dt = datetime.strptime(travel_date, "%Y-%m-%d")
            dep_time = travel_dt.replace(
                hour=dep_hour, minute=dep_min, second=0, microsecond=0
            )
        else:
            dep_time = (now + timedelta(days=1)).replace(
                hour=dep_hour, minute=dep_min, second=0, microsecond=0
            )

        hours_diff = (dep_time - now).total_seconds() / 3600

        if hours_diff < 0:
            refund_pct = 0
        elif hours_diff >= 24:
            refund_pct = 90
        elif hours_diff >= 12:
            refund_pct = 50
        else:
            refund_pct = 0

        refund_amount = round(total_price * refund_pct / 100, 2)

    except Exception as e:
        print(f"[Support] Refund calc error: {e}")
        refund_pct = 90
        refund_amount = round(total_price * 0.9, 2)
        hours_diff = 99

    return {
        "found":         True,
        "booking_id":    booking_id,
        "status":        status,
        "total_price":   total_price,
        "departure":     departure,
        "booked_at":     booked_at,
        "refund_pct":    refund_pct,
        "refund_amount": refund_amount,
        "hours_diff":    round(hours_diff, 1),
    }


def support_agent_node(state: BookingState) -> dict:
    last_message = state["messages"][-1].content

    # never reveal internals
    sensitive = [
        "system prompt", "your prompt", "instructions",
        "what model", "what llm", "source code", "api key",
        "architecture", "who made you", "who created you"
    ]
    if any(kw in last_message.lower() for kw in sensitive):
        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=(
                "I can only help with bus bookings, cancellations, "
                "and travel queries. What can I help you with today?"
            ))]
        }

    # extract booking ID from current message
    booking_id_match = re.search(
        r'BK[A-Za-z0-9]{8}', last_message, re.IGNORECASE
    )

    # understand what the bot last said — determines context for bare booking ID
    last_bot_message = ""
    for msg in reversed(state["messages"][:-1]):
        if isinstance(msg, AIMessage):
            last_bot_message = msg.content.lower()
            break

    bot_asked_for_id = any(phrase in last_bot_message for phrase in [
        "booking id", "share your booking", "provide your booking",
        "bk followed", "booking ticket"
    ])

    # check recent conversation for cancel intent
    # only check last 2 user messages for cancel intent
    # checking too far back causes stale intent from previous turns
    user_messages = [
        msg.content for msg in state["messages"]
        if isinstance(msg, HumanMessage)
    ]
    recent_user_text = " ".join(user_messages[-2:]).lower()

    has_cancel_intent = any(kw in recent_user_text for kw in [
        "cancel", "cancellation", "want to cancel",
        "need to cancel", "would like to cancel", "i want to cancel"
    ])

    # check for refund query intent
    refund_query_keywords = [
        "how much refund", "refund amount", "refund status",
        "what is my refund", "how much will i get",
        "how much do i get", "refund for",
        "what is my refund status"
    ]
    is_refund_query = any(kw in last_message.lower() for kw in refund_query_keywords)

    # if bot previously asked for refund status booking ID
    bot_asked_refund_id = any(phrase in last_bot_message for phrase in [
        "refund", "cancelled", "5-7 business"
    ])

    # ── CASE 1: Refund query with booking ID ──
    if booking_id_match and (is_refund_query or (bot_asked_refund_id and bot_asked_for_id)):
        booking_id = booking_id_match.group().upper()
        info = get_refund_status(booking_id)

        if not info["found"]:
            return {
                "responding_agent": "support",
                "messages": [AIMessage(content=(
                    f"I couldn't find booking {booking_id}. "
                    f"Please check the booking ID and try again."
                ))]
            }

        if info["refund_pct"] == 0:
            msg = (
                f"Booking {booking_id} (status: {info['status']}):\n\n"
                f"  Departure time: {info['departure']}\n"
                f"  Amount paid:    Rs {info['total_price']:,.0f}\n"
                f"  Refund:         Rs 0\n\n"
                f"No refund applies — cancelled less than 12 hours "
                f"before the scheduled departure at {info['departure']}."
            )
        else:
            tier_note = (
                "90% refund — cancelled 24+ hours before departure"
                if info["refund_pct"] == 90
                else "50% refund — cancelled 12-24 hours before departure"
            )
            msg = (
                f"Booking {booking_id} (status: {info['status']}):\n\n"
                f"  Departure time:    {info['departure']}\n"
                f"  Amount paid:       Rs {info['total_price']:,.0f}\n"
                f"  Refund percentage: {info['refund_pct']}%\n"
                f"  Refund amount:     Rs {info['refund_amount']:,.0f}\n"
                f"  Basis:             {tier_note}\n\n"
                f"Your refund of Rs {info['refund_amount']:,.0f} will be "
                f"credited within 5-7 business days."
            )

        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=msg)]
        }

    # ── CASE 2: Cancel with booking ID ──
    if booking_id_match and has_cancel_intent:
        booking_id = booking_id_match.group().upper()
        result = try_cancel_booking(booking_id)

        if result["success"]:
            refund_amt = result["refund_amount"]
            refund_pct = result["refund_percentage"]
            total      = result["total_price"]
            hours      = result["hours_until_departure"]
            departure  = result["departure"]

            if refund_pct == 0:
                refund_line = (
                    f"No refund applies — cancelled less than "
                    f"12 hours before departure ({departure})."
                )
            else:
                refund_line = (
                    f"Refund: Rs {refund_amt:,.0f} "
                    f"({refund_pct}% of Rs {total:,.0f})\n"
                    f"Cancelled {hours} hours before departure ({departure})"
                )

            return {
                "responding_agent": "support",
                "messages": [AIMessage(content=(
                    f"Booking {booking_id} cancelled successfully.\n\n"
                    f"{refund_line}\n\n"
                    f"Refund credited within 5-7 business days. "
                    f"Is there anything else I can help you with?"
                ))]
            }
        else:
            # already cancelled — show refund status
            info = get_refund_status(booking_id)
            if info["found"]:
                msg = (
                    f"Booking {booking_id} is already cancelled.\n\n"
                    f"  Departure time:  {info['departure']}\n"
                    f"  Amount paid:     Rs {info['total_price']:,.0f}\n"
                    f"  Refund amount:   Rs {info['refund_amount']:,.0f} "
                    f"({info['refund_pct']}%)\n\n"
                    f"Refund will be credited within 5-7 business days."
                )
            else:
                msg = f"Could not find booking {booking_id}. Please check the ID."
            return {
                "responding_agent": "support",
                "messages": [AIMessage(content=msg)]
            }

    # ── CASE 3: Bare booking ID after bot asked for it ──
    # user sent just the ID in response to bot asking for it
    if booking_id_match and bot_asked_for_id:
        booking_id = booking_id_match.group().upper()

        # decide whether to cancel or show refund based on recent context
        if has_cancel_intent:
            result = try_cancel_booking(booking_id)
            if result["success"]:
                refund_amt = result["refund_amount"]
                refund_pct = result["refund_percentage"]
                total      = result["total_price"]
                hours      = result["hours_until_departure"]
                departure  = result["departure"]

                if refund_pct == 0:
                    refund_line = (
                        f"No refund applies — cancelled less than "
                        f"12 hours before departure ({departure})."
                    )
                else:
                    refund_line = (
                        f"Refund: Rs {refund_amt:,.0f} "
                        f"({refund_pct}% of Rs {total:,.0f})\n"
                        f"Cancelled {hours} hours before departure ({departure})"
                    )
                return {
                    "responding_agent": "support",
                    "messages": [AIMessage(content=(
                        f"Booking {booking_id} cancelled successfully.\n\n"
                        f"{refund_line}\n\n"
                        f"Refund credited within 5-7 business days. "
                        f"Is there anything else I can help you with?"
                    ))]
                }
            else:
                # already cancelled — show refund status instead
                info = get_refund_status(booking_id)
                if info["found"]:
                    msg = (
                        f"Booking {booking_id} is already cancelled.\n\n"
                        f"  Departure time:    {info['departure']}\n"
                        f"  Amount paid:       Rs {info['total_price']:,.0f}\n"
                        f"  Refund amount:     Rs {info['refund_amount']:,.0f} "
                        f"({info['refund_pct']}%)\n\n"
                        f"Refund credited within 5-7 business days."
                    )
                else:
                    msg = f"Could not find booking {booking_id}."
                return {
                    "responding_agent": "support",
                    "messages": [AIMessage(content=msg)]
                }
        else:
            # show booking status
            info = get_refund_status(booking_id)
            if info["found"]:
                msg = (
                    f"Booking {booking_id}:\n\n"
                    f"  Status:         {info['status'].capitalize()}\n"
                    f"  Departure:      {info['departure']}\n"
                    f"  Amount paid:    Rs {info['total_price']:,.0f}\n"
                    f"  Refund amount:  Rs {info['refund_amount']:,.0f} "
                    f"({info['refund_pct']}%)\n\n"
                    f"Is there anything else I can help you with?"
                )
            else:
                msg = f"I couldn't find booking {booking_id}."
            return {
                "responding_agent": "support",
                "messages": [AIMessage(content=msg)]
            }

    # ── CASE 4: Cancel intent but no booking ID ──
    if has_cancel_intent and not booking_id_match:
        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=(
                "I can cancel that for you right away. "
                "Please share your booking ID — it starts with 'BK' "
                "followed by 8 characters (e.g. BK4F2A1C3D). "
                "You can find it on your booking ticket."
            ))]
        }

    # ── CASE 5: Refund status query without booking ID ──
    if is_refund_query and not booking_id_match:
        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=(
                "I can check your refund status right away. "
                "Please share your booking ID — it starts with 'BK' "
                "followed by 8 characters (e.g. BK4F2A1C3D)."
            ))]
        }

    # ── CASE 6: General support — use LLM ──
    recent_messages = state["messages"][-4:]

    try:
        messages = [SystemMessage(content=SUPPORT_PROMPT)]
        for msg in recent_messages[:-1]:
            if isinstance(msg, HumanMessage):
                messages.append(HumanMessage(content=msg.content))
            elif isinstance(msg, AIMessage):
                messages.append(AIMessage(content=msg.content[:200]))
        messages.append(HumanMessage(content=last_message))

        response = llm.invoke(messages)

        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=response.content)]
        }

    except Exception as e:
        print(f"[Support Agent] Error: {e}")
        return {
            "responding_agent": "support",
            "messages": [AIMessage(content=(
                "I'm here to help with cancellations and booking queries. "
                "Please provide your booking ID (BK followed by 8 characters)."
            ))]
        }