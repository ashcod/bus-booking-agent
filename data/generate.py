# data/generate.py
# Purpose: generate realistic Indian bus route data and save to SQLite
# Why synthetic data: lets us control edge cases (full buses, night routes,
# surge pricing) that real data might not have enough of for testing

import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

# --- Constants you can explain in interviews ---

CITIES = [
    "Hyderabad", "Bangalore", "Chennai", "Mumbai", "Pune",
    "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
    "Nagpur", "Visakhapatnam", "Kochi", "Coimbatore", "Madurai"
]

OPERATORS = [
    "TSRTC", "KSRTC", "MSRTC", "Orange Travels",
    "VRL Travels", "SRS Travels", "Parveen Travels", "IntrCity"
]

# seat types mirror real redbus categories
SEAT_TYPES = ["Seater", "Sleeper", "AC Seater", "AC Sleeper"]

# base prices per km by seat type - surge multiplier applied later
PRICE_PER_KM = {
    "Seater": 0.8,
    "Sleeper": 1.2,
    "AC Seater": 1.5,
    "AC Sleeper": 2.0
}

# approximate distances in km between city pairs
# why hardcode: avoids geocoding API dependency, fully offline
CITY_DISTANCES = {
    ("Hyderabad", "Bangalore"): 570,
    ("Hyderabad", "Chennai"): 630,
    ("Hyderabad", "Mumbai"): 710,
    ("Hyderabad", "Pune"): 560,
    ("Bangalore", "Chennai"): 350,
    ("Bangalore", "Mumbai"): 980,
    ("Bangalore", "Coimbatore"): 360,
    ("Chennai", "Madurai"): 460,
    ("Chennai", "Kochi"): 680,
    ("Mumbai", "Pune"): 150,
    ("Mumbai", "Ahmedabad"): 530,
    ("Delhi", "Jaipur"): 280,
    ("Delhi", "Lucknow"): 550,
    ("Delhi", "Ahmedabad"): 940,
    ("Kolkata", "Visakhapatnam"): 800,
}


def get_distance(origin: str, destination: str) -> int:
    """Look up distance, try reverse pair too, else estimate."""
    key = (origin, destination)
    reverse = (destination, origin)
    if key in CITY_DISTANCES:
        return CITY_DISTANCES[key]
    if reverse in CITY_DISTANCES:
        return CITY_DISTANCES[reverse]
    # fallback: random realistic distance so no route fails
    return random.randint(200, 900)


def calculate_price(distance: int, seat_type: str) -> float:
    """
    Base price = distance * rate per km.
    Surge: weekends cost 20% more (simulates dynamic pricing).
    Interview point: in production this would query a pricing agent
    with demand signals, time of day, remaining seats.
    """
    base = distance * PRICE_PER_KM[seat_type]
    surge = random.choice([1.0, 1.0, 1.0, 1.2, 1.5])  # 40% chance of surge
    return round(base * surge, 2)


def calculate_arrival(departure: str, distance: int) -> str:
    """Estimate arrival assuming avg 60 km/h including stops."""
    dep = datetime.strptime(departure, "%H:%M")
    hours = distance / 60
    arrival = dep + timedelta(hours=hours)
    return arrival.strftime("%H:%M")


# --- Database setup ---

def create_tables(conn: sqlite3.Connection):
    """
    Three tables only. Kept minimal on purpose.
    Interview point: normalization - routes are separate from schedules
    so the same Hyderabad->Bangalore route can have multiple daily buses.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS routes (
            route_id    TEXT PRIMARY KEY,
            origin      TEXT NOT NULL,
            destination TEXT NOT NULL,
            distance_km INTEGER NOT NULL,
            operator    TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schedules (
            schedule_id  TEXT PRIMARY KEY,
            route_id     TEXT NOT NULL,
            departure    TEXT NOT NULL,
            arrival      TEXT NOT NULL,
            seat_type    TEXT NOT NULL,
            total_seats  INTEGER NOT NULL,
            available    INTEGER NOT NULL,
            price        REAL NOT NULL,
            days         TEXT NOT NULL,
            has_upper    INTEGER DEFAULT 0,
            FOREIGN KEY (route_id) REFERENCES routes(route_id)
        );

        CREATE TABLE IF NOT EXISTS bookings (
            booking_id   TEXT PRIMARY KEY,
            schedule_id  TEXT NOT NULL,
            user_name    TEXT NOT NULL,
            user_email   TEXT NOT NULL,
            gender       TEXT NOT NULL DEFAULT 'M',
            seats_booked INTEGER NOT NULL,
            total_price  REAL NOT NULL,
            booked_at    TEXT NOT NULL,
            status       TEXT DEFAULT 'confirmed',
            FOREIGN KEY (schedule_id) REFERENCES schedules(schedule_id)
        );
                       
        CREATE TABLE IF NOT EXISTS seat_inventory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            schedule_id TEXT NOT NULL,
            seat_number INTEGER NOT NULL,
            deck        TEXT NOT NULL DEFAULT 'lower',
            is_window   INTEGER DEFAULT 0,
            status      TEXT DEFAULT 'available',
            booked_by   TEXT,
            gender      TEXT,
            FOREIGN KEY (schedule_id) REFERENCES schedules(schedule_id)
        );
        
    """)
    conn.commit()


def generate_data(conn: sqlite3.Connection):
    """Generate routes and schedules. No bookings yet - agents create those."""
    
    route_count = 0
    schedule_count = 0
    departures = ["06:00", "07:30", "09:00", "11:00", "13:30",
                  "16:00", "18:30", "20:00", "21:30", "23:00"]
    days_options = ["Mon-Sun", "Mon-Fri", "Fri-Sun", "Mon-Sat"]

    for i, origin in enumerate(CITIES):
        for destination in CITIES[i+1:]:          # avoid duplicates & self
            operator = random.choice(OPERATORS)
            distance = get_distance(origin, destination)
            route_id = f"RT{route_count:04d}"

            conn.execute(
                "INSERT INTO routes VALUES (?, ?, ?, ?, ?)",
                (route_id, origin, destination, distance, operator)
            )

            # 2-4 schedules per route, different seat types and timings
            num_schedules = random.randint(2, 4)
            used_times = []

            for _ in range(num_schedules):
                # pick a departure not already used on this route
                dep = random.choice(departures)
                while dep in used_times and len(used_times) < len(departures):
                    dep = random.choice(departures)
                used_times.append(dep)

                seat_type = random.choice(SEAT_TYPES)
                total = random.choice([36, 40, 42, 54])
                available = random.randint(0, total)   # some buses nearly full
                price = calculate_price(distance, seat_type)
                arrival = calculate_arrival(dep, distance)
                days = random.choice(days_options)
                schedule_id = f"SC{schedule_count:05d}"

                has_upper = 1 if "Sleeper" in seat_type else 0

                conn.execute(
                    "INSERT INTO schedules VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (schedule_id, route_id, dep, arrival,
                     seat_type, total, available, price, days, has_upper)
                )
                schedule_count += 1

            route_count += 1

    conn.commit()
    print(f"Generated {route_count} routes and {schedule_count} schedules.")

def generate_seat_inventory(conn: sqlite3.Connection):
    """
    Generate individual seat records for every schedule.
    Interview point: normalising seats into their own table
    enables per-seat operations — booking specific seats,
    showing gender of occupant, upper/lower deck selection.
    Without this table you can only track total available count.
    """
    import random

    schedules = conn.execute(
        "SELECT schedule_id, total_seats, seat_type, has_upper FROM schedules"
    ).fetchall()

    for schedule_id, total_seats, seat_type, has_upper in schedules:
        half = total_seats // 2

        for seat_num in range(1, total_seats + 1):
            # determine deck
            if has_upper:
                deck = "lower" if seat_num <= half else "upper"
            else:
                deck = "lower"

            # window seats: first and last column in each row
            # in a 2+2 layout: seats 1,2 per row — seat 1 and 4 are windows
            position_in_row = ((seat_num - 1) % 4)
            is_window = 1 if position_in_row in [0, 3] else 0

            # randomly pre-book some seats with gender
            rand = random.random()
            if rand < 0.3:
                status = "booked"
                gender = random.choice(["M", "F"])
                booked_by = f"user_{random.randint(1000,9999)}"
            else:
                status = "available"
                gender = None
                booked_by = None

            conn.execute(
                """INSERT INTO seat_inventory
                   (schedule_id, seat_number, deck, is_window, status, booked_by, gender)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (schedule_id, seat_num, deck, is_window,
                 status, booked_by, gender)
            )

    conn.commit()
    print("Generated seat inventory for all schedules.")


if __name__ == "__main__":
    db_path = Path("data/db/bus_booking.db")

    # delete existing DB and regenerate fresh
    if db_path.exists():
        db_path.unlink()
        print("Deleted existing database.")

    conn = sqlite3.connect(db_path)
    create_tables(conn)
    generate_data(conn)
    generate_seat_inventory(conn)
    conn.close()
    print(f"Database saved to {db_path}")