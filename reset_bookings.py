import sqlite3

conn = sqlite3.connect('data/db/bus_booking.db')

conn.execute('DELETE FROM bookings')
print('Bookings deleted')

conn.execute('UPDATE schedules SET available = total_seats')
print('Seat counts restored')

conn.execute("""
    UPDATE seat_inventory
    SET status = 'available', gender = NULL, booked_by = NULL
    WHERE booked_by = 'Guest User'
""")
print('Seat inventory restored')

conn.commit()
conn.close()
print('Done — all test bookings cleared')