#!/usr/bin/env python3
import sqlite3
from datetime import datetime

def test_attendance_logging():
    """Test manual attendance logging"""
    
    # Connect to database
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Test data
    employee_id = 'EMPN6126'  # This is the employee in the database
    current_time = datetime.now()
    today = current_time.strftime('%Y-%m-%d')
    
    print(f"Testing attendance logging for employee: {employee_id}")
    print(f"Date: {today}")
    print(f"Time: {current_time}")
    
    # Check if employee exists
    cursor.execute("SELECT employee_id, first_name, last_name FROM employees WHERE employee_id = ?", (employee_id,))
    employee_data = cursor.fetchone()
    
    if not employee_data:
        print(f"❌ Employee {employee_id} not found!")
        conn.close()
        return
    
    print(f"✅ Employee found: {employee_data[1]} {employee_data[2]}")
    
    # Check for existing session today
    cursor.execute("""
        SELECT id, start_time, end_time FROM work_log 
        WHERE employee_id = ? AND date = ? AND end_time IS NULL
        ORDER BY start_time DESC LIMIT 1
    """, (employee_id, today))
    
    active_session = cursor.fetchone()
    print(f"Active session check: {active_session}")
    
    if active_session:
        print("Employee already has an active session today")
        # Simulate check-out
        session_id, start_time, end_time = active_session
        start_datetime = datetime.fromisoformat(start_time)
        work_duration = (current_time - start_datetime).total_seconds() / 3600
        
        if work_duration >= 0.1:  # Reduced to 0.1 hours for testing
            cursor.execute("""
                UPDATE work_log SET end_time = ?, hours = ? WHERE id = ?
            """, (current_time.isoformat(), round(work_duration, 2), session_id))
            print(f"✅ Check-out logged! Duration: {work_duration:.2f} hours")
        else:
            print(f"Work duration too short: {work_duration:.2f} hours")
    else:
        # Simulate check-in
        cursor.execute("""
            INSERT INTO work_log (employee_id, start_time, date) 
            VALUES (?, ?, ?)
        """, (employee_id, current_time.isoformat(), today))
        print("✅ Check-in logged!")
    
    conn.commit()
    
    # Verify the entry
    cursor.execute("SELECT * FROM work_log WHERE employee_id = ? ORDER BY start_time DESC LIMIT 1", (employee_id,))
    latest_entry = cursor.fetchone()
    print(f"Latest work_log entry: {latest_entry}")
    
    conn.close()
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    test_attendance_logging()
