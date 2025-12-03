#!/usr/bin/env python3
import sqlite3

def check_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    print("=== WORK_LOG TABLE STRUCTURE ===")
    cursor.execute('PRAGMA table_info(work_log)')
    work_log_columns = cursor.fetchall()
    for i, col in enumerate(work_log_columns):
        print(f"{i}: {col}")
    
    print("\n=== EMPLOYEES TABLE STRUCTURE ===")
    cursor.execute('PRAGMA table_info(employees)')
    employee_columns = cursor.fetchall()
    for i, col in enumerate(employee_columns):
        print(f"{i}: {col}")
    
    print("\n=== WORK_LOG DATA ===")
    cursor.execute('SELECT * FROM work_log LIMIT 5')
    work_log_data = cursor.fetchall()
    for row in work_log_data:
        print(row)
    
    print(f"\nTotal work_log entries: {len(work_log_data)}")
    
    print("\n=== EMPLOYEES DATA ===")
    cursor.execute('SELECT employee_id, first_name, last_name FROM employees LIMIT 5')
    employee_data = cursor.fetchall()
    for row in employee_data:
        print(row)
    
    conn.close()

if __name__ == "__main__":
    check_database()
