#!/usr/bin/env python3

import sqlite3
import datetime
import os
import sys

# Add current directory to path to import our modules
sys.path.insert(0, '.')

from database import DatabaseManager

def test_dashboard_data():
    """Test dashboard data queries directly"""
    print("Testing Dashboard Data Queries...")
    print("=" * 50)
    
    # Initialize database
    db = DatabaseManager('attendance.db')
    
    try:
        # Test 1: Today's sessions
        print("\n1. Today's Sessions:")
        today_sessions = db.get_today_sessions()
        print(f"   Found {len(today_sessions)} sessions today")
        for session in today_sessions[:3]:  # Show first 3
            print(f"   Session: ID={session.get('id')}, Employee={session.get('employee_id')}, Hours={session.get('hours')}")
        
        # Test 2: Today's summary
        print("\n2. Today's Summary:")
        today_summary = db.get_today_summary()
        print(f"   Total hours: {today_summary.get('total_hours', 0):.2f}")
        print(f"   Average hours: {today_summary.get('avg_hours', 0):.2f}")
        print(f"   Sessions count: {today_summary.get('sessions_count', 0)}")
        print(f"   Unique employees: {today_summary.get('unique_employees', 0)}")
        
        # Test 3: Active employee count
        print("\n3. Active Employee Count:")
        active_count = db.get_active_employee_count()
        print(f"   Active employees: {active_count}")
        
        # Test 4: Recent sessions
        print("\n4. Recent Sessions:")
        recent_sessions = db.get_recent_sessions(limit=5)
        print(f"   Found {len(recent_sessions)} recent sessions")
        for session in recent_sessions:
            print(f"   Session: Employee={session.get('employee_id')}, Date={session.get('date')}, Hours={session.get('hours')}")
        
        # Test 5: Weekly data
        print("\n5. Weekly Data:")
        weekly_hours = []
        for i in range(7):
            date = datetime.datetime.now() - datetime.timedelta(days=6-i)
            day_sessions = db.get_sessions_by_date(date.strftime('%Y-%m-%d'))
            total_hours = sum(float(session.get('hours', 0)) for session in day_sessions if session.get('hours'))
            weekly_hours.append(round(total_hours, 1))
            print(f"   {date.strftime('%Y-%m-%d')}: {total_hours:.1f} hours")
        
        print(f"\n   Weekly hours: {weekly_hours}")
        
        # Test 6: Employee lookup test
        print("\n6. Employee Lookup Test:")
        all_employees = db.get_all_employees()
        print(f"   Total employees in database: {len(all_employees)}")
        for emp in all_employees[:3]:  # Show first 3
            print(f"   Employee: ID={emp.get('id')}, EmpID={emp.get('employee_id')}, Name={emp.get('name')}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Dashboard Data Testing Complete")

if __name__ == "__main__":
    test_dashboard_data()
