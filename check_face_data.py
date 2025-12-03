#!/usr/bin/env python3

import os
import sqlite3

def check_face_recognition_data():
    """Check face recognition data without starting video"""
    print("Checking Face Recognition Setup...")
    print("=" * 50)
    
    # Check if face images exist
    face_images_dir = 'static/face_images'
    if os.path.exists(face_images_dir):
        face_files = [f for f in os.listdir(face_images_dir) if f.endswith('.jpg')]
        print(f"Found {len(face_files)} face image files:")
        for f in face_files:
            print(f"  - {f}")
    else:
        print("Face images directory not found!")
        return
    
    # Check employee database
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, employee_id, first_name, last_name FROM employees WHERE status = 'active'")
        employees = cursor.fetchall()
        conn.close()
        
        print(f"\nFound {len(employees)} active employees in database:")
        for emp in employees:
            print(f"  - DB_ID: {emp[0]}, Employee_ID: {emp[1]}, Name: {emp[2]} {emp[3]}")
            
            # Check if this employee has face images
            emp_face_files = [f for f in face_files if f.startswith(f"{emp[1]}_face_")]
            print(f"    Face files: {len(emp_face_files)} - {emp_face_files}")
            
    except Exception as e:
        print(f"Error accessing employee database: {e}")
        return

if __name__ == "__main__":
    check_face_recognition_data()
