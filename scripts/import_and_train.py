#!/usr/bin/env python3
"""
Script: import_and_train.py

Usage:
  python scripts/import_and_train.py /path/to/photo1.jpg /path/to/photo2.jpg ...

This script copies provided images into `static/face_images/` and names them using
the employee identifier inferred from the filename (before the first dot). If an
employee with that employee_id doesn't exist in the `employees` table, a minimal
employee record will be created. After copying images, the script imports the
app's PersonDetector and calls `load_employee_faces()` to retrain the recognizer.

This is a convenience helper intended for local development. Do not run on
production systems without verifying file ownership and permissions.
"""
import sys
import os
import shutil
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_FACE_DIR = REPO_ROOT / 'static' / 'face_images'
DB_PATH = REPO_ROOT / 'attendance.db'

def ensure_dirs():
    STATIC_FACE_DIR.mkdir(parents=True, exist_ok=True)

def create_employee_if_missing(employee_code, first_name='Unknown', last_name=''):
    """Create an employee row if missing. Handles UNIQUE email conflicts and DB locks.

    Returns the DB primary id for the employee.
    """
    attempts = 0
    while attempts < 6:
        conn = None
        try:
            # Increase timeout to reduce "database is locked" errors
            conn = sqlite3.connect(str(DB_PATH), timeout=30)
            cursor = conn.cursor()

            # Fast path: if employee_id exists, return it
            cursor.execute("SELECT id FROM employees WHERE employee_id = ?", (employee_code,))
            row = cursor.fetchone()
            if row:
                return row[0]

            # Try to insert; if email is already taken generate a unique email suffix
            base_email = f"{employee_code.lower()}@example.local"
            email = base_email
            suffix = 0
            while True:
                try:
                    cursor.execute(
                        "INSERT INTO employees (employee_id, first_name, last_name, email, status) VALUES (?, ?, ?, ?, 'active')",
                        (employee_code, first_name, last_name, email)
                    )
                    conn.commit()
                    return cursor.lastrowid
                except sqlite3.IntegrityError as ie:
                    msg = str(ie).lower()
                    # If email uniqueness caused the problem, try a different email
                    if 'email' in msg or 'unique' in msg:
                        suffix += 1
                        email = f"{employee_code.lower()}{suffix}@example.local"
                        continue
                    else:
                        # Re-raise other integrity errors
                        raise

        except sqlite3.OperationalError as oe:
            # Retry on locked DB
            if 'locked' in str(oe).lower():
                attempts += 1
                import time
                time.sleep(0.5 * attempts)
                continue
            else:
                raise
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    raise Exception('Failed to create/find employee after multiple retries (database may be locked)')

def copy_and_name(src_path, employee_code, index):
    ext = Path(src_path).suffix.lower() or '.jpg'
    dst_name = f"{employee_code}_face_{index}{ext}"
    dst_path = STATIC_FACE_DIR / dst_name
    shutil.copy2(src_path, dst_path)
    print(f"Copied {src_path} -> {dst_path}")
    return dst_path

def main(args):
    if len(args) < 1:
        print("Usage: import_and_train.py <image1> <image2> ...")
        return 1

    ensure_dirs()

    # Map employee_code -> list of files
    files_by_emp = {}
    for p in args:
        pth = Path(p)
        if not pth.exists():
            print(f"File not found: {p}")
            continue
        # Infer employee code and name from filename
        name = pth.stem
        parts = name.split('_')
        if name.startswith('training_') and len(parts) > 1:
            emp_code = parts[1]
        elif len(parts) > 1:
            emp_code = parts[0]
        else:
            emp_code = name.split('.')[0]

        emp_code = emp_code.strip() or 'unknown'
        first_name = emp_code  # Use emp_code as first_name
        files_by_emp.setdefault((emp_code, first_name), []).append(str(pth))

    # Insert missing employees and copy files
    total = 0
    for (emp_code, first_name), files in files_by_emp.items():
        emp_db_id = create_employee_if_missing(emp_code, first_name)
        for i, src in enumerate(files):
            copy_and_name(src, emp_code, i+1)
            total += 1

    print(f"Imported {total} images into {STATIC_FACE_DIR}")

    # Trigger model reload via the app's PersonDetector
    try:
        # Import minimal app components without starting the web server
        sys.path.insert(0, str(REPO_ROOT))
        from person_detector import PersonDetector
        detector = PersonDetector()
        detector.load_employee_faces()
        print("Triggered detector.load_employee_faces()")
    except Exception as e:
        print(f"Failed to trigger retrain: {e}")

    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
