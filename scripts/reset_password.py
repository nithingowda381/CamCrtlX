#!/usr/bin/env python3
"""
Reset a user's password in attendance.db to a generated temporary password.
Usage: python scripts/reset_password.py <username>
Prints: RESET_OK <temporary-password> on success, or an error message.
"""
import sqlite3
import hashlib
import secrets
import string
import sys

if len(sys.argv) < 2:
    print("Usage: reset_password.py <username>")
    sys.exit(1)

username = sys.argv[1]
DB_PATH = 'attendance.db'

try:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, username FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if not row:
        print(f"ERROR: user '{username}' not found")
        sys.exit(1)
    user_id = row[0]

    alphabet = string.ascii_letters + string.digits + "@#$%&*?-"
    new_password = ''.join(secrets.choice(alphabet) for _ in range(12))
    password_hash = hashlib.sha256(new_password.encode()).hexdigest()

    cur.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
    conn.commit()
    conn.close()

    print("RESET_OK", new_password)
except Exception as e:
    print("ERROR:", e)
    sys.exit(1)
