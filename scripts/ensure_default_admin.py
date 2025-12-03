#!/usr/bin/env python3
"""
Ensure a default company admin user exists.

Creates a user with username 'admin' and password 'admin1234' and role 'company_admin'
if it does not already exist. Intended for development only. Do NOT keep default
credentials in production.
"""
import os
from werkzeug.security import generate_password_hash
from database import DatabaseManager
import config


def ensure_default_admin(username='admin', password='admin1234', email='admin@example.com'):
    db = DatabaseManager(config.DATABASE_PATH)

    existing = db.get_user_by_username(username)
    if existing:
        print(f"User '{username}' already exists (id={existing.get('id')}). No changes made.")
        return False

    existing_email = db.get_user_by_email(email)
    if existing_email:
        print(f"A user with email '{email}' already exists (id={existing_email.get('id')}). No changes made.")
        return False

    # create password hash using werkzeug to match login verification
    password_hash = generate_password_hash(password)
    user_id = db.create_company_admin(username, email, password_hash)
    print(f"Created company_admin user: id={user_id}, username={username}, password={password}")
    return True


if __name__ == '__main__':
    print("Ensuring default company admin exists...")
    ensure_default_admin()
