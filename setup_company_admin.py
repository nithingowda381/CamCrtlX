#!/usr/bin/env python3
"""
Company Admin Setup Script
Creates a company admin user for the CamCrtlX system
"""

import os
import sys
from werkzeug.security import generate_password_hash
from database import DatabaseManager
import config

def create_company_admin():
    """Create a company admin user"""
    print("=== CamCrtlX Company Admin Setup ===\n")
    
    # Initialize database
    db = DatabaseManager(config.DATABASE_PATH)
    
    # Get admin credentials
    print("Enter details for the company administrator account:")
    username = input("Admin Username: ").strip()
    if not username:
        print("Error: Username cannot be empty")
        return False
    
    email = input("Admin Email: ").strip()
    if not email:
        print("Error: Email cannot be empty")
        return False
    
    password = input("Admin Password: ").strip()
    if not password:
        print("Error: Password cannot be empty")
        return False
    
    # Confirm password
    confirm_password = input("Confirm Password: ").strip()
    if password != confirm_password:
        print("Error: Passwords do not match")
        return False
    
    # Check if user already exists
    existing_user = db.get_user_by_username(username)
    if existing_user:
        print(f"Error: Username '{username}' already exists")
        return False
    
    existing_email = db.get_user_by_email(email)
    if existing_email:
        print(f"Error: Email '{email}' is already registered")
        return False
    
    # Create password hash
    password_hash = generate_password_hash(password)
    
    try:
        # Create company admin user
        user_id = db.create_company_admin(username, email, password_hash)
        
        print(f"\n✅ Company admin user created successfully!")
        print(f"User ID: {user_id}")
        print(f"Username: {username}")
        print(f"Email: {email}")
        print(f"Role: company_admin")
        print(f"\nYou can now login at: http://localhost:5000/company_login")
        
        return True
        
    except Exception as e:
        print(f"Error creating admin user: {e}")
        return False

def list_existing_users():
    """List all existing users"""
    print("\n=== Existing Users ===")
    db = DatabaseManager(config.DATABASE_PATH)
    
    try:
        # Get all users
        conn = db._get_connection() if hasattr(db, '_get_connection') else None
        if not conn:
            import sqlite3
            conn = sqlite3.connect(config.DATABASE_PATH)
        
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, email, role FROM users ORDER BY created_at")
        users = cursor.fetchall()
        
        if users:
            print(f"{'ID':<5} {'Username':<20} {'Email':<30} {'Role':<15}")
            print("-" * 70)
            for user in users:
                user_id, username, email, role = user
                role = role or 'user'
                print(f"{user_id:<5} {username:<20} {email:<30} {role:<15}")
        else:
            print("No users found in the database.")
        
        conn.close()
        
    except Exception as e:
        print(f"Error listing users: {e}")

def main():
    """Main function"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--list":
            list_existing_users()
            return
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python setup_company_admin.py          - Create company admin")
            print("  python setup_company_admin.py --list   - List existing users")
            print("  python setup_company_admin.py --help   - Show this help")
            return
    
    # Create company admin
    success = create_company_admin()
    
    if success:
        print("\n=== Next Steps ===")
        print("1. Start the application: python app.py")
        print("2. Go to: http://localhost:5000/company_login")
        print("3. Login with your admin credentials")
        print("4. Access ML models and advanced features")
        
        # Ask if user wants to see existing users
        show_users = input("\nWould you like to see all existing users? (y/N): ").strip().lower()
        if show_users in ['y', 'yes']:
            list_existing_users()
    else:
        print("\nFailed to create company admin user.")
        sys.exit(1)

if __name__ == "__main__":
    main()
