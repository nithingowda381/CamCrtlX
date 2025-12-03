#!/usr/bin/env python3
"""
Database and File Cleanup Script for CamCrtlX
Removes all stored data, files, and optionally resets the database
"""

import os
import sys
import sqlite3
import shutil
from datetime import datetime
import config

def get_database_info():
    """Get current database information"""
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        info = {}
        
        # Get table counts
        tables = ['users', 'employees', 'work_log', 'daily_summary', 'user_settings', 'profile_settings']
        for table in tables:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                info[table] = cursor.fetchone()[0]
            except sqlite3.OperationalError:
                info[table] = 0
        
        conn.close()
        return info
    except Exception as e:
        print(f"Error getting database info: {e}")
        return {}

def get_file_info():
    """Get current file system information"""
    file_info = {}
    
    directories = [
        'static/face_images',
        'static/profile_photos', 
        'static/employee_photos',
        'static/recordings',
        'backups',
        'reports'
    ]
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                file_info[dir_path] = len(files)
            except Exception as e:
                file_info[dir_path] = f"Error: {e}"
        else:
            file_info[dir_path] = 0
    
    return file_info

def delete_static_files():
    """Delete all static files"""
    print("\n=== Deleting Static Files ===")
    
    directories = [
        'static/face_images',
        'static/profile_photos', 
        'static/employee_photos',
        'static/recordings'
    ]
    
    deleted_count = 0
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                for file in files:
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"  Deleted: {file_path}")
                print(f"✅ Cleaned directory: {dir_path} ({len(files)} files)")
            except Exception as e:
                print(f"❌ Error cleaning {dir_path}: {e}")
        else:
            print(f"⚠️  Directory not found: {dir_path}")
    
    print(f"\nTotal files deleted: {deleted_count}")

def delete_backups_and_reports():
    """Delete backup and report files"""
    print("\n=== Deleting Backups and Reports ===")
    
    directories = ['backups', 'reports']
    deleted_count = 0
    
    for dir_path in directories:
        if os.path.exists(dir_path):
            try:
                files = os.listdir(dir_path)
                for file in files:
                    file_path = os.path.join(dir_path, file)
                    if os.path.isfile(file_path) and not file.endswith('.pdf'):  # Keep PDF documentation
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"  Deleted: {file_path}")
                print(f"✅ Cleaned directory: {dir_path}")
            except Exception as e:
                print(f"❌ Error cleaning {dir_path}: {e}")
    
    print(f"Backup/Report files deleted: {deleted_count}")

def clear_database_data():
    """Clear all data from database tables"""
    print("\n=== Clearing Database Data ===")
    
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Tables to clear (in order to avoid foreign key issues)
        tables_to_clear = [
            'work_log',
            'daily_summary', 
            'user_settings',
            'profile_settings',
            'employees'
        ]
        
        for table in tables_to_clear:
            try:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count_before = cursor.fetchone()[0]
                
                cursor.execute(f'DELETE FROM {table}')
                
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count_after = cursor.fetchone()[0]
                
                deleted = count_before - count_after
                print(f"  {table}: Deleted {deleted} records")
                
            except sqlite3.OperationalError as e:
                print(f"  {table}: Table not found or error - {e}")
        
        conn.commit()
        conn.close()
        print("✅ Database data cleared successfully")
        
    except Exception as e:
        print(f"❌ Error clearing database: {e}")

def delete_users(keep_admin=True):
    """Delete users from database"""
    print(f"\n=== {'Deleting Non-Admin Users' if keep_admin else 'Deleting All Users'} ===")
    
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        if keep_admin:
            # Keep company admin users
            cursor.execute('SELECT COUNT(*) FROM users WHERE role != "company_admin"')
            count_to_delete = cursor.fetchone()[0]
            
            cursor.execute('DELETE FROM users WHERE role != "company_admin"')
            print(f"  Deleted {count_to_delete} regular users")
            print("  Kept company admin users")
        else:
            cursor.execute('SELECT COUNT(*) FROM users')
            count_to_delete = cursor.fetchone()[0]
            
            cursor.execute('DELETE FROM users')
            print(f"  Deleted {count_to_delete} users")
        
        conn.commit()
        conn.close()
        print("✅ User deletion completed")
        
    except Exception as e:
        print(f"❌ Error deleting users: {e}")

def reset_database_completely():
    """Completely reset the database"""
    print("\n=== Completely Resetting Database ===")
    
    try:
        # Delete the database file
        if os.path.exists(config.DATABASE_PATH):
            os.remove(config.DATABASE_PATH)
            print("✅ Database file deleted")
        
        # Delete face recognition database if it exists
        if os.path.exists('face_recognition.db'):
            os.remove('face_recognition.db')
            print("✅ Face recognition database deleted")
        
        # Recreate database with fresh structure
        from database import DatabaseManager
        db = DatabaseManager(config.DATABASE_PATH)
        print("✅ Fresh database created")
        
    except Exception as e:
        print(f"❌ Error resetting database: {e}")

def main():
    """Main cleanup function"""
    print("=== CamCrtlX Database and File Cleanup Tool ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show current state
    print("\n=== Current State ===")
    db_info = get_database_info()
    file_info = get_file_info()
    
    print("Database tables:")
    for table, count in db_info.items():
        print(f"  {table}: {count} records")
    
    print("\nFile directories:")
    for dir_path, count in file_info.items():
        print(f"  {dir_path}: {count} files")
    
    # Interactive menu
    print("\n=== Cleanup Options ===")
    print("1. Delete only static files (face images, photos)")
    print("2. Clear database data (keep users and structure)")
    print("3. Delete non-admin users only")
    print("4. Delete all files + clear data (keep admin users)")
    print("5. Complete reset (delete everything including database)")
    print("6. Show current state only (no deletion)")
    print("0. Exit")
    
    try:
        if len(sys.argv) > 1:
            choice = sys.argv[1]
        else:
            choice = input("\nEnter your choice (0-6): ").strip()
        
        if choice == '1':
            delete_static_files()
        elif choice == '2':
            clear_database_data()
        elif choice == '3':
            delete_users(keep_admin=True)
        elif choice == '4':
            delete_static_files()
            delete_backups_and_reports()
            clear_database_data()
            print("\n✅ Files and data cleared (admin users preserved)")
        elif choice == '5':
            confirm = input("⚠️  This will delete EVERYTHING. Type 'DELETE ALL' to confirm: ")
            if confirm == 'DELETE ALL':
                delete_static_files()
                delete_backups_and_reports()
                reset_database_completely()
                print("\n✅ Complete system reset completed")
            else:
                print("❌ Reset cancelled")
        elif choice == '6':
            print("\n✅ Current state displayed above")
        elif choice == '0':
            print("👋 Exiting without changes")
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Cleanup cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during cleanup: {e}")
    
    # Show final state if changes were made
    if choice in ['1', '2', '3', '4', '5']:
        print("\n=== Final State ===")
        final_db_info = get_database_info()
        final_file_info = get_file_info()
        
        print("Database tables:")
        for table, count in final_db_info.items():
            print(f"  {table}: {count} records")
        
        print("\nFile directories:")
        for dir_path, count in final_file_info.items():
            print(f"  {dir_path}: {count} files")

if __name__ == "__main__":
    main()
