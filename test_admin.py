from database import DatabaseManager
from werkzeug.security import check_password_hash
import config
import sqlite3

# Connect directly to database
conn = sqlite3.connect(config.DATABASE_PATH)
cursor = conn.cursor()

# Get admin user
cursor.execute('SELECT username, password_hash FROM users WHERE username = ?', ('admin',))
user = cursor.fetchone()

if user:
    username, password_hash = user
    print(f'Username: {username}')
    print(f'Password hash: {password_hash[:50]}...')
    
    # Test password verification
    test_passwords = ['admin123', 'admin', 'password']
    for pwd in test_passwords:
        result = check_password_hash(password_hash, pwd)
        print(f'Password "{pwd}": {"✓" if result else "✗"}')
else:
    print('Admin user not found')

conn.close()
