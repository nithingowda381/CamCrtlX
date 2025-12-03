from database import DatabaseManager
from werkzeug.security import check_password_hash
import config

# Test the full authentication flow
db = DatabaseManager(config.DATABASE_PATH)

# Test admin user
user_data = db.get_user_by_username('admin')
print(f"User data for 'admin': {user_data}")

if user_data:
    print(f"Role: {user_data.get('role')}")
    password_check = check_password_hash(user_data['password'], 'admin123')
    print(f"Password 'admin123' check: {password_check}")
    
    # Test role checking
    role = user_data.get('role', 'user')
    is_company_admin = role == 'company_admin'
    print(f"Is company admin: {is_company_admin}")
else:
    print("Admin user not found!")
