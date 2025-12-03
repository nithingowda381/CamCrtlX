import sqlite3
import os
import time
from datetime import datetime
from typing import List, Tuple, Optional

class DatabaseManager:
    def __init__(self, db_path: str):
        # Support shared in-memory DB for tests: map ':memory:' to a shared URI
        if db_path == ':memory:':
            # use a named in-memory DB with shared cache so multiple connections
            # in the same process see the same database
            self.db_path = 'file:memdb1?mode=memory&cache=shared'
            self._use_uri = True
            # Keep a persistent connection open to keep the in-memory DB alive
            self._keepalive_conn = sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        else:
            self.db_path = db_path
            self._use_uri = False

        self.init_database()

    def _connect(self):
        """Return a new sqlite3 connection, using URI mode when appropriate."""
        if getattr(self, '_use_uri', False):
            return sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        return sqlite3.connect(self.db_path, check_same_thread=False)
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Create work_log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS work_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                hours REAL,
                total_seconds INTEGER,
                date TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if employee_id column exists, if not add it
        cursor.execute("PRAGMA table_info(work_log)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'employee_id' not in columns:
            cursor.execute("ALTER TABLE work_log ADD COLUMN employee_id TEXT")
        if 'hours' not in columns:
            cursor.execute("ALTER TABLE work_log ADD COLUMN hours REAL")
        
        # Create daily_summary table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                total_work_seconds INTEGER DEFAULT 0,
                sessions_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if role column exists, if not add it
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [column[1] for column in cursor.fetchall()]
        if 'role' not in user_columns:
            cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'user'")
        
        # Create user_settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                dvr_url TEXT,
                confidence_threshold REAL,
                check_interval INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Create profile_settings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profile_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER UNIQUE NOT NULL,
                first_name TEXT,
                last_name TEXT,
                phone TEXT,
                profile_photo TEXT,
                timezone TEXT DEFAULT 'UTC',
                theme TEXT DEFAULT 'dark',
                email_notifications BOOLEAN DEFAULT 1,
                sound_alerts BOOLEAN DEFAULT 1,
                auto_backup BOOLEAN DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)

        # Create employees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id TEXT UNIQUE NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                phone TEXT,
                designation TEXT,
                department TEXT,
                hire_date TEXT,
                salary REAL,
                status TEXT DEFAULT 'active',
                profile_photo TEXT,
                address TEXT,
                emergency_contact TEXT,
                emergency_phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    # Backwards-compatible alias for test suites / older code
    def create_tables(self):
        """Compatibility wrapper for older code that expects create_tables()."""
        return self.init_database()

    def close_connection(self):
        """Compatibility method used by tests to reset or remove the database file.

        For in-memory databases this is a no-op. For file-backed DBs, the file
        will be removed so subsequent tests/startup get a fresh DB.
        """
        try:
            # For file-based DBs, remove the file so tests get a fresh DB
            if self.db_path and not getattr(self, '_use_uri', False) and os.path.exists(self.db_path):
                os.remove(self.db_path)

            # For shared in-memory DBs, restart the keepalive connection to reset contents
            if getattr(self, '_use_uri', False):
                try:
                    if hasattr(self, '_keepalive_conn') and self._keepalive_conn:
                        self._keepalive_conn.close()
                except Exception:
                    pass
                # Recreate a fresh keepalive connection (new empty in-memory DB)
                self._keepalive_conn = sqlite3.connect(self.db_path, uri=True, check_same_thread=False)
        except Exception:
            # Non-fatal - tests will recreate tables
            pass
        return True
    
    def insert_work_session(self, start_time: datetime) -> int:
        """Insert a new work session"""
        conn = self._connect()
        cursor = conn.cursor()
        
        date_str = start_time.strftime('%Y-%m-%d')
        cursor.execute(
            "INSERT INTO work_log (start_time, date) VALUES (?, ?)",
            (start_time.isoformat(), date_str)
        )
        
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return session_id
    
    def update_work_session(self, session_id: int, end_time: datetime):
        """Update work session with end time and calculate duration"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Get start time
        cursor.execute("SELECT start_time FROM work_log WHERE id = ?", (session_id,))
        result = cursor.fetchone()
        
        if result:
            start_time = datetime.fromisoformat(result[0])
            total_seconds = int((end_time - start_time).total_seconds())
            hours = round(total_seconds / 3600.0, 2)

            # Fetch previous total_seconds if any (to avoid double-counting summaries)
            cursor.execute("SELECT total_seconds, date FROM work_log WHERE id = ?", (session_id,))
            prev = cursor.fetchone()
            prev_seconds = prev[0] if prev and prev[0] else 0
            session_date = prev[1] if prev and prev[1] else start_time.strftime('%Y-%m-%d')

            cursor.execute(
                "UPDATE work_log SET end_time = ?, total_seconds = ?, hours = ? WHERE id = ?",
                (end_time.isoformat(), total_seconds, hours, session_id)
            )

            # Update daily_summary table: adjust by delta compared to previous stored seconds
            try:
                delta_seconds = total_seconds - (prev_seconds or 0)
                if delta_seconds != 0:
                    # Ensure a daily_summary row exists
                    cursor.execute("SELECT total_work_seconds, sessions_count FROM daily_summary WHERE date = ?", (session_date,))
                    ds = cursor.fetchone()
                    if ds:
                        new_total = (ds[0] or 0) + delta_seconds
                        new_sessions = ds[1] or 0
                        # If this session previously had no end_time (prev_seconds==0), increment sessions_count
                        if not prev_seconds:
                            new_sessions += 1
                        cursor.execute("UPDATE daily_summary SET total_work_seconds = ?, sessions_count = ?, last_updated = CURRENT_TIMESTAMP WHERE date = ?", (new_total, new_sessions, session_date))
                    else:
                        sessions_count = 1 if not prev_seconds else 0
                        cursor.execute("INSERT INTO daily_summary (date, total_work_seconds, sessions_count) VALUES (?, ?, ?)", (session_date, delta_seconds, sessions_count))
            except Exception:
                # Non-fatal: don't prevent session update if summary update fails
                pass

            conn.commit()
        
        conn.close()
    
    def get_today_sessions(self) -> List[dict]:
        """Get all work sessions for today"""
        conn = self._connect()
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(
            "SELECT * FROM work_log WHERE date = ? ORDER BY start_time",
            (today,)
        )
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_all_sessions(self) -> List[dict]:
        """Get all work sessions"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM work_log ORDER BY date DESC, start_time DESC LIMIT 1000"
        )
        
        columns = [description[0] for description in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_today_summary(self) -> dict:
        """Get today's work summary"""
        conn = self._connect()
        cursor = conn.cursor()
        today = datetime.now().strftime('%Y-%m-%d')
        # Get total hours and sessions count
        cursor.execute(
            "SELECT COALESCE(SUM(total_seconds), 0) as total_seconds, COUNT(*) as sessions_count FROM work_log WHERE date = ?",
            (today,)
        )
        result = cursor.fetchone()
        total_seconds = result[0] or 0
        sessions_count = result[1] or 0
        # Get unique employees who worked today for average calculation
        cursor.execute(
            "SELECT COUNT(DISTINCT employee_id) as unique_employees FROM work_log WHERE date = ? AND employee_id IS NOT NULL",
            (today,)
        )
        unique_employees = cursor.fetchone()[0] or 0
        conn.close()
        total_hours = total_seconds / 3600
        avg_hours = total_hours / max(unique_employees, 1) if unique_employees > 0 else 0
        return {
            'total_seconds': total_seconds,
            'sessions_count': sessions_count,
            'total_hours': total_hours,
            'avg_hours': avg_hours,
            'unique_employees': unique_employees
        }
    
    def get_last_session(self) -> Optional[dict]:
        """Get the last work session"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM work_log ORDER BY id DESC LIMIT 1"
        )
        
        columns = [description[0] for description in cursor.description]
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return dict(zip(columns, result))
        return None

    # User management methods
    def create_user(self, username: str, email: str, password_hash: str, role: str = 'user') -> int:
        """Create a new user with specified role"""
        # Create table if missing and attempt insert with retry and duplicate handling
        attempts = 0
        while True:
            try:
                conn = self._connect()
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        email TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # Use INSERT OR IGNORE to avoid IntegrityError on duplicates, then SELECT the id
                cursor.execute(
                    "INSERT OR IGNORE INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                    (username, email, password_hash, role)
                )

                # Get the user id whether newly inserted or pre-existing
                cursor.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
                row = cursor.fetchone()
                if row:
                    user_id = row[0]
                else:
                    user_id = None

                conn.commit()
                conn.close()
                if user_id is not None:
                    return user_id
                # Unexpected - retry
                attempts += 1
                if attempts > 3:
                    raise Exception('Failed to create or locate user')
                time.sleep(0.01)

            except sqlite3.OperationalError as oe:
                # Retry on locked DB
                attempts += 1
                try:
                    conn.close()
                except Exception:
                    pass
                if attempts > 5:
                    raise
                time.sleep(0.05)
        
    def create_company_admin(self, username: str, email: str, password_hash: str) -> int:
        """Create a company admin user"""
        return self.create_user(username, email, password_hash, 'company_admin')

    def get_user_by_username(self, username: str) -> Optional[dict]:
        """Get user by username"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, password_hash, role FROM users WHERE username = ?",
            (username,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'email': result[2],
                'password': result[3],  # password_hash field
                'role': result[4] if result[4] else 'user'
            }
        
        return None

    def get_user_by_email(self, email: str) -> Optional[dict]:
        """Get user by email"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, password_hash FROM users WHERE email = ?",
            (email,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'email': result[2],
                'password': result[3]
            }
        return None

    def get_user_by_id(self, user_id: int) -> Optional[dict]:
        """Get user by ID"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, username, email, password_hash, role FROM users WHERE id = ?",
            (user_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'username': result[1],
                'email': result[2],
                'password': result[3],
                'role': result[4] if result[4] else 'user'
            }
        return None

    def update_user_settings(self, user_id: int, settings: dict):
        """Update user settings"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                dvr_url TEXT,
                confidence_threshold REAL,
                check_interval INTEGER,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        cursor.execute(
            "INSERT OR REPLACE INTO user_settings (user_id, dvr_url, confidence_threshold, check_interval) VALUES (?, ?, ?, ?)",
            (user_id, settings.get('dvr_url'), settings.get('confidence_threshold'), settings.get('check_interval'))
        )
        
        conn.commit()
        conn.close()

    def get_user_settings(self, user_id: int) -> Optional[dict]:
        """Get user settings"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT dvr_url, confidence_threshold, check_interval FROM user_settings WHERE user_id = ?",
            (user_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'dvr_url': result[0],
                'confidence_threshold': result[1],
                'check_interval': result[2]
            }
        return None

    def get_sessions_by_date(self, date: str) -> List[dict]:
        """Get all sessions for a specific date"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT * FROM work_log 
            WHERE DATE(start_time) = ? OR DATE(end_time) = ?
            ORDER BY start_time DESC
            """,
            (date, date)
        )
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        conn.close()
        
        sessions = []
        for result in results:
            session = dict(zip(columns, result))
            # Calculate hours if both start and end time exist
            if session['start_time'] and session['end_time']:
                try:
                    start = datetime.fromisoformat(session['start_time'])
                    end = datetime.fromisoformat(session['end_time'])
                    hours = (end - start).total_seconds() / 3600
                    session['hours'] = round(hours, 2)
                except:
                    session['hours'] = 0
            else:
                session['hours'] = 0
            sessions.append(session)
        
        return sessions

    def get_recent_sessions(self, limit: int = 10) -> List[dict]:
        """Get recent work sessions"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT * FROM work_log 
            ORDER BY COALESCE(end_time, start_time) DESC 
            LIMIT ?
            """,
            (limit,)
        )
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        conn.close()
        
        sessions = []
        for result in results:
            session = dict(zip(columns, result))
            sessions.append(session)
        
        return sessions

    # Employee Management Methods
    def get_all_employees(self) -> List[dict]:
        """Get all employees"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM employees 
            ORDER BY first_name, last_name
        """)
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        conn.close()
        
        employees = []
        for result in results:
            employee = dict(zip(columns, result))
            # Add computed fields for template compatibility
            employee['name'] = f"{employee.get('first_name', '')} {employee.get('last_name', '')}".strip()
            employee['emp_id'] = employee.get('employee_id')
            employee['position'] = employee.get('designation')
            employee['photo_path'] = employee.get('profile_photo')
            employees.append(employee)
        
        return employees

    def get_active_employee_count(self) -> int:
        """Get count of active employees"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status = ?", ('active',))
        count = cursor.fetchone()[0]
        
        conn.close()
        return count

    def get_employee_by_id(self, employee_id: int) -> Optional[dict]:
        """Get employee by ID"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
        
        columns = [description[0] for description in cursor.description]
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return dict(zip(columns, result))
        return None

    def get_employee_by_employee_id(self, employee_id: str) -> Optional[dict]:
        """Get employee by employee ID"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM employees WHERE employee_id = ?", (employee_id,))
        
        columns = [description[0] for description in cursor.description]
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return dict(zip(columns, result))
        return None

    def get_employee_by_name(self, name: str) -> Optional[dict]:
        """Get employee by full name (first_name + last_name)"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Split name into parts
        name_parts = name.strip().split()
        if len(name_parts) == 1:
            # Single name - search in both first_name and last_name
            cursor.execute("""
                SELECT * FROM employees 
                WHERE first_name LIKE ? OR last_name LIKE ?
                OR (first_name || ' ' || last_name) LIKE ?
            """, (f"%{name}%", f"%{name}%", f"%{name}%"))
        else:
            # Multiple parts - try to match full name
            first_name = name_parts[0]
            last_name = ' '.join(name_parts[1:])
            cursor.execute("""
                SELECT * FROM employees 
                WHERE (first_name LIKE ? AND last_name LIKE ?)
                OR (first_name || ' ' || last_name) LIKE ?
            """, (f"%{first_name}%", f"%{last_name}%", f"%{name}%"))
        
        columns = [description[0] for description in cursor.description]
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return dict(zip(columns, result))
        return None

    def create_employee(self, employee_data: dict) -> int:
        """Create a new employee"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO employees (
                employee_id, first_name, last_name, email, phone, 
                designation, department, hire_date, salary, status,
                profile_photo, address, emergency_contact, emergency_phone
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            employee_data.get('employee_id'),
            employee_data.get('first_name'),
            employee_data.get('last_name'),
            employee_data.get('email'),
            employee_data.get('phone'),
            employee_data.get('designation'),
            employee_data.get('department'),
            employee_data.get('hire_date'),
            employee_data.get('salary'),
            employee_data.get('status', 'active'),
            employee_data.get('profile_photo'),
            employee_data.get('address'),
            employee_data.get('emergency_contact'),
            employee_data.get('emergency_phone')
        ))
        
        employee_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return employee_id

    def update_employee(self, employee_id: int, employee_data: dict) -> bool:
        """Update employee information"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE employees SET
                employee_id = ?, first_name = ?, last_name = ?, email = ?, phone = ?,
                designation = ?, department = ?, hire_date = ?, salary = ?, status = ?,
                profile_photo = ?, address = ?, emergency_contact = ?, 
                emergency_phone = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            employee_data.get('employee_id'),
            employee_data.get('first_name'),
            employee_data.get('last_name'),
            employee_data.get('email'),
            employee_data.get('phone'),
            employee_data.get('designation'),
            employee_data.get('department'),
            employee_data.get('hire_date'),
            employee_data.get('salary'),
            employee_data.get('status'),
            employee_data.get('profile_photo'),
            employee_data.get('address'),
            employee_data.get('emergency_contact'),
            employee_data.get('emergency_phone'),
            employee_id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success

    def delete_employee(self, employee_id: str) -> bool:
        """Delete an employee by employee_id"""
        conn = self._connect()
        cursor = conn.cursor()
        # Accept either numeric primary key (id) or employee_id code (e.g. 'emp001')
        # If employee_id looks like an integer, delete by primary key
        if isinstance(employee_id, int) or (isinstance(employee_id, str) and employee_id.isdigit()):
            cursor.execute("DELETE FROM employees WHERE id = ?", (int(employee_id),))
        else:
            cursor.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success

    def get_all_users(self) -> List[dict]:
        """Get all users (for backward compatibility)"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM users ORDER BY username")
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        conn.close()
        
        users = []
        for result in results:
            user = dict(zip(columns, result))
            users.append(user)
        
        return users

    def verify_user_password(self, user_id: int, password: str) -> bool:
        """Verify user's current password"""
        import bcrypt
        
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute("SELECT password_hash FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            stored_hash = result[0]
            return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
        
        return False

    def update_user_password(self, user_id: int, new_password: str) -> bool:
        """Update user's password"""
        import bcrypt
        
        # Hash the new password
        password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (password_hash, user_id)
        )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success

    def update_user_profile(self, user_id: int, profile_data: dict) -> bool:
        """Update user profile information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if profile_photo column exists, if not add it
        cursor.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'profile_photo' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN profile_photo TEXT")
        
        if 'full_name' not in columns:
            cursor.execute("ALTER TABLE users ADD COLUMN full_name TEXT")
        
        # Update profile
        update_fields = []
        values = []
        
        if 'username' in profile_data:
            update_fields.append("username = ?")
            values.append(profile_data['username'])
        
        if 'email' in profile_data:
            update_fields.append("email = ?")
            values.append(profile_data['email'])
        
        if 'full_name' in profile_data:
            update_fields.append("full_name = ?")
            values.append(profile_data['full_name'])
        
        if 'profile_photo' in profile_data:
            update_fields.append("profile_photo = ?")
            values.append(profile_data['profile_photo'])
        
        if update_fields:
            values.append(user_id)
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE id = ?"
            cursor.execute(query, values)
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success

    def get_all_sessions(self) -> List[dict]:
        """Get all attendance sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT w.*, 'Unknown' as name, '' as employee_id 
            FROM work_log w
            ORDER BY w.start_time DESC
        """)
        
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        
        conn.close()
        
        sessions = []
        for result in results:
            session = dict(zip(columns, result))
            sessions.append(session)
        
        return sessions

    def cleanup_old_sessions(self, cutoff_date: str) -> int:
        """Delete sessions older than cutoff date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM work_log WHERE start_time < ?", (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count

    def create_or_update_profile_settings(self, user_id: int, profile_data: dict) -> bool:
        """Create or update profile settings for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Check if profile settings exist for this user
            cursor.execute("SELECT id FROM profile_settings WHERE user_id = ?", (user_id,))
            existing = cursor.fetchone()
            
            if existing:
                # Update existing profile settings
                update_fields = []
                values = []
                
                for field in ['first_name', 'last_name', 'phone', 'profile_photo', 'timezone', 
                             'theme', 'email_notifications', 'sound_alerts', 'auto_backup']:
                    if field in profile_data:
                        update_fields.append(f"{field} = ?")
                        values.append(profile_data[field])
                
                if update_fields:
                    update_fields.append("updated_at = CURRENT_TIMESTAMP")
                    values.append(user_id)
                    query = f"UPDATE profile_settings SET {', '.join(update_fields)} WHERE user_id = ?"
                    cursor.execute(query, values)
            else:
                # Create new profile settings
                fields = ['user_id']
                values = [user_id]
                placeholders = ['?']
                
                for field in ['first_name', 'last_name', 'phone', 'profile_photo', 'timezone', 
                             'theme', 'email_notifications', 'sound_alerts', 'auto_backup']:
                    if field in profile_data:
                        fields.append(field)
                        values.append(profile_data[field])
                        placeholders.append('?')
                
                query = f"INSERT INTO profile_settings ({', '.join(fields)}) VALUES ({', '.join(placeholders)})"
                cursor.execute(query, values)
            
            conn.commit()
            return True
            
        except Exception as e:
            print(f"Error updating profile settings: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    def get_profile_settings(self, user_id: int) -> dict:
        """Get profile settings for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ps.*, u.username, u.email 
            FROM profile_settings ps
            JOIN users u ON ps.user_id = u.id
            WHERE ps.user_id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        
        if result:
            columns = [description[0] for description in cursor.description]
            profile_data = dict(zip(columns, result))
        else:
            # Return default profile settings if none exist
            cursor.execute("SELECT username, email FROM users WHERE id = ?", (user_id,))
            user_data = cursor.fetchone()
            
            if user_data:
                profile_data = {
                    'user_id': user_id,
                    'username': user_data[0],
                    'email': user_data[1],
                    'first_name': '',
                    'last_name': '',
                    'phone': '',
                    'profile_photo': '',
                    'timezone': 'UTC',
                    'theme': 'dark',
                    'email_notifications': True,
                    'sound_alerts': True,
                    'auto_backup': True
                }
            else:
                profile_data = {}
        
        conn.close()
        return profile_data

    def delete_profile_settings(self, user_id: int) -> bool:
        """Delete profile settings for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM profile_settings WHERE user_id = ?", (user_id,))
            success = cursor.rowcount > 0
            conn.commit()
            return success
        except Exception as e:
            print(f"Error deleting profile settings: {e}")
            return False
        finally:
            conn.close()

    def get_user_profile_data(self, user_id: int) -> dict:
        """Get combined user and profile data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.created_at,
                   ps.first_name, ps.last_name, ps.phone, ps.profile_photo,
                   ps.timezone, ps.theme, ps.email_notifications, 
                   ps.sound_alerts, ps.auto_backup
            FROM users u
            LEFT JOIN profile_settings ps ON u.id = ps.user_id
            WHERE u.id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        
        if result:
            columns = [description[0] for description in cursor.description]
            user_data = dict(zip(columns, result))
            
            # Convert boolean values
            if user_data.get('email_notifications') is not None:
                user_data['email_notifications'] = bool(user_data['email_notifications'])
            if user_data.get('sound_alerts') is not None:
                user_data['sound_alerts'] = bool(user_data['sound_alerts'])
            if user_data.get('auto_backup') is not None:
                user_data['auto_backup'] = bool(user_data['auto_backup'])
                
            return user_data
        
        conn.close()
        return {}

    def verify_password(self, username: str, password: str) -> bool:
        """Verify user password"""
        import hashlib
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
        result = cursor.fetchone()
        conn.close()
        
        if result:
            stored_hash = result[0]
            # Hash the provided password and compare
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return password_hash == stored_hash
        
        return False

    def update_password(self, user_id: int, new_password: str) -> bool:
        """Update user password"""
        import hashlib
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hash the new password
            password_hash = hashlib.sha256(new_password.encode()).hexdigest()
            
            cursor.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (password_hash, user_id)
            )
            
            success = cursor.rowcount > 0
            conn.commit()
            conn.close()
            
            return success
        except Exception as e:
            print(f"Error updating password: {e}")
            return False

# Standalone functions for testing and app integration
def init_db():
    """Initialize database using the default database path"""
    db = DatabaseManager('attendance.db')
    return True

def create_or_update_profile_settings(user_id, profile_data=None, **kwargs):
    """Create or update profile settings for a user"""
    db = DatabaseManager('attendance.db')
    if profile_data is None:
        profile_data = kwargs
    return db.create_or_update_profile_settings(user_id, profile_data)

def get_profile_settings(user_id):
    """Get profile settings for a user"""
    db = DatabaseManager('attendance.db')
    return db.get_profile_settings(user_id)

def get_user_profile_data(user_id):
    """Get complete user profile data"""
    db = DatabaseManager('attendance.db')
    return db.get_user_profile_data(user_id)

def verify_password(username, password):
    """Verify user password"""
    db = DatabaseManager('attendance.db')
    return db.verify_password(username, password)

def update_password(user_id, new_password):
    """Update user password"""
    db = DatabaseManager('attendance.db')
    return db.update_password(user_id, new_password)
