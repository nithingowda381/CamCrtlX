import pytest
import os
import json
from unittest.mock import patch, MagicMock, ANY

# Set environment to testing before importing the app
os.environ['FLASK_ENV'] = 'testing'

# We need to patch the config before the app and its modules are imported
from unittest.mock import patch

# Patch the database path to use an in-memory SQLite database for tests
@patch('config.DATABASE_PATH', ':memory:')
def app_import():
    from app import app as flask_app
    from database import DatabaseManager

    # After app import, re-initialize the database for the test environment
    db = DatabaseManager(':memory:')
    db.create_tables()
    
    # Monkeypatch the app's db instance
    flask_app.db = db
    
    return flask_app, db

flask_app, db = app_import()

from auth import User

@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "testing",
        "WTF_CSRF_ENABLED": False,
        "LOGIN_DISABLED": False,
    })

    # Clean and recreate tables for each test
    db.close_connection()
    db.create_tables()

    yield flask_app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture(autouse=True)
def mock_detector():
    """Auto-used fixture to mock the PersonDetector."""
    with patch('app.detector') as mock_detector_instance:
        mock_detector_instance.start_stream.return_value = True
        mock_detector_instance.get_frame_with_detection.return_value = b'fake_frame'
        mock_detector_instance.cap.isOpened.return_value = True
        mock_detector_instance.cap.get.return_value = 30 # for FPS
        yield mock_detector_instance


def login(client, username, password):
    """Helper function to log in a user."""
    return client.post('/login', data=dict(
        username=username,
        password=password
    ), follow_redirects=True)


@pytest.fixture
def authenticated_client(client):
    """A test client that is pre-authenticated."""
    with client:
        # Create a test user in the database
        db.create_user('testuser', 'test@example.com', 'password', 'user')
        user_data = db.get_user_by_username('testuser')
        
        # Use test_request_context to simulate a login
        with flask_app.test_request_context():
            from flask_login import login_user
            user = User(user_data['id'], user_data['username'], user_data['email'], user_data['role'])
            login_user(user)

        # The session is now populated, so the client is "logged in"
        yield client


class TestBasicRoutes:
    def test_health_check(self, client):
        """Test the health check endpoint."""
        response = client.get('/_health')
        assert response.status_code == 200
        assert response.json == {'status': 'ok'}

    def test_index_redirects_to_login(self, client):
        """Test that the root URL redirects to the login page for unauthenticated users."""
        response = client.get('/', follow_redirects=False)
        assert response.status_code == 302
        assert '/login' in response.location

    def test_index_redirects_to_dashboard_for_authenticated_user(self, authenticated_client):
        """Test that the root URL redirects to the dashboard for authenticated users."""
        response = authenticated_client.get('/', follow_redirects=False)
        assert response.status_code == 302
        assert '/dashboard' in response.location

    def test_about_page(self, client):
        """Test the about page."""
        response = client.get('/about')
        assert response.status_code == 200
        assert b"About Us" in response.data


class TestDashboard:
    def test_dashboard_unauthenticated(self, client):
        """Test that the dashboard requires login."""
        response = client.get('/dashboard', follow_redirects=False)
        assert response.status_code == 302
        assert '/login' in response.location

    def test_dashboard_authenticated(self, authenticated_client):
        """Test that the dashboard is accessible to authenticated users."""
        response = authenticated_client.get('/dashboard')
        assert response.status_code == 200
        assert b"Dashboard" in response.data
        assert b"testuser" in response.data


class TestEmployeeRoutes:
    def test_add_employee(self, authenticated_client):
        """Test adding a new employee."""
        employee_data = {
            'emp_id': 'E123',
            'name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '1234567890',
            'position': 'Engineer',
            'department': 'Tech',
            'join_date': '2023-01-01',
            'salary': '80000',
            'address': '123 Main St',
            'emergency_contact_name': 'Jane Doe',
            'emergency_contact_phone': '0987654321'
        }
        response = authenticated_client.post('/add_employee', data=employee_data, follow_redirects=True)
        assert response.status_code == 200
        assert b"Employee added successfully!" in response.data
        assert b"E123" in response.data # Check if new employee is in the list

        # Verify in DB
        employee = db.get_employee_by_employee_id('E123')
        assert employee is not None
        assert employee['first_name'] == 'John'

    def test_employees_page(self, authenticated_client):
        """Test the main employees listing page."""
        # First add an employee to see
        db.create_employee({
            'employee_id': 'E456', 'first_name': 'Jane', 'last_name': 'Smith',
            'email': 'jane@test.com', 'status': 'active'
        })
        response = authenticated_client.get('/employees')
        assert response.status_code == 200
        assert b"Employees" in response.data
        assert b"E456" in response.data
        assert b"Jane Smith" in response.data


class TestAPIRoutes:
    def test_api_status(self, authenticated_client):
        """Test the /api/status endpoint."""
        response = authenticated_client.get('/api/status')
        assert response.status_code == 200
        data = response.json
        assert 'current_status' in data
        assert 'today_summary' in data
        assert 'settings' in data
        assert data['settings']['overlays_enabled'] is True # Default value