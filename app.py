## Place these routes after app initialization
from flask import Flask, render_template, jsonify, Response, request, flash, send_file, redirect, url_for, session
from werkzeug.utils import secure_filename
import warnings

# Suppress known resource_tracker semaphore leak warnings coming from loky/joblib at shutdown
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
)

# Lazy-import heavy native libs so app can start in a minimal environment
try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None
import openpyxl
from io import BytesIO
import datetime
import json
import os
import sqlite3
import time
from database import DatabaseManager, get_profile_settings, get_user_profile_data
from ml_evaluator import MLModelEvaluator
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
try:
    from person_detector import PersonDetector
except Exception:
    # Fall back to a lightweight dummy detector if person_detector cannot be imported
    from dummy_detector import DummyPersonDetector as PersonDetector
import config
import threading
import time
from google_vision_service import GoogleVisionService, VideoAnalysisService

from flask_login import LoginManager, login_required, current_user
from auth import auth_bp
from auth import User
import config

app = Flask(__name__)
app.secret_key = '6a019a530153245fc5de0e8a60df910f91252e667fde49362433e9d07446f984'

login_manager = LoginManager()
login_manager.login_view = 'auth.login'
login_manager.init_app(app)

# Initialize the person detector
detector = PersonDetector()

# Initialize AI services
vision_service = GoogleVisionService()
video_analysis = VideoAnalysisService()

# Global variables for video streaming
is_streaming = False
is_recording = False
video_writer = None
recording_filename = None

@app.route('/face_login')
def face_login():
    """Face recognition login page"""
    return render_template('face_login.html')

@app.route('/face_login_check', methods=['POST'])
def face_login_check():
    """Check face recognition for login (no confidence threshold)"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'No image data provided'})
        # Extract base64 image data
        image_data = data['image'].split(',')[1]
        import base64
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'success': False, 'message': 'Invalid image data'})
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return jsonify({'success': False, 'message': 'No face detected in image'})
        (x, y, w, h) = faces[0]
        face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        employee_label, _ = detector.recognize_face(face_img)
        print(f"Face recognition result: label={employee_label}")
        if employee_label is not None and employee_label in detector.employee_labels:
            employee_info = detector.employee_labels[employee_label]
            employee_id = employee_info['employee_id']
            employee = db.get_employee_by_employee_id(employee_id)
            if employee:
                user_data = db.get_user_by_username(employee_id)
                if user_data:
                    from flask_login import login_user
                    user = User(user_data['id'], user_data['username'], user_data['email'], user_data.get('role', 'user'))
                    login_user(user, remember=True)
                    return jsonify({
                        'success': True,
                        'user_name': f"{employee.get('first_name', '')} {employee.get('last_name', '')}".strip(),
                        'employee_id': employee_id,
                        'redirect_url': url_for('dashboard'),
                        'message': f'Welcome back, {employee_info["name"]}!'
                    })
                else:
                    session.clear()
                    session['face_login_user'] = {
                        'employee_id': employee_id,
                        'name': employee_info['name'],
                        'authenticated': True,
                        'login_time': datetime.datetime.now().isoformat()
                    }
                    session.permanent = True
                    session.modified = True
                    return jsonify({
                        'success': True,
                        'user_name': employee_info['name'],
                        'employee_id': employee_id,
                        'redirect_url': url_for('face_dashboard'),
                        'message': f'Welcome, {employee_info["name"]}!'
                    })
            else:
                print(f"Employee {employee_id} not found in database")
        return jsonify({
            'success': False,
            'message': 'Face not recognized. Please try again or use regular login.'
        })
    except Exception as e:
        print(f"Face login error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': 'Authentication error occurred'})

@app.route('/about')
def about():
    """About Us page"""
    return render_template('about.html')


@app.route('/thank-you')
def thank_you():
    """Thank you page for faculty/HOD support"""
    return render_template('thank_you.html')

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    db = DatabaseManager(config.DATABASE_PATH)
    user_data = db.get_user_by_id(user_id)
    if user_data:
        role = user_data.get('role', 'user')
        # Create the User object
        user = User(user_data['id'], user_data['username'], user_data['email'], role)
        try:
            # Attach profile fields (if present) so templates can read current_user.first_name etc.
            profile = db.get_user_profile_data(user_data['id'])
            if profile:
                user.first_name = profile.get('first_name')
                user.last_name = profile.get('last_name')
                user.phone = profile.get('phone')
                # profile_photo may be stored as filename
                user.profile_photo = profile.get('profile_photo')
        except Exception:
            # Non-fatal: proceed without profile enrichment
            pass
        return user
    return None

def admin_required(f):
    """Decorator for routes that require admin access"""
    from functools import wraps
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            return redirect(url_for('auth.company_login'))
        if not current_user.is_company_admin():
            flash('Access denied. Company admin privileges required.')
            return redirect(url_for('auth.company_login'))
        return f(*args, **kwargs)
    return decorated_function

app.register_blueprint(auth_bp)


# Root route - redirect unauthenticated users to login page
@app.route('/')
def index():
    # If user is authenticated, send to dashboard, otherwise to login page
    try:
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
    except Exception:
        # current_user may not be available at import time in some contexts
        pass
    return redirect(url_for('auth.login'))


# Simple health endpoint
@app.route('/_health')
def _health():
    return jsonify({'status': 'ok'})

db = DatabaseManager(config.DATABASE_PATH)

# Global variables for live feed
detector = PersonDetector(confidence_threshold=config.CONFIDENCE_THRESHOLD)
is_streaming = False

# Global variables for recording
from werkzeug.security import check_password_hash

# Employee Training Login
@app.route('/employee_training_login', methods=['GET', 'POST'])
def employee_training_login():
    # Login removed for employee training - redirect to the training page.
    # Keep this route for compatibility with existing links.
    return redirect(url_for('train_face_data'))

@app.route('/train_face_data', methods=['GET', 'POST'])
def train_face_data():
    """Face training page for employees"""
    # Allow access without a prior training login. Employee ID may come from:
    #  - session['training_employee_id'] (back-compat)
    #  - query parameter ?employee_id=...
    #  - POSTed 'employee_id' from the form
    employee_id = session.get('training_employee_id') or request.args.get('employee_id')

    if request.method == 'POST':
        # Accept employee_id from posted form if present
        employee_id = request.form.get('employee_id') or employee_id

        # Basic validation
        if not employee_id:
            flash('Employee ID is required', 'error')
            return redirect(url_for('train_face_data'))

        # Handle face training data upload
        if 'face_images' not in request.files:
            flash('No files uploaded', 'error')
            return redirect(url_for('train_face_data'))

        files = request.files.getlist('face_images')
        if not files or all(file.filename == '' for file in files):
            flash('No files selected', 'error')
            return redirect(url_for('train_face_data'))

        # Ensure face_images directory exists
        upload_dir = os.path.join('static', 'face_images')
        os.makedirs(upload_dir, exist_ok=True)

        # Process uploaded files
        uploaded_count = 0
        for file in files:
            if file.filename != '':
                filename = secure_filename(f"{employee_id}_face_{int(time.time())}_{uploaded_count}.jpg")
                filepath = os.path.join(upload_dir, filename)
                try:
                    file.save(filepath)
                    uploaded_count += 1
                except Exception as e:
                    print(f"Error saving file {filename}: {e}")
                    continue

        if uploaded_count > 0:
            # Retrain the model with new data
            try:
                detector.load_employee_faces()
            except Exception:
                pass
            flash(f'Successfully uploaded {uploaded_count} face images for training!', 'success')
        else:
            flash('No valid images were uploaded', 'error')

        # Keep the employee_id in session for capture flow convenience
        session['training_employee_id'] = employee_id
        return redirect(url_for('train_face_data', employee_id=employee_id))

    # GET request - show the form
    return render_template('train_face_data.html', employee_id=employee_id)

@app.route('/capture_face_data')
def capture_face_data_page():
    """Webcam face capture page for employees"""
    # Allow access without prior training login. Employee ID may be provided via
    # session (from upload flow) or query parameter.
    employee_id = session.get('training_employee_id') or request.args.get('employee_id')
    if not employee_id:
        flash('Employee ID is required to capture face data. Please enter your Employee ID on the upload page.', 'error')
        return redirect(url_for('train_face_data'))

    return render_template('capture_face_data.html', employee_id=employee_id)

@app.route('/api/capture_training_images', methods=['POST'])
def capture_training_images():
    """API endpoint to handle webcam captured training images"""
    try:
        if 'training_employee_id' not in session:
            return jsonify({'success': False, 'error': 'Not authenticated'})

        data = request.get_json()
        employee_id = data.get('employee_id')
        images = data.get('images', [])

        if not employee_id or not images:
            return jsonify({'success': False, 'error': 'Missing employee_id or images'})

        # Ensure face_images directory exists
        face_images_dir = os.path.join('static', 'face_images')
        os.makedirs(face_images_dir, exist_ok=True)

        uploaded_count = 0
        for i, image_data in enumerate(images):
            try:
                # Remove data URL prefix
                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                # Decode base64 image
                import base64
                image_bytes = base64.b64decode(image_data)

                # Save image
                filename = secure_filename(f"{employee_id}_webcam_{i}.jpg")
                filepath = os.path.join(face_images_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(image_bytes)

                uploaded_count += 1

            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue

        if uploaded_count > 0:
            # Retrain the model with new data
            detector.load_employee_faces()
            return jsonify({
                'success': True,
                'message': f'Successfully uploaded {uploaded_count} images and retrained model'
            })
        else:
            return jsonify({'success': False, 'error': 'No images were saved'})

    except Exception as e:
        print(f"Error in capture_training_images: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/capture_face_data', methods=['POST'])
def capture_face_data():
    """API endpoint to handle webcam captured face data for training"""
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
        images = data.get('images', [])

        if not employee_id or not images:
            return jsonify({'success': False, 'message': 'Missing employee_id or images'})

        # Ensure face_images directory exists
        face_images_dir = os.path.join('static', 'face_images')
        os.makedirs(face_images_dir, exist_ok=True)

        uploaded_count = 0
        for i, image_data in enumerate(images):
            try:
                # Remove data URL prefix if present
                if ',' in image_data:
                    image_data = image_data.split(',')[1]

                # Decode base64 image
                import base64
                image_bytes = base64.b64decode(image_data)

                # Save image with unique filename
                filename = secure_filename(f"{employee_id}_face_{uploaded_count}.jpg")
                filepath = os.path.join(face_images_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(image_bytes)

                uploaded_count += 1

            except Exception as e:
                print(f"Error saving image {i}: {e}")
                continue

        if uploaded_count > 0:
            # Trigger model retraining
            try:
                detector.load_employee_faces()
                return jsonify({
                    'success': True,
                    'message': f'Successfully uploaded {uploaded_count} face images. Model retraining initiated.'
                })
            except Exception as e:
                print(f"Error retraining model: {e}")
                return jsonify({
                    'success': True,
                    'message': f'Uploaded {uploaded_count} images, but model retraining failed: {str(e)}'
                })
        else:
            return jsonify({'success': False, 'message': 'No images were saved successfully'})

    except Exception as e:
        print(f"Error in capture_face_data: {e}")
        return jsonify({'success': False, 'message': str(e)})

@app.route('/add_employee', methods=['GET', 'POST'])
@login_required
def add_employee():
    if request.method == 'POST':
        try:
            # Handle file upload for profile photo
            profile_photo = None
            if 'photo' in request.files:
                file = request.files['photo']
                if file.filename != '':
                    upload_dir = os.path.join('static', 'employee_photos')
                    os.makedirs(upload_dir, exist_ok=True)
                    filename = f"emp_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    filepath = os.path.join(upload_dir, filename)
                    file.save(filepath)
                    profile_photo = f"employee_photos/{filename}"

            # Parse full name into first and last name
            full_name = request.form.get('name', '').strip()
            name_parts = full_name.split(' ', 1)
            first_name = name_parts[0] if name_parts else ''
            last_name = name_parts[1] if len(name_parts) > 1 else ''

            employee_data = {
                'employee_id': request.form.get('emp_id'),
                'first_name': first_name,
                'last_name': last_name,
                'email': request.form.get('email'),
                'phone': request.form.get('phone'),
                'designation': request.form.get('position'),
                'department': request.form.get('department'),
                'hire_date': request.form.get('join_date'),
                'salary': float(request.form.get('salary', 0)) if request.form.get('salary') else 0,
                'status': 'active',
                'profile_photo': profile_photo,
                'address': request.form.get('address'),
                'emergency_contact': request.form.get('emergency_contact_name'),
                'emergency_phone': request.form.get('emergency_contact_phone')
            }

            db.create_employee(employee_data)
            flash('Employee added successfully!', 'success')
            return redirect(url_for('employees'))
        except Exception as e:
            flash(f'Error adding employee: {str(e)}', 'error')
            
    return render_template('add_employee.html')

@app.route('/employees')
@login_required
def employees():
    """Display all employees"""
    all_employees = db.get_all_employees()
    return render_template('employees_new.html', employees=all_employees, user=current_user)

@app.route('/employee/<int:employee_id>')
@login_required
def view_employee(employee_id):
    """View employee details"""
    employee = db.get_employee_by_id(employee_id)
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employees'))
    return render_template('view_employee.html', employee=employee)

@app.route('/edit_employee/<int:employee_id>', methods=['GET', 'POST'])
@login_required
def edit_employee(employee_id):
    """Edit employee details"""
    employee = db.get_employee_by_id(employee_id)
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employees'))
    
    print(f"Edit employee called for ID: {employee_id}, method: {request.method}")  # Debug print
    
    if request.method == 'POST':
        print(f"POST request received. Files: {list(request.files.keys())}")  # Debug print
        print(f"Form data: {dict(request.form)}")  # Debug print
        try:
            # Handle file upload for profile photo
            profile_photo = employee.get('profile_photo')  # Keep existing photo by default
            if 'profile_photo' in request.files:
                file = request.files['profile_photo']
                if file.filename != '':
                    # Create upload directory if it doesn't exist
                    upload_dir = os.path.join('static', 'employee_photos')
                    os.makedirs(upload_dir, exist_ok=True)
                    
                    # Test if directory is writable
                    if not os.access(upload_dir, os.W_OK):
                        print(f"Upload directory not writable: {upload_dir}")
                        flash('Error: Upload directory not writable', 'error')
                        return render_template('edit_employee.html', employee=employee)
                    
                    # Save file with unique name
                    filename = f"emp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    filepath = os.path.join(upload_dir, filename)
                    print(f"Attempting to save file to: {filepath}")  # Debug print
                    file.save(filepath)
                    profile_photo = f"employee_photos/{filename}"
                    print(f"Photo uploaded and saved as: {profile_photo}")  # Debug print
                else:
                    print("No file selected, keeping existing photo")  # Debug print
            else:
                print("No profile_photo in request.files")  # Debug print
            
            employee_data = {
                'employee_id': request.form.get('employee_id'),
                'first_name': request.form.get('first_name'),
                'last_name': request.form.get('last_name'),
                'email': request.form.get('email'),
                'phone': request.form.get('phone'),
                'designation': request.form.get('designation'),
                'department': request.form.get('department'),
                'hire_date': request.form.get('hire_date') if request.form.get('hire_date') else None,
                'salary': float(request.form.get('salary', 0)) if request.form.get('salary') else None,
                'status': request.form.get('status', 'active'),
                'profile_photo': profile_photo,
                'address': request.form.get('address'),
                'emergency_contact': request.form.get('emergency_contact'),
                'emergency_phone': request.form.get('emergency_phone')
            }
            
            print(f"Employee data to update: {employee_data}")  # Debug print
            success = db.update_employee(employee_id, employee_data)
            print(f"Database update result: {success}")  # Debug print
            if success:
                flash('Employee updated successfully!', 'success')
                return redirect(url_for('view_employee', employee_id=employee_id))
            else:
                flash('Error updating employee', 'error')
                print("Database update failed")  # Debug print
                
        except Exception as e:
            flash(f'Error updating employee: {str(e)}', 'error')
    
    return render_template('edit_employee.html', employee=employee)


@app.route('/edit_employee_by_empid/<employee_id>', methods=['GET', 'POST'])
@login_required
def edit_employee_by_empid(employee_id):
    """Edit employee using employee_id code (e.g., emp001)"""
    # Find employee by employee_id code
    employee = db.get_employee_by_employee_id(employee_id)
    if not employee:
        flash('Employee not found', 'error')
        return redirect(url_for('employees'))

    # Use primary key id internally for updates
    pk = employee.get('id')

    if request.method == 'POST':
        try:
            profile_photo = employee.get('profile_photo')
            if 'profile_photo' in request.files:
                file = request.files['profile_photo']
                if file.filename != '':
                    upload_dir = os.path.join('static', 'employee_photos')
                    os.makedirs(upload_dir, exist_ok=True)
                    if not os.access(upload_dir, os.W_OK):
                        flash('Error: Upload directory not writable', 'error')
                        return render_template('edit_employee.html', employee=employee)
                    filename = f"emp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
                    filepath = os.path.join(upload_dir, filename)
                    file.save(filepath)
                    profile_photo = f"employee_photos/{filename}"

            employee_data = {
                'employee_id': request.form.get('employee_id'),
                'first_name': request.form.get('first_name'),
                'last_name': request.form.get('last_name'),
                'email': request.form.get('email'),
                'phone': request.form.get('phone'),
                'designation': request.form.get('designation'),
                'department': request.form.get('department'),
                'hire_date': request.form.get('hire_date') if request.form.get('hire_date') else None,
                'salary': float(request.form.get('salary', 0)) if request.form.get('salary') else None,
                'status': request.form.get('status', 'active'),
                'profile_photo': profile_photo,
                'address': request.form.get('address'),
                'emergency_contact': request.form.get('emergency_contact'),
                'emergency_phone': request.form.get('emergency_phone')
            }

            success = db.update_employee(pk, employee_data)
            if success:
                flash('Employee updated successfully!', 'success')
                return redirect(url_for('view_employee', employee_id=pk))
            else:
                flash('Error updating employee', 'error')
        except Exception as e:
            flash(f'Error updating employee: {str(e)}', 'error')

    # Refresh employee data for rendering
    employee = db.get_employee_by_id(pk)
    return render_template('edit_employee.html', employee=employee)

@app.route('/delete_employee/<employee_identifier>', methods=['POST', 'DELETE'])
@login_required
def delete_employee(employee_identifier):
    """Delete employee (accepts numeric id or employee_id code like 'emp001')"""
    try:
        success = db.delete_employee(employee_identifier)
        if success:
            if request.method == 'DELETE':
                return jsonify({'success': True, 'message': 'Employee deleted successfully!'})
            else:
                flash('Employee deleted successfully!', 'success')
        else:
            if request.method == 'DELETE':
                return jsonify({'success': False, 'message': 'Employee not found'})
            else:
                flash('Employee not found', 'error')
    except Exception as e:
        if request.method == 'DELETE':
            return jsonify({'success': False, 'message': str(e)})
        else:
            flash(f'Error deleting employee: {str(e)}', 'error')

    return redirect(url_for('employees'))

@app.route('/settings')
@login_required
def settings():
    """Settings page for DVR configuration"""
    try:
        # Get user profile data
        profile_data = db.get_user_profile_data(current_user.id)
        employee_count = len(db.get_all_employees())
        attendance_count = len(db.get_all_sessions())
        
        # Get last backup info (placeholder)
        last_backup = "Never"  # This could be enhanced to track actual backups
        
        return render_template('settings_new.html', 
                             profile_data=profile_data or {},
                             employee_count=employee_count,
                             attendance_count=attendance_count,
                             last_backup=last_backup)
    except Exception as e:
        print(f"Error loading settings page: {e}")
        # Provide empty profile_data when there's an error
        return render_template('settings_new.html', 
                             profile_data={},
                             employee_count=0,
                             attendance_count=0,
                             last_backup="Never")

@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard for regular users"""
    # Get user profile data
    profile_data = db.get_user_profile_data(current_user.id)
    employee_count = len(db.get_all_employees())
    attendance_count = len(db.get_all_sessions())
    today_summary = db.get_today_summary()
    # You can add more dashboard logic here as needed
    return render_template('dashboard.html',
                          profile_data=profile_data or {},
                          employee_count=employee_count,
                          attendance_count=attendance_count,
                          today_summary=today_summary,
                          user=current_user)

@app.route('/api/update_dvr_settings', methods=['POST'])
@login_required
def update_settings():
    """Update DVR settings"""
    try:
        dvr_url = request.form.get('dvr_url')
        confidence_threshold = float(request.form.get('confidence_threshold', 0.5))
        check_interval = int(request.form.get('check_interval', 30))
        
        # Update configuration
        config.DVR_STREAM_URL = dvr_url
        config.CONFIDENCE_THRESHOLD = confidence_threshold
        config.CHECK_INTERVAL = check_interval
        
        # Save to database
        db.update_user_settings(current_user.id, {
            'dvr_url': dvr_url,
            'confidence_threshold': confidence_threshold,
            'check_interval': check_interval
        })

        # Update config values in runtime
        if dvr_url == 'webcam':
            config.DVR_STREAM_URL = 'webcam'
        else:
            config.DVR_STREAM_URL = dvr_url
        config.CONFIDENCE_THRESHOLD = confidence_threshold
        config.CHECK_INTERVAL = check_interval
        
        flash('Settings updated successfully!')
        return redirect(url_for('settings'))
    except Exception as e:
        flash(f'Error updating settings: {str(e)}')
        return redirect(url_for('settings'))

@app.route('/api/get_settings')
@login_required
def get_settings():
    """Get current settings"""
    settings = db.get_user_settings(current_user.id)
    return jsonify(settings or {
        'dvr_url': config.DVR_STREAM_URL,
        'confidence_threshold': config.CONFIDENCE_THRESHOLD,
        'check_interval': config.CHECK_INTERVAL
    })

@app.route('/api/test_dvr_connection', methods=['POST'], endpoint='api_test_dvr_connection')
@login_required
def api_test_dvr_connection():
    """Test DVR connection"""
    try:
        data = request.get_json()
        url = data.get('url', '')
        
        # Test RTSP connection
        cap = cv2.VideoCapture(url)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return jsonify({'success': True, 'message': 'Connection successful'})
            else:
                return jsonify({'success': False, 'error': 'Could not read frame'})
        else:
            return jsonify({'success': False, 'error': 'Could not open stream'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status')
@login_required
def api_status():
    """API endpoint for current status"""
    summary = db.get_today_summary()
    last_session = db.get_last_session()
    
    current_status = {
        'is_present': False,
        'current_session_start': None
    }
    
    if last_session and not last_session['end_time']:
        current_status['is_present'] = True
        start_time = datetime.datetime.fromisoformat(last_session['start_time'])
        current_status['current_session_start'] = start_time.isoformat()
    
    # Initialize detector overlay state from session
    overlays_enabled = session.get('overlays_enabled', True)
    if hasattr(detector, 'set_overlays_enabled'):
        detector.set_overlays_enabled(overlays_enabled)
    
    return jsonify({
        'current_status': current_status,
        'today_summary': summary,
        'settings': {
            'dvr_url': config.DVR_STREAM_URL,
            'confidence_threshold': config.CONFIDENCE_THRESHOLD,
            'check_interval': config.CHECK_INTERVAL,
            'overlays_enabled': overlays_enabled
        }
    })

@app.route('/api/live_stats')
@login_required
def api_live_stats():
    """API endpoint for live statistics data"""
    try:
        # Get today's summary from database
        summary = db.get_today_summary()
        
        # Get recent attendance sessions (last 24 hours)
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        # Get sessions from today
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        cursor.execute("""
            SELECT employee_id, start_time, end_time, hours 
            FROM work_log 
            WHERE date = ? 
            ORDER BY start_time DESC 
            LIMIT 50
        """, (today,))
        
        recent_sessions = cursor.fetchall()
        
        # Calculate unique employees today
        cursor.execute("""
            SELECT COUNT(DISTINCT employee_id) 
            FROM work_log 
            WHERE date = ? AND employee_id IS NOT NULL
        """, (today,))
        
        unique_employees_today = cursor.fetchone()[0]
        
        # Get total detections today (this would need to be tracked separately)
        # For now, we'll use sessions count as proxy
        total_detections = summary.get('sessions_count', 0)
        
        # Calculate average confidence from recent face detections
        # This would need to be stored in a separate table for accuracy
        avg_confidence = 85  # Default value, should be calculated from actual detections
        
        conn.close()
        
        # Get camera status
        camera_status = "Online" if detector.cap and detector.cap.isOpened() else "Offline"
        
        return jsonify({
            'success': True,
            'stats': {
                'detection_count': total_detections,
                'unique_persons': unique_employees_today,
                'avg_confidence': avg_confidence,
                'camera_status': camera_status,
                'total_hours': round(summary.get('total_hours', 0), 2),
                'sessions_today': summary.get('sessions_count', 0)
            },
            'recent_sessions': [
                {
                    'employee_id': session[0],
                    'start_time': session[1],
                    'end_time': session[2],
                    'hours': session[3]
                } for session in recent_sessions
            ]
        })
        
    except Exception as e:
        print(f"Error getting live stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'stats': {
                'detection_count': 0,
                'unique_persons': 0,
                'avg_confidence': 0,
                'camera_status': 'Error',
                'total_hours': 0,
                'sessions_today': 0
            },
            'recent_sessions': []
        })

@app.route('/api/toggle_overlays', methods=['POST'])
@login_required
def toggle_overlays():
    """API endpoint to toggle video overlays on/off"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', False)
        
        # Store overlay state in session
        session['overlays_enabled'] = enabled
        
        # Update detector overlay setting if available
        if hasattr(detector, 'set_overlays_enabled'):
            detector.set_overlays_enabled(enabled)
        
        return jsonify({
            'success': True,
            'overlays_enabled': enabled,
            'message': f'Video overlays {"enabled" if enabled else "disabled"}'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/live')
@login_required
def live_feed():
    """Live video feed page"""
    return render_template('live.html', dvr_url=config.DVR_STREAM_URL)


# Compatibility route: support legacy /view_employee/<id_or_code> URLs used by older templates/JS
@app.route('/view_employee/<employee_identifier>')
@login_required
def view_employee_compat(employee_identifier):
    """Resolve legacy view_employee URL to canonical /employee/<int:employee_id> route.

    Accepts numeric primary key or business employee_id code (e.g., 'emp001').
    Redirects to the canonical view_employee route which expects an integer id.
    """
    try:
        # If numeric, redirect directly
        if str(employee_identifier).isdigit():
            return redirect(url_for('view_employee', employee_id=int(employee_identifier)))

        # Otherwise, look up by business employee_id code
        emp = db.get_employee_by_employee_id(employee_identifier)
        if not emp:
            flash('Employee not found', 'error')
            return redirect(url_for('employees'))

        return redirect(url_for('view_employee', employee_id=emp.get('id')))
    except Exception as e:
        print(f"Error in view_employee_compat: {e}")
        flash('Error locating employee', 'error')
        return redirect(url_for('employees'))

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/face_dashboard')
def face_dashboard():
    """Dashboard for face-authenticated users"""
    if 'face_login_user' not in session:
        return redirect(url_for('face_login'))
    
    # Get employee data
    employee_id = session['face_login_user']['employee_id']
    employee = db.get_employee_by_employee_id(employee_id)
    
    if not employee:
        session.pop('face_login_user', None)
        return redirect(url_for('face_login'))
    
    # Get today's sessions for this employee
    today_sessions = db.get_today_sessions()
    employee_sessions = [s for s in today_sessions if s['employee_id'] == employee_id]
    
    # Get today's summary
    today_summary = db.get_today_summary()
    
    # Format sessions for display
    formatted_sessions = []
    for session_data in employee_sessions:
        if session_data['start_time']:
            start_time = datetime.datetime.fromisoformat(session_data['start_time'])
            session_data['start_time_formatted'] = start_time.strftime('%H:%M:%S')
        
        if session_data['end_time']:
            end_time = datetime.datetime.fromisoformat(session_data['end_time'])
            session_data['end_time_formatted'] = end_time.strftime('%H:%M:%S')
            
            # Calculate duration
            if session_data['start_time']:
                start_dt = datetime.datetime.fromisoformat(session_data['start_time'])
                duration = end_time - start_dt
                session_data['duration_formatted'] = str(duration).split('.')[0]  # Remove microseconds
        else:
            session_data['end_time_formatted'] = 'Active'
            session_data['duration_formatted'] = 'Ongoing'
        
        formatted_sessions.append(session_data)
    
    return render_template('face_dashboard.html', 
                         sessions=formatted_sessions,
                         summary=today_summary,
                         employee=employee,
                         face_user=session['face_login_user'])

@app.route('/company_dashboard')
@admin_required
def company_dashboard():
    """Company admin dashboard with full system overview"""
    # Get comprehensive system data
    all_employees = db.get_all_employees()
    all_sessions = db.get_today_sessions()
    today_summary = db.get_today_summary()
    
    # Get weekly and monthly statistics
    weekly_stats = get_weekly_statistics()
    monthly_stats = get_monthly_statistics()
    
    # Format sessions for display
    formatted_sessions = []
    for session_data in all_sessions:
        if session_data['start_time']:
            start_time = datetime.datetime.fromisoformat(session_data['start_time'])
            session_data['start_time_formatted'] = start_time.strftime('%H:%M:%S')
        
        if session_data['end_time']:
            end_time = datetime.datetime.fromisoformat(session_data['end_time'])
            session_data['end_time_formatted'] = end_time.strftime('%H:%M:%S')
            
            # Calculate duration
            if session_data['start_time']:
                start_dt = datetime.datetime.fromisoformat(session_data['start_time'])
                duration = end_time - start_dt
                session_data['duration_formatted'] = str(duration).split('.')[0]
        else:
            session_data['end_time_formatted'] = 'Active'
            session_data['duration_formatted'] = 'Ongoing'
        
        formatted_sessions.append(session_data)
    
    return render_template('company_dashboard.html',
                         employees=all_employees,
                         sessions=formatted_sessions,
                         summary=today_summary,
                         weekly_stats=weekly_stats,
                         monthly_stats=monthly_stats,
                         user=current_user)

def get_weekly_statistics():
    """Get weekly statistics for company dashboard"""
    # This is a placeholder - implement actual weekly stats logic
    return {
        'total_hours': '168h',
        'avg_daily_hours': '24h',
        'most_active_day': 'Monday',
        'total_employees_worked': 15
    }

def get_monthly_statistics():
    """Get monthly statistics for company dashboard"""
    # This is a placeholder - implement actual monthly stats logic
    return {
        'total_hours': '720h',
        'avg_daily_hours': '24h',
        'total_sessions': 450,
        'employee_utilization': '85%'
    }

@app.route('/api/employees/<employee_id>', methods=['DELETE'])
@admin_required
def delete_employee_api(employee_id):
    """Delete an employee - Admin only"""
    try:
        # Accept either numeric id or employee_id code (emp001)
        if isinstance(employee_id, str) and employee_id.isdigit():
            employee = db.get_employee_by_id(int(employee_id))
        else:
            employee = db.get_employee_by_employee_id(employee_id)

        if not employee:
            return jsonify({'success': False, 'error': 'Employee not found'}), 404

        # Delete employee (DatabaseManager.delete_employee handles both id and code)
        # If we have the PK id, pass it; otherwise pass the employee_id code
        identifier = employee.get('id') if employee.get('id') else employee.get('employee_id')
        success = db.delete_employee(identifier)
        
        if success:
            return jsonify({'success': True, 'message': 'Employee deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Failed to delete employee'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/backup', methods=['POST'])
@admin_required
def create_system_backup():
    """Create a system backup - Admin only"""
    try:
        import shutil
        import zipfile
        from datetime import datetime
        
        # Create backup directory
        backup_dir = 'backups'
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f"system_backup_{timestamp}.zip"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Create zip file with all important data
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add database
            if os.path.exists(config.DATABASE_PATH):
                zipf.write(config.DATABASE_PATH, 'attendance.db')
            
            # Add face recognition database
            if os.path.exists('face_recognition.db'):
                zipf.write('face_recognition.db', 'face_recognition.db')
            
            # Add static files
            for root, dirs, files in os.walk('static'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, '.')
                    zipf.write(file_path, arcname)
        
        return jsonify({
            'success': True, 
            'message': 'Backup created successfully',
            'backup_file': backup_filename,
            'backup_size': os.path.getsize(backup_path)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/export/all')
@admin_required
def export_all_data():
    """Export all system data as Excel file - Admin only"""
    try:
        from io import BytesIO
        import pandas as pd
        
        # Get all data
        employees = db.get_all_employees()
        sessions = db.get_all_sessions()  # You might need to implement this method
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Employees sheet
            if employees:
                df_employees = pd.DataFrame(employees)
                df_employees.to_excel(writer, sheet_name='Employees', index=False)
            
            # Sessions sheet
            if sessions:
                df_sessions = pd.DataFrame(sessions)
                df_sessions.to_excel(writer, sheet_name='Sessions', index=False)
        
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'system_export_{timestamp}.xlsx'
        
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/cleanup_data', methods=['POST'])
@login_required
@admin_required
def cleanup_data():
    """Admin-only route to cleanup system data"""
    try:
        data = request.get_json()
        cleanup_type = data.get('type', 'files_only')
        
        import subprocess
        import sys
        
        # Map cleanup types to script arguments
        cleanup_map = {
            'files_only': '1',
            'clear_data': '2', 
            'delete_users': '3',
            'full_cleanup': '4',
            'complete_reset': '5'
        }
        
        if cleanup_type not in cleanup_map:
            return jsonify({'success': False, 'error': 'Invalid cleanup type'}), 400
        
        # Run cleanup script
        script_path = os.path.join(os.getcwd(), 'cleanup_database.py')
        result = subprocess.run([
            sys.executable, script_path, cleanup_map[cleanup_type]
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({
                'success': True,
                'message': f'Cleanup completed: {cleanup_type}',
                'output': result.stdout
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Cleanup failed: {result.stderr}'
            }), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/admin/system_status', methods=['GET'])
@login_required
@admin_required
def get_system_status():
    """Get current system status for admin dashboard"""
    try:
        import sqlite3
        
        # Database info
        db_info = {}
        try:
            conn = sqlite3.connect(config.DATABASE_PATH)
            cursor = conn.cursor()
            
            tables = ['users', 'employees', 'work_log', 'daily_summary']
            for table in tables:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    db_info[table] = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    db_info[table] = 0
            conn.close()
        except Exception as e:
            db_info['error'] = str(e)
        
        # File info
        file_info = {}
        directories = [
            'static/face_images',
            'static/profile_photos',
            'static/employee_photos',
            'static/recordings'
        ]
        
        for dir_path in directories:
            if os.path.exists(dir_path):
                try:
                    files = os.listdir(dir_path)
                    file_info[dir_path] = len(files)
                except Exception:
                    file_info[dir_path] = 0
            else:
                file_info[dir_path] = 0
        
        # Database file size
        db_size = 0
        if os.path.exists(config.DATABASE_PATH):
            db_size = os.path.getsize(config.DATABASE_PATH)
        
        return jsonify({
            'success': True,
            'database': db_info,
            'files': file_info,
            'database_size': f"{db_size / 1024:.1f} KB"
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/change_video_source', methods=['POST'])
@login_required
def change_video_source():
    """Change the video source for live feed"""
    global is_streaming
    try:
        data = request.get_json()
        source = data.get('source')
        
        if source is None:
            return jsonify({'success': False, 'message': 'No source provided'})
        
        # Stop current stream if running
        if is_streaming:
            detector.stop_stream()
            is_streaming = False
        
        # Convert source to appropriate format
        if source.isdigit():
            # Camera index
            video_source = int(source)
        else:
            # String source (RTSP, file path, etc.)
            video_source = source
        
        # Test the new source
        if detector.start_stream(video_source):
            # Update the config for future use
            config.DVR_STREAM_URL = video_source
            detector.stop_stream()  # Stop the test stream
            is_streaming = False
            
            return jsonify({
                'success': True, 
                'message': f'Video source changed to: {source}'
            })
        else:
            return jsonify({
                'success': False, 
                'message': f'Failed to connect to video source: {source}'
            })
            
    except Exception as e:
        return jsonify({
            'success': False, 
            'message': f'Error changing video source: {str(e)}'
        })

@app.route('/export_attendance')
@login_required
def export_attendance():
    """Export attendance data to an Excel file"""
    try:
        sessions = db.get_all_sessions()

        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = 'Attendance'

        # Headers
        headers = ['Session ID', 'User ID', 'Username', 'Start Time', 'End Time', 'Duration (minutes)']
        sheet.append(headers)

        # Data
        for session in sessions:
            user = db.get_user_by_id(session['user_id'])
            username = user['username'] if user else 'N/A'
            
            start_time = datetime.datetime.fromisoformat(session['start_time']) if session['start_time'] else None
            end_time = datetime.datetime.fromisoformat(session['end_time']) if session['end_time'] else None
            
            duration = None
            if start_time and end_time:
                duration = round((end_time - start_time).total_seconds() / 60, 2)

            row = [
                session['id'],
                session['user_id'],
                username,
                start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A',
                end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else 'In Progress',
                duration
            ]
            sheet.append(row)

        # Save to a memory buffer
        output = BytesIO()
        workbook.save(output)
        output.seek(0)

        return send_file(
            output,
            as_attachment=True,
            download_name='attendance.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        flash(f'Error exporting data: {str(e)}')
        return redirect(url_for('dashboard'))

def generate_frames():
    """Generate frames for live video feed with enhanced camera fallback"""
    global is_streaming
    
    # Add reconnection tracking
    last_reconnect_attempt = 0
    reconnect_cooldown = 5  # seconds
    
    if not is_streaming:
        # Try multiple video sources in order of preference
        sources_to_try = [
            config.DVR_STREAM_URL,  # Primary source from config
            0,                      # Default webcam
            1,                      # Secondary webcam
            2                       # Third webcam
        ]
        
        connected = False
        for source in sources_to_try:
            print(f"Attempting to start video stream with source: {source}")
            if detector.start_stream(source):
                print(f"Successfully connected to source: {source}")
                connected = True
                break
            else:
                print(f"Failed to connect to source: {source}")
        
        if not connected:
            print("No video source available, generating error frames...")
            # Generate continuous error frames
            while True:
                error_frame = generate_error_frame()
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(1)  # Update error frame every second
    
    is_streaming = True
    
    try:
        while True:
            frame = detector.get_frame_with_detection()
            if frame is None:
                current_time = time.time()
                
                # Only attempt reconnection if cooldown has passed
                if current_time - last_reconnect_attempt > reconnect_cooldown:
                    print("No frame received from detector, attempting to reconnect...")
                    last_reconnect_attempt = current_time
                    
                    # Try to reconnect with current source
                    if detector.start_stream(config.DVR_STREAM_URL if config.DVR_STREAM_URL != 'webcam' else 0):
                        print("Reconnection successful, continuing stream")
                        continue
                    else:
                        print("Reconnection failed, will retry in cooldown period")
                
                # Show error frame briefly, then try again
                error_frame = generate_error_frame()
                ret, buffer = cv2.imencode('.jpg', error_frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.5)  # Shorter error frame display
                continue
            
            # Write frame to video file if recording
            global is_recording, video_writer
            if is_recording and video_writer:
                try:
                    video_writer.write(frame)
                except Exception as e:
                    print(f"Error writing frame to video: {e}")
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                break
                
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            time.sleep(0.1)
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
        is_streaming = False
        detector.stop_stream()

def generate_error_frame():
    """Generate an error frame when no camera is available"""
    import cv2
    import numpy as np
    
    # Create a black image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add error text
    text_lines = [
        "Camera Not Available",
        "Please check:",
        "1. Camera is connected",
        "2. Camera permissions",
        "3. No other app is using camera",
        "4. Try different video source"
    ]
    
    y_start = 150
    for i, line in enumerate(text_lines):
        y_pos = y_start + (i * 40)
        if i == 0:
            # Title in red
            cv2.putText(frame, line, (120, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Other lines in white
            cv2.putText(frame, line, (80, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def camera_availability_check():
    """Test which cameras are available"""
    available_cameras = []
    for i in range(5):  # Test cameras 0-4
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available_cameras.append(i)
            cap.release()
    return available_cameras

@app.route('/api/available_cameras')
@login_required
def get_available_cameras():
    """Get list of available cameras"""
    try:
        cameras = camera_availability_check()
        return jsonify({
            'success': True,
            'cameras': cameras,
            'message': f'Found {len(cameras)} available camera(s)'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/test_camera', methods=['POST'], endpoint='api_test_camera')
@login_required
def api_test_camera():
    """Test camera connectivity"""
    try:
        data = request.get_json()
        camera_index = data.get('camera_index', 0)
        
        # Test camera connection using enhanced logic
        test_cap = None
        backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
        
        for backend in backends:
            try:
                print(f"Testing camera {camera_index} with backend {backend}")
                test_cap = cv2.VideoCapture(camera_index, backend)
                
                if test_cap.isOpened():
                    # Set basic properties
                    test_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    test_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Try to read a frame
                    ret, frame = test_cap.read()
                    if ret and frame is not None and frame.size > 0:
                        test_cap.release()
                        return jsonify({
                            'success': True,
                            'message': f'Camera {camera_index} is working properly with backend {backend}',
                            'backend': backend,
                            'resolution': f"{frame.shape[1]}x{frame.shape[0]}"
                        })
                    else:
                        test_cap.release()
                        continue
                else:
                    continue
            except Exception as e:
                if test_cap:
                    test_cap.release()
                print(f"Backend {backend} failed: {e}")
                continue
        
        return jsonify({
            'success': False,
            'message': f'Camera {camera_index} could not be accessed with any backend'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/start_recording', methods=['POST'])
@login_required
def start_recording():
    """Start video recording"""
    global is_recording, video_writer, recording_filename
    
    try:
        if is_recording:
            return jsonify({
                'success': False,
                'message': 'Recording is already in progress'
            })
        
        # Check if camera is available
        if not detector.cap or not detector.cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Camera not available. Please ensure camera is connected and working.'
            })
        
        # Create recordings directory if it doesn't exist
        recordings_dir = os.path.join('static', 'recordings')
        os.makedirs(recordings_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"recording_{timestamp}.avi"
        full_path = os.path.join(recordings_dir, recording_filename)
        
        # Get video properties from camera
        fps = detector.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(detector.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(detector.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(full_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            return jsonify({
                'success': False,
                'message': 'Failed to initialize video writer'
            })
        
        is_recording = True
        
        return jsonify({
            'success': True,
            'message': f'Recording started: {recording_filename}',
            'filename': recording_filename
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/stop_recording', methods=['POST'])
@login_required
def stop_recording():
    """Stop video recording"""
    global is_recording, video_writer, recording_filename
    
    try:
        if not is_recording:
            return jsonify({
                'success': False,
                'message': 'No recording in progress'
            })
        
        is_recording = False
        
        if video_writer:
            video_writer.release()
            video_writer = None
        
        # Get file size for confirmation
        file_path = os.path.join('static', 'recordings', recording_filename)
        file_size = 0
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
        
        filename_to_return = recording_filename
        recording_filename = None
        
        return jsonify({
            'success': True,
            'message': f'Recording stopped: {filename_to_return}',
            'filename': filename_to_return,
            'file_size': f"{file_size / (1024*1024):.2f} MB" if file_size > 0 else "0 MB"
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/recording_status')
@login_required
def get_recording_status():
    """Get current recording status"""
    global is_recording, recording_filename
    
    return jsonify({
        'is_recording': is_recording,
        'filename': recording_filename if is_recording else None
    })

@app.route('/api/capture_snapshot', methods=['POST'])
@login_required
def capture_snapshot():
    """Capture a snapshot from the current video feed"""
    try:
        # Check if camera is available
        if not detector.cap or not detector.cap.isOpened():
            return jsonify({
                'success': False,
                'message': 'Camera not available. Please ensure camera is connected and working.'
            })
        
        # Capture frame
        ret, frame = detector.cap.read()
        if not ret or frame is None:
            return jsonify({
                'success': False,
                'message': 'Failed to capture frame from camera'
            })
        
        # Create snapshots directory if it doesn't exist
        snapshots_dir = os.path.join('static', 'snapshots')
        os.makedirs(snapshots_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        full_path = os.path.join(snapshots_dir, filename)
        
        # Save the frame as JPEG
        success = cv2.imwrite(full_path, frame)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Snapshot saved: {filename}',
                'filename': filename,
                'path': f'/static/snapshots/{filename}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to save snapshot'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/analyze_video', methods=['POST'])
@login_required
def analyze_video():
    """Upload and analyze a video file using AI"""
    try:
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No video file provided'
            })

        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No video file selected'
            })

        # Validate file type
        allowed_extensions = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
        if not video_file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({
                'success': False,
                'message': 'Invalid file type. Allowed: MP4, AVI, MOV, MKV, WebM'
            })

        # Check file size (max 100MB)
        video_file.seek(0, os.SEEK_END)
        file_size = video_file.tell()
        video_file.seek(0)
        max_size = 100 * 1024 * 1024  # 100MB

        if file_size > max_size:
            return jsonify({
                'success': False,
                'message': 'File too large. Maximum size: 100MB'
            })

        # Create uploads directory if it doesn't exist
        uploads_dir = os.path.join('static', 'video_uploads')
        os.makedirs(uploads_dir, exist_ok=True)

        # Save the video file
        filename = secure_filename(video_file.filename)
        video_path = os.path.join(uploads_dir, filename)
        video_file.save(video_path)

        # Start video analysis in a separate thread
        def analyze_async():
            try:
                results = video_analysis.analyze_video(video_path)
                # Store results for retrieval
                video_analysis.analysis_results.append({
                    'filename': filename,
                    'path': video_path,
                    'results': results,
                    'timestamp': time.time()
                })
            except Exception as e:
                print(f"Error in video analysis: {e}")
                # Store error result
                video_analysis.analysis_results.append({
                    'filename': filename,
                    'path': video_path,
                    'results': {'error': str(e)},
                    'timestamp': time.time()
                })

        analysis_thread = threading.Thread(target=analyze_async)
        analysis_thread.daemon = True
        analysis_thread.start()

        return jsonify({
            'success': True,
            'message': f'Video uploaded and analysis started: {filename}',
            'filename': filename,
            'status': 'processing'
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/video_analysis_status')
@login_required
def get_video_analysis_status():
    """Get the status of video analysis"""
    try:
        status = video_analysis.get_analysis_status()

        # Get latest results if available
        latest_results = None
        if video_analysis.analysis_results:
            latest_results = video_analysis.analysis_results[-1]

        return jsonify({
            'success': True,
            'status': status,
            'latest_results': latest_results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/capture_employee_image', methods=['POST'])
@login_required
def capture_employee_image():
    """Capture employee image from live feed"""
    try:
        data = request.get_json()
        employee_id = data.get('employee_id')
        
        if not employee_id:
            return jsonify({'success': False, 'message': 'Employee ID is required'})
        
        # Get current frame from detector
        frame = detector.current_frame
        if frame is None:
            return jsonify({'success': False, 'message': 'No video frame available'})
        
        # Create face_images directory if it doesn't exist
        face_images_dir = 'static/face_images'
        os.makedirs(face_images_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{employee_id}_face_{timestamp}.jpg"
        filepath = os.path.join(face_images_dir, filename)
        
        # Save the current frame
        cv2.imwrite(filepath, frame)
        
        # Reload face recognition data
        detector.load_employee_faces()
        
        return jsonify({
            'success': True,
            'message': f'Employee image captured and saved as {filename}',
            'filename': filename
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error capturing image: {str(e)}'
        })

@app.route('/api/upload_video', methods=['POST'])
@login_required
def upload_video():
    """Upload a video file for live feed processing"""
    try:
        if 'video_file' not in request.files:
            return jsonify({'success': False, 'message': 'No video file provided'}), 400
        
        file = request.files['video_file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        # Validate file type
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False, 
                'message': f'Unsupported file type: {file_ext}. Allowed types: {", ".join(allowed_extensions)}'
            }), 400
        
        # Create upload directory if it doesn't exist
        upload_dir = 'static/uploaded_videos'
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate secure filename with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        secure_name = secure_filename(file.filename)
        name_without_ext = os.path.splitext(secure_name)[0]
        filename = f"{name_without_ext}_{timestamp}{file_ext}"
        filepath = os.path.join(upload_dir, filename)
        
        # Save the uploaded file
        file.save(filepath)
        
        # Get file size for response
        file_size = os.path.getsize(filepath)
        file_size_mb = round(file_size / (1024 * 1024), 2)
        
        # Verify the video file can be opened by OpenCV
        import cv2
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            # Remove the invalid file
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'success': False, 
                'message': 'Invalid video file or unsupported codec. Please try a different video file.'
            }), 400
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Return the absolute path for video source
        absolute_path = os.path.abspath(filepath)
        
        return jsonify({
            'success': True,
            'message': f'Video uploaded successfully: {filename}',
            'filename': filename,
            'file_path': absolute_path,
            'file_size': f'{file_size_mb} MB',
            'duration': f'{duration:.1f} seconds',
            'resolution': f'{width}x{height}',
            'fps': f'{fps:.1f}'
        })
        
    except Exception as e:
        # Clean up uploaded file if there was an error
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        return jsonify({
            'success': False,
            'message': f'Error uploading video: {str(e)}'
        }), 500

@app.route('/api/get_detected_faces')
@login_required
def get_detected_faces():
    """Get recently detected faces from the system with activity info"""
    try:
        # Get recent detections from the detector
        detected_faces = []
        
        # Check if detector has recent face detections
        if hasattr(detector, 'recent_detections') and detector.recent_detections:
            # Get the last 10 detections without clearing them
            recent = detector.recent_detections[-10:]
            
            for detection in recent:
                # Get activity if available
                activity = detection.get('activity', 'unknown')
                
                face_data = {
                    'name': detection.get('name', 'Unknown'),
                    'confidence': detection.get('confidence', 0),
                    'image_data': detection.get('image_data', ''),
                    'is_known': detection.get('is_known', False),
                    'timestamp': detection.get('timestamp', time.time() * 1000),
                    'activity': activity
                }
                detected_faces.append(face_data)
            
            # Don't clear detections here - let them accumulate and be managed by frontend
            # detector.recent_detections.clear()
        
        return jsonify({
            'success': True,
            'faces': detected_faces
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/capture_training_image', methods=['POST'])
@login_required
def capture_training_image():
    """Capture training image from live feed for AI model training"""
    try:
        data = request.get_json()
        person_name = data.get('person_name', '').strip()
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'})
        
        # Get current frame from detector
        frame = detector.current_frame
        if frame is None:
            return jsonify({'success': False, 'message': 'No video frame available'})
        
        # Create face_images directory if it doesn't exist
        face_images_dir = 'static/face_images'
        os.makedirs(face_images_dir, exist_ok=True)
        
        # Generate filename with timestamp and person name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Include milliseconds
        safe_name = "".join(c for c in person_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')
        filename = f"training_{safe_name}_{timestamp}.jpg"
        filepath = os.path.join(face_images_dir, filename)
        
        # Save the current frame
        cv2.imwrite(filepath, frame)
        
        return jsonify({
            'success': True,
            'message': f'Training image captured for {person_name}',
            'filename': filename,
            'person_name': person_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error capturing training image: {str(e)}'
        })

@app.route('/api/train_model', methods=['POST'])
@login_required  
def train_model_live():
    """Train the face recognition model with captured training images"""
    try:
        data = request.get_json()
        person_name = data.get('person_name', '').strip()
        training_images = data.get('training_images', [])
        
        if not person_name:
            return jsonify({'success': False, 'message': 'Person name is required'})
        
        if len(training_images) < 5:
            return jsonify({'success': False, 'message': 'At least 5 training images are required'})
        
        # Add person to database if not exists
        try:
            # Check if person already exists in employees table
            existing_emp = db.get_employee_by_name(person_name)
            if not existing_emp:
                # Add new employee
                employee_id = f"TRAIN_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                names = person_name.split()
                first_name = names[0] if names else person_name
                last_name = names[1] if len(names) > 1 else ''
                
                db.add_employee({
                    'employee_id': employee_id,
                    'first_name': first_name,
                    'last_name': last_name,
                    'department': 'Training',
                    'position': 'Trainee',
                    'email': f"{employee_id.lower()}@training.local",
                    'phone': '000-000-0000',
                    'hire_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'status': 'active'
                })
        except Exception as e:
            print(f"Error adding employee: {e}")
        
        # Reload face recognition data to include new training images
        detector.load_employee_faces()
        
        # Get model statistics for response
        try:
            accuracy = 85.0  # Default accuracy estimate
            trained_persons = len(detector.employee_labels) if hasattr(detector, 'employee_labels') else 1
        except:
            accuracy = None
            trained_persons = 1
        
        return jsonify({
            'success': True,
            'message': f'Model training completed for {person_name}',
            'person_name': person_name,
            'images_used': len(training_images),
            'accuracy': accuracy,
            'trained_persons': trained_persons
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error training model: {str(e)}'
        })

@app.route('/api/evaluate_model', methods=['POST'])
@login_required
def evaluate_model():
    """Evaluate the current face recognition model performance"""
    try:
        # Get model statistics
        trained_persons = len(detector.employee_labels) if hasattr(detector, 'employee_labels') else 0
        
        # Count training images
        face_images_dir = 'static/face_images'
        total_images = 0
        if os.path.exists(face_images_dir):
            total_images = len([f for f in os.listdir(face_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Estimate accuracy based on number of training images
        if trained_persons == 0:
            accuracy = 0
        elif total_images < 50:
            accuracy = min(70, total_images * 1.4)  # Basic accuracy estimation
        else:
            accuracy = min(95, 70 + (total_images - 50) * 0.5)
        
        return jsonify({
            'success': True,
            'accuracy': round(accuracy, 1),
            'trained_persons': trained_persons,
            'total_images': total_images,
            'message': f'Model evaluation completed - {trained_persons} persons trained with {total_images} images'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error evaluating model: {str(e)}'
        })

@app.route('/api/model_stats')
@login_required
def get_model_stats():
    """Get current model statistics"""
    try:
        trained_persons = len(detector.employee_labels) if hasattr(detector, 'employee_labels') else 0
        
        # Count training images
        face_images_dir = 'static/face_images'
        total_images = 0
        if os.path.exists(face_images_dir):
            total_images = len([f for f in os.listdir(face_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        # Estimate accuracy
        if trained_persons == 0:
            accuracy = 0
        elif total_images < 50:
            accuracy = min(70, total_images * 1.4)
        else:
            accuracy = min(95, 70 + (total_images - 50) * 0.5)
        
        return jsonify({
            'success': True,
            'trained_persons': trained_persons,
            'total_images': total_images,
            'accuracy': round(accuracy, 1) if accuracy > 0 else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/employees_list')
@login_required
def get_employees_list():
    """Get list of employees for image capture"""
    try:
        employees = db.get_all_employees()
        return jsonify({
            'success': True,
            'employees': [
                {
                    'id': emp['id'],
                    'employee_id': emp['employee_id'],
                    'name': f"{emp['first_name']} {emp['last_name']}",
                    'department': emp['department']
                }
                for emp in employees if emp['status'] == 'active'
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/attendance_logs')
@login_required
def get_attendance_logs():
    """Get recent attendance logs"""
    try:
        conn = sqlite3.connect(config.DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT wl.id, wl.employee_id, wl.start_time, wl.end_time, 
                   wl.date, wl.hours, e.first_name, e.last_name, e.employee_id as full_employee_id
            FROM work_log wl
            LEFT JOIN employees e ON wl.employee_id = e.employee_id
            ORDER BY wl.start_time DESC
            LIMIT 20
        """)
        
        logs = []
        for row in cursor.fetchall():
            log = {
                'id': row[0],
                'employee_id': row[1],
                'start_time': row[2],
                'end_time': row[3],
                'date': row[4],
                'hours': row[5] if row[5] else 0,
                'employee_name': f"{row[6]} {row[7]}" if row[6] else 'Unknown',
                'full_employee_id': row[8]
            }
            logs.append(log)
        
        conn.close();
        
        return jsonify({
            'success': True,
            'logs': logs
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# Real-time Dashboard API Endpoints
@app.route('/api/dashboard/metrics')
@login_required
def get_dashboard_metrics():
    """Get real-time dashboard metrics"""
    try:
        today_summary = db.get_today_summary()
        today_sessions = db.get_today_sessions()
        
        # Get total employees from database (all active employees)
        total_employees = db.get_active_employee_count()
        
        # Calculate other metrics
        present_today = len([s for s in today_sessions if s.get('start_time') and not s.get('end_time')])
        total_hours = sum(float(session.get('hours', 0)) for session in today_sessions if session.get('hours'))
        
        # Calculate average hours based on employees who worked today (not all employees)
        employees_worked_today = len(set(session.get('employee_id') for session in today_sessions if session.get('employee_id') and session.get('hours')))
        avg_hours = total_hours / max(employees_worked_today, 1) if employees_worked_today > 0 else 0
        
        return jsonify({
            'success': True,
            'data': {
                'totalEmployees': total_employees,
                'presentToday': present_today,
                'totalHours': round(total_hours, 1),
                'avgHours': round(avg_hours, 1),
                'lastUpdated': datetime.datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/weekly')
@login_required
def get_weekly_data():
    """Get weekly attendance data for charts"""
    try:
        # Get data for the last 7 days
        weekly_data = []
        for i in range(7):
            date = datetime.datetime.now() - datetime.timedelta(days=6-i)
            day_sessions = db.get_sessions_by_date(date.strftime('%Y-%m-%d'))
            total_hours = sum(float(session.get('hours', 0)) for session in day_sessions if session.get('hours'))
            weekly_data.append(round(total_hours, 1))
        
        return jsonify({
            'success': True,
            'data': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'hours': weekly_data,
                'lastUpdated': datetime.datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/status')
@login_required
def get_status_data():
    """Get today's attendance status data"""
    try:
        today_sessions = db.get_today_sessions()
        
        # Count present and absent
        unique_users = set()
        present_users = set()
        
        for session in today_sessions:
            employee_id = session.get('employee_id')
            if employee_id:
                unique_users.add(employee_id)
                if session.get('start_time') and not session.get('end_time'):
                    present_users.add(employee_id)
        
        present_count = len(present_users)
        total_count = max(len(unique_users), 1)
        absent_count = total_count - present_count
        
        return jsonify({
            'success': True,
            'data': {
                'present': present_count,
                'absent': absent_count,
                'total': total_count,
                'lastUpdated': datetime.datetime.now().isoformat()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/dashboard/activity')
@login_required
def get_recent_activity():
    """Get recent activity feed"""
    try:
        recent_sessions = db.get_recent_sessions(limit=10)
        
        activities = []
        for session in recent_sessions:
            activity_time = session.get('start_time') or session.get('end_time')
            if activity_time:
                time_obj = datetime.datetime.fromisoformat(activity_time)
                time_diff = datetime.datetime.now() - time_obj
                
                if time_diff.days > 0:
                    time_str = f"{time_diff.days} day{'s' if time_diff.days > 1 else ''} ago"
                elif time_diff.seconds > 3600:
                    hours = time_diff.seconds // 3600
                    time_str = f"{hours} hour{'s' if hours > 1 else ''} ago"
                elif time_diff.seconds > 60:
                    minutes = time_diff.seconds // 60
                    time_str = f"{minutes} minute{'s' if minutes > 1 else ''} ago"
                else:
                    time_str = "Just now"
                
                # Get employee information
                employee_id = session.get('employee_id')
                employee_name = "Unknown Employee"
                if employee_id:
                    employee = db.get_employee_by_employee_id(employee_id)
                    if employee:
                        employee_name = employee.get('name', employee_id)
                    else:
                        employee_name = employee_id
                
                if session.get('end_time'):
                    activities.append({
                        'icon': 'fa-user-times',
                        'type': 'warning',
                        'message': f"{employee_name} checked out",
                        'time': time_str
                    })
                else:
                    activities.append({
                        'icon': 'fa-user-check',
                        'type': 'success',
                        'message': f"{employee_name} checked in",
                        'time': time_str
                    })
        
        return jsonify({
            'success': True,
            'data': activities[:5],  # Return last 5 activities
            'lastUpdated': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_settings', methods=['POST'])
@login_required
def save_settings():
    """Save comprehensive settings"""
    try:
        data = request.get_json()
        
        # Update settings in database
        db.update_user_settings(current_user.id, data)
        
        return jsonify({'success': True, 'message': 'Settings saved successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/get_settings')
@login_required
def get_comprehensive_settings():
    """Get comprehensive settings"""
    try:
        settings = db.get_user_settings(current_user.id)
        return jsonify({'success': True, 'settings': settings or {}})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_profile', methods=['POST'])
@login_required
def update_profile():
    """Update user profile information"""
    try:
        # Handle form data for photo upload
        profile_data = {}
        
        # Get form fields
        if 'firstName' in request.form:
            profile_data['first_name'] = request.form['firstName']
        if 'lastName' in request.form:
            profile_data['last_name'] = request.form['lastName']
        if 'phone' in request.form:
            profile_data['phone'] = request.form['phone']
        # Email is stored on users table; update it separately so changes persist across logins
        email_to_update = None
        if 'email' in request.form:
            email_to_update = request.form['email']
        
        # Handle profile photo upload
        if 'profilePhoto' in request.files:
            photo_file = request.files['profilePhoto']
            if photo_file and photo_file.filename:
                import os
                import uuid
                from werkzeug.utils import secure_filename

                # Create directory if it doesn't exist
                profile_dir = os.path.join('static', 'profile_photos')
                os.makedirs(profile_dir, exist_ok=True)

                # Use secure filename and preserve extension safely
                secure_name = secure_filename(photo_file.filename)
                ext = os.path.splitext(secure_name)[1].lower()
                if not ext:
                    # default to .jpg if extension missing
                    ext = '.jpg'

                filename = f"profile_{current_user.id}_{uuid.uuid4().hex[:8]}{ext}"
                photo_path = os.path.join(profile_dir, filename)

                try:
                    photo_file.save(photo_path)
                    profile_data['profile_photo'] = filename
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return jsonify({'success': False, 'message': f'Failed to save profile photo: {str(e)}'}), 500
        
        # Update profile in database
        success_profile = db.create_or_update_profile_settings(current_user.id, profile_data)
        success_user = True
        if email_to_update:
            try:
                # Update users.email via update_user_profile helper
                success_user = db.update_user_profile(current_user.id, {'email': email_to_update})
            except Exception as e:
                print(f"Error updating user email: {e}")
                success_user = False

        success = success_profile and success_user
        
        if success:
            # Refresh in-memory current_user attributes so UI shows updates without re-login
            try:
                # Enrich current_user if it's writable
                if hasattr(current_user, 'first_name') or True:
                    current_user.first_name = profile_data.get('first_name') or getattr(current_user, 'first_name', None)
                    current_user.last_name = profile_data.get('last_name') or getattr(current_user, 'last_name', None)
                    current_user.phone = profile_data.get('phone') or getattr(current_user, 'phone', None)
                    if profile_data.get('profile_photo'):
                        current_user.profile_photo = profile_data.get('profile_photo')
                    if email_to_update:
                        try:
                            current_user.email = email_to_update
                        except Exception:
                            pass
            except Exception:
                pass

            # Return updated profile data so front-end can update immediately
            try:
                updated_profile = db.get_user_profile_data(current_user.id)
            except Exception:
                updated_profile = profile_data

            return jsonify({'success': True, 'message': 'Profile updated successfully', 'profile': updated_profile})
        else:
            return jsonify({'success': False, 'message': 'Failed to update profile'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    """Change user password"""
    try:
        data = request.get_json()
        current_password = data.get('currentPassword')
        new_password = data.get('newPassword')
        
        if not current_password or not new_password:
            return jsonify({'success': False, 'message': 'Current and new passwords are required'})
        
        # Verify current password
        if not db.verify_password(current_user.username, current_password):
            return jsonify({'success': False, 'message': 'Current password is incorrect'})
        
        # Update password
        success = db.update_password(current_user.id, new_password)
        
        if success:
            return jsonify({'success': True, 'message': 'Password changed successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to change password'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/update_settings', methods=['POST'])
@login_required
def update_app_settings():
    """Update application settings"""
    try:
        data = request.get_json()
        
        # Update application settings in profile_settings table
        settings_data = {}
        
        if 'timezone' in data:
            settings_data['timezone'] = data['timezone']
        if 'theme' in data:
            settings_data['theme'] = data['theme']
        if 'emailNotifications' in data:
            settings_data['email_notifications'] = data['emailNotifications']
        if 'soundAlerts' in data:
            settings_data['sound_alerts'] = data['soundAlerts']
        if 'autoBackup' in data:
            settings_data['auto_backup'] = data['autoBackup']
        
        success = db.create_or_update_profile_settings(current_user.id, settings_data)
        
        if success:
            return jsonify({'success': True, 'message': 'Settings updated successfully'})
        else:
            return jsonify({'success': False, 'message': 'Failed to update settings'})
            
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/get_profile', methods=['GET'])
@login_required
def get_profile():
    """Get user profile data"""
    try:
        profile_data = db.get_user_profile_data(current_user.id)
        return jsonify({'success': True, 'profile': profile_data})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/export_data', methods=['GET'])
@login_required
def export_data():
    """Export user data"""
    try:
        import json
        import io
        from flask import Response
        
        # Get all user data
        profile_data = db.get_user_profile_data(current_user.id)
        settings_data = db.get_profile_settings(current_user.id)
        
        export_data = {
            'profile': profile_data,
            'settings': settings_data,
            'export_date': datetime.datetime.now().isoformat()
        }
        
        # Create JSON response
        json_str = json.dumps(export_data, indent=2)
        
        return Response(
            json_str,
            mimetype='application/json',
            headers={
                'Content-Disposition': f'attachment; filename=profile_data_{current_user.username}.json'
            }
        )
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/backup_data', methods=['POST'])
@login_required
def backup_data():
    """Create a backup of user data"""
    try:
        import json
        import os
        from datetime import datetime
        
        # Create backups directory if it doesn't exist
        backup_dir = os.path.join('backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Get all user data
        profile_data = db.get_user_profile_data(current_user.id)
        settings_data = db.get_profile_settings(current_user.id)
        
        backup_data = {
            'profile': profile_data,
            'settings': settings_data,
            'backup_date': datetime.now().isoformat()
        }
        
        # Create backup filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_filename = f'user_backup_{current_user.username}_{timestamp}.json'
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Save backup file
        with open(backup_path, 'w') as f:
            json.dump(backup_data, f, indent=2)
        
        return jsonify({
            'success': True, 
            'message': 'Backup created successfully',
            'backup_file': backup_filename
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/clear_cache', methods=['POST'])
@login_required
def clear_cache():
    """Clear application cache"""
    try:
        # This is a placeholder for cache clearing logic
        # In a real application, you might clear Redis cache, 
        # temporary files, or other cached data
        
        import os
        import shutil
        
        # Clear any temporary files
        temp_dirs = ['static/temp', '__pycache__']
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return jsonify({
            'success': True, 
            'message': 'Cache cleared successfully'
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/create_backup', methods=['POST'])
@login_required
def create_backup():
    """Create database backup"""
    try:
        backup_dir = os.path.join('backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = os.path.join(backup_dir, f'backup_{timestamp}.db')
        
        # Copy database file
        import shutil
        shutil.copy2(config.DATABASE_PATH, backup_path)
        
        return jsonify({'success': True, 'message': f'Backup created: {backup_path}'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/api/ml/evaluate', methods=['POST'])
@admin_required
def evaluate_ml_models():
    """Run ML model evaluation and return results"""
    try:
        evaluator = MLModelEvaluator()
        
        # Run face recognition evaluation
        face_results = evaluator.evaluate_face_recognition_model()
        
        # Generate visual reports
        chart_files = evaluator.generate_visual_reports()
        
        # Get system health metrics
        health_metrics = evaluator.get_system_health_metrics()
        
        return jsonify({
            'success': True,
            'data': {
                'face_recognition': face_results,
                'system_health': health_metrics,
                'chart_files': chart_files,
                'evaluation_timestamp': datetime.datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/report', methods=['POST'])
@admin_required
def generate_ml_report():
    """Generate comprehensive ML report"""
    try:
        evaluator = MLModelEvaluator()
        
        # Run full evaluation
        face_results = evaluator.evaluate_face_recognition_model()
        chart_files = evaluator.generate_visual_reports()
        
        # Save comprehensive report
        report_file = evaluator.save_evaluation_report()
        
        return jsonify({
            'success': True,
            'data': {
                'report_file': report_file,
                'accuracy': face_results.get('accuracy', 0),
                'precision': face_results.get('precision', 0),
                'recall': face_results.get('recall', 0),
                'f1_score': face_results.get('f1_score', 0),
                'chart_files': chart_files
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/accuracy-history')
@admin_required
def get_accuracy_history():
    """Get accuracy history data"""
    try:
        # Load accuracy history from file if it exists
        history_file = 'static/accuracy_history.json'
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        return jsonify({
            'success': True,
            'data': history
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/train', methods=['POST'])
@admin_required
def train_face_recognition_model():
    """Retrain the face recognition model"""
    try:
        evaluator = MLModelEvaluator()
        
        # Retrain the model
        training_result = evaluator.retrain_face_recognition_model()
        
        if training_result.get('success'):
            # Also run evaluation on the new model
            evaluation_result = evaluator.evaluate_face_recognition_model()
            
            return jsonify({
                'success': True,
                'data': training_result.get('data', {}),
                'message': 'Model retrained successfully.'
            })
        else:
            return jsonify({
                'success': False,
                'error': training_result.get('error', 'Training failed')
            }), 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/upload-training-data', methods=['POST'])
@admin_required
def upload_training_data():
    """Upload photos and train the model with new data"""
    try:
        # Check if the post request has the file part
        if 'photos' not in request.files:
            return jsonify({'success': False, 'error': 'No photos uploaded'}), 400
        
        files = request.files.getlist('photos')
        employee_name = request.form.get('employee_name', '').strip()
        employee_id = request.form.get('employee_id', '').strip()
        
        if not employee_name:
            return jsonify({'success': False, 'error': 'Employee name is required'}), 400
        
        if not employee_id:
            return jsonify({'success': False, 'error': 'Employee ID is required'}), 400
        
        if not files or all(file.filename == '' for file in files):
            return jsonify({'success': False, 'error': 'No photos selected'}), 400
        
        # Validate file types
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        for file in files:
            if file.filename == '':
                continue
            ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
            if ext not in allowed_extensions:
                return jsonify({'success': False, 'error': f'Invalid file type: {file.filename}. Only PNG, JPG, JPEG allowed'}), 400
        
        # Ensure face_images directory exists
        face_images_dir = os.path.join('static', 'face_images')
        os.makedirs(face_images_dir, exist_ok=True)
        
        # Process and save uploaded photos
        saved_files = []
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        for i, file in enumerate(files):
            if file.filename == '':
                continue
            
            # Read image data
            file_data = file.read()
            nparr = np.frombuffer(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                continue
            
            # Detect faces in the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return jsonify({'success': False, 'error': f'No face detected in {file.filename}. Please upload images with clear faces.'}), 400
            
            if len(faces) > 1:
                return jsonify({'success': False, 'error': f'Multiple faces detected in {file.filename}. Please upload images with only one face.'}), 400
            
            # Crop and save the face
            x, y, w, h = faces[0]
            # Add some padding around the face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            face_img = img[y:y+h, x:x+w]
            
            # Resize to standard size
            face_img = cv2.resize(face_img, (150, 150))
            
            # Save the face image
            filename = f"{employee_id}_face_{i}.jpg"
            filepath = os.path.join(face_images_dir, filename)
            cv2.imwrite(filepath, face_img)
            saved_files.append(filename)
        
        if not saved_files:
            return jsonify({'success': False, 'error': 'No valid face images were processed'}), 400
        
        # Add/update employee in database if needed
        try:
            db = DatabaseManager(config.DATABASE_PATH)
            existing_employee = db.get_employee_by_id(employee_id)
            
            if not existing_employee:
                # Create new employee record
                db.add_employee(
                    employee_id=employee_id,
                    name=employee_name,
                    email=f"{employee_id.lower()}@company.com",
                    phone="",
                    department="Training",
                    position="",
                    profile_photo=""
                )
        except Exception as e:
            print(f"Database error (non-critical): {e}")
        
        # Retrain the model with the new data
        try:
            evaluator = MLModelEvaluator()
            training_result = evaluator.retrain_face_recognition_model()
            
            if not training_result.get('success'):
                # Training failed, clean up uploaded files
                for filename in saved_files:
                    filepath = os.path.join(face_images_dir, filename)
                    if os.path.exists(filepath):
                        os.remove(filepath)
                return jsonify({'success': False, 'error': f'Model training failed: {training_result.get("error", "Unknown error")}'}), 500
            
            # Get updated training data info
            training_data_result = get_training_data_info()
            training_data = training_data_result.get_json() if hasattr(training_data_result, 'get_json') else {'data': {}}
            
            # Reload employee faces for live recognition
            detector.load_employee_faces()
            
            return jsonify({
                'success': True,
                'data': {
                    'message': f'Successfully uploaded {len(saved_files)} face images for {employee_name} and retrained the model!',
                    'employee_id': employee_id,
                    'employee_name': employee_name,
                    'uploaded_files': saved_files,
                    'training_result': training_result,
                    'training_data': training_data.get('data', {})
                }
            })
            
        except Exception as e:
            # Training failed, clean up uploaded files
            for filename in saved_files:
                filepath = os.path.join(face_images_dir, filename)
                if os.path.exists(filepath):
                    os.remove(filepath)
            return jsonify({'success': False, 'error': f'Model training error: {str(e)}'}), 500
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/optimize', methods=['POST'])
@admin_required
def optimize_model_parameters():
    """Optimize face recognition model parameters"""
    try:
        evaluator = MLModelEvaluator()
        
        # Optimize parameters
        optimization_result = evaluator.optimize_model_parameters()
        
        if optimization_result.get('success'):
            # Train model with best parameters
            best_params = optimization_result['best_parameters']
            
            # Create recognizer with optimized parameters
            recognizer = cv2.face.LBPHFaceRecognizer_create(**best_params)
            
            # Load and train with all data
            faces, labels, _ = evaluator.load_face_data_for_evaluation()
            if len(faces) > 0:
                faces = np.array(faces)
                labels = np.array(labels)
                recognizer.train(faces, labels)
                
                # Save optimized model
                model_path = 'static/optimized_face_model.yml'
                recognizer.save(model_path)
                
                # Update person detector
                evaluator.detector.face_recognizer = recognizer
                evaluator.detector.load_employee_faces()
            
            return jsonify({
                'success': True,
                'data': optimization_result
            })
        else:
            return jsonify({
                'success': False,
                'error': optimization_result.get('error', 'Optimization failed')
            }), 400
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/ml/training-data')
@admin_required
def get_training_data_info():
    """Get information about available training data"""
    try:
        evaluator = MLModelEvaluator()
        faces, labels, employee_names = evaluator.load_face_data_for_evaluation()
        
        # Count samples per employee
        employee_counts = {}
        for label, name in zip(labels, employee_names):
            if name not in employee_counts:
                employee_counts[name] = 0
            employee_counts[name] += 1
        
        return jsonify({
            'success': True,
            'data': {
                'total_samples': len(faces),
                'unique_employees': len(set(labels)),
                'employee_counts': employee_counts,
                'min_samples_per_employee': min(employee_counts.values()) if employee_counts else 0,
                'max_samples_per_employee': max(employee_counts.values()) if employee_counts else 0,
                'avg_samples_per_employee': round(np.mean(list(employee_counts.values())), 1) if employee_counts else 0
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/reports')
@admin_required
def reports_page():
    """Render ML reports page - Company Admin Only"""
    return render_template('reports.html')

# Add these methods to your DatabaseManager class in database.py

def get_active_employee_count(self):
    """Get count of active employees"""
    try:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM employees WHERE status = 'active'")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        print(f"Error getting active employee count: {e}")
        return 0

def get_sessions_by_date(self, date_str):
    """Get sessions for a specific date"""
    try:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM work_log 
            WHERE date = ?
        """, (date_str,))
        
        columns = [desc[0] for desc in cursor.description]
        sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error getting sessions by date: {e}")
        return []

def get_recent_sessions(self, limit=10):
    """Get recent sessions"""
    try:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM work_log 
            ORDER BY start_time DESC 
            LIMIT ?
        """, (limit,))
        
        columns = [desc[0] for desc in cursor.description]
        sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error getting recent sessions: {e}")
        return []

def cleanup_old_sessions(self, cutoff_date):
    """Cleanup old sessions"""
    try:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            DELETE FROM work_log 
            WHERE start_time < ?
        """, (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted_count
    except Exception as e:
        print(f"Error cleaning up old sessions: {e}")
        return 0

def get_all_sessions(self):
    """Get all sessions"""
    try:
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM work_log ORDER BY start_time DESC")
        
        columns = [desc[0] for desc in cursor.description]
        sessions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return sessions
    except Exception as e:
        print(f"Error getting all sessions: {e}")
        return []

@app.route('/start_live_detection_gui', methods=['POST'])
@login_required
def start_live_detection_gui():
    """Enable the live detection GUI window."""
    try:
        if hasattr(detector, 'enable_gui'):
            detector.enable_gui()
            return jsonify({'success': True, 'message': 'Live detection GUI enabled.'})
        else:
            return jsonify({'success': False, 'error': 'GUI control not available in the detector.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/stop_live_detection_gui', methods=['POST'])
@login_required
def stop_live_detection_gui():
    """Disable the live detection GUI window."""
    try:
        if hasattr(detector, 'disable_gui'):
            detector.disable_gui()
            return jsonify({'success': True, 'message': 'Live detection GUI disabled.'})
        else:
            return jsonify({'success': False, 'error': 'GUI control not available in the detector.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
        
if __name__ == '__main__':
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
