from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from database import DatabaseManager
import config
import datetime

auth_bp = Blueprint('auth', __name__)
login_manager = LoginManager()
# Lazy-initialize FaceManager to avoid importing CV dependencies at module import time
face_manager = None

def get_face_manager():
    global face_manager
    if face_manager is None:
        try:
            # Import FaceManager lazily to avoid importing cv2 at module import time
            from face_manager import FaceManager
            face_manager = FaceManager()
        except Exception as e:
            print(f"Warning: could not initialize FaceManager: {e}")
            face_manager = None
    return face_manager

class User(UserMixin):
    def __init__(self, id, username, email, role='user'):
        self.id = id
        self.username = username
        self.email = email
        self.role = role
        
    def is_admin(self):
        return self.role == 'admin'
        
    def is_company_admin(self):
        return self.role == 'company_admin'
        
    def has_role(self, role):
        return self.role == role

@login_manager.user_loader
def load_user(user_id):
    db = DatabaseManager(config.DATABASE_PATH)
    user_data = db.get_user_by_id(user_id)
    if user_data:
        role = user_data.get('role', 'user')
        return User(user_data['id'], user_data['username'], user_data['email'], role)
    return None

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = DatabaseManager(config.DATABASE_PATH)
        user_data = db.get_user_by_username(username)
        
        if user_data and check_password_hash(user_data['password'], password):
            role = user_data.get('role', 'user')
            user = User(user_data['id'], user_data['username'], user_data['email'], role)
            # Enrich user with profile fields from profile_settings so current_user has first_name, last_name, phone, profile_photo
            try:
                db_instance = DatabaseManager(config.DATABASE_PATH)
                profile = db_instance.get_user_profile_data(user_data['id'])
                if profile:
                    user.first_name = profile.get('first_name')
                    user.last_name = profile.get('last_name')
                    user.phone = profile.get('phone')
                    user.profile_photo = profile.get('profile_photo')
            except Exception:
                pass
            login_user(user)
            next_page = request.args.get('next')
            
            # Redirect based on role
            if role == 'company_admin':
                return redirect(next_page or url_for('company_dashboard'))
            else:
                return redirect(next_page or url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@auth_bp.route('/company_login', methods=['GET', 'POST'])
def company_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        db = DatabaseManager(config.DATABASE_PATH)
        user_data = db.get_user_by_username(username)
        
        if user_data and check_password_hash(user_data['password'], password):
            role = user_data.get('role', 'user')
            if role != 'company_admin':
                flash('Access denied. Company admin privileges required.')
                return render_template('company_login.html')
            user = User(user_data['id'], user_data['username'], user_data['email'], role)
            # Enrich user with profile fields
            try:
                db_instance = DatabaseManager(config.DATABASE_PATH)
                profile = db_instance.get_user_profile_data(user_data['id'])
                if profile:
                    user.first_name = profile.get('first_name')
                    user.last_name = profile.get('last_name')
                    user.phone = profile.get('phone')
                    user.profile_photo = profile.get('profile_photo')
            except Exception:
                pass
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('company_dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('company_login.html')

@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        db = DatabaseManager(config.DATABASE_PATH)
        
        # Check if username already exists
        if db.get_user_by_username(username):
            flash('Username already exists')
            return render_template('register.html')
        
        # Check if email already exists
        if db.get_user_by_email(email):
            flash('Email already registered')
            return render_template('register.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        user_id = db.create_user(username, email, hashed_password)
        
        if user_id:
            flash('Registration successful! Please login.')
            return redirect(url_for('auth.login'))
        else:
            flash('Registration failed. Please try again.')
    
    return render_template('register.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.')
    return redirect(url_for('auth.login'))

@auth_bp.route('/register_face', methods=['GET', 'POST'])
@login_required
def register_face():
    """Register face for the current user"""
    if request.method == 'POST':
        # Get person name from form
        person_name = request.form.get('person_name', current_user.username)
        source = request.form.get('source', 'upload')  # 'upload' or 'camera'
        
        # Handle multiple face images from camera capture
        face_images = []
        for key in request.files:
            if key.startswith('face_image'):
                file = request.files[key]
                if file.filename != '':
                    face_images.append(file)
        
        # Handle single file upload
        if not face_images and 'face_image' in request.files:
            file = request.files['face_image']
            if file.filename != '':
                face_images.append(file)
        
        if face_images:
            success_count = 0
            error_messages = []
            
            for i, file in enumerate(face_images):
                fm = get_face_manager()
                if fm is None:
                    error_messages.append('Face manager not available')
                    continue

                result = fm.upload_face_image(
                    current_user.id, 
                    current_user.username, 
                    person_name, 
                    file
                )
                
                if result['success']:
                    success_count += 1
                else:
                    error_messages.append(result.get('message', 'Unknown error'))
            
            # Return JSON response for AJAX requests (camera capture)
            if source == 'camera' or request.headers.get('Content-Type') == 'application/json':
                if success_count > 0:
                    # Reload employee faces for live recognition
                    try:
                        from app import detector
                        detector.load_employee_faces()
                    except Exception:
                        pass
                    return jsonify({
                        'success': True,
                        'message': f'Successfully registered {success_count} face image(s)'
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': '; '.join(error_messages)
                    })
            else:
                # Traditional form submission (file upload)
                if success_count > 0:
                    # Reload employee faces for live recognition
                    try:
                        from app import detector
                        detector.load_employee_faces()
                    except Exception:
                        pass
                    flash(f'Successfully registered {success_count} face image(s)')
                else:
                    flash('Failed to register face: ' + '; '.join(error_messages))
        else:
            if source == 'camera':
                return jsonify({'success': False, 'error': 'No images provided'})
            else:
                flash('No image uploaded')
    
    return render_template('register_face.html')

@auth_bp.route('/register_employee_face', methods=['POST'])
@login_required
def register_employee_face():
    """Register new employee with face recognition"""
    try:
        # Extract form data
        employee_data = {
            'employee_id': request.form.get('employee_id'),
            'first_name': request.form.get('username'),  # Using username as first_name for now
            'last_name': '',  # We'll extract this if needed
            'email': request.form.get('email'),
            'phone': '',  # Not provided in form
            'designation': request.form.get('position', ''),
            'department': request.form.get('department'),
            'hire_date': datetime.datetime.now().strftime('%Y-%m-%d'),
            'salary': 0.0,  # Default salary
            'status': 'active',
            'profile_photo': None,
            'address': '',
            'emergency_contact': '',
            'emergency_phone': ''
        }
        
        # Create database instance
        db_instance = DatabaseManager(config.DATABASE_PATH)
        
        # Create employee record
        employee_result = db_instance.create_employee(employee_data)
        if not employee_result:
            return jsonify({
                'success': False,
                'message': 'Failed to create employee record'
            })
        
        # Process face images
        face_images_processed = 0
        face_registration_success = False
        
        # Look for face images in the request
        for key in request.files:
            if key.startswith('face_image_'):
                file = request.files[key]
                if file.filename != '':
                    # Use the employee_id or username as person_name
                    person_name = request.form.get('username', request.form.get('employee_id'))
                    fm = get_face_manager()
                    if fm is None:
                        continue
                    result = fm.upload_face_image(
                        employee_result,  # Use the created employee ID
                        person_name,
                        person_name,
                        file
                    )
                    
                    if result['success']:
                        face_images_processed += 1
                        face_registration_success = True
        
        if face_images_processed > 0:
            # Reload employee faces for live recognition
            from app import detector
            detector.load_employee_faces()
            return jsonify({
                'success': True,
                'message': f'Employee registered successfully with {face_images_processed} face images!'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Employee created but no face images were processed'
            })
            
    except Exception as e:
        print(f"Registration error: {str(e)}")  # Debug logging
        return jsonify({
            'success': False,
            'message': f'Registration failed: {str(e)}'
        })

@auth_bp.route('/manage_faces')
@login_required  
def manage_faces():
    """Manage face registrations (admin only)"""
    fm = get_face_manager()
    registered_users = fm.get_registered_users() if fm else []
    login_history = fm.get_face_login_history(20) if fm else []
    return render_template('manage_faces.html', 
                         registered_users=registered_users,
                         login_history=login_history)

@auth_bp.route('/remove_face/<int:user_id>', methods=['POST'])
@login_required
def remove_face(user_id):
    """Remove face registration for a user"""
    fm = get_face_manager()
    if fm is None:
        flash('Face manager not available')
        return redirect(url_for('auth.manage_faces'))

    success = fm.remove_user_face(user_id)
    if success:
        # Reload employee faces for live recognition
        from app import detector
        detector.load_employee_faces()
        flash('Face registration removed successfully')
    else:
        flash('Failed to remove face registration')
    return redirect(url_for('auth.manage_faces'))
