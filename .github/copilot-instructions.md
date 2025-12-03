# CamCtrlX - AI-Powered Attendance Management System

## System Architecture

**Core Components:**
- **Flask Web Application** (`app.py`) - Main server with role-based auth, dual dashboard system
- **PersonDetector** (`person_detector.py`) - YOLO + OpenCV face recognition pipeline
- **DatabaseManager** (`database.py`) - SQLite with attendance tracking, user management, ML data
- **MLModelEvaluator** (`ml_evaluator.py`) - Model training, optimization, performance analytics
- **Authentication System** (`auth.py`) - Flask-Login with company admin vs regular user roles

**Data Flow:**
1. Live video → YOLO person detection → Face cascade → LBPH recognition → Attendance logging
2. Face images stored in `static/face_images/` with pattern `{employee_id}_face_{n}.jpg`
3. Training triggers model retraining across all stored face data

## Role-Based Access Pattern

**Two-Tier Authentication:**
```python
# Regular users: dashboard, employees, settings
@login_required
def route_name():

# Company admins: system management, ML training, data cleanup
@admin_required  # Checks current_user.is_company_admin()
def admin_route():
```

**User Types:**
- `role='user'` → Standard dashboard (`/dashboard`)
- `role='company_admin'` → Admin dashboard (`/company_dashboard`) 
- Face-only login → Session-based auth (`face_login_user` in session)

## Critical Development Workflows

**Setup Commands:**
```bash
# Create company admin (required for ML features)
python setup_company_admin.py

# Test face recognition accuracy
python test_face_recognition.py

# Database cleanup (preserves admin users)
python cleanup_database.py 4
```

**ML Training Pipeline:**
1. Upload photos via `/reports` (admin-only ML dashboard)
2. Images saved to `static/face_images/` with auto-naming
3. Model retrained on all available face data
4. Evaluation metrics generated and stored

## Project-Specific Conventions

**Database Management:**
- All DB operations through `DatabaseManager` class methods
- SQLite with attendance.db (main) + face_recognition.db (ML data)
- Session tracking: start_time → attendance logging → end_time calculation

**Face Recognition Integration:**
- Employee data in employees table links to face images by `employee_id`
- Face recognition cooldown: 15 seconds between recognitions per person
- Confidence threshold: 50+ similarity score for positive identification

**Static File Organization:**
```
static/
├── face_images/        # Training data: {employee_id}_face_{n}.jpg
├── profile_photos/     # User avatars
├── employee_photos/    # Employee profile images  
├── recordings/         # Video captures
└── trained_face_model.yml  # OpenCV LBPH model
```

**Template Structure:**
- `dashboard.html` - Regular user dashboard
- `company_dashboard.html` - Admin system overview
- `face_dashboard.html` - Face-only authenticated users
- All use glass-morphism CSS with gradient backgrounds

## Integration Points

**Video Processing:**
*** Begin Patch
## CamCtrlX — concise AI instructions for contributors

This file explains the minimal, high-value knowledge an AI coding agent needs to be productive in this repo.

- Big picture: a single-process Flask app (`app.py`) orchestrates
	video ingestion → detection (`person_detector.py`) → face recognition (OpenCV LBPH)
	→ attendance persistence (`database.py`) and optional ML training (`ml_evaluator.py`).

- Key files to read first: `app.py`, `person_detector.py`, `database.py`, `ml_evaluator.py`, `auth.py`, `face_manager.py`, `config.py`.

- Dev setup (quick, exact):
	1. Create venv: `python3 -m venv .venv && source .venv/bin/activate`
	2. Ensure build tooling: `python -m pip install --upgrade pip setuptools wheel build`
	3. Install deps: `pip install -r requirements.txt` (note: `opencv-contrib-python` is required for `cv2.face`)
	4. Initialize DB: `python -c "from database import init_db; init_db()"`
	5. (Optional) create admin: `python setup_company_admin.py`

- Runtime notes: start server with `python app.py` — host/port come from `config.py`.

- Important conventions and patterns (examples):
	- Face images: `static/face_images/{employee_id}_face_{n}.jpg`. Functions that retrain always call `detector.load_employee_faces()` after writes.
	- Trained model: `static/trained_face_model.yml` (OpenCV LBPH). Code expects `cv2.face.LBPHFaceRecognizer_create()` (needs opencv-contrib).
	- Authentication: Flask-Login + a custom `@admin_required` decorator in `app.py`. Face-only login stores `session['face_login_user']` and renders `face_dashboard`.
	- DB access: use `DatabaseManager` in `database.py`. Helpful methods: `get_all_employees()`, `get_today_sessions()`, `get_today_summary()`, `get_user_by_username()`.

- Integration & external dependencies to watch for:
	- ultralytics / YOLO model file `yolov8n.pt` (large). If you don’t need real-time detection, avoid installing `ultralytics` until needed.
	- Google Vision: `GOOGLE_APPLICATION_CREDENTIALS` env var or `your-project-credentials.json` in repo root (see `config.py` fallback).
	- OpenCV backends: camera tests in `app.py` try backends like `cv2.CAP_MSMF`/`CAP_DSHOW`/`CAP_ANY` — on macOS camera behavior differs.

- Tests & debugging:
	- Quick syntax check: `python3 -m py_compile app.py`
	- Init DB and run tests: `pip install pytest && pytest -q` (fix failures iteratively).
	- When facing install errors, ensure `setuptools`/`wheel` are present in the venv before installing packages.

- Small gotchas discovered in code:
	- `sqlite3` is stdlib; don’t add it to `requirements.txt`.
	- The app writes many `static/` subfolders (face_images, recordings, snapshots) — ensure the process has write permissions.
	- There are a few helper scripts (`setup_company_admin.py`, `cleanup_database.py`, `setup_face_recognition.py`) used by workflows. Prefer those for one-shot tasks.

If any section is unclear or you want the file expanded with examples (e.g., a short reproducible runbook for macOS camera testing or a recommended Python version), tell me which area to expand.
