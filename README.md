# CamCtrlX - AI-Powered Attendance Management System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-lightgrey.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1.78-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CamCtrlX is a comprehensive AI-powered attendance management system that combines computer vision, face recognition, and real-time video processing to automate employee attendance tracking. The system provides multiple authentication methods, live video monitoring, and detailed analytics for workforce management.

## 🌟 Key Features

### 🔐 Multi-Mode Authentication
- **Face Recognition Login**: AI-powered facial recognition for seamless authentication
- **Traditional Login**: Username/password authentication for standard access
- **Role-Based Access**: Company admin vs regular user permissions

### 📹 Real-Time Video Processing
- **Live Video Feed**: Real-time camera streaming with face detection overlays
- **YOLO Integration**: Advanced person detection using YOLOv8
- **Multi-Camera Support**: Support for multiple video sources (webcam, RTSP, IP cameras)
- **Video Recording**: Built-in recording capabilities with timestamping

### 👥 Face Recognition & Attendance
- **LBPH Face Recognition**: Local Binary Patterns Histograms for accurate face identification
- **Employee Management**: Complete CRUD operations for employee records
- **Attendance Tracking**: Automatic check-in/check-out with session management
- **Confidence Thresholding**: Configurable recognition accuracy settings

### 📊 Analytics & Reporting
- **Real-Time Dashboard**: Live statistics and attendance metrics
- **ML Model Evaluation**: Performance analytics for face recognition accuracy
- **Export Capabilities**: Excel export for attendance reports
- **System Monitoring**: Camera status, database health, and system diagnostics

### 🛠️ Administrative Features
- **System Backup**: Automated backup creation and management
- **Data Cleanup**: Database maintenance and cleanup utilities
- **User Management**: Admin panel for system configuration
- **API Endpoints**: RESTful API for integration with other systems

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │   Flask Server  │    │   Database      │
│   (HTML/CSS/JS) │◄──►│   (app.py)      │◄──►│   (SQLite)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Video Processing│    │Face Recognition │    │ ML Evaluation  │
│   (OpenCV)      │    │   (LBPH)        │    │   (Analytics)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Camera**: Webcam or IP camera (RTSP/HTTP stream)
- **Storage**: Minimum 2GB free space for face images and database
- **Operating System**: Windows 10+, macOS 10.15+, or Linux

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/nithingowda381/CamCrtlX-DVR-Automation.git
cd CamCrtlX-DVR-Automation
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
# Twilio Configuration (optional)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_FROM_PHONE=+1234567890
TWILIO_TO_PHONE=+0987654321

# Video Stream Configuration
DVR_STREAM_URL=0  # 0 for webcam, or RTSP URL

# Detection Settings
CONFIDENCE_THRESHOLD=0.45
DETECTION_INTERVAL=2
CHECK_INTERVAL=30

# Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=True
```

### 5. Initialize the System
```bash
# Create company admin user
python setup_company_admin.py

# Set up face recognition database
python setup_face_recognition.py
```

### 6. Run the Application
```bash
python app.py
```

Visit `http://localhost:5000` in your browser to access the application.

## 📖 Usage Guide

### First-Time Setup
1. **Create Admin Account**: Run `setup_company_admin.py` to create the first admin user
2. **Configure Camera**: Test camera connectivity through the settings page
3. **Add Employees**: Use the admin dashboard to add employee records
4. **Train Face Recognition**: Upload employee photos for face training

### Daily Operations
1. **Live Monitoring**: Access `/live` for real-time video feed with face detection
2. **Employee Check-in**: Employees can log in using face recognition or credentials
3. **View Reports**: Access attendance reports and analytics from the dashboard
4. **System Management**: Admins can manage users, backup data, and configure settings

### Face Recognition Training
1. Navigate to `/reports` (admin only)
2. Upload employee photos (multiple angles recommended)
3. The system automatically retrains the face recognition model
4. Monitor accuracy through the ML evaluation dashboard

## 🔧 Configuration

### Video Sources
The system supports multiple video input sources:
- **Webcam**: `0`, `1`, `2` (camera indices)
- **RTSP Stream**: `rtsp://username:password@ip:port/stream`
- **HTTP Stream**: `http://ip:port/video`
- **File**: Path to video file for testing

### Face Recognition Settings
- **Confidence Threshold**: Minimum similarity score (0.0-1.0)
- **Detection Scale Factor**: Face detection sensitivity (1.01-1.5)
- **Minimum Neighbors**: Face detection strictness (1-10)
- **Face Size Limits**: Minimum/maximum face dimensions

### Database Configuration
- **SQLite Database**: `attendance.db` for attendance records
- **Face Database**: `face_recognition.db` for ML model data
- **Auto-backup**: Configurable backup intervals

## 📁 Project Structure

```
CamCtrlX/
├── app.py                      # Main Flask application
├── config.py                   # Configuration settings
├── database.py                 # Database management
├── person_detector.py          # Face detection and recognition
├── ml_evaluator.py            # ML model evaluation
├── auth.py                     # Authentication system
├── requirements.txt            # Python dependencies
├── static/                     # Static files (CSS, JS, images)
│   ├── face_images/           # Training face images
│   ├── profile_photos/        # User profile images
│   ├── employee_photos/       # Employee photos
│   └── trained_face_model.yml # Trained ML model
├── templates/                  # HTML templates
│   ├── dashboard.html         # Main dashboard
│   ├── company_dashboard.html # Admin dashboard
│   ├── face_dashboard.html    # Face-login dashboard
│   └── live.html             # Live video feed
├── backups/                    # System backups
├── reports/                    # Generated reports
└── tests/                      # Test files
    ├── test_face_recognition.py
    ├── test_attendance.py
    └── test_admin.py
```

## 🔌 API Endpoints

### Authentication
- `POST /face_login_check` - Face recognition authentication
- `POST /auth/login` - Traditional login
- `POST /auth/logout` - User logout

### Video Processing
- `GET /video_feed` - Live video stream with detection
- `POST /api/start_recording` - Start video recording
- `POST /api/stop_recording` - Stop video recording
- `POST /api/capture_snapshot` - Capture image snapshot

### Employee Management
- `GET /employees` - List all employees
- `POST /add_employee` - Add new employee
- `PUT /edit_employee/<id>` - Update employee
- `DELETE /delete_employee/<id>` - Delete employee

### Analytics
- `GET /api/live_stats` - Real-time attendance statistics
- `GET /api/status` - System status information
- `POST /api/export/all` - Export all data to Excel

## 🧪 Testing

Run the test suite:
```bash
# Test face recognition
python test_face_recognition.py

# Test attendance system
python test_attendance.py

# Test admin functions
python test_admin.py

# Test live face detection
python test_live_face_detection.py
```

## 🔧 Troubleshooting

### Common Issues

**Camera Not Working**
```bash
# Test camera availability
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera available:', cap.isOpened())"
```

**Face Recognition Not Working**
1. Check if face images are in `static/face_images/`
2. Verify employee records exist in database
3. Retrain the model: `python setup_face_recognition.py`

**Database Errors**
```bash
# Check database integrity
python check_db.py

# Clean up database
python cleanup_database.py 1
```

**Permission Issues**
- Ensure camera permissions are granted
- Check file system permissions for `static/` directory
- Verify Python virtual environment is activated

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Write unit tests for new features
- Update documentation for API changes
- Test on multiple camera sources

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV** for computer vision capabilities
- **Ultralytics YOLO** for person detection
- **Flask** for web framework
- **SQLite** for database management
- **Google Cloud Vision** for advanced image analysis

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration documentation

## 🔄 Version History

### v1.0.0 (Current)
- Complete face recognition system
- Multi-camera support
- Real-time video processing
- Admin dashboard with analytics
- Automated attendance tracking
- ML model evaluation system

---

**Built with ❤️ for efficient workforce management**
