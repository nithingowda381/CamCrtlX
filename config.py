import os

# Twilio Configuration
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', 'your_sid_here')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', 'your_token_here')
TWILIO_FROM_PHONE = os.getenv('TWILIO_FROM_PHONE', '+123456789')
TWILIO_TO_PHONE = os.getenv('TWILIO_TO_PHONE', '+987654321')

# DVR Stream Configuration
DVR_STREAM_URL = os.getenv('DVR_STREAM_URL', '0')  # Default to camera 0

# Detection Configuration
DETECTION_INTERVAL = int(os.getenv('DETECTION_INTERVAL', '2'))  # seconds
ABSENCE_TIMEOUT = int(os.getenv('ABSENCE_TIMEOUT', '300'))  # 5 minutes
CONFIDENCE_THRESHOLD = 0.45  # Changed from 45 to 0.45 (45% confidence)
CHECK_INTERVAL = int(os.getenv('CHECK_INTERVAL', '30')) # seconds

# Database Configuration
DATABASE_PATH = 'attendance.db'

# Flask Configuration
FLASK_HOST = os.getenv('FLASK_HOST', '0.0.0.0')
FLASK_PORT = int(os.getenv('FLASK_PORT', '5000'))
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'

# Google Cloud Vision API Configuration
# Google Cloud Vision API Configuration
# Option 1: Use environment variable (recommended)
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')

# Option 2: Fallback to a credentials file included in project root (if present)
# This avoids a hard-coded Windows path and works on macOS/Linux when the file is present.
local_creds = os.path.join(os.getcwd(), 'your-project-credentials.json')
if not GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(local_creds):
	GOOGLE_APPLICATION_CREDENTIALS = local_creds

# If neither env var nor local file are available, leave as None and the
# Google client libraries will use the default credentials or raise an error
# when attempting to use the Vision API.

