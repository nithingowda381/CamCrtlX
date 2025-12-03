"""
Simple script to test and setup face recognition
"""
import cv2
import os
from face_recognition_db import FaceRecognitionDB
from database import DatabaseManager
import config

def setup_face_recognition():
    print("CamCrtlX Face Recognition Setup")
    print("=" * 40)
    
    # Initialize databases
    db = DatabaseManager(config.DATABASE_PATH)
    face_db = FaceRecognitionDB()
    
    print("✓ Databases initialized")
    
    # Check if we have any users
    print("\nChecking for existing users...")
    try:
        # Try to get the first user (for testing)
        user = db.get_user_by_id(1)
        if user:
            print(f"✓ Found user: {user['username']}")
            
            # Ask if they want to register a face
            response = input(f"Do you want to register a face for {user['username']}? (y/n): ")
            if response.lower() == 'y':
                capture_face_for_user(face_db, user['id'], user['username'], user['username'])
        else:
            print("No users found. Please register a user first through the web interface.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

def capture_face_for_user(face_db, user_id, username, full_name):
    """Capture face from webcam for a user"""
    print(f"\nCapturing face for {full_name}...")
    print("Instructions:")
    print("- Look directly at the camera")
    print("- Ensure good lighting")
    print("- Press SPACE to capture")
    print("- Press ESC to cancel")
    
    # Create directory if it doesn't exist
    os.makedirs('static/face_images', exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
        
        # Detect faces and draw rectangle
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.putText(frame, 'Press SPACE to capture, ESC to cancel', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Face Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            print("Capture cancelled")
            break
        elif key == 32:  # SPACE
            if len(faces) > 0:
                # Save the image
                filename = f'static/face_images/{username}_{user_id}.jpg'
                cv2.imwrite(filename, frame)
                
                # Add to database
                success = face_db.add_face_data(user_id, username, full_name, filename)
                
                if success:
                    print(f"✓ Face registered successfully for {full_name}")
                else:
                    print("✗ Failed to register face")
                break
            else:
                print("No face detected. Please position your face in the camera.")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    setup_face_recognition()
