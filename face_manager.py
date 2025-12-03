import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from face_recognition_db import FaceRecognitionDB

class FaceManager:
    def __init__(self):
        self.face_db = FaceRecognitionDB()
        self.upload_folder = 'static/face_images'
        self.ensure_upload_folder()
    
    def ensure_upload_folder(self):
        """Ensure the upload folder exists"""
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)
    
    def capture_and_save_face(self, user_id: int, username: str, full_name: str) -> dict:
        """Capture face from webcam and save to database"""
        try:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return {"success": False, "message": "Could not open webcam"}
            
            print("Press SPACE to capture your face, ESC to cancel")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Display the frame
                cv2.imshow('Face Capture - Press SPACE to capture, ESC to cancel', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    return {"success": False, "message": "Capture cancelled"}
                elif key == 32:  # SPACE key
                    # Save the captured frame
                    filename = f"{username}_{user_id}.jpg"
                    filepath = os.path.join(self.upload_folder, filename)
                    cv2.imwrite(filepath, frame)
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Add to database
                    success = self.face_db.add_face_encoding(user_id, username, full_name, filepath)
                    
                    if success:
                        return {"success": True, "message": f"Face registered successfully for {full_name}"}
                    else:
                        return {"success": False, "message": "Failed to register face"}
            
            cap.release()
            cv2.destroyAllWindows()
            return {"success": False, "message": "Capture failed"}
            
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def upload_face_image(self, user_id: int, username: str, full_name: str, image_file) -> dict:
        """Upload face image from file"""
        try:
            if image_file and image_file.filename:
                filename = secure_filename(f"{username}_{user_id}_{image_file.filename}")
                filepath = os.path.join(self.upload_folder, filename)
                image_file.save(filepath)
                
                # Add to database
                success = self.face_db.add_face_data(user_id, username, full_name, filepath)
                
                if success:
                    return {"success": True, "message": f"Face registered successfully for {full_name}"}
                else:
                    # Clean up file if database operation failed
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return {"success": False, "message": "Failed to register face - no face detected or multiple faces found"}
            
            return {"success": False, "message": "No file uploaded"}
            
        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
    
    def authenticate_face(self, image_data: str) -> dict:
        """Authenticate face from camera image"""
        return self.face_db.recognize_face(image_data)
    
    def get_registered_users(self) -> list:
        """Get list of users with registered faces"""
        return self.face_db.get_all_face_users()
    
    def remove_user_face(self, user_id: int) -> bool:
        """Remove face registration for a user"""
        return self.face_db.remove_face_data(user_id)
    
    def get_face_login_history(self, limit: int = 50) -> list:
        """Get face login attempt history"""
        return self.face_db.get_face_login_history(limit)
