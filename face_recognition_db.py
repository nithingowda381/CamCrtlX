import sqlite3
import os
import base64
try:
    import numpy as np
except Exception:
    np = None
import cv2
from datetime import datetime
from typing import Optional, List, Tuple
import pickle

class FaceRecognitionDB:
    def __init__(self, db_path: str = 'face_recognition.db'):
        self.db_path = db_path
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Some OpenCV builds (non-contrib) don't expose cv2.face. Provide a lightweight
        # fallback recognizer so the app can run without opencv-contrib installed.
        if hasattr(cv2, 'face'):
            try:
                self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            except Exception:
                self.recognizer = None
        else:
            # Dummy recognizer implementation: stores faces and labels and does a simple
            # nearest-match by mean absolute difference. This is NOT a replacement for
            # LBPH but allows the app to run in environments without opencv-contrib.
            class DummyRecognizer:
                def __init__(self):
                    self.faces = []
                    self.labels = []

                def train(self, faces, labels):
                    # store copies
                    self.faces = [f.copy() if hasattr(f, 'copy') else f for f in faces]
                    # labels may be numpy array or list
                    try:
                        self.labels = list(labels.tolist())
                    except Exception:
                        self.labels = list(labels)

                def predict(self, face):
                    if not self.faces:
                        raise Exception('No trained data')
                    best_idx = None
                    best_score = float('inf')
                    for i, f in enumerate(self.faces):
                        try:
                            if np is not None:
                                # use mean absolute difference
                                score = float(np.mean(np.abs(f.astype('int16') - face.astype('int16'))))
                            else:
                                # fallback numeric comparison
                                score = 0.0 if f == face else 1000.0
                        except Exception:
                            score = 1000.0
                        if score < best_score:
                            best_score = score
                            best_idx = i
                    return self.labels[best_idx], float(best_score)

            self.recognizer = DummyRecognizer()
        self.init_database()
        self.load_model()
    
    def init_database(self):
        """Initialize the face recognition database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create face_data table to store face training data
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                full_name TEXT NOT NULL,
                face_data BLOB NOT NULL,
                face_image BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        # Create face_login_attempts table for logging
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face_login_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                success BOOLEAN NOT NULL,
                confidence_score REAL,
                attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def load_model(self):
        """Load the trained face recognition model"""
        try:
            # Check if we have any training data
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM face_data WHERE is_active = 1")
            count = cursor.fetchone()[0]
            conn.close()
            
            if count > 0:
                self.train_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def add_face_data(self, user_id: int, username: str, full_name: str, 
                     face_image_path: str) -> bool:
        """Add face data for a user"""
        try:
            # Load and process the image
            image = cv2.imread(face_image_path)
            if image is None:
                print("Could not load image")
                return False
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                print("No face found in the image")
                return False
            
            if len(faces) > 1:
                print("Multiple faces found. Please use an image with only one face")
                return False
            
            # Extract face region
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Store face data
            face_data_blob = pickle.dumps(face_roi)
            
            # Read image as blob
            with open(face_image_path, 'rb') as f:
                image_blob = f.read()
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Don't deactivate existing data - keep multiple face samples for better recognition
            # cursor.execute("UPDATE face_data SET is_active = 0 WHERE user_id = ?", (user_id,))
            
            cursor.execute("""
                INSERT INTO face_data 
                (user_id, username, full_name, face_data, face_image)
                VALUES (?, ?, ?, ?, ?)
            """, (user_id, username, full_name, face_data_blob, image_blob))
            
            conn.commit()
            conn.close()
            
            # Retrain the model
            self.train_model()
            
            print(f"Face data added successfully for {full_name}")
            return True
            
        except Exception as e:
            print(f"Error adding face data: {str(e)}")
            return False
    
    def train_model(self):
        """Train the face recognition model with current data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT user_id, face_data FROM face_data WHERE is_active = 1
            """)
            
            results = cursor.fetchall()
            conn.close()
            
            if len(results) == 0:
                return
            
            faces = []
            labels = []
            
            for user_id, face_data_blob in results:
                face_data = pickle.loads(face_data_blob)
                faces.append(face_data)
                labels.append(user_id)
            
            # Train the recognizer. LBPH expects numpy arrays; our DummyRecognizer
            # accepts Python lists. Try the LBPH form first, fall back to list form.
            try:
                if np is not None:
                    self.recognizer.train(faces, np.array(labels))
                else:
                    # If numpy isn't available, pass labels as list
                    self.recognizer.train(faces, labels)
                print("Model trained successfully")
            except Exception:
                try:
                    self.recognizer.train(faces, labels)
                    print("Model trained successfully (dummy recognizer)")
                except Exception as e:
                    print(f"Error training recognizer: {e}")
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
    
    def get_all_face_users(self) -> List[Tuple]:
        """Get all users with active face data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, full_name 
            FROM face_data 
            WHERE is_active = 1
        """)
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    def recognize_face(self, image_data: str, threshold: int = 70) -> Optional[dict]:
        """Recognize face from base64 image data"""
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
            image_bytes = base64.b64decode(image_data)
            
            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                return {"success": False, "message": "No face detected"}
            
            if len(faces) > 1:
                return {"success": False, "message": "Multiple faces detected. Please ensure only one face is visible"}
            
            # Extract face region
            (x, y, w, h) = faces[0]
            face_roi = gray[y:y+h, x:x+w]
            face_roi = cv2.resize(face_roi, (200, 200))
            
            # Check if we have trained data
            try:
                # Predict using the trained model
                label, confidence = self.recognizer.predict(face_roi)
                
                # Lower confidence value means better match
                if confidence < threshold:
                    # Get user info
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT username, full_name FROM face_data 
                        WHERE user_id = ? AND is_active = 1
                    """, (int(label),))
                    
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        username, full_name = result
                        confidence_percent = max(0, (threshold - confidence) / threshold)
                        
                        # Log successful attempt
                        self.log_face_attempt(int(label), username, True, confidence_percent)
                        
                        return {
                            "success": True,
                            "user_id": int(label),
                            "username": username,
                            "full_name": full_name,
                            "confidence": confidence_percent,
                            "message": f"Welcome, {full_name}!"
                        }
                
                # Log failed attempt
                self.log_face_attempt(None, None, False, 0.0)
                return {"success": False, "message": "Face not recognized. Access denied."}
                
            except Exception as e:
                return {"success": False, "message": "No trained model available. Please register faces first."}
            
        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            return {"success": False, "message": f"Error processing image: {str(e)}"}
    
    def log_face_attempt(self, user_id: Optional[int], username: Optional[str], 
                        success: bool, confidence: float, ip_address: str = None):
        """Log face login attempt"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO face_login_attempts 
            (user_id, username, success, confidence_score, ip_address)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, username, success, confidence, ip_address))
        
        conn.commit()
        conn.close()
    
    def get_face_login_history(self, limit: int = 50) -> List[dict]:
        """Get face login attempt history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, success, confidence_score, attempt_time, ip_address
            FROM face_login_attempts 
            ORDER BY attempt_time DESC 
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "user_id": row[0],
                "username": row[1],
                "success": bool(row[2]),
                "confidence": row[3],
                "attempt_time": row[4],
                "ip_address": row[5]
            }
            for row in results
        ]
    
    def remove_face_data(self, user_id: int) -> bool:
        """Remove/deactivate face data for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE face_data 
                SET is_active = 0 
                WHERE user_id = ?
            """, (user_id,))
            
            conn.commit()
            conn.close()
            
            # Retrain model
            self.train_model()
            return True
        except Exception as e:
            print(f"Error removing face data: {str(e)}")
            return False
