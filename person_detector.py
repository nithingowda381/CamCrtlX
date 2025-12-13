import cv2
import numpy as np
import os
import torch
import warnings
import pickle
import time
import sqlite3
from datetime import datetime, timedelta
import base64

# Patch torch.load for PyTorch 2.6+ to support YOLO models
original_torch_load = torch.load

def patched_torch_load(f, *args, **kwargs):
    # Set weights_only=False for YOLO model compatibility
    kwargs['weights_only'] = False
    return original_torch_load(f, *args, **kwargs)

torch.load = patched_torch_load

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False
    print(f"[WARNING] Ultralytics not found. Object detection will be disabled. Error: {e}")

# Suppress known resource_tracker semaphore leak warnings
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
)
warnings.filterwarnings("ignore", category=UserWarning)

class PersonDetector:
    def __init__(self, model_path: str = 'yolov8n.pt', confidence_threshold: float = 0.5):
        # Initialize YOLO model if available; otherwise run in OpenCV-only mode
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"[INFO] YOLO model loaded successfully. Object detection enabled.")
            except Exception as e:
                print(f"[ERROR] YOLO model load failed: {e}. Falling back to OpenCV-only detection.")
                self.model = None
        else:
            print(f"[WARNING] YOLO not available. Object detection disabled.")
        self.confidence_threshold = confidence_threshold
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.detection_result = None

        # Face recognition setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if self.face_cascade.empty():
             print("[ERROR] Failed to load haarcascade_frontalface_default.xml")

        # Try to load deep learning model first, fall back to LBPH
        self.deep_learning_model = None
        self.face_recognizer = None

        # Load deep learning model if available
        self._load_deep_learning_model()

        # If deep learning model not available, use traditional LBPH
        if self.deep_learning_model is None:
            self._setup_lbph_recognizer()

        self.employee_labels = {}  # Maps label IDs to employee data
        self.last_recognition_time = {}  # Track last recognition time per employee
        self.recognition_cooldown = 15  # Reduced to 15 seconds for more responsive attendance

        # Action display tracking
        self.current_actions = {}  # Store current actions for each person
        self.action_display_time = {}  # When to stop showing action text
        
        # Activity detection with OpenCV
        self.person_activities = {}  # Track activity for each detected person
        # Note: Previous version used 'lbpcascade_anon_face.xml' for hand detection but it was unused and caused errors. Removed.
        
        # Object detection (bags, phones, laptops, TV)
        self.detected_objects = {}  # Track detected objects
        self.object_classes = ['bag', 'handbag', 'backpack', 'phone', 'cell phone', 'laptop', 'tv', 'monitor', 'screen', 'smartphone']
        self.last_objects_frame_id = None  # Cache frame ID for object detection
        self.last_objects_results = {}  # Cache results to avoid re-detecting same frame

        # Recent face detections for live display
        self.recent_detections = []
        self.max_recent_detections = 50  # Keep last 50 detections
        # Track whether the last ROI contained a face (used to avoid drawing NOT-RECOGNIZED on non-face detections)
        self._last_face_found = False

        # Overlay toggle setting
        self.overlays_enabled = True  # Default to enabled

        self.load_employee_faces()
        # Allow low-confidence matches when explicitly enabled via env var (for debugging)
        self.allow_low_confidence = os.environ.get('ALLOW_LOW_CONFIDENCE', '0') == '1'
        # Minimum similarity percent to accept a recognized label as known
        try:
            env_val = float(os.environ.get('MIN_SIMILARITY_PERCENT', '5'))
            # Use 5% as absolute minimum for live video
            self.min_similarity_percent = max(5.0, min(env_val, 10.0))
        except Exception:
            self.min_similarity_percent = 5.0
        # Confirmation streaks
        try:
            self.confirmation_required = int(os.environ.get('CONFIRMATION_REQUIRED', '3'))
        except Exception:
            self.confirmation_required = 3
        # How long (s) between sightings to continue a streak
        try:
            self.confirmation_window = float(os.environ.get('CONFIRMATION_WINDOW', '2.0'))
        except Exception:
            self.confirmation_window = 2.0
        # Tracking structures
        # recognition_streaks: {db_id: {'count': int, 'last_seen': timestamp}}
        self.recognition_streaks = {}
        # confirmed_ids: map db_id -> expiry timestamp (to allow repeated logging after expiry)
        self.confirmed_ids = {}
        self.show_gui_window = False  # Control GUI window visibility

        # Tracking structures
        self.tracked_people = {}  # id -> {box, first_seen, last_seen, ...}
        self.tracked_objects = {} # id -> {box, first_seen, last_seen, label, ...}
        self.next_person_id = 0
        self.next_object_id = 0

    def _load_deep_learning_model(self):
        """Load deep learning face recognition model if available"""
        try:
            # Check if dlib is available first
            import dlib

            if os.path.exists('deep_face_model.pkl'):
                with open('deep_face_model.pkl', 'rb') as f:
                    model_data = pickle.load(f)

                if model_data.get('model_type') == 'deep_learning_knn':
                    self.deep_learning_model = model_data
                    self.employee_labels = model_data.get('employee_data', {})
                    print("✓ Deep learning face recognition model loaded successfully")
                    print(f"  Model type: {model_data.get('model_type')}")
                    print(f"  Trained employees: {len(self.employee_labels)}")
                    return True
        except ImportError:
            print("ℹ dlib not installed. Deep learning features disabled.")
        except Exception as e:
            print(f"Warning: Could not load deep learning model: {e}")

        print("ℹ Using traditional LBPH face recognition (deep learning model not available)")
        return False

    def _setup_lbph_recognizer(self):
        """Set up traditional LBPH face recognizer"""
        # Create LBPH recognizer if available, otherwise use a lightweight dummy
        if hasattr(cv2, 'face'):
            try:
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            except Exception:
                self.face_recognizer = None
        else:
            class DummyRecognizer:
                def __init__(self):
                    self.faces = []
                    self.labels = []

                def train(self, faces, labels):
                    self.faces = [f.copy() if hasattr(f, 'copy') else f for f in faces]
                    try:
                        self.labels = list(labels.tolist())
                    except Exception:
                        self.labels = list(labels)

                def predict(self, face):
                    if not self.faces:
                        raise Exception('No trained data')

                    # Compute L2 distances between probe and each stored face
                    dists = []
                    for f in self.faces:
                        try:
                            # Ensure numeric type for distance computation
                            d = np.linalg.norm((f.astype('float32') - face.astype('float32')).ravel())
                        except Exception:
                            d = float('inf')
                        dists.append(d)
                    dists = np.array(dists)

                    # Aggregate distances per label (mean distance per label)
                    label_dist = {}
                    for lbl in set(self.labels):
                        idxs = [i for i, l in enumerate(self.labels) if l == lbl]
                        if not idxs:
                            continue
                        label_dist[lbl] = float(np.mean(dists[idxs]))

                    # Select label with minimum mean distance
                    best_label = min(label_dist, key=label_dist.get)
                    best_score = float(label_dist[best_label])

                    # Improved normalization
                    try:
                        max_possible_dist = 255.0 * np.sqrt(100*100)  # ≈ 25500
                        normalized_confidence = 100.0 - (best_score / max_possible_dist) * 100.0
                        normalized_confidence = max(0.0, min(100.0, normalized_confidence))
                    except Exception:
                        normalized_confidence = max(0.0, min(100.0, 100.0 - best_score))

                    return best_label, float(normalized_confidence)

            self.face_recognizer = DummyRecognizer()

    def __del__(self):
        try:
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            if hasattr(self, 'model') and self.model is not None:
                try:
                    if hasattr(self.model, 'close'):
                        self.model.close()
                except Exception:
                    pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        except Exception:
            pass
        
    def start_stream(self, stream_url) -> bool:
        """Start the video stream with enhanced camera connection logic"""
        try:
            if isinstance(stream_url, str) and stream_url.isdigit():
                video_source = int(stream_url)
            elif isinstance(stream_url, int):
                video_source = stream_url
            else:
                video_source = stream_url
            
            print(f"Attempting to connect to video source: {video_source}")
            
            if isinstance(video_source, int):
                backends = [cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY]
                camera_indices = [video_source] if video_source != 0 else [0, 1, 2]
                
                success = False
                for cam_index in camera_indices:
                    if success:
                        break
                    for backend in backends:
                        try:
                            print(f"Trying camera {cam_index} with backend {backend}")
                            self.cap = cv2.VideoCapture(cam_index, backend)

                            if self.cap.isOpened():
                                import time
                                time.sleep(0.5)
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.cap.set(cv2.CAP_PROP_FPS, 30)
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                                ret, frame = self.cap.read()
                                if ret and frame is not None and frame.size > 0:
                                    print(f"Successfully connected to camera {cam_index} with backend {backend}")
                                    self.is_running = True
                                    return True
                                else:
                                    print(f"Camera {cam_index} opened but no frame received")
                                    self.cap.release()
                        except Exception as backend_error:
                            print(f"Backend {backend} failed for camera {cam_index}: {backend_error}")
                            if hasattr(self, 'cap') and self.cap:
                                self.cap.release()
                            continue
                
                print("Failed to connect to any webcam with any backend")
                return False
            else:
                self.cap = cv2.VideoCapture(video_source)
                if not self.cap.isOpened():
                    print(f"Failed to open video source: {video_source}")
                    return False
                
                ret, frame = self.cap.read()
                if not ret or frame is None or frame.size == 0:
                    print(f"Failed to read frame from video source: {video_source}")
                    self.cap.release()
                    return False
                
                print(f"Successfully connected to video source: {video_source}")
                self.is_running = True
                return True
            
        except Exception as e:
            print(f"Error starting stream: {e}")
            if hasattr(self, 'cap') and self.cap:
                self.cap.release()
            return False
    
    def set_overlays_enabled(self, enabled: bool):
        """Enable or disable video overlays"""
        self.overlays_enabled = enabled
        print(f"Video overlays {'enabled' if enabled else 'disabled'}")
    
    def detect_person_activity(self, frame_roi: np.ndarray) -> str:
        """
        Detect person's activity/pose from ROI using OpenCV-based analysis
        Returns: 'standing', 'sitting', 'phone', 'looking_down', 'unknown'
        """
        if frame_roi is None or frame_roi.size == 0:
            return 'unknown'
        
        try:
            h, w = frame_roi.shape[:2]
            
            # Face detection in ROI to find head position
            gray_roi = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray_roi, 1.05, 5, minSize=(30, 30), maxSize=(w-10, h-10))
            
            if len(faces) == 0:
                return 'unknown'
            
            # Get face position (highest confidence = largest face)
            face = max(faces, key=lambda f: f[2] * f[3])  # Sort by area
            fx, fy, fw, fh = face
            face_center_y = fy + fh / 2
            
            # Analyze ROI structure to determine activity
            # Check if face is in upper part of ROI (looking down/at phone)
            face_position_ratio = face_center_y / h
            
            # Check for bright/illuminated areas (phone screen effect)
            lower_roi = gray_roi[int(fy + fh):, :]
            if len(lower_roi) > 0 and lower_roi.size > 0:
                lower_brightness = np.mean(lower_roi)
            else:
                lower_brightness = 0
            
            full_brightness = np.mean(gray_roi)
            
            # Activity classification based on position and lighting
            if face_position_ratio < 0.3:
                # Face in upper part - likely looking down at phone
                return 'phone'
            elif face_position_ratio > 0.7:
                # Face in lower part - likely sitting
                return 'sitting'
            elif lower_brightness > full_brightness + 20:
                # Bright area below face (phone screen glow)
                return 'phone'
            elif 0.3 <= face_position_ratio <= 0.7:
                # Face in middle - standing or walking
                return 'standing'
            else:
                return 'unknown'
                
        except Exception as e:
            print(f"Error in activity detection: {e}")
            return 'unknown'
    
    def detect_objects_in_frame(self, frame: np.ndarray) -> dict:
        """
        Detect objects like bags, phones, laptops, TV in the frame
        Caches results to avoid detecting the same frame multiple times
        Returns dict with object names and their bounding boxes
        """
        objects_found = {}
        
        if not YOLO_AVAILABLE or self.model is None:
            return objects_found
        
        try:
            # Use frame ID based on memory address to detect frame changes
            frame_id = id(frame)
            
            # Return cached results if same frame
            if frame_id == self.last_objects_frame_id:
                return self.last_objects_results
            
            # Run YOLO detection on entire frame
            results = self.model(frame, conf=0.25)  # Lower confidence for better phone detection
            
            detected_count = 0
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        class_name = result.names.get(class_id, f"Class {class_id}")
                        confidence = float(box.conf[0])
                        detected_count += 1
                        
                        # Check if detected object is in our interest list
                        is_match = any(obj_type.lower() in class_name.lower() for obj_type in self.object_classes)
                        if is_match:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            key = f"{class_name}_{confidence:.2f}"
                            objects_found[key] = {
                                'name': class_name,
                                'confidence': confidence,
                                'box': (x1, y1, x2, y2)
                            }
            
            # Store detected objects for tracking
            if objects_found:
                self.detected_objects = objects_found
            
            # Cache the results
            self.last_objects_frame_id = frame_id
            self.last_objects_results = objects_found
                
        except Exception as e:
            print(f"Error in object detection: {e}")
        
        return objects_found
    
    def load_employee_faces(self):
        """Load employee face data for recognition - SIMPLIFIED"""
        try:
            face_images_dir = 'static/face_images'
            if not os.path.exists(face_images_dir):
                print(f"Warning: Face images directory not found at '{face_images_dir}'")
                return

            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT id, employee_id, first_name, last_name FROM employees WHERE status = 'active'")
            employees = cursor.fetchall()
            conn.close()

            faces = []
            labels = []
            self.employee_labels.clear() # Clear previous data

            print(f"Found {len(employees)} active employees. Loading faces...")

            for db_id, employee_id, first_name, last_name in employees:
                images_loaded_for_employee = 0
                for filename in os.listdir(face_images_dir):
                    # Strict matching: {employee_id}_face_...
                    if filename.startswith(f"{employee_id}_face_"):
                        image_path = os.path.join(face_images_dir, filename)
                        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        
                        if img is not None:
                            img_resized = cv2.resize(img, (100, 100))
                            faces.append(img_resized)
                            labels.append(db_id)
                            images_loaded_for_employee += 1

                if images_loaded_for_employee > 0:
                    self.employee_labels[db_id] = {
                        'employee_id': employee_id,
                        'name': f"{first_name} {last_name}",
                        'first_name': first_name,
                        'last_name': last_name
                    }
                    print(f"  - Loaded {images_loaded_for_employee} images for {first_name} {last_name} (DB ID: {db_id})")

            if faces and labels:
                print(f"Training recognizer with {len(faces)} faces for {len(set(labels))} employees.")
                try:
                    self.face_recognizer.train(faces, np.array(labels, dtype=np.int32))
                    # Save the trained model for debugging and external use
                    model_save_path = 'static/trained_face_model.yml'
                    self.face_recognizer.save(model_save_path)
                    print(f"Successfully trained and saved model to '{model_save_path}'")
                except Exception as e:
                    print(f"Error during face recognizer training or saving: {e}")
            else:
                print("No faces were loaded for training. Face recognition will be inactive.")
                # If no faces are found, clear the recognizer
                if hasattr(cv2, 'face'):
                    self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                else:
                    self.face_recognizer.__init__()


        except Exception as e:
            print(f"An error occurred in load_employee_faces: {e}")

    def _load_recognition_threshold(self) -> float:
        """Load recognition threshold from saved trained model config if available."""
        try:
            model_yaml = 'static/trained_face_model.yml'
            if os.path.exists(model_yaml):
                try:
                    import importlib
                    yaml_spec = importlib.util.find_spec('yaml')
                    yaml = importlib.import_module('yaml') if yaml_spec is not None else None

                    if yaml is not None:
                        with open(model_yaml, 'r') as f:
                            data = yaml.safe_load(f)
                            cfg = data.get('opencv_lbphfaces', {})
                            thr = cfg.get('threshold')
                            if thr is not None:
                                return float(thr)
                    else:
                        import re
                        with open(model_yaml, 'r') as f:
                            content = f.read()
                        m = re.search(r"threshold\s*:\s*([0-9]+(?:\.[0-9]+)?)", content)
                        if m:
                            return float(m.group(1))
                except Exception:
                    pass
        except Exception:
            pass
        return 70.0
    
    def recognize_face(self, face_img) -> tuple:
        """Recognize face and return employee DB id and confidence percent.
        Returns None or tuple (db_id, confidence).
        """
        try:
            # If we don't have any labeled employees loaded, bail out early
            if len(self.employee_labels) == 0:
                return None, 0.0

            # Use deep learning model if available AND dlib is installed
            if self.deep_learning_model is not None:
                try:
                    import dlib
                    return self._recognize_face_deep_learning(face_img)
                except ImportError:
                    pass # Fallback to LBPH

            return self._recognize_face_lbph(face_img)

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, 0.0

    def _recognize_face_deep_learning(self, face_img) -> tuple:
        """Recognize face using deep learning embeddings"""
        try:
            import dlib

            # Initialize dlib models if not already done
            if not hasattr(self, '_dlib_detector'):
                self._dlib_detector = dlib.get_frontal_face_detector()
                self._dlib_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                self._dlib_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

            # Convert grayscale face_img to RGB for dlib
            if len(face_img.shape) == 2:
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_face = face_img

            # Detect faces
            faces = self._dlib_detector(rgb_face, 1)
            if len(faces) == 0:
                return None, 0.0

            # Use the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # Get facial landmarks and compute embedding
            shape = self._dlib_predictor(rgb_face, face)
            face_embedding = self._dlib_recognizer.compute_face_descriptor(rgb_face, shape)
            embedding = np.array(face_embedding)

            # Normalize embedding
            scaler = self.deep_learning_model['scaler']
            embedding_normalized = scaler.transform(embedding.reshape(1, -1))

            # Predict using KNN classifier
            knn = self.deep_learning_model['knn_classifier']
            distances, indices = knn.kneighbors(embedding_normalized, n_neighbors=5)

            # Get the closest match
            closest_distance = distances[0][0]
            predicted_label = self.deep_learning_model['labels'][indices[0][0]]

            # Convert distance to confidence
            max_distance = 1.0  # Empirical threshold
            confidence_percent = max(0.0, 100.0 - (closest_distance / max_distance) * 100.0)

            # Only return match if confidence is above threshold
            if confidence_percent >= self.min_similarity_percent:
                return int(predicted_label), confidence_percent
            else:
                return None, confidence_percent

        except Exception as e:
            print(f"Error in deep learning recognition: {e}")
            return None, 0.0

    def _recognize_face_lbph(self, face_img) -> tuple:
        """Recognize face using traditional LBPH method"""
        try:
            try:
                label, confidence = self.face_recognizer.predict(face_img)
            except Exception as e:
                # print(f"Recognizer predict error: {e}")
                return None, 0.0

            if hasattr(self.face_recognizer, 'predict') and 'DummyRecognizer' in str(type(self.face_recognizer)):
                confidence_percent = float(confidence)
            else:
                try:
                    # Map the distance to a 0-100 similarity score.
                    # LBPH distance: 0 (exact) -> 200 (completely different)
                    distance = float(confidence)
                    max_distance_for_scaling = 200.0
                    similarity = 100.0 - (distance / max_distance_for_scaling) * 100.0
                    confidence_percent = max(0.0, min(100.0, similarity))

                except Exception:
                    confidence_percent = 0.0

            return int(label), confidence_percent
        except Exception as e:
            print(f"Error in LBPH face recognition: {e}")
            return None, 0.0
    
    def log_attendance(self, employee_db_id: int, employee_data: dict, confidence: float):
        """Log attendance for recognized employee and return action performed"""
        try:
            current_time = datetime.now()
            action_performed = None
            
            # Check cooldown period
            if employee_db_id in self.last_recognition_time:
                time_diff = (current_time - self.last_recognition_time[employee_db_id]).total_seconds()
                if time_diff < self.recognition_cooldown:
                    return self.current_actions.get(employee_db_id, None)
            
            # Update last recognition time
            self.last_recognition_time[employee_db_id] = current_time
            
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            
            # Check if employee already has an active session today
            today = current_time.strftime('%Y-%m-%d')
            cursor.execute("""
                SELECT id, start_time, end_time FROM work_log 
                WHERE employee_id = ? AND date = ? AND end_time IS NULL
                ORDER BY start_time DESC LIMIT 1
            """, (employee_data['employee_id'], today))
            
            active_session = cursor.fetchone()
            
            if active_session:
                # Employee has active session - this could be checkout
                session_id, start_time, end_time = active_session
                start_datetime = datetime.fromisoformat(start_time)
                work_duration = (current_time - start_datetime).total_seconds() / 3600  # hours
                
                # Allow checkout after minimum 5 minutes
                if work_duration >= 0.083:
                    cursor.execute("""
                        UPDATE work_log SET end_time = ?, hours = ? WHERE id = ?
                    """, (current_time.isoformat(), round(work_duration, 2), session_id))
                    
                    action_performed = "CHECK-OUT"
                else:
                    action_performed = "CHECKED-IN"
            else:
                # No active session - this is check-in
                cursor.execute("""
                    INSERT INTO work_log (employee_id, start_time, date, total_seconds) 
                    VALUES (?, ?, ?, 0)
                """, (employee_data['employee_id'], current_time.isoformat(), today))
                
                action_performed = "CHECK-IN"
            
            self.current_actions[employee_db_id] = action_performed
            self.action_display_time[employee_db_id] = current_time + timedelta(seconds=5)
            
            conn.commit()
            conn.close()
            
            print(f"Attendance logged: {employee_data['name']} - {action_performed}")
            return action_performed
            
        except Exception as e:
            print(f"Error logging attendance: {e}")
            return None
    
    def stop_stream(self):
        """Stop the video stream"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def detect_person(self) -> tuple:
        """Detect if person is present in current frame. Returns (bool, float)."""
        if not self.cap or not self.cap.isOpened():
            return False, None
        
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        
        self.current_frame = frame
        
        # If YOLO model is available use it, otherwise fall back to face cascade detection
        if self.model is not None:
            try:
                results = self.model(frame, conf=self.confidence_threshold)
                person_detected = False
                max_confidence = 0.0

                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            if int(box.cls[0]) == 0:  # Person class
                                person_detected = True
                                confidence = float(box.conf[0])
                                max_confidence = max(max_confidence, confidence)

                return person_detected, max_confidence if person_detected else None
            except Exception as e:
                print(f"Error running YOLO detection: {e}. Falling back to OpenCV face detection.")

        # Fallback: detect faces in the frame using Haar cascade
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=6,
                minSize=(48, 48)
            )
            if len(faces) > 0:
                return True, 0.6
            return False, None
        except Exception as e:
            return False, None
    
    def _update_trackers(self, current_detections, tracker_dict, next_id_attr_name):
        updated_trackers = {}
        used_detections = set()

        # Match existing trackers to new detections
        for obj_id, tracked_obj in tracker_dict.items():
            tx1, ty1, tx2, ty2 = tracked_obj['box']
            t_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)

            best_match_idx = -1
            min_dist = float('inf')

            for i, det in enumerate(current_detections):
                if i in used_detections:
                    continue

                dx1, dy1, dx2, dy2 = det['box']
                d_center = ((dx1 + dx2) / 2, (dy1 + dy2) / 2)

                dist = ((t_center[0] - d_center[0])**2 + (t_center[1] - d_center[1])**2)**0.5

                if dist < 150 and dist < min_dist:
                    min_dist = dist
                    best_match_idx = i

            if best_match_idx != -1:
                det = current_detections[best_match_idx]
                tracked_obj['box'] = det['box']
                tracked_obj['last_seen'] = time.time()
                tracked_obj['confidence'] = det.get('confidence', tracked_obj.get('confidence', 0))
                for k, v in det.items():
                    if k not in ['box', 'confidence']:
                        tracked_obj[k] = v

                updated_trackers[obj_id] = tracked_obj
                used_detections.add(best_match_idx)
            else:
                if time.time() - tracked_obj['last_seen'] < 1.0:
                    updated_trackers[obj_id] = tracked_obj

        # Create new trackers
        for i, det in enumerate(current_detections):
            if i not in used_detections:
                next_id = getattr(self, next_id_attr_name)
                setattr(self, next_id_attr_name, next_id + 1)

                new_obj = {
                    'id': next_id,
                    'box': det['box'],
                    'first_seen': time.time(),
                    'last_seen': time.time(),
                    'confidence': det.get('confidence', 0)
                }
                for k, v in det.items():
                    if k not in ['box', 'confidence']:
                        new_obj[k] = v

                updated_trackers[next_id] = new_obj

        tracker_dict.clear()
        tracker_dict.update(updated_trackers)

    def draw_object_overlay(self, frame, x1, y1, x2, y2, label, confidence, duration):
        """Draw overlay for generic objects"""
        try:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
            label_text = f"{label} {confidence:.2f}"
            time_text = f"Time: {int(duration)}s"

            cv2.putText(frame, label_text, (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(frame, time_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        except Exception as e:
            print(f"Error drawing object overlay: {e}")

    def get_frame_with_detection(self) -> np.ndarray:
        """Get current frame with detection overlay and enhanced error handling"""
        if not self.cap or not self.cap.isOpened() or not self.is_running:
            return None
            
        try:
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                return None
                
            self.current_frame = frame
            display_frame = self.current_frame.copy()
            
            # --- Detection ---
            person_detections = []
            
            if self.model is not None:
                try:
                    results = self.model(display_frame, conf=self.confidence_threshold, classes=[0]) # Class 0 is 'person'
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf[0])
                            person_detections.append({'box': (x1, y1, x2, y2), 'confidence': confidence, 'label': 'person'})
                except Exception as e:
                    print(f"Error during YOLO detection: {e}")
            else:
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                for (fx, fy, fw, fh) in faces:
                    person_detections.append({'box': (fx, fy, fx + fw, fy + fh), 'confidence': 0.0, 'label': 'person'})

            # --- Update Person Tracker ---
            self._update_trackers(person_detections, self.tracked_people, 'next_person_id')

            # --- Process tracked people ---
            self._decay_streaks()

            for p_id, p_data in self.tracked_people.items():
                if time.time() - p_data['last_seen'] > 0.5:
                    continue

                x1, y1, x2, y2 = p_data['box']
                # Clip coordinates
                h, w = display_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                person_roi = display_frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Recognize face within the person ROI
                employee_info = self.recognize_person_face(person_roi, full_frame=display_frame, person_box=(x1, y1, x2, y2))
                
                # Update streak
                confirmed = False
                tentative = False
                if employee_info and isinstance(employee_info, dict):
                    db_id = employee_info.get('db_id')
                    if db_id is not None:
                        confirmed = self._update_recognition_streak(db_id)
                        tentative = not confirmed
                        employee_info['tentative'] = tentative

                duration = time.time() - p_data['first_seen']

                self.draw_person_overlay(display_frame, x1, y1, x2, y2, p_data['confidence'], employee_info, duration=duration)

                # Log attendance
                if confirmed and employee_info:
                    db_id = employee_info.get('db_id')
                    now = time.time()
                    last_confirm_expiry = self.confirmed_ids.get(db_id, 0)
                    if now > last_confirm_expiry:
                        self.confirmed_ids[db_id] = now + 10.0
                        self.log_attendance(
                            employee_info['db_id'],
                            employee_info,
                            employee_info['confidence']
                        )

            # --- Process Objects ---
            objects_found = self.detect_objects_in_frame(display_frame)
            object_detections = []
            if objects_found:
                for obj_key, obj_val in objects_found.items():
                    object_detections.append({
                        'box': obj_val['box'],
                        'confidence': obj_val['confidence'],
                        'label': obj_val['name']
                    })

            self._update_trackers(object_detections, self.tracked_objects, 'next_object_id')

            for o_id, o_data in self.tracked_objects.items():
                if time.time() - o_data['last_seen'] > 0.5:
                    continue

                x1, y1, x2, y2 = o_data['box']
                duration = time.time() - o_data['first_seen']
                self.draw_object_overlay(display_frame, x1, y1, x2, y2, o_data['label'], o_data['confidence'], duration)
            
            if self.overlays_enabled:
                self.add_attendance_overlay(display_frame)
            
            self.cleanup_expired_actions()

            if self.show_gui_window:
                try:
                    h, w = display_frame.shape[:2]
                    if h > 720:
                        display_frame_resized = cv2.resize(display_frame, (int(w * 720 / h), 720))
                    else:
                        display_frame_resized = display_frame
                    cv2.imshow('CamCtrlX Live Detection', display_frame_resized)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.disable_gui()
                except Exception:
                    self.show_gui_window = False
            
            return display_frame
            
        except Exception as e:
            print(f"Error in get_frame_with_detection: {e}")
            return None
    
    def cleanup_expired_actions(self):
        try:
            current_time = datetime.now()
            expired_employees = []
            for employee_id, expire_time in self.action_display_time.items():
                if current_time > expire_time:
                    expired_employees.append(employee_id)
            for employee_id in expired_employees:
                self.current_actions.pop(employee_id, None)
                self.action_display_time.pop(employee_id, None)
        except Exception:
            pass

    def _decay_streaks(self):
        try:
            now = time.time()
            remove = []
            for db_id, data in list(self.recognition_streaks.items()):
                last = data.get('last_seen', 0)
                if now - last > self.confirmation_window:
                    remove.append(db_id)
            for db_id in remove:
                self.recognition_streaks.pop(db_id, None)
        except Exception:
            pass

    def _update_recognition_streak(self, db_id: int) -> bool:
        try:
            now = time.time()
            data = self.recognition_streaks.get(db_id)
            if data is None:
                self.recognition_streaks[db_id] = {'count': 1, 'last_seen': now}
                return False
            else:
                last_seen = data.get('last_seen', 0)
                if now - last_seen <= self.confirmation_window:
                    data['count'] = data.get('count', 0) + 1
                else:
                    data['count'] = 1
                data['last_seen'] = now
                self.recognition_streaks[db_id] = data
                if data['count'] >= self.confirmation_required:
                    return True
                return False
        except Exception:
            return False
    
    def add_attendance_overlay(self, frame):
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            cv2.putText(frame, "SMART ATTENDANCE SYSTEM", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Time: {current_time}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            face_status = "Face Recognition: ACTIVE" if len(self.employee_labels) > 0 else "Face Recognition: NO TRAINING DATA"
            face_color = (0, 255, 0) if len(self.employee_labels) > 0 else (0, 0, 255)
            cv2.putText(frame, face_status, 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
            
            active_count = len([action for action in self.current_actions.values() if action])
            cv2.putText(frame, f"Active Actions: {active_count} | Trained Employees: {len(self.employee_labels)}", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        except Exception:
            pass
    
    def draw_person_overlay(self, frame, x1, y1, x2, y2, confidence, employee_info, duration=None):
        try:
            tentative = False
            if isinstance(employee_info, dict) and employee_info.get('tentative'):
                tentative = True

            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            activity = self.detect_person_activity(person_roi) if person_roi.size > 0 else 'unknown'
            
            duration_str = ""
            if duration is not None:
                minutes = int(duration // 60)
                seconds = int(duration % 60)
                duration_str = f"Time: {minutes:02d}:{seconds:02d}"

            # Known and confirmed
            if employee_info and not tentative:
                box_color = (0, 255, 0)  # Green
                name_label = f"{employee_info.get('name', 'Unknown')}"

                cv2.putText(frame, name_label, (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                confidence_label = f"Confidence: {employee_info.get('confidence', 0):.1f}%"
                cv2.putText(frame, confidence_label, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                activity_label = f"Activity: {activity.title()}"
                cv2.putText(frame, activity_label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                nearby_objects = employee_info.get('nearby_objects', [])
                if nearby_objects:
                    objects_text = "Objects: " + ", ".join(nearby_objects[:2])
                    cv2.putText(frame, objects_text, (int(x1), int(y1)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)

                if duration_str:
                     cv2.putText(frame, duration_str, (int(x1), int(y1)+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Tentative
            elif employee_info and tentative:
                box_color = (0, 140, 255)  # Orange
                name_label = f"{employee_info.get('name', 'Unknown')} (tentative)"

                cv2.putText(frame, name_label, (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 3)

                confidence_label = f"Confidence: {employee_info.get('confidence', 0):.1f}%"
                cv2.putText(frame, confidence_label, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

                activity_label = f"Activity: {activity.title()}"
                cv2.putText(frame, activity_label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                
                if duration_str:
                     cv2.putText(frame, duration_str, (int(x1), int(y1)+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)

            # Unknown
            else:
                box_color = (0, 0, 255)  # Red
                unknown_label = "NOT RECOGNIZED"

                cv2.putText(frame, unknown_label, (int(x1), int(y1)-60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                confidence_label = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, confidence_label, (int(x1), int(y1)-35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                activity_label = f"Activity: {activity.title()}"
                cv2.putText(frame, activity_label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                nearby_objects = employee_info.get('nearby_objects', []) if employee_info else []
                if nearby_objects:
                    objects_text = "Objects: " + ", ".join(nearby_objects[:2])
                    cv2.putText(frame, objects_text, (int(x1), int(y1)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 2)

                if duration_str:
                     cv2.putText(frame, duration_str, (int(x1), int(y1)+35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 3)

        except Exception as e:
            print(f"Error drawing person overlay: {e}")
    
    def recognize_person_face(self, person_roi, full_frame=None, person_box=None) -> dict:
        """Recognize face in person region of interest with enhanced detection, activity and nearby objects"""
        try:
            if person_roi is None or person_roi.size == 0:
                self._last_face_found = False
                return None
            
            activity = self.detect_person_activity(person_roi)
            
            nearby_objects = []
            if YOLO_AVAILABLE and self.model is not None and full_frame is not None:
                try:
                    objects = self.detect_objects_in_frame(full_frame)
                    if person_box and objects:
                        px1, py1, px2, py2 = person_box
                        person_center_x = (px1 + px2) / 2
                        person_center_y = (py1 + py2) / 2
                        
                        for obj_key, obj_data in objects.items():
                            ox1, oy1, ox2, oy2 = obj_data['box']
                            obj_center_x = (ox1 + ox2) / 2
                            obj_center_y = (oy1 + oy2) / 2
                            
                            distance = ((person_center_x - obj_center_x)**2 + (person_center_y - obj_center_y)**2)**0.5
                            if distance < 300:
                                nearby_objects.append(obj_data['name'])
                        
                        nearby_objects = list(set(nearby_objects))[:3]
                except Exception as e:
                    print(f"Error detecting nearby objects: {e}")

            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_roi = clahe.apply(gray_roi)
                gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            except Exception:
                gray_roi = cv2.equalizeHist(gray_roi)

            faces = self.face_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(48, 48),
                maxSize=(400, 400)
            )

            if len(faces) == 0:
                self._last_face_found = False
                return None

            face = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = face

            padding = max(5, int(0.1 * min(fw, fh)))
            fx = max(0, fx - padding)
            fy = max(0, fy - padding)
            fw = min(gray_roi.shape[1] - fx, fw + 2 * padding)
            fh = min(gray_roi.shape[0] - fy, fh + 2 * padding)

            face_img = gray_roi[fy:fy + fh, fx:fx + fw]

            if face_img.size == 0:
                self._last_face_found = False
                return None

            face_img = cv2.resize(face_img, (100, 100))

            try:
                if fw < 50 or fh < 50:
                    self._last_face_found = False
                    return None
                aspect = float(fw) / float(fh) if fh != 0 else 0.0
                if aspect < 0.8 or aspect > 1.4:
                    self._last_face_found = False
                    return None
                if np.std(face_img) < 20.0:
                    self._last_face_found = False
                    return None
            except Exception:
                pass

            result = self.recognize_face(face_img)
            if result:
                employee_db_id, confidence = result
            else:
                employee_db_id, confidence = None, 0.0

            self._last_face_found = True

            color_crop = person_roi[fy:fy + fh, fx:fx + fw]

            is_known = False
            try:
                if employee_db_id is not None and employee_db_id in self.employee_labels:
                    if confidence >= self.min_similarity_percent:
                        is_known = True
            except Exception:
                is_known = False

            self.store_detected_face(face_img, employee_db_id, confidence, color_crop, is_known=is_known, activity=activity, nearby_objects=nearby_objects)

            if is_known:
                employee_data = self.employee_labels[employee_db_id].copy()
                employee_data['confidence'] = confidence
                employee_data['db_id'] = employee_db_id
                employee_data['activity'] = activity
                employee_data['nearby_objects'] = nearby_objects
                return employee_data
            else:
                return None

        except Exception as e:
            print(f"Error in recognition loop: {e}")
            self._last_face_found = False
            return None
    
    def store_detected_face(self, face_gray, employee_db_id, confidence, face_color, is_known=None, activity='unknown', nearby_objects=None):
        try:
            _, buffer = cv2.imencode('.jpg', face_color)
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            
            if is_known is None:
                is_known = employee_db_id is not None and employee_db_id in self.employee_labels
            name = "Unknown"
            
            if is_known:
                employee_data = self.employee_labels[employee_db_id]
                name = f"{employee_data.get('first_name', '')} {employee_data.get('last_name', '')}".strip()
                if not name:
                    name = employee_data.get('name', 'Unknown')
            
            detection_data = {
                'name': name,
                'confidence': confidence if confidence else 0,
                'image_data': face_base64,
                'is_known': is_known,
                'timestamp': time.time() * 1000,
                'activity': activity,
                'nearby_objects': nearby_objects if nearby_objects else []
            }
            
            self.recent_detections.append(detection_data)
            if len(self.recent_detections) > self.max_recent_detections:
                self.recent_detections.pop(0)
                
        except Exception:
            pass
    
    def test_detection(self, image_path: str) -> bool:
        """Test detection on a single image"""
        try:
            results = self.model(image_path, conf=self.confidence_threshold)
            return any(int(box.cls[0]) == 0 for result in results for box in result.boxes)
        except Exception:
            return False
    
    def enable_gui(self):
        """Enable the GUI window for live detection view."""
        self.show_gui_window = True
        print("GUI window enabled.")

    def disable_gui(self):
        """Disable and close the GUI window."""
        self.show_gui_window = False
        try:
            cv2.destroyWindow('CamCtrlX Live Detection')
            cv2.waitKey(1)
        except Exception:
            pass
