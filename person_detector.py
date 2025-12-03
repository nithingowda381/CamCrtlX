import cv2
import numpy as np
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False
import threading
import time
from typing import Optional, Tuple, List
import sqlite3
from datetime import datetime, timedelta
import os
import base64
import warnings
import pickle

# Suppress known resource_tracker semaphore leak warnings coming from loky/joblib at shutdown
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects to clean up at shutdown",
)
warnings.filterwarnings("ignore", category=UserWarning)  # Suppress other UserWarnings from loky

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
        self.pose_detector = None
        self.person_activities = {}  # Track activity for each detected person
        self._init_pose_detector()
        
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
        # This prevents false positives where the recognizer always returns the
        # best label even for poor matches. Higher = stricter.
        # Live video LBPH distances (160-200) map to 0-20% similarity
        # Training images LBPH distances (0) map to 100% similarity
        # So we need a VERY low threshold to accept matches in live video
        try:
            env_val = float(os.environ.get('MIN_SIMILARITY_PERCENT', '5'))
            # Ignore unreasonably high values from env (like 60 from .env)
            # Use 5% as absolute minimum for live video - LBPH distances 170-180 = 10-15%
            self.min_similarity_percent = max(5.0, min(env_val, 10.0))
        except Exception:
            self.min_similarity_percent = 5.0
        # Confirmation streaks: require the same label to be seen N times within
        # a short window before treating it as confirmed. Helps avoid random
        # single-frame false positives.
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

    def _load_deep_learning_model(self):
        """Load deep learning face recognition model if available"""
        try:
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

                    # Improved normalization: convert L2 distance to confidence where higher is better
                    # Lower distance = higher confidence, scale to 0-100 range
                    try:
                        # Use a more accurate scaling based on typical face distances
                        # Max possible L2 distance for 100x100 grayscale image is ~255*sqrt(10000) ≈ 25500
                        max_possible_dist = 255.0 * np.sqrt(100*100)  # ≈ 25500
                        # Higher confidence for lower distance
                        normalized_confidence = 100.0 - (best_score / max_possible_dist) * 100.0
                        # Clamp to reasonable range
                        normalized_confidence = max(0.0, min(100.0, normalized_confidence))
                    except Exception:
                        normalized_confidence = max(0.0, min(100.0, 100.0 - best_score))  # fallback

                    print(f"DummyRecognizer: best_label={best_label}, best_score={best_score:.2f}, normalized_confidence={normalized_confidence:.2f}")

                    # Return label and a normalized confidence (higher is better)
                    return best_label, float(normalized_confidence)

            self.face_recognizer = DummyRecognizer()

    def __del__(self):
        # Attempt to clean up resources to avoid leaked semaphores on shutdown
        try:
            if hasattr(self, 'cap') and self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            if hasattr(self, 'model') and self.model is not None:
                try:
                    # ultralytics YOLO has a close method in some versions
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
            # Handle different types of video sources
            if isinstance(stream_url, str) and stream_url.isdigit():
                # String that represents a camera index
                video_source = int(stream_url)
            elif isinstance(stream_url, int):
                # Already an integer (camera index)
                video_source = stream_url
            else:
                # String URL (RTSP, HTTP, file path, etc.)
                video_source = stream_url
            
            print(f"Attempting to connect to video source: {video_source}")
            
            # For webcam sources, try multiple backends and camera indices
            if isinstance(video_source, int):
                # List of backends to try in order of preference for Windows
                backends = [
                    cv2.CAP_MSMF,       # Microsoft Media Foundation (works on this system)
                    cv2.CAP_DSHOW,      # DirectShow
                    cv2.CAP_ANY         # Any available backend
                ]
                
                # Try different camera indices if source is 0
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
                                # Add a small delay to let camera initialize
                                import time
                                time.sleep(0.5)

                                # Set properties for better performance
                                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                self.cap.set(cv2.CAP_PROP_FPS, 30)
                                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for real-time

                                # Test if we can read a frame
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
                # For URL/file sources, use default backend
                self.cap = cv2.VideoCapture(video_source)
                
                # Test if source is accessible
                if not self.cap.isOpened():
                    print(f"Failed to open video source: {video_source}")
                    return False
                
                # Try to read a frame to verify the connection
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
    
    def _init_pose_detector(self):
        """Initialize activity detector (OpenCV-based without external dependencies)"""
        try:
            # Load hand cascade for detecting if person is using phone
            hand_cascade_path = cv2.data.haarcascades + 'lbpcascade_anon_face.xml'
            self.hand_cascade = cv2.CascadeClassifier(hand_cascade_path)
            print("✓ Activity detector initialized (using OpenCV)")
        except Exception as e:
            print(f"Warning: Could not initialize activity detector: {e}")
            self.hand_cascade = None
    
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
            face_center_x = fx + fw / 2
            
            # Analyze ROI structure to determine activity
            # Check if face is in upper part of ROI (looking down/at phone)
            face_position_ratio = face_center_y / h
            
            # Check for bright/illuminated areas (phone screen effect)
            lower_roi = gray_roi[int(fy + fh):, :]
            if len(lower_roi) > 0 and lower_roi.size > 0:
                lower_brightness = np.mean(lower_roi)
            else:
                lower_brightness = 0
            
            upper_roi = gray_roi[:int(fy), :]
            if len(upper_roi) > 0 and upper_roi.size > 0:
                upper_brightness = np.mean(upper_roi)
            else:
                upper_brightness = 0
            
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
                # Check for motion blur or body visibility
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
                        
                        # Debug: Log all detected objects
                        print(f"[DEBUG] Detected: {class_name} (conf: {confidence:.2f})")
                        
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
                            # Debug: Log matched objects
                            print(f"[DEBUG] MATCHED: {class_name}")
                        else:
                            print(f"[DEBUG] NO MATCH: {class_name} (not in {self.object_classes})")
            
            # Log detection summary
            if detected_count == 0:
                print(f"[DEBUG] No objects detected in frame")
            
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
                    # Re-initialize the dummy recognizer
                    self.face_recognizer.__init__()


        except Exception as e:
            print(f"An error occurred in load_employee_faces: {e}")
            import traceback
            traceback.print_exc()

    def _load_recognition_threshold(self) -> float:
        """Load recognition threshold from saved trained model config if available.

        Returns a threshold value where LOWER recognizer confidence is better (LBPH).
        Default to 70 (compatible with FaceRecognitionDB defaults).
        """
        try:
            model_yaml = 'static/trained_face_model.yml'
            if os.path.exists(model_yaml):
                # Prefer yaml if installed, otherwise fallback to a simple regex parse
                try:
                    # import yaml via importlib so static analyzers don't require it
                    import importlib
                    yaml_spec = importlib.util.find_spec('yaml')
                    yaml = importlib.import_module('yaml') if yaml_spec is not None else None

                    if yaml is not None:
                        with open(model_yaml, 'r') as f:
                            data = yaml.safe_load(f)
                            cfg = data.get('opencv_lbphfaces', {})
                            thr = cfg.get('threshold')
                            if thr is not None:
                                try:
                                    return float(thr)
                                except Exception:
                                    pass
                    else:
                        # simple regex fallback to find a line like 'threshold: 80.0'
                        import re
                        with open(model_yaml, 'r') as f:
                            content = f.read()
                        m = re.search(r"threshold\s*:\s*([0-9]+(?:\.[0-9]+)?)", content)
                        if m:
                            try:
                                return float(m.group(1))
                            except Exception:
                                pass
                except Exception:
                    # file read / parse error - ignore and fall back
                    pass
        except Exception:
            pass
        # sensible default
        return 70.0
    
    def recognize_face(self, face_img) -> Tuple[Optional[int], float]:
        """Recognize face and return employee DB id and confidence percent.

        Uses deep learning model if available, otherwise falls back to LBPH.
        """
        try:
            # If we don't have any labeled employees loaded, bail out early
            if len(self.employee_labels) == 0:
                return None, 0.0

            # Use deep learning model if available
            if self.deep_learning_model is not None:
                return self._recognize_face_deep_learning(face_img)
            else:
                return self._recognize_face_lbph(face_img)

        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, 0.0

    def _recognize_face_deep_learning(self, face_img) -> Tuple[Optional[int], float]:
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
                # Convert grayscale to RGB by duplicating channels
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

            # Convert distance to confidence (lower distance = higher confidence)
            # Typical good match distance is < 0.6, bad match > 0.6
            max_distance = 1.0  # Empirical threshold
            confidence_percent = max(0.0, 100.0 - (closest_distance / max_distance) * 100.0)

            print(f"Deep learning recognition: label={predicted_label}, distance={closest_distance:.3f}, confidence={confidence_percent:.2f}%")

            # Only return match if confidence is above threshold
            if confidence_percent >= self.min_similarity_percent:
                return int(predicted_label), confidence_percent
            else:
                print(f"Rejected match: confidence {confidence_percent:.2f}% < {self.min_similarity_percent}%")
                return None, confidence_percent

        except Exception as e:
            print(f"Error in deep learning recognition: {e}")
            return None, 0.0

    def _recognize_face_lbph(self, face_img) -> Tuple[Optional[int], float]:
        """Recognize face using traditional LBPH method"""
        try:
            # Determine threshold (lower is better for LBPH/predict confidence)
            threshold = self._load_recognition_threshold()

            try:
                label, confidence = self.face_recognizer.predict(face_img)
            except Exception as e:
                print(f"Recognizer predict error: {e}")
                return None, 0.0

            # Log raw prediction for debugging
            print(f"LBPH predict -> label: {label}, confidence: {confidence:.2f}, threshold: {threshold:.2f}, min_similarity: {self.min_similarity_percent:.2f}")

            # Handle confidence mapping based on recognizer type
            if hasattr(self.face_recognizer, 'predict') and 'DummyRecognizer' in str(type(self.face_recognizer)):
                # DummyRecognizer returns confidence where higher is better (0-100)
                confidence_percent = float(confidence)
            else:
                # LBPH: lower numeric 'confidence' is better. Convert to higher is better percentage.
                # A good match has a low distance (e.g., < 50). A bad match has a high distance (e.g., > 200).
                # We will map this so that a distance of 0 is 100% confidence, and a distance of `max_distance_for_scaling` is 0% confidence.
                # This creates a more linear and predictable similarity score.
                try:
                    # Map the distance to a 0-100 similarity score.
                    # A lower distance score from predict() is a better match.
                    distance = float(confidence)

                    # If distance is 0, it's a perfect match.
                    # If distance is high, it's a poor match.
                    # Empirically, LBPH distances for same person typically < 50-100
                    # For different persons or poor quality: 150-500+
                    # Using 200 as the threshold for 0% similarity (more generous than 150)
                    max_distance_for_scaling = 200.0

                    similarity = 100.0 - (distance / max_distance_for_scaling) * 100.0

                    # Clamp the value between 0 and 100.
                    confidence_percent = max(0.0, min(100.0, similarity))

                except Exception:
                    confidence_percent = 0.0

            # confidence_percent is already set above

            # If USE_THRESHOLD env var is set, fall back to the previous rejection behavior
            if os.environ.get('USE_THRESHOLD', '0') == '1':
                if confidence < threshold:
                    return int(label), confidence_percent
                else:
                    # Not a good match per legacy threshold
                    if self.allow_low_confidence:
                        try:
                            return int(label), 0.0
                        except Exception:
                            return None, 0.0
                    return None, 0.0

            # Default behavior: always return the best label and the computed percent
            try:
                return int(label), confidence_percent
            except Exception:
                return None, 0.0
        except Exception as e:
            print(f"Error in LBPH face recognition: {e}")
            return None, 0.0
    
    def log_attendance(self, employee_db_id: int, employee_data: dict, confidence: float):
        """Log attendance for recognized employee and return action performed"""
        try:
            current_time = datetime.now()
            action_performed = None
            
            print(f"Attempting to log attendance for employee: {employee_data.get('name')} (ID: {employee_data.get('employee_id')})")
            
            # Check cooldown period
            if employee_db_id in self.last_recognition_time:
                time_diff = (current_time - self.last_recognition_time[employee_db_id]).total_seconds()
                if time_diff < self.recognition_cooldown:
                    # Return the current stored action if within cooldown
                    print(f"Within cooldown period ({time_diff:.1f}s < {self.recognition_cooldown}s)")
                    return self.current_actions.get(employee_db_id, None)
            
            # Update last recognition time
            self.last_recognition_time[employee_db_id] = current_time
            
            # Log to attendance database
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
            print(f"Active session check for {employee_data['employee_id']} on {today}: {active_session}")
            
            if active_session:
                # Employee has active session - this could be checkout
                session_id, start_time, end_time = active_session
                start_datetime = datetime.fromisoformat(start_time)
                work_duration = (current_time - start_datetime).total_seconds() / 3600  # hours
                
                print(f"Found active session. Work duration: {work_duration:.2f} hours")
                
                # Allow checkout after minimum 5 minutes (more reasonable for testing)
                if work_duration >= 0.083:  # 5 minutes = 0.083 hours
                    cursor.execute("""
                        UPDATE work_log SET end_time = ?, hours = ? WHERE id = ?
                    """, (current_time.isoformat(), round(work_duration, 2), session_id))
                    
                    action_performed = "CHECK-OUT"
                    print(f"Checkout logged for {employee_data['name']} - Duration: {work_duration:.2f} hours")
                else:
                    action_performed = "CHECKED-IN"  # Already checked in, show current status
                    print(f"Employee already checked in (duration too short: {work_duration:.2f}h)")
            else:
                # No active session - this is check-in
                cursor.execute("""
                    INSERT INTO work_log (employee_id, start_time, date, total_seconds) 
                    VALUES (?, ?, ?, 0)
                """, (employee_data['employee_id'], current_time.isoformat(), today))
                
                action_performed = "CHECK-IN"
                print(f"Check-in logged for {employee_data['name']} at {current_time}")
            
            # Store the action and set display timeout
            self.current_actions[employee_db_id] = action_performed
            self.action_display_time[employee_db_id] = current_time + timedelta(seconds=5)
            
            conn.commit()
            conn.close()
            
            print(f"Attendance logged successfully. Action: {action_performed}")
            return action_performed
            
        except Exception as e:
            print(f"Error logging attendance: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def stop_stream(self):
        """Stop the video stream"""
        self.is_running = False
        if self.cap:
            self.cap.release()
    
    def detect_person(self) -> Tuple[bool, Optional[float]]:
        """Detect if person is present in current frame"""
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

                # Check for person class (class 0 in COCO dataset)
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

        # Fallback: detect faces in the frame using Haar cascade - treat any face as a person
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Be strict at full-frame detection to reduce false positives
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.12,
                minNeighbors=6,
                minSize=(48, 48)
            )
            if len(faces) > 0:
                # Estimate confidence as 0.6 for faces (heuristic)
                return True, 0.6
            return False, None
        except Exception as e:
            print(f"Fallback face detection error: {e}")
            return False, None
    
    def get_frame_with_detection(self) -> Optional[np.ndarray]:
        """Get current frame with detection overlay and enhanced error handling"""
        if not self.cap or not self.cap.isOpened() or not self.is_running:
            print("Camera not available or not running")
            return None
            
        try:
            # Read a new frame with timeout handling
            ret, frame = self.cap.read()
            if not ret or frame is None or frame.size == 0:
                print("Failed to read frame from camera")
                return None
                
            self.current_frame = frame
            
            # Create a copy of the frame for drawing
            display_frame = self.current_frame.copy()
            
            # --- Centralized Detection and Recognition Logic ---
            detected_people = []
            
            # Use YOLO if available, otherwise fall back to Haar Cascade
            if self.model is not None:
                try:
                    results = self.model(display_frame, conf=self.confidence_threshold, classes=[0]) # Class 0 is 'person'
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            confidence = float(box.conf[0])
                            detected_people.append({'box': (x1, y1, x2, y2), 'confidence': confidence})
                except Exception as e:
                    print(f"Error during YOLO detection: {e}")
            else:
                # Fallback to Haar Cascade for face detection if YOLO is not used for person detection
                gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
                for (fx, fy, fw, fh) in faces:
                    # Treat the detected face area as a "person" box
                    detected_people.append({'box': (fx, fy, fx + fw, fy + fh), 'confidence': 0.0})

            # --- Process all detected people ---
            self._decay_streaks() # Decay all streaks once per frame

            for person in detected_people:
                x1, y1, x2, y2 = person['box']
                person_roi = display_frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Recognize face within the person ROI, passing full frame and person box for object detection
                employee_info = self.recognize_person_face(person_roi, full_frame=display_frame, person_box=(x1, y1, x2, y2))
                
                # Update and check confirmation streak
                confirmed = False
                tentative = False
                if employee_info and isinstance(employee_info, dict):
                    db_id = employee_info.get('db_id')
                    if db_id is not None:
                        confirmed = self._update_recognition_streak(db_id)
                        tentative = not confirmed
                        employee_info['tentative'] = tentative

                # Draw overlay for this person
                self.draw_person_overlay(display_frame, x1, y1, x2, y2, person['confidence'], employee_info)

                # Log attendance only when confirmed
                if confirmed and employee_info:
                    db_id = employee_info.get('db_id')
                    now = time.time()
                    last_confirm_expiry = self.confirmed_ids.get(db_id, 0)
                    if now > last_confirm_expiry:
                        self.confirmed_ids[db_id] = now + 10.0 # 10-second cooldown for logging
                        self.log_attendance(
                            employee_info['db_id'],
                            employee_info,
                            employee_info['confidence']
                        )
            
            # Add general status overlay
            if self.overlays_enabled:
                self.add_attendance_overlay(display_frame)
            
            # Clean up expired action messages
            self.cleanup_expired_actions()

            # Show GUI window if enabled
            if self.show_gui_window:
                try:
                    # Resize for a more manageable window size
                    h, w = display_frame.shape[:2]
                    if h > 720:
                        display_frame_resized = cv2.resize(display_frame, (int(w * 720 / h), 720))
                    else:
                        display_frame_resized = display_frame
                    cv2.imshow('CamCtrlX Live Detection', display_frame_resized)
                    # Check for 'q' key press to close the window
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.disable_gui()

                except Exception as e:
                    # Catch errors if the display environment is not available (e.g., on a headless server)
                    print(f"Could not display GUI window: {e}")
                    self.show_gui_window = False # Disable to prevent further errors
            
            return display_frame
            
        except Exception as e:
            print(f"Error in get_frame_with_detection: {e}")
            return None
    
    def cleanup_expired_actions(self):
        """Clean up expired action displays"""
        try:
            current_time = datetime.now()
            expired_employees = []
            
            for employee_id, expire_time in self.action_display_time.items():
                if current_time > expire_time:
                    expired_employees.append(employee_id)
            
            for employee_id in expired_employees:
                if employee_id in self.current_actions:
                    del self.current_actions[employee_id]
                if employee_id in self.action_display_time:
                    del self.action_display_time[employee_id]
                    
            if expired_employees:
                print(f"Cleaned up {len(expired_employees)} expired actions")
                
        except Exception as e:
            print(f"Error cleaning up actions: {e}")

    def _decay_streaks(self):
        """Remove recognition streaks that have not been updated within the confirmation window."""
        try:
            now = time.time()
            remove = []
            for db_id, data in list(self.recognition_streaks.items()):
                last = data.get('last_seen', 0)
                if now - last > self.confirmation_window:
                    remove.append(db_id)
            for db_id in remove:
                try:
                    del self.recognition_streaks[db_id]
                except Exception:
                    pass
        except Exception as e:
            print(f"Error decaying streaks: {e}")

    def _update_recognition_streak(self, db_id: int) -> bool:
        """Update the streak counter for a given db_id. Returns True when the
        streak reaches the confirmation_required threshold (i.e., confirmed).
        """
        try:
            now = time.time()
            data = self.recognition_streaks.get(db_id)
            if data is None:
                # start a new streak
                self.recognition_streaks[db_id] = {'count': 1, 'last_seen': now}
                return False
            else:
                last_seen = data.get('last_seen', 0)
                if now - last_seen <= self.confirmation_window:
                    data['count'] = data.get('count', 0) + 1
                else:
                    data['count'] = 1
                data['last_seen'] = now
                # update back
                self.recognition_streaks[db_id] = data
                if data['count'] >= self.confirmation_required:
                    return True
                return False
        except Exception as e:
            print(f"Error updating recognition streak: {e}")
            return False
    
    def add_attendance_overlay(self, frame):
        """Add enhanced attendance status overlay to the frame"""
        try:
            # Add a semi-transparent background for status
            overlay = frame.copy()
            h, w = frame.shape[:2]
            
            # Status background - make it larger for more info
            cv2.rectangle(overlay, (10, 10), (450, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Title
            cv2.putText(frame, "SMART ATTENDANCE SYSTEM", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, f"Time: {current_time}", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Face detection status
            face_status = "Face Recognition: ACTIVE" if len(self.employee_labels) > 0 else "Face Recognition: NO TRAINING DATA"
            face_color = (0, 255, 0) if len(self.employee_labels) > 0 else (0, 0, 255)
            cv2.putText(frame, face_status, 
                       (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 1)
            
            # Active recognitions count
            active_count = len([action for action in self.current_actions.values() if action])
            cv2.putText(frame, f"Active Actions: {active_count} | Trained Employees: {len(self.employee_labels)}", 
                       (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        except Exception as e:
            print(f"Error adding overlay: {e}")
    
    def draw_person_overlay(self, frame, x1, y1, x2, y2, confidence, employee_info):
        """Draw person overlay with name, activity and status on the video frame"""
        try:
            tentative = False
            if isinstance(employee_info, dict) and employee_info.get('tentative'):
                tentative = True

            # Detect activity from the ROI
            person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
            activity = self.detect_person_activity(person_roi) if person_roi.size > 0 else 'unknown'
            
            # Activity emoji mapping
            activity_emoji = {
                'standing': '🧑‍💼',
                'sitting': '🪑',
                'phone': '📱',
                'walking': '🚶',
                'unknown': '❓'
            }
            activity_display = activity_emoji.get(activity, '❓')

            # Recognized and confirmed
            if employee_info and not tentative:
                box_color = (0, 255, 0)  # Green
                name_label = f"{employee_info.get('name', 'Unknown')}"
                cv2.putText(frame, name_label,
                            (int(x1), int(y1)-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3)
                (name_width, name_height), _ = cv2.getTextSize(name_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.rectangle(frame,
                              (int(x1), int(y1)-70),
                              (int(x1) + name_width, int(y1)-60 + name_height),
                              (0, 0, 0), -1)
                cv2.putText(frame, name_label,
                            (int(x1), int(y1)-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 255, 0), 3)
                confidence_label = f"Confidence: {employee_info.get('confidence', 0):.1f}%"
                cv2.putText(frame, confidence_label,
                            (int(x1), int(y1)-35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                # Activity label
                activity_label = f"Activity: {activity.title()}"
                cv2.putText(frame, activity_label,
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                
                # Objects label (if any detected)
                nearby_objects = employee_info.get('nearby_objects', [])
                if nearby_objects:
                    objects_text = "Objects: " + ", ".join(nearby_objects[:2])  # Show first 2 objects
                    cv2.putText(frame, objects_text,
                                (int(x1), int(y1)+15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 200, 200), 2)

            # Tentative (not yet confirmed)
            elif employee_info and tentative:
                box_color = (0, 140, 255)  # Orange
                name_label = f"{employee_info.get('name', 'Unknown')} (tentative)"
                cv2.putText(frame, name_label,
                            (int(x1), int(y1)-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 140, 255), 3)
                confidence_label = f"Confidence: {employee_info.get('confidence', 0):.1f}%"
                cv2.putText(frame, confidence_label,
                            (int(x1), int(y1)-35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 140, 255), 2)
                # Activity label
                activity_label = f"Activity: {activity.title()}"
                cv2.putText(frame, activity_label,
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 140, 255), 2)
                
                # Objects label (if any detected)
                nearby_objects = employee_info.get('nearby_objects', [])
                if nearby_objects:
                    objects_text = "Objects: " + ", ".join(nearby_objects[:2])  # Show first 2 objects
                    cv2.putText(frame, objects_text,
                                (int(x1), int(y1)+15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 200, 200), 2)

            # Unknown
            else:
                box_color = (0, 0, 255)  # Red
                unknown_label = "NOT RECOGNIZED"
                cv2.putText(frame, unknown_label,
                            (int(x1), int(y1)-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3)
                (unknown_width, unknown_height), _ = cv2.getTextSize(unknown_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
                cv2.rectangle(frame,
                              (int(x1), int(y1)-70),
                              (int(x1) + unknown_width, int(y1)-60 + unknown_height),
                              (0, 0, 0), -1)
                cv2.putText(frame, unknown_label,
                            (int(x1), int(y1)-60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 255), 3)
                confidence_label = f"Confidence: {confidence:.2f}"
                cv2.putText(frame, confidence_label,
                            (int(x1), int(y1)-35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                # Activity label for unknown
                activity_label = f"Activity: {activity.title()}"
                cv2.putText(frame, activity_label,
                            (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)
                
                # Objects label for unknown (if any detected)
                nearby_objects = employee_info.get('nearby_objects', []) if employee_info else []
                if nearby_objects:
                    objects_text = "Objects: " + ", ".join(nearby_objects[:2])  # Show first 2 objects
                    cv2.putText(frame, objects_text,
                                (int(x1), int(y1)+15),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 150, 255), 2)

            # Draw rectangle around person with appropriate color
            cv2.rectangle(frame,
                          (int(x1), int(y1)),
                          (int(x2), int(y2)),
                          box_color, 3)

        except Exception as e:
            print(f"Error drawing person overlay: {e}")
    
    def recognize_person_face(self, person_roi, full_frame=None, person_box=None) -> Optional[dict]:
        """Recognize face in person region of interest with enhanced detection, activity and nearby objects"""
        try:
            if person_roi is None or person_roi.size == 0:
                self._last_face_found = False
                return None
            
            # Detect activity early so we can include it in the return data
            activity = self.detect_person_activity(person_roi)
            
            # Detect nearby objects from the full frame (not just the small ROI)
            nearby_objects = []
            if YOLO_AVAILABLE and self.model is not None and full_frame is not None:
                try:
                    # Detect objects in full frame only once per frame
                    objects = self.detect_objects_in_frame(full_frame)
                    # Filter objects that are near this person's bounding box
                    if person_box and objects:
                        px1, py1, px2, py2 = person_box
                        person_center_x = (px1 + px2) / 2
                        person_center_y = (py1 + py2) / 2
                        
                        for obj_key, obj_data in objects.items():
                            ox1, oy1, ox2, oy2 = obj_data['box']
                            obj_center_x = (ox1 + ox2) / 2
                            obj_center_y = (oy1 + oy2) / 2
                            
                            # Check if object is within 300 pixels of person (nearby)
                            distance = ((person_center_x - obj_center_x)**2 + (person_center_y - obj_center_y)**2)**0.5
                            if distance < 300:
                                nearby_objects.append(obj_data['name'])
                        
                        # Remove duplicates and limit to 3 objects
                        nearby_objects = list(set(nearby_objects))[:3]
                except Exception as e:
                    print(f"Error detecting nearby objects: {e}")

            # Convert to grayscale for face detection
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)

            # Enhanced preprocessing: CLAHE for better lighting robustness and Gaussian blur to reduce noise
            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_roi = clahe.apply(gray_roi)
                gray_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
            except Exception:
                gray_roi = cv2.equalizeHist(gray_roi)

            # Detect faces with multiple scale parameters for better detection
            # Be stricter here to reduce false positives
            faces = self.face_cascade.detectMultiScale(
                gray_roi,
                scaleFactor=1.05,
                minNeighbors=5,
                minSize=(48, 48),
                maxSize=(400, 400)
            )

            print(f"Face detection: Found {len(faces)} faces in person ROI")

            if len(faces) == 0:
                print("No faces detected in person ROI")
                self._last_face_found = False
                return None

            # Use the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            fx, fy, fw, fh = face

            print(f"Processing face at coordinates: ({fx}, {fy}, {fw}, {fh})")

            # Extract face region with padding proportional to face size
            padding = max(5, int(0.1 * min(fw, fh)))
            fx = max(0, fx - padding)
            fy = max(0, fy - padding)
            fw = min(gray_roi.shape[1] - fx, fw + 2 * padding)
            fh = min(gray_roi.shape[0] - fy, fh + 2 * padding)

            face_img = gray_roi[fy:fy + fh, fx:fx + fw]

            # Ensure face image is valid
            if face_img.size == 0:
                print("Invalid face image extracted")
                self._last_face_found = False
                return None

            # Resize face for recognition
            face_img = cv2.resize(face_img, (100, 100))

            # Reject faces that are too small or have unlikely aspect ratios (likely false positives)
            try:
                if fw < 70 or fh < 70:
                    print(f"Rejected face due to small size: ({fw}x{fh})")
                    self._last_face_found = False
                    return None
                aspect = float(fw) / float(fh) if fh != 0 else 0.0
                if aspect < 0.8 or aspect > 1.4:
                    print(f"Rejected face due to unusual aspect ratio: {aspect:.2f}")
                    self._last_face_found = False
                    return None

                # Reject if face crop has very low variance (blank/textureless region)
                if np.std(face_img) < 20.0:
                    print(f"Rejected face due to low variance: std={np.std(face_img):.2f}")
                    self._last_face_found = False
                    return None
            except Exception:
                pass

            # Recognize face
            employee_db_id, confidence = self.recognize_face(face_img)

            print(f"Face recognition result: employee_db_id={employee_db_id}, confidence={confidence:.2f}")

            # Indicate that a face was found in this ROI
            self._last_face_found = True

            # Store face data for live display (use color crop)
            color_crop = person_roi[fy:fy + fh, fx:fx + fw]

            # Decide whether this prediction is considered a known match
            is_known = False
            try:
                # If recognizer returned None label treat as unknown
                if employee_db_id is not None and employee_db_id in self.employee_labels:
                    # Compare against minimum similarity percent
                    if confidence >= self.min_similarity_percent:
                        is_known = True
                        print(f"Accepted match for employee {employee_db_id} with confidence {confidence:.2f}% >= {self.min_similarity_percent}%")
                    else:
                        print(f"Rejected match for employee {employee_db_id} with confidence {confidence:.2f}% < {self.min_similarity_percent}%")
            except Exception:
                is_known = False

            # Store detection with explicit known/unknown flag and activity
            self.store_detected_face(face_img, employee_db_id, confidence, color_crop, is_known=is_known, activity=activity, nearby_objects=nearby_objects)
            if is_known:
                employee_data = self.employee_labels[employee_db_id].copy()
                employee_data['confidence'] = confidence
                employee_data['db_id'] = employee_db_id
                employee_data['activity'] = activity
                employee_data['nearby_objects'] = nearby_objects
                objects_str = ", ".join(nearby_objects) if nearby_objects else "none"
                print(f"Employee recognized: {employee_data.get('name', 'Unknown')} (confidence: {confidence:.2f}%, activity: {activity}, objects: {objects_str})")
                return employee_data
            else:
                print(f"Prediction for label {employee_db_id} below similarity threshold ({confidence:.2f}% < {self.min_similarity_percent}). Treating as unknown")
                return None

        except Exception as e:
            print(f"Error in face recognition: {e}")
            self._last_face_found = False
            return None
    
    def store_detected_face(self, face_gray, employee_db_id, confidence, face_color, is_known=None, activity='unknown', nearby_objects=None):
        """Store detected face data for live display

        is_known: optional boolean to explicitly mark whether the detection is a
        known employee (True) or not (False). If None, the function falls back
        to checking `employee_db_id in self.employee_labels`.
        """
        try:
            import base64
            
            # Convert face image to base64 for frontend display
            _, buffer = cv2.imencode('.jpg', face_color)
            face_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Determine if face is known or unknown
            if is_known is None:
                is_known = employee_db_id is not None and employee_db_id in self.employee_labels
            name = "Unknown"
            
            if is_known:
                employee_data = self.employee_labels[employee_db_id]
                name = f"{employee_data.get('first_name', '')} {employee_data.get('last_name', '')}".strip()
                if not name:
                    name = employee_data.get('name', 'Unknown')
            
            # Create detection data
            detection_data = {
                'name': name,
                'confidence': confidence if confidence else 0,
                'image_data': face_base64,
                'is_known': is_known,
                'timestamp': time.time() * 1000,  # JavaScript timestamp
                'activity': activity,
                'nearby_objects': nearby_objects if nearby_objects else []
            }
            
            # Add to recent detections
            self.recent_detections.append(detection_data)
            
            # Keep only the most recent detections
            if len(self.recent_detections) > self.max_recent_detections:
                self.recent_detections.pop(0)
            
            print(f"Face stored for display: {name} (known: {is_known}, confidence: {confidence:.2f})")
                
        except Exception as e:
            print(f"Error storing detected face: {e}")
    
    def test_detection(self, image_path: str) -> bool:
        """Test detection on a single image"""
        try:
            results = self.model(image_path, conf=self.confidence_threshold)
            return any(int(box.cls[0]) == 0 for result in results for box in result.boxes)
        except Exception as e:
            print(f"Error testing detection: {e}")
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
            cv2.waitKey(1) # Process window close event
            print("GUI window disabled and destroyed.")
        except Exception as e:
            print(f"Error destroying GUI window: {e}")
