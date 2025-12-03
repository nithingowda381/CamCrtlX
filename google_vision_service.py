import os
import cv2
import numpy as np
from config import GOOGLE_APPLICATION_CREDENTIALS
import threading
import time
from typing import List, Dict, Any

# Try to import Google Vision API
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    print("Google Cloud Vision not installed. AI features disabled.")
    GOOGLE_VISION_AVAILABLE = False
    vision = None

# Set credentials if available
if GOOGLE_APPLICATION_CREDENTIALS and os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GOOGLE_APPLICATION_CREDENTIALS
else:
    print(f"Warning: Google Vision credentials not found at {GOOGLE_APPLICATION_CREDENTIALS}")
    print("AI video analysis features will be disabled.")

class GoogleVisionService:
    def __init__(self):
        if not GOOGLE_VISION_AVAILABLE:
            self.api_available = False
            self.client = None
            return

        try:
            self.client = vision.ImageAnnotatorClient()
            self.api_available = True
        except Exception as e:
            print(f"Warning: Google Vision API not available: {e}")
            print("AI video analysis features will be disabled.")
            self.client = None
            self.api_available = False

    def get_face_embedding(self, image_content):
        """
        Detects faces in an image and returns the first face's embedding.
        image_content: The binary content of the image.
        """
        image = vision.Image(content=image_content)
        response = self.client.face_detection(image=image)
        face_annotations = response.face_annotations

        if not face_annotations:
            return None

        # For simplicity, we're taking the first detected face's embedding.
        # In a real-world scenario, you might want to handle multiple faces.
        embedding = face_annotations[0].face_embedding
        return np.array(embedding)

    def compare_faces(self, embedding1, embedding2, threshold=0.7):
        """
        Compares two face embeddings using cosine similarity.
        Returns True if they are similar enough, False otherwise.
        """
        if embedding1 is None or embedding2 is None:
            return False

        # Cosine similarity calculation
        cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

        return cosine_similarity >= threshold

    def detect_objects(self, image_content, max_results=10):
        """
        Detects objects in an image using Google Vision API.
        Returns a list of detected objects with their confidence scores.
        """
        try:
            image = vision.Image(content=image_content)
            response = self.client.object_localization(image=image)
            objects = response.localized_object_annotations

            detected_objects = []
            for obj in objects[:max_results]:
                detected_objects.append({
                    'name': obj.name,
                    'confidence': obj.score,
                    'bounding_box': {
                        'vertices': [
                            {'x': vertex.x, 'y': vertex.y}
                            for vertex in obj.bounding_poly.normalized_vertices
                        ]
                    }
                })

            return detected_objects
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return []

    def detect_labels(self, image_content, max_results=10):
        """
        Detects labels (objects, scenes, actions) in an image.
        Returns a list of labels with confidence scores.
        """
        try:
            image = vision.Image(content=image_content)
            response = self.client.label_detection(image=image)
            labels = response.label_annotations

            detected_labels = []
            for label in labels[:max_results]:
                detected_labels.append({
                    'description': label.description,
                    'confidence': label.score,
                    'mid': label.mid
                })

            return detected_labels
        except Exception as e:
            print(f"Error detecting labels: {e}")
            return []

    def analyze_image(self, image_content):
        """
        Comprehensive image analysis including objects, labels, and text.
        Returns a dictionary with all detection results.
        """
        try:
            results = {
                'objects': self.detect_objects(image_content),
                'labels': self.detect_labels(image_content),
                'faces': len(self.client.face_detection(image=vision.Image(content=image_content)).face_annotations),
                'timestamp': str(np.datetime64('now'))
            }
            return results
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                'objects': [],
                'labels': [],
                'faces': 0,
                'error': str(e),
                'timestamp': str(np.datetime64('now'))
            }

class VideoAnalysisService:
    def __init__(self):
        try:
            self.vision_service = GoogleVisionService()
            self.api_available = True
        except Exception as e:
            print(f"Warning: Google Vision API not available: {e}")
            self.vision_service = None
            self.api_available = False

        self.analysis_results = []
        self.is_analyzing = False
        self.current_video_path = None

    def analyze_video(self, video_path: str, frame_interval: int = 30, max_frames: int = 50) -> Dict[str, Any]:
        """
        Analyze a video file by processing frames at regular intervals.
        Returns comprehensive analysis results.
        """
        if not self.api_available:
            return {
                'error': 'Google Vision API not configured. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable.',
                'api_available': False
            }

        self.is_analyzing = True
        self.current_video_path = video_path
        self.analysis_results = []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {'error': 'Could not open video file'}

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            frame_count = 0
            processed_frames = 0
            analysis_summary = {
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'processed_frames': 0,
                'objects_detected': {},
                'labels_detected': {},
                'faces_detected': 0,
                'frames': []
            }

            while cap.isOpened() and processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame
                if frame_count % frame_interval == 0:
                    try:
                        # Convert frame to JPEG
                        success, buffer = cv2.imencode('.jpg', frame)
                        if success:
                            frame_data = buffer.tobytes()

                            # Analyze the frame
                            analysis = self.vision_service.analyze_image(frame_data)
                            analysis['frame_number'] = frame_count
                            analysis['timestamp'] = frame_count / fps if fps > 0 else frame_count

                            # Update summary statistics
                            for obj in analysis.get('objects', []):
                                obj_name = obj['name']
                                analysis_summary['objects_detected'][obj_name] = analysis_summary['objects_detected'].get(obj_name, 0) + 1

                            for label in analysis.get('labels', []):
                                label_desc = label['description']
                                analysis_summary['labels_detected'][label_desc] = analysis_summary['labels_detected'].get(label_desc, 0) + 1

                            analysis_summary['faces_detected'] += analysis.get('faces', 0)
                            analysis_summary['frames'].append(analysis)
                            processed_frames += 1

                    except Exception as e:
                        print(f"Error processing frame {frame_count}: {e}")

                frame_count += 1

            cap.release()
            analysis_summary['processed_frames'] = processed_frames

            # Sort detected items by frequency
            analysis_summary['top_objects'] = sorted(
                analysis_summary['objects_detected'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            analysis_summary['top_labels'] = sorted(
                analysis_summary['labels_detected'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return analysis_summary

        except Exception as e:
            return {'error': str(e)}
        finally:
            self.is_analyzing = False

    def get_analysis_status(self) -> Dict[str, Any]:
        """Get current analysis status"""
        return {
            'is_analyzing': self.is_analyzing,
            'current_video': self.current_video_path,
            'results_count': len(self.analysis_results)
        }

class GoogleVisionService:
    def __init__(self):
        if not GOOGLE_VISION_AVAILABLE:
            self.api_available = False
            self.client = None
            return

        try:
            self.client = vision.ImageAnnotatorClient()
            self.api_available = True
        except Exception as e:
            print(f"Warning: Google Vision API not available: {e}")
            print("AI video analysis features will be disabled.")
            self.client = None
            self.api_available = False

    def get_face_embedding(self, image_content):
        """
        Detects faces in an image and returns the first face's embedding.
        image_content: The binary content of the image.
        """
        if not self.api_available:
            return None

        try:
            image = vision.Image(content=image_content)
            response = self.client.face_detection(image=image)
            face_annotations = response.face_annotations

            if not face_annotations:
                return None

            # For simplicity, we're taking the first detected face's embedding.
            # In a real-world scenario, you might want to handle multiple faces.
            embedding = face_annotations[0].face_embedding
            return np.array(embedding)
        except Exception as e:
            print(f"Error getting face embedding: {e}")
            return None

    def compare_faces(self, embedding1, embedding2, threshold=0.7):
        """
        Compares two face embeddings using cosine similarity.
        Returns True if they are similar enough, False otherwise.
        """
        if not self.api_available:
            return False

        if embedding1 is None or embedding2 is None:
            return False

        try:
            # Cosine similarity calculation
            cosine_similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

            return cosine_similarity >= threshold
        except Exception as e:
            print(f"Error comparing faces: {e}")
            return False

    def detect_objects(self, image_content, max_results=10):
        """
        Detects objects in an image using Google Vision API.
        Returns a list of detected objects with their confidence scores.
        """
        if not self.api_available:
            return []

        try:
            image = vision.Image(content=image_content)
            response = self.client.object_localization(image=image)
            objects = response.localized_object_annotations

            detected_objects = []
            for obj in objects[:max_results]:
                detected_objects.append({
                    'name': obj.name,
                    'confidence': obj.score,
                    'bounding_box': {
                        'x': obj.bounding_poly.normalized_vertices[0].x,
                        'y': obj.bounding_poly.normalized_vertices[0].y,
                        'width': obj.bounding_poly.normalized_vertices[2].x - obj.bounding_poly.normalized_vertices[0].x,
                        'height': obj.bounding_poly.normalized_vertices[2].y - obj.bounding_poly.normalized_vertices[0].y
                    }
                })

            return detected_objects
        except Exception as e:
            print(f"Error detecting objects: {e}")
            return []

    def detect_labels(self, image_content, max_results=10):
        """
        Detects labels in an image using Google Vision API.
        Returns a list of detected labels with their confidence scores.
        """
        if not self.api_available:
            return []

        try:
            image = vision.Image(content=image_content)
            response = self.client.label_detection(image=image)
            labels = response.label_annotations

            detected_labels = []
            for label in labels[:max_results]:
                detected_labels.append({
                    'description': label.description,
                    'confidence': label.score,
                    'mid': label.mid
                })

            return detected_labels
        except Exception as e:
            print(f"Error detecting labels: {e}")
            return []

    def analyze_image(self, image_content):
        """
        Comprehensive image analysis including objects, labels, and text.
        Returns a dictionary with all detection results.
        """
        if not self.api_available:
            return {
                'objects': [],
                'labels': [],
                'faces': 0,
                'error': 'Google Vision API not available',
                'timestamp': str(np.datetime64('now'))
            }

        try:
            results = {
                'objects': self.detect_objects(image_content),
                'labels': self.detect_labels(image_content),
                'faces': len(self.client.face_detection(image=vision.Image(content=image_content)).face_annotations),
                'timestamp': str(np.datetime64('now'))
            }
            return results
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return {
                'objects': [],
                'labels': [],
                'faces': 0,
                'error': str(e),
                'timestamp': str(np.datetime64('now'))
            }
