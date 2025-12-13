#!/usr/bin/env python3
"""
Deep Learning Face Recognition Training Script

This script trains a face recognition model using deep learning embeddings
for more accurate face identification. It uses dlib's face recognition model
to extract 128D face embeddings from training images.

Usage:
    python train_face_model.py

Requirements:
    - dlib
    - numpy
    - opencv-python
    - scikit-learn (for KNN classifier)
"""

import os
import cv2
import numpy as np
import pickle
from typing import List, Tuple, Dict
import sqlite3
from pathlib import Path
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

class DeepFaceTrainer:
    def __init__(self):
        self.dlib_available = False
        # Initialize dlib's face detector and recognition model
        try:
            import dlib
            self.dlib = dlib
            self.detector = dlib.get_frontal_face_detector()

            # Use dlib's face recognition model (128D embeddings)
            try:
                self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                self.face_recognizer = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")
                self.dlib_available = True
                print("✓ Deep learning face recognition models loaded successfully")
            except Exception as e:
                print(f"✗ Error loading dlib models: {e}")
                print("Please download the required models:")
                print("  - shape_predictor_68_face_landmarks.dat")
                print("  - dlib_face_recognition_resnet_model_v1.dat")
                print("From: http://dlib.net/files/")
                self.detector = None
                self.face_recognizer = None
        except ImportError:
             print("✗ dlib library not found. Deep learning training unavailable.")
             print("ℹ Using standard LBPH recognition only.")
             self.detector = None
             self.face_recognizer = None

        self.embeddings = []
        self.labels = []
        self.employee_data = {}

    def load_employee_data(self) -> Dict[int, dict]:
        """Load employee data from database"""
        try:
            conn = sqlite3.connect('attendance.db')
            cursor = conn.cursor()
            cursor.execute("SELECT id, employee_id, first_name, last_name FROM employees WHERE status = 'active'")
            employees = cursor.fetchall()
            conn.close()

            employee_dict = {}
            for db_id, employee_id, first_name, last_name in employees:
                employee_dict[db_id] = {
                    'employee_id': employee_id,
                    'name': f"{first_name} {last_name}",
                    'first_name': first_name,
                    'last_name': last_name
                }

            print(f"✓ Loaded {len(employee_dict)} active employees from database")
            return employee_dict

        except Exception as e:
            print(f"✗ Error loading employee data: {e}")
            return {}

    def extract_face_embedding(self, image_path: str) -> Tuple[np.ndarray, bool]:
        """Extract 128D face embedding from an image"""
        if self.detector is None:
            return None, False

        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None, False

            # Convert to RGB (dlib expects RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect faces
            faces = self.detector(rgb_img, 1)

            if len(faces) == 0:
                return None, False

            # Use the largest face if multiple faces detected
            face = max(faces, key=lambda rect: rect.width() * rect.height())

            # Get facial landmarks
            shape = self.predictor(rgb_img, face)

            # Extract face embedding (128D vector)
            face_embedding = self.face_recognizer.compute_face_descriptor(rgb_img, shape)

            # Convert to numpy array
            embedding = np.array(face_embedding)

            return embedding, True

        except Exception as e:
            print(f"Error extracting embedding from {image_path}: {e}")
            return None, False

    def load_training_data(self) -> bool:
        """Load all face images and extract embeddings"""
        face_images_dir = 'static/face_images'

        if not os.path.exists(face_images_dir):
            print(f"✗ Face images directory not found: {face_images_dir}")
            return False

        self.employee_data = self.load_employee_data()
        if not self.employee_data:
            print("✗ No employee data found")
            return False

        print(f"Loading face images from {face_images_dir}...")

        total_images = 0
        successful_extractions = 0

        for filename in os.listdir(face_images_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Parse filename to get employee_id
            # Expected format: {employee_id}_face_{index}.jpg
            try:
                parts = filename.split('_face_')
                if len(parts) != 2:
                    continue

                employee_id_str = parts[0]
                # Find matching employee by employee_id
                employee_db_id = None
                for db_id, emp_data in self.employee_data.items():
                    if emp_data['employee_id'] == employee_id_str:
                        employee_db_id = db_id
                        break

                if employee_db_id is None:
                    print(f"Warning: No employee found for ID {employee_id_str}")
                    continue

            except Exception as e:
                print(f"Error parsing filename {filename}: {e}")
                continue

            image_path = os.path.join(face_images_dir, filename)
            total_images += 1

            # Extract embedding
            embedding, success = self.extract_face_embedding(image_path)

            if success:
                self.embeddings.append(embedding)
                self.labels.append(employee_db_id)
                successful_extractions += 1
                print(f"✓ Extracted embedding for {self.employee_data[employee_db_id]['name']} ({filename})")
            else:
                print(f"✗ Failed to extract embedding from {filename}")

        print(f"\nTraining Summary:")
        print(f"  Total images processed: {total_images}")
        print(f"  Successful extractions: {successful_extractions}")
        print(f"  Unique employees: {len(set(self.labels))}")

        if successful_extractions == 0:
            print("✗ No face embeddings extracted. Training failed.")
            return False

        return True

    def train_knn_classifier(self):
        """Train KNN classifier for face recognition"""
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        if not self.embeddings or not self.labels:
            print("✗ No training data available")
            return None

        # Convert to numpy arrays
        X = np.array(self.embeddings)
        y = np.array(self.labels)

        # Normalize embeddings
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)

        # Train KNN classifier (k=5, distance-based)
        knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
        knn.fit(X_normalized, y)

        print("✓ KNN classifier trained successfully")
        return knn, scaler

    def save_model(self, model_data: dict, filename: str = 'deep_face_model.pkl'):
        """Save the trained model to disk"""
        try:
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✓ Model saved to {filename}")
            return True
        except Exception as e:
            print(f"✗ Error saving model: {e}")
            return False

    def train_and_save(self):
        """Main training function"""
        print("=== Deep Learning Face Recognition Training ===\n")

        if not self.dlib_available:
             print("✗ dlib is not available. Cannot proceed with DEEP LEARNING training.")
             print("ℹ Standard LBPH training is handled automatically by the system.")
             return False

        if self.detector is None:
             print("✗ dlib components failed to load. Cannot proceed.")
             return False

        # Load training data
        if not self.load_training_data():
            return False

        # Train classifier
        result = self.train_knn_classifier()
        if result is None:
            return False

        knn_classifier, scaler = result

        # Prepare model data
        model_data = {
            'knn_classifier': knn_classifier,
            'scaler': scaler,
            'embeddings': self.embeddings,
            'labels': self.labels,
            'employee_data': self.employee_data,
            'model_type': 'deep_learning_knn',
            'embedding_dim': 128
        }

        # Save model
        success = self.save_model(model_data, 'deep_face_model.pkl')

        if success:
            print("\n✓ Training completed successfully!")
            print(f"  Model saved as: deep_face_model.pkl")
            print(f"  Trained on {len(self.labels)} face images")
            print(f"  Recognizes {len(set(self.labels))} employees")
        else:
            print("\n✗ Training failed - could not save model")

        return success

def main():
    """Main function"""
    try:
        trainer = DeepFaceTrainer()
        success = trainer.train_and_save()

        if success:
            print("\n🎉 Deep learning face recognition model trained successfully!")
            print("You can now use this model for more accurate face recognition.")
        else:
            print("\n❌ Training failed. Please check the errors above.")
            # If dlib is missing, this is expected.

    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
