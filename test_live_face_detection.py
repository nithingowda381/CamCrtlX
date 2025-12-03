#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sqlite3
from person_detector import PersonDetector
import time

def test_live_face_detection():
    """Test face detection in live feed with debug information"""
    print("Testing Live Face Detection...")
    print("=" * 50)
    
    # Initialize detector
    detector = PersonDetector(confidence_threshold=0.5)
    
    # Check training data
    print(f"Trained employees: {len(detector.employee_labels)}")
    for label_id, employee_data in detector.employee_labels.items():
        print(f"  - Label {label_id}: {employee_data['first_name']} {employee_data['last_name']} (ID: {employee_data['employee_id']})")
    
    # Test camera connection
    if detector.start_stream(0):
        print("\nCamera connected successfully!")
        print("Testing face recognition... (Press 'q' to quit)")
        
        frame_count = 0
        face_detections = 0
        recognitions = 0
        
        while True:
            frame = detector.get_frame_with_detection()
            if frame is not None:
                frame_count += 1
                
                # Show frame
                cv2.imshow('Live Face Detection Test', frame)
                
                # Test face recognition on current frame
                if frame_count % 30 == 0:  # Test every 30 frames (about 1 per second)
                    # Convert frame to grayscale and detect faces
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(30, 30))
                    
                    if len(faces) > 0:
                        face_detections += 1
                        print(f"Frame {frame_count}: Detected {len(faces)} face(s)")
                        
                        # Test recognition on the largest face
                        face = max(faces, key=lambda x: x[2] * x[3])
                        x, y, w, h = face
                        face_img = gray[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, (100, 100))
                        
                        # Try recognition
                        if len(detector.employee_labels) > 0:
                            try:
                                label, confidence = detector.face_recognizer.predict(face_img)
                                similarity = max(0, 100 - confidence)
                                
                                if similarity > 50 and label in detector.employee_labels:
                                    recognitions += 1
                                    employee = detector.employee_labels[label]
                                    print(f"  -> RECOGNIZED: {employee['first_name']} {employee['last_name']} (Confidence: {similarity:.1f}%)")
                                else:
                                    print(f"  -> UNKNOWN PERSON (Similarity: {similarity:.1f}%)")
                            except Exception as e:
                                print(f"  -> Recognition error: {e}")
                    else:
                        print(f"Frame {frame_count}: No faces detected")
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                print("No frame received")
                break
        
        detector.stop_stream()
        cv2.destroyAllWindows()
        
        # Print statistics
        print(f"\nTest Statistics:")
        print(f"Total frames processed: {frame_count}")
        print(f"Face detections: {face_detections}")
        print(f"Successful recognitions: {recognitions}")
        print(f"Recognition rate: {(recognitions/max(face_detections, 1)*100):.1f}%")
        
    else:
        print("Failed to connect to camera!")

def test_face_recognition_accuracy():
    """Test face recognition accuracy using existing face images"""
    print("\nTesting Face Recognition Accuracy...")
    print("=" * 50)
    
    detector = PersonDetector()
    
    face_images_dir = 'static/face_images'
    if not os.path.exists(face_images_dir):
        print("Face images directory not found!")
        return
    
    # Get all face images
    face_files = [f for f in os.listdir(face_images_dir) if f.endswith('.jpg')]
    print(f"Found {len(face_files)} face images")
    
    correct_predictions = 0
    total_predictions = 0
    
    for face_file in face_files:
        # Extract employee ID from filename
        employee_id = face_file.split('_face_')[0]
        
        # Load and process image
        image_path = os.path.join(face_images_dir, face_file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # Resize to standard size
            img_resized = cv2.resize(img, (100, 100))
            
            # Try to recognize
            try:
                label, confidence = detector.face_recognizer.predict(img_resized)
                similarity = max(0, 100 - confidence)
                
                total_predictions += 1
                
                if similarity > 50 and label in detector.employee_labels:
                    predicted_employee_id = detector.employee_labels[label]['employee_id']
                    if predicted_employee_id == employee_id:
                        correct_predictions += 1
                        print(f"✓ {face_file}: Correctly recognized as {predicted_employee_id} (Confidence: {similarity:.1f}%)")
                    else:
                        print(f"✗ {face_file}: Incorrectly recognized as {predicted_employee_id}, expected {employee_id} (Confidence: {similarity:.1f}%)")
                else:
                    print(f"✗ {face_file}: Not recognized (Similarity: {similarity:.1f}%)")
                    
            except Exception as e:
                print(f"✗ {face_file}: Recognition failed - {e}")
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nAccuracy: {correct_predictions}/{total_predictions} = {accuracy:.1f}%")
    else:
        print("No predictions made")

if __name__ == "__main__":
    print("Face Recognition Test Suite")
    print("=" * 50)
    
    # Test static accuracy first
    test_face_recognition_accuracy()
    
    # Ask user if they want to test live detection
    response = input("\nDo you want to test live face detection? (y/n): ")
    if response.lower() == 'y':
        test_live_face_detection()
    
    print("Testing completed!")
