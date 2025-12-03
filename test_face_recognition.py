import cv2
import os
import sqlite3
from person_detector import PersonDetector

def test_face_recognition():
    """Test face recognition setup and capabilities"""
    print("Testing Face Recognition System...")
    print("=" * 50)
    
    # Initialize the detector
    detector = PersonDetector()
    
    # Check if face images exist
    face_images_dir = 'static/face_images'
    if os.path.exists(face_images_dir):
        face_files = [f for f in os.listdir(face_images_dir) if f.endswith('.jpg')]
        print(f"Found {len(face_files)} face image files:")
        for f in face_files:
            print(f"  - {f}")
    else:
        print("Face images directory not found!")
        return
    
    # Check employee database
    try:
        conn = sqlite3.connect('attendance.db')
        cursor = conn.cursor()
        cursor.execute("SELECT id, employee_id, first_name, last_name FROM employees WHERE status = 'active'")
        employees = cursor.fetchall()
        conn.close()
        
        print(f"\nFound {len(employees)} active employees:")
        for emp in employees:
            print(f"  - ID: {emp[1]}, Name: {emp[2]} {emp[3]}")
    except Exception as e:
        print(f"Error accessing employee database: {e}")
        return
    
    # Check loaded employee labels
    print(f"\nFace recognizer trained with {len(detector.employee_labels)} employees:")
    for label_id, data in detector.employee_labels.items():
        print(f"  - Label {label_id}: {data['first_name']} {data['last_name']} (ID: {data['employee_id']})")
    
    # Test face cascade
    if detector.face_cascade.empty():
        print("\nERROR: Face cascade classifier not loaded!")
    else:
        print("\nFace cascade classifier loaded successfully.")
    
    # Test with webcam if available
    print("\nTesting with webcam...")
    if detector.start_stream(0):
        print("Webcam connected successfully!")
        print("Press 'q' to quit the test")
        
        while True:
            frame = detector.get_frame_with_detection()
            if frame is not None:
                cv2.imshow('Face Recognition Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                print("No frame received")
                break
        
        detector.stop_stream()
        cv2.destroyAllWindows()
    else:
        print("Failed to connect to webcam!")

if __name__ == "__main__":
    test_face_recognition()

