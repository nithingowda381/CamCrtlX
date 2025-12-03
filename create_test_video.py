import os
import cv2
import numpy as np

def create_test_video():
    """Create a simple test video for testing the analysis functionality"""
    # Create a simple video with some basic content
    width, height = 640, 480
    fps = 30
    duration = 5  # seconds

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = 'static/video_uploads/test_video.mp4'
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Generate frames
    for i in range(fps * duration):
        # Create a frame with some basic shapes
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Add some colored rectangles (simulating objects)
        cv2.rectangle(frame, (50, 50), (150, 150), (0, 255, 0), -1)  # Green rectangle
        cv2.rectangle(frame, (200, 100), (300, 200), (255, 0, 0), -1)  # Blue rectangle
        cv2.circle(frame, (400, 300), 50, (0, 0, 255), -1)  # Red circle

        # Add some text
        cv2.putText(frame, f'Frame {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    out.release()
    print(f"Test video created: {video_path}")
    return video_path

if __name__ == "__main__":
    # Create uploads directory
    os.makedirs('static/video_uploads', exist_ok=True)
    create_test_video()