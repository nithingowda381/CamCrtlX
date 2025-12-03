import cv2
import time
from person_detector import PersonDetector
import config

def main():
    """
    Runs the live face detection and recognition process in a dedicated GUI window.
    """
    print("Initializing live GUI...")
    
    # Use the same detector configuration as the web app
    detector = PersonDetector(confidence_threshold=config.CONFIDENCE_THRESHOLD)
    
    # Start the video stream using the source from the main config
    # Defaulting to '0' (webcam) if not specified
    stream_source = getattr(config, 'DVR_STREAM_URL', '0')
    if stream_source == 'webcam':
        stream_source = 0
        
    print(f"Attempting to start video stream from: {stream_source}")
    if not detector.start_stream(stream_source):
        print("Failed to open video stream. Please check the camera or stream URL.")
        return

    print("\n" + "="*50)
    print("  Live Detection GUI is running.")
    print("  Press 'q' in the window to quit.")
    print("="*50 + "\n")

    window_name = 'CamCtrlX - Live AI Detection'

    while detector.is_running:
        try:
            # Get the processed frame with all detection overlays
            frame = detector.get_frame_with_detection()

            if frame is not None:
                # Display the frame in a GUI window
                cv2.imshow(window_name, frame)
            else:
                print("Received an empty frame. Attempting to continue...")
                time.sleep(0.5) # Wait a bit before trying again

            # Wait for key press. If 'q' is pressed, exit the loop.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("'q' key pressed. Shutting down live GUI.")
                break
        
        except KeyboardInterrupt:
            print("Keyboard interrupt received. Shutting down.")
            break
        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            break

    # Cleanup
    print("Releasing resources...")
    detector.stop_stream()
    cv2.destroyAllWindows()
    print("GUI closed.")

if __name__ == '__main__':
    main()
