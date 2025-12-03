import datetime
import time
from typing import Optional
from database import DatabaseManager
from person_detector import PersonDetector
from sms_notifier import SMSNotifier
import config

class WorkHoursTracker:
    def __init__(self):
        self.db = DatabaseManager(config.DATABASE_PATH)
        self.detector = PersonDetector(confidence_threshold=config.CONFIDENCE_THRESHOLD)
        self.notifier = SMSNotifier()
        
        self.is_running = False
        self.current_session_id = None
        self.person_present = False
        self.last_detection_time = None
        self.absence_start_time = None
        
    def start_tracking(self) -> bool:
        """Start the work hours tracking system"""
        if not self.detector.start_stream(config.DVR_STREAM_URL):
            print("Failed to start video stream")
            return False
        
        self.is_running = True
        print("Work hours tracking started...")
        
        try:
            while self.is_running:
                # Detect person
                person_detected, confidence = self.detector.detect_person()
                
                if person_detected:
                    self.handle_person_detected()
                else:
                    self.handle_person_absent()
                
                # Sleep for detection interval
                time.sleep(config.DETECTION_INTERVAL)
                
        except KeyboardInterrupt:
            print("\nStopping work tracking...")
        finally:
            self.stop_tracking()
        
        return True
    
    def handle_person_detected(self):
        """Handle when person is detected"""
        self.last_detection_time = datetime.datetime.now()
        
        if not self.person_present:
            # Person just arrived
            self.person_present = True
            self.absence_start_time = None
            
            # Start new work session
            self.current_session_id = self.db.insert_work_session(self.last_detection_time)
            
            # Send SMS notification
            self.notifier.send_work_started(self.last_detection_time)
            
            print(f"Person started work at {self.last_detection_time.strftime('%H:%M:%S')}")
    
    def handle_person_absent(self):
        """Handle when person is not detected"""
        current_time = datetime.datetime.now()
        
        if self.person_present:
            # Person was present, now absent
            if self.absence_start_time is None:
                self.absence_start_time = current_time
            
            # Check if absence timeout has been reached
            absence_duration = (current_time - self.absence_start_time).total_seconds()
            
            if absence_duration >= config.ABSENCE_TIMEOUT:
                # Person has left
                self.person_present = False
                
                if self.current_session_id:
                    # End current work session
                    self.db.update_work_session(self.current_session_id, current_time)
                    
                    # Get session details
                    session = self.db.get_last_session()
                    if session and session['total_seconds']:
                        self.notifier.send_work_ended(
                            current_time, 
                            session['total_seconds']
                        )
                    
                    self.current_session_id = None
                
                print(f"Person left work at {current_time.strftime('%H:%M:%S')}")
    
    def stop_tracking(self):
        """Stop the work hours tracking"""
        self.is_running = False
        self.detector.stop_stream()
        
        # End current session if active
        if self.current_session_id and self.person_present:
            end_time = datetime.datetime.now()
            self.db.update_work_session(self.current_session_id, end_time)
            
            session = self.db.get_last_session()
            if session and session['total_seconds']:
                self.notifier.send_work_ended(end_time, session['total_seconds'])
    
    def get_status(self) -> dict:
        """Get current tracking status"""
        return {
            'is_running': self.is_running,
            'person_present': self.person_present,
            'current_session_id': self.current_session_id,
            'last_detection_time': self.last_detection_time
        }
    
    def send_daily_summary(self):
        """Send daily work summary"""
        summary = self.db.get_today_summary()
        self.notifier.send_daily_summary(
            summary['total_seconds'], 
            summary['sessions_count']
        )

if __name__ == "__main__":
    tracker = WorkHoursTracker()
    tracker.start_tracking()
