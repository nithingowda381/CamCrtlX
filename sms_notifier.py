from twilio.rest import Client
import config
import datetime

class SMSNotifier:
    def __init__(self):
        self.client = Client(config.TWILIO_ACCOUNT_SID, config.TWILIO_AUTH_TOKEN)
        self.from_phone = config.TWILIO_FROM_PHONE
        self.to_phone = config.TWILIO_TO_PHONE
    
    def send_sms(self, message: str) -> bool:
        """Send SMS notification"""
        try:
            message = self.client.messages.create(
                body=message,
                from_=self.from_phone,
                to=self.to_phone
            )
            print(f"SMS sent: {message.sid}")
            return True
        except Exception as e:
            print(f"Error sending SMS: {e}")
            return False
    
    def send_work_started(self, start_time: datetime.datetime):
        """Send SMS when person starts work"""
        message = f"Person started work at {start_time.strftime('%H:%M:%S')}"
        return self.send_sms(message)
    
    def send_work_ended(self, end_time: datetime.datetime, total_seconds: int):
        """Send SMS when person leaves work"""
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        message = f"Person left work at {end_time.strftime('%H:%M:%S')}. Total work time: {hours}h {minutes}m"
        return self.send_sms(message)
    
    def send_daily_summary(self, total_seconds: int, sessions_count: int):
        """Send daily work summary"""
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        
        message = f"Daily work summary: {hours}h {minutes}m total work time across {sessions_count} sessions"
        return self.send_sms(message)
