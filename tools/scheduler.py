import datetime
import os.path
import sys

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dateutil import parser, tz

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def get_credentials():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists('credentials.json'):
                print("Error: credentials.json not found. Please download OAuth client ID credentials from Google Cloud Console.")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def list_events(service, calendar_id='primary', max_results=10):
    now = datetime.datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
    print(f"Getting the upcoming {max_results} events for {calendar_id}")
    events_result = service.events().list(calendarId=calendar_id, timeMin=now,
                                          maxResults=max_results, singleEvents=True,
                                          orderBy='startTime').execute()
    events = events_result.get('items', [])
    return events

def find_common_slots(events1, events2, duration_minutes=60):
    # This is a simplified logic. In a real scenario, we would merge free/busy intervals.
    # For now, let's just list the busy times to help the user decide.
    
    print("\n--- BUSY TIMES (UTC) ---")
    print("User 1:")
    for event in events1:
        start = event['start'].get('dateTime', event['start'].get('date'))
        print(f"  {start} - {event['summary']}")
        
    print("\nUser 2 (Francois):")
    # Note: Accessing another user's calendar requires sharing permissions.
    # We will attempt to read 'francois.smit.fs@gmail.com' if shared, or prompt user.
    try:
        if not events2:
             print("  [No access or no events]")
        else:
            for event in events2:
                start = event['start'].get('dateTime', event['start'].get('date'))
                print(f"  {start} - {event['summary']}")
    except Exception as e:
        print(f"  Could not read Francois's calendar: {e}")

def main():
    creds = get_credentials()
    if not creds:
        return

    try:
        service = build('calendar', 'v3', credentials=creds)

        # 1. Get User's Calendar
        my_events = list_events(service, 'primary')

        # 2. Get Francois's Calendar (Simulated attempt, requires permission)
        # In a real app, this would be a secondary calendar ID or a service account lookup
        francois_email = 'francois.smit.fs@gmail.com'
        try:
            francois_events = list_events(service, francois_email)
        except Exception:
            print(f"\nWarning: Cannot access {francois_email} directly. Make sure he has shared his calendar with you.")
            francois_events = []

        # 3. Analyze
        find_common_slots(my_events, francois_events)
        
        print("\n[Scheduler] Analysis Complete. Please check the busy times above to propose a meeting slot.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
