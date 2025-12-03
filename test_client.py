"""
API Test Client - Demonstrates complete workflow
Replicates the terminal structure of test.py
"""

import sys
import requests
import time
import json
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# API Configuration
BASE_URL = "http://localhost:8000"

def print_banner(text):
    """Print formatted banner"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def create_session():
    """Step 1: Create a new session"""
    print_banner("STEP 1: CREATE SESSION")
    
    response = requests.post(f"{BASE_URL}/session/create")
    response.raise_for_status()
    
    data = response.json()
    session_id = data['session_id']
    
    print(f"✓ Session Created")
    print(f"  Session ID: {session_id}")
    print(f"  Created At: {data['created_at']}")
    
    return session_id

def upload_video(session_id, exercise_type, video_path):
    """Step 2: Upload video file"""
    print_banner("STEP 2: UPLOAD VIDEO")
    
    print(f"Session ID: {session_id}")
    print(f"Exercise: {exercise_type.upper()}")
    print(f"Video: {video_path}")
    print()
    
    # Open and upload file
    with open(video_path, 'rb') as f:
        files = {'video': f}
        params = {'exercise_type': exercise_type}
        
        response = requests.post(
            f"{BASE_URL}/session/{session_id}/upload",
            files=files,
            params=params
        )
        response.raise_for_status()
    
    data = response.json()
    
    print(f"✓ Video Uploaded Successfully")
    print(f"  Filename: {data['file_info']['filename']}")
    print(f"  Size: {data['file_info']['size_mb']} MB")
    
    return data

def start_analysis(session_id):
    """Step 3: Start video analysis"""
    print_banner("STEP 3: START ANALYSIS")
    
    response = requests.post(f"{BASE_URL}/session/{session_id}/analyze")
    response.raise_for_status()
    
    data = response.json()
    
    print(f"✓ Analysis Started")
    print(f"  Session ID: {session_id}")
    print(f"  Status: {data['status']}")
    print()
    
    return data

def poll_status(session_id, interval=2):
    """Step 4: Poll analysis status until complete"""
    print_banner("STEP 4: MONITOR PROGRESS")
    
    print(f"Session ID: {session_id}")
    print(f"Polling every {interval}s...\n")
    
    while True:
        response = requests.get(f"{BASE_URL}/session/{session_id}/status")
        response.raise_for_status()
        
        data = response.json()
        status = data['status']
        progress = data['progress']
        message = data['message']
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\r[{bar}] {progress*100:.0f}% - {message}", end='', flush=True)
        
        if status == 'completed':
            print("\n\n✓ Analysis Complete!")
            break
        elif status == 'failed':
            print(f"\n\n✗ Analysis Failed: {message}")
            return None
        
        time.sleep(interval)
    
    return data

def get_report(session_id):
    """Step 5: Retrieve performance report"""
    print_banner("STEP 5: RETRIEVE REPORT")
    
    response = requests.get(f"{BASE_URL}/session/{session_id}/report")
    response.raise_for_status()
    
    data = response.json()
    
    print(f"✓ Report Retrieved")
    print(f"  Exercise: {data['exercise'].upper()}")
    print(f"  Timestamp: {data['timestamp']}")
    print()
    
    return data['report']

def print_report_summary(report):
    """Print formatted report summary"""
    print_banner("PERFORMANCE REPORT SUMMARY")
    
    # Key metrics
    if 'exercise' in report:
        print(f"Exercise: {report['exercise']}")
    if 'duration_seconds' in report:
        print(f"Duration: {report['duration_seconds']}s")
    if 'total_jumps' in report:
        print(f"Total Jumps: {report['total_jumps']}")
    if 'valid_jumps' in report:
        print(f"Valid Jumps: {report['valid_jumps']}")
    if 'accuracy' in report:
        print(f"Accuracy: {report['accuracy']}%")
    if 'final_score' in report:
        print(f"\nFinal Score: {report['final_score']}")
    if 'rating' in report:
        print(f"Rating: {report['rating']}")
    
    print()

def save_report(session_id, report, filename="api_report.json"):
    """Save report to file"""
    report_data = {
        'session_id': session_id,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'report': report
    }
    
    with open(filename, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"✓ Report saved to {filename}")

def test_workflow():
    """Complete test workflow"""
    print("\n" + "="*60)
    print("  AI EXERCISE TRAINER - API TEST CLIENT")
    print("="*60)
    
    # Configuration
    EXERCISE_TYPE = "bjump"  # Change this: pushup, squat, situp, etc.
    VIDEO_PATH = r"C:\Users\avijn_th5xjtu\Desktop\code\application\b1.mp4"
    
    # Check if video exists
    if not Path(VIDEO_PATH).exists():
        print(f"\n✗ ERROR: Video file not found: {VIDEO_PATH}")
        print("Please update VIDEO_PATH in the script.")
        return
    
    try:
        # Step 1: Create session
        session_id = create_session()
        
        # Step 2: Upload video
        upload_video(session_id, EXERCISE_TYPE, VIDEO_PATH)
        
        # Step 3: Start analysis
        start_analysis(session_id)
        
        # Step 4: Monitor progress
        poll_status(session_id, interval=1)
        
        # Step 5: Get report
        report = get_report(session_id)
        
        # Print summary
        print_report_summary(report)
        
        # Save report
        save_report(session_id, report, f"{EXERCISE_TYPE}_report.json")
        
        print_banner("TEST COMPLETE")
        print(f"Session ID: {session_id}")
        print(f"Exercise: {EXERCISE_TYPE.upper()}")
        print(f"Status: SUCCESS ✓")
        
    except requests.exceptions.RequestException as e:
        print(f"\n\n✗ ERROR: {str(e)}")
        print("\nMake sure the API server is running:")
        print("  python server.py")
    except Exception as e:
        print(f"\n\n✗ UNEXPECTED ERROR: {str(e)}")

def list_all_sessions():
    """List all active sessions"""
    print_banner("ALL ACTIVE SESSIONS")
    
    response = requests.get(f"{BASE_URL}/sessions")
    response.raise_for_status()
    
    data = response.json()
    
    print(f"Total Sessions: {data['total_sessions']}\n")
    
    if data['total_sessions'] == 0:
        print("No active sessions.")
    else:
        for session in data['sessions']:
            print(f"Session ID: {session['session_id']}")
            print(f"  Exercise: {session['exercise_type']}")
            print(f"  Status: {session['status']}")
            print(f"  Progress: {session['progress']*100:.0f}%")
            print(f"  Created: {session['created_at']}")
            print()

def delete_session(session_id):
    """Delete a session"""
    print_banner("DELETE SESSION")
    
    response = requests.delete(f"{BASE_URL}/session/{session_id}")
    response.raise_for_status()
    
    data = response.json()
    print(f"✓ {data['message']}")
    print(f"  Session ID: {session_id}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "list":
            list_all_sessions()
        elif command == "delete" and len(sys.argv) > 2:
            delete_session(sys.argv[2])
        elif command == "test":
            test_workflow()
        else:
            print("Usage:")
            print("  python test_client.py test      - Run complete test workflow")
            print("  python test_client.py list      - List all sessions")
            print("  python test_client.py delete ID - Delete session")
    else:
        # Default: run test workflow
        test_workflow()
