# AI Exercise Trainer - FastAPI Server

Complete REST API for exercise performance analysis with automatic session management.

## Features

- **8 Exercise Types**: Push-ups, Squats, Sit-ups, Sit-and-Reach, Skipping, Jumping Jacks, Vertical Jump, Broad Jump
- **Auto Session Management**: UUID-based sessions with automatic cleanup
- **Video Analysis**: Upload videos for frame-by-frame analysis
- **Real-time Progress**: Track analysis progress in real-time
- **Performance Reports**: Detailed metrics matching test.py terminal output
- **Background Processing**: Non-blocking video analysis

## Quick Start

### 1. Start Server

```bash
# Option 1: Using startup script (recommended)
python start_server.py

# Option 2: Direct uvicorn
python server.py

# Option 3: Manual uvicorn
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Server runs at: **http://localhost:8000**

### 2. Test API

```bash
# Run complete test workflow
python test_client.py test

# List all sessions
python test_client.py list

# Delete a session
python test_client.py delete <session_id>
```

## API Endpoints

### Health Check
```http
GET /
```
Returns API status and info

### Create Session
```http
POST /session/create
```
**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created successfully",
  "created_at": "2025-12-03T10:30:00"
}
```

### Upload Video
```http
POST /session/{session_id}/upload?exercise_type=bjump
Content-Type: multipart/form-data

video: <file>
```
**Parameters:**
- `exercise_type`: pushup, squat, situp, sitnreach, skipping, jumpingjacks, vjump, bjump

**Response:**
```json
{
  "message": "Video uploaded successfully",
  "session_id": "550e8400-...",
  "exercise_type": "bjump",
  "file_info": {
    "filename": "b1.mp4",
    "size_bytes": 12458752,
    "size_mb": 11.88
  }
}
```

### Start Analysis
```http
POST /session/{session_id}/analyze
```
Starts background video analysis

**Response:**
```json
{
  "message": "Analysis started",
  "session_id": "550e8400-...",
  "exercise_type": "bjump",
  "status": "processing"
}
```

### Check Status
```http
GET /session/{session_id}/status
```
**Response:**
```json
{
  "session_id": "550e8400-...",
  "status": "processing",
  "progress": 0.75,
  "message": "Analyzing video... 75% complete"
}
```

**Status Values:**
- `created`: Session created, awaiting upload
- `uploaded`: Video uploaded, ready for analysis
- `processing`: Analysis in progress
- `completed`: Analysis complete, report ready
- `failed`: Analysis failed

### Get Report
```http
GET /session/{session_id}/report
```
**Response:**
```json
{
  "session_id": "550e8400-...",
  "exercise": "bjump",
  "report": {
    "exercise": "Broad Jump",
    "duration_seconds": 6.4,
    "total_jumps": 2,
    "valid_jumps": 1,
    "accuracy": 50.0,
    "distance_score": 1.0,
    "final_score": 0.97,
    "rating": "KEEP PRACTICING",
    ...
  },
  "timestamp": "2025-12-03T10:31:45"
}
```

### List Sessions
```http
GET /sessions
```
Returns all active sessions

### Delete Session
```http
DELETE /session/{session_id}
```
Deletes session and cleanup files

## Complete Workflow Example

### Python (using requests)

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Create session
response = requests.post(f"{BASE_URL}/session/create")
session = response.json()
session_id = session['session_id']

# 2. Upload video
with open('video.mp4', 'rb') as f:
    files = {'video': f}
    params = {'exercise_type': 'bjump'}
    requests.post(
        f"{BASE_URL}/session/{session_id}/upload",
        files=files,
        params=params
    )

# 3. Start analysis
requests.post(f"{BASE_URL}/session/{session_id}/analyze")

# 4. Poll status until complete
import time
while True:
    status = requests.get(f"{BASE_URL}/session/{session_id}/status").json()
    if status['status'] == 'completed':
        break
    time.sleep(2)

# 5. Get report
report = requests.get(f"{BASE_URL}/session/{session_id}/report").json()
print(report)
```

### cURL

```bash
# 1. Create session
SESSION_ID=$(curl -X POST http://localhost:8000/session/create | jq -r '.session_id')

# 2. Upload video
curl -X POST "http://localhost:8000/session/$SESSION_ID/upload?exercise_type=bjump" \
  -F "video=@video.mp4"

# 3. Start analysis
curl -X POST "http://localhost:8000/session/$SESSION_ID/analyze"

# 4. Check status
curl "http://localhost:8000/session/$SESSION_ID/status"

# 5. Get report
curl "http://localhost:8000/session/$SESSION_ID/report"
```

## Exercise Types

| Code | Exercise | Description |
|------|----------|-------------|
| `pushup` | Push-ups | Upper body strength |
| `squat` | Squats | Lower body strength |
| `situp` | Sit-ups | Core strength |
| `sitnreach` | Sit-and-Reach | Flexibility |
| `skipping` | Jump Rope | Cardio endurance |
| `jumpingjacks` | Jumping Jacks | Full body cardio |
| `vjump` | Vertical Jump | Explosive power (vertical) |
| `bjump` | Broad Jump | Explosive power (horizontal) |

## Report Structure

### Broad Jump Report
```json
{
  "exercise": "Broad Jump",
  "duration_seconds": 6.4,
  "total_jumps": 2,
  "valid_jumps": 1,
  "accuracy": 50.0,
  
  "distance_score": 1.0,
  "countermovement_score": 1.0,
  "arm_swing_score": 1.0,
  "symmetry_score": 1.0,
  "landing_stability_score": 0.7,
  
  "max_jump_distance": 1380.5,
  "avg_jump_distance": 725.6,
  "avg_countermovement": 96.0,
  "avg_arm_swing": 169.5,
  "avg_symmetry_error": 0.0,
  "avg_landing_stability": 34.8,
  
  "final_score": 0.97,
  "rating": "KEEP PRACTICING"
}
```

## Terminal Output

The server prints formatted reports matching test.py output:

```
==================================================
STANDING BROAD JUMP PERFORMANCE REPORT
==================================================
Duration: 6.4s
Total Jumps: 2 | Valid Jumps: 1
Accuracy: 50.0%

METRIC BREAKDOWN:
--------------------------------------------------
Jump Distance Score: 1.0 | KEEP PRACTICING
Countermovement Score: 1.0 | KEEP PRACTICING
Arm Swing Score: 1.0 | KEEP PRACTICING
Takeoff Symmetry: 1.0 | KEEP PRACTICING
Landing Stability: 0.7 | KEEP PRACTICING

DETAILED MEASUREMENTS:
--------------------------------------------------
Max Jump Distance: 1380.5 px
Average Jump Distance: 725.6 px
Average Countermovement: 96.0° (Good: 90-120°)
Average Arm Swing: 169.5° (Good: 140+°)
Average Symmetry Error: 0.0s (Good: <0.1s)
Average Landing Stability: 34.8px (Good: <30px)

==================================================
FINAL PERFORMANCE SCORE: 0.97 | KEEP PRACTICING
==================================================
```

## Session Management

- **Auto IDs**: UUID v4 automatically generated
- **Isolation**: Each session independent with own storage
- **Cleanup**: Delete session removes video files
- **Sets Support**: Create new session for each exercise set

### Multiple Sessions Example

```python
# Set 1
session1 = create_session()
upload_video(session1, 'bjump', 'set1.mp4')
analyze(session1)

# Set 2
session2 = create_session()
upload_video(session2, 'bjump', 'set2.mp4')
analyze(session2)

# Compare results
report1 = get_report(session1)
report2 = get_report(session2)
```

## File Structure

```
fast/
├── server.py              # Main FastAPI server
├── test_client.py         # Test client script
├── start_server.py        # Quick start script
├── utils.py               # Pose detection (PoseCalibrator)
├── metrics.py             # Performance metrics
├── requirements.txt       # Dependencies
├── yolo11n-pose.pt       # YOLO model
├── uploads/              # Uploaded videos (auto-created)
└── results/              # Analysis results (auto-created)
```

## API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Error Handling

The API returns standard HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid exercise type, wrong status, etc.)
- `404`: Not found (session doesn't exist)
- `500`: Server error

Example error response:
```json
{
  "detail": "Session not found"
}
```

## Performance

- **Background Processing**: Video analysis runs async
- **Progress Tracking**: Real-time progress updates
- **Memory Efficient**: Processes frame-by-frame
- **Fast Response**: API endpoints respond immediately

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill process
taskkill /PID <pid> /F

# Use different port
uvicorn server:app --port 8001
```

### Model Not Found
Ensure `yolo11n-pose.pt` is in the same directory as `server.py`

### Import Errors
```bash
pip install -r requirements.txt
```

### CORS Issues
CORS is enabled for all origins by default. Modify `server.py` if needed.

## Production Deployment

### Using Gunicorn (Linux/Mac)
```bash
pip install gunicorn
gunicorn server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables
```bash
# Port configuration
export PORT=8000

# Upload directory
export UPLOAD_DIR=/path/to/uploads

# Results directory
export RESULTS_DIR=/path/to/results
```

## License

Part of AI Exercise Trainer project.
