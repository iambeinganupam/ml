# FastAPI Testing Guide - Interactive Docs

## 🌐 Server is Running!

**API Documentation**: http://localhost:8000/docs
**Alternative Docs**: http://localhost:8000/redoc
**Health Check**: http://localhost:8000/

---

## 📋 Complete Testing Checklist

### ✅ Test 1: Health Check
**URL**: http://localhost:8000/
**Method**: GET
**Expected Response**:
```json
{
  "status": "online",
  "service": "AI Exercise Trainer API",
  "version": "2.0",
  "exercises": ["pushup", "squat", "situp", "sitnreach", "skipping", "jumpingjacks", "vjump", "bjump"],
  "active_sessions": 2
}
```

---

## 🧪 Testing in FastAPI Docs (Swagger UI)

### Step 1: Open Interactive Docs
1. Go to: **http://localhost:8000/docs**
2. You'll see all 8 endpoints listed

### Step 2: Test Each Endpoint (In Order)

---

### ✅ **Test 2: Create Session**

**Endpoint**: `POST /session/create`

1. Click on **"POST /session/create"**
2. Click **"Try it out"** button
3. Click **"Execute"** button
4. **Expected Response** (200):
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created successfully",
  "created_at": "2025-12-03T22:30:00"
}
```
5. **Copy the `session_id`** - you'll need it for next tests!

---

### ✅ **Test 3: Upload Video**

**Endpoint**: `POST /session/{session_id}/upload`

1. Click on **"POST /session/{session_id}/upload"**
2. Click **"Try it out"**
3. **Parameters**:
   - `session_id`: Paste the session ID from Test 2
   - `exercise_type`: Select **"bjump"** from dropdown
   - `video`: Click **"Choose File"** and upload a video
     - Use: `C:\Users\avijn_th5xjtu\Desktop\code\application\b1.mp4`
4. Click **"Execute"**
5. **Expected Response** (200):
```json
{
  "message": "Video uploaded successfully",
  "session_id": "550e8400-...",
  "exercise_type": "bjump",
  "file_info": {
    "filename": "b1.mp4",
    "size_bytes": 14028800,
    "size_mb": 13.38
  }
}
```

---

### ✅ **Test 4: Start Analysis**

**Endpoint**: `POST /session/{session_id}/analyze`

1. Click on **"POST /session/{session_id}/analyze"**
2. Click **"Try it out"**
3. **Parameters**:
   - `session_id`: Paste your session ID
4. Click **"Execute"**
5. **Expected Response** (200):
```json
{
  "message": "Analysis started",
  "session_id": "550e8400-...",
  "exercise_type": "bjump",
  "status": "processing"
}
```

---

### ✅ **Test 5: Check Status** (Poll multiple times)

**Endpoint**: `GET /session/{session_id}/status`

1. Click on **"GET /session/{session_id}/status"**
2. Click **"Try it out"**
3. **Parameters**:
   - `session_id`: Paste your session ID
4. Click **"Execute"**
5. **Wait 2 seconds and click Execute again** (repeat until completed)

**Response Progression**:

**First call** (0%):
```json
{
  "session_id": "550e8400-...",
  "status": "processing",
  "progress": 0.0,
  "message": "Analyzing video... 0% complete"
}
```

**Second call** (50%):
```json
{
  "session_id": "550e8400-...",
  "status": "processing",
  "progress": 0.5,
  "message": "Analyzing video... 50% complete"
}
```

**Final call** (100%):
```json
{
  "session_id": "550e8400-...",
  "status": "completed",
  "progress": 1.0,
  "message": "Analysis complete! Report ready"
}
```

---

### ✅ **Test 6: Get Performance Report**

**Endpoint**: `GET /session/{session_id}/report`

1. Click on **"GET /session/{session_id}/report"**
2. Click **"Try it out"**
3. **Parameters**:
   - `session_id`: Paste your session ID
4. Click **"Execute"**
5. **Expected Response** (200):
```json
{
  "session_id": "550e8400-...",
  "exercise": "bjump",
  "report": {
    "exercise": "Standing Broad Jump",
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
    "rating": "KEEP PRACTICING",
    "processing_time": 5.62,
    "frames_processed": 236,
    "video_fps": 30
  },
  "timestamp": "2025-12-03T22:31:45"
}
```

**Also check server console** - it will print the formatted report!

---

### ✅ **Test 7: List All Sessions**

**Endpoint**: `GET /sessions`

1. Click on **"GET /sessions"**
2. Click **"Try it out"**
3. Click **"Execute"**
4. **Expected Response** (200):
```json
{
  "total_sessions": 2,
  "sessions": [
    {
      "session_id": "63ee7b28-...",
      "exercise_type": "bjump",
      "status": "completed",
      "created_at": "2025-12-03T21:59:34",
      "progress": 1.0
    },
    {
      "session_id": "4248ab31-...",
      "exercise_type": "bjump",
      "status": "completed",
      "created_at": "2025-12-03T22:01:06",
      "progress": 1.0
    }
  ]
}
```

---

### ✅ **Test 8: Delete Session**

**Endpoint**: `DELETE /session/{session_id}`

1. Click on **"DELETE /session/{session_id}"**
2. Click **"Try it out"**
3. **Parameters**:
   - `session_id`: Paste a session ID you want to delete
4. Click **"Execute"**
5. **Expected Response** (200):
```json
{
  "message": "Session deleted successfully",
  "session_id": "550e8400-..."
}
```

---

## 🎯 Testing All 8 Exercises

Repeat Tests 2-6 for each exercise type:

| Exercise | Code | Video File |
|----------|------|------------|
| **Broad Jump** | `bjump` | b1.mp4 |
| **Vertical Jump** | `vjump` | (any jump video) |
| **Push-ups** | `pushup` | (pushup video) |
| **Squats** | `squat` | (squat video) |
| **Sit-ups** | `situp` | (situp video) |
| **Sit-and-Reach** | `sitnreach` | (flexibility video) |
| **Skipping** | `skipping` | (jump rope video) |
| **Jumping Jacks** | `jumpingjacks` | (jumping jack video) |

### Quick Test for Each Exercise:

1. Create session
2. Upload video with exercise type (e.g., `vjump`, `pushup`)
3. Start analysis
4. Poll status until completed
5. Get report

---

## 🔍 Visual Testing Guide

### FastAPI Docs Interface:

```
┌──────────────────────────────────────────────────┐
│  AI Exercise Trainer API - v2.0                  │
├──────────────────────────────────────────────────┤
│                                                   │
│  [GET]    /                    Health check      │
│  [POST]   /session/create      Create session ◄──┐
│  [POST]   /session/{id}/upload Upload video   ◄──┤
│  [POST]   /session/{id}/analyze Start analysis◄──┤ Complete
│  [GET]    /session/{id}/status Check progress ◄──┤ Workflow
│  [GET]    /session/{id}/report Get report     ◄──┤
│  [DELETE] /session/{id}        Delete session ◄──┘
│  [GET]    /sessions            List all          │
│                                                   │
└──────────────────────────────────────────────────┘
```

---

## 📸 Screenshots Guide

### 1. Open Docs Page
- Navigate to: **http://localhost:8000/docs**
- You'll see Swagger UI interface

### 2. Create Session
- Expand **POST /session/create**
- Click **"Try it out"**
- Click **"Execute"**
- See green response with session ID

### 3. Upload Video
- Expand **POST /session/{session_id}/upload**
- Click **"Try it out"**
- Fill in:
  - session_id: (paste from step 2)
  - exercise_type: bjump
  - video: (choose file)
- Click **"Execute"**

### 4. Monitor Progress
- Keep clicking Execute on **GET /session/{session_id}/status**
- Watch progress: 0% → 25% → 50% → 75% → 100%

### 5. Get Report
- Click **GET /session/{session_id}/report**
- See complete performance metrics

---

## 🧪 Automated Test Script

If you prefer command line testing:

```bash
# Test complete workflow
python test_client.py test

# Test multiple sets
python demo.py sets

# List all sessions
python test_client.py list
```

---

## ✅ Success Criteria

All tests pass if:

1. ✅ **Health Check** returns status "online"
2. ✅ **Create Session** returns unique session_id
3. ✅ **Upload Video** accepts file and returns size
4. ✅ **Start Analysis** returns status "processing"
5. ✅ **Check Status** shows progress 0% → 100%
6. ✅ **Get Report** returns complete metrics
7. ✅ **List Sessions** shows all sessions
8. ✅ **Delete Session** removes session

---

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Kill all Python processes
Get-Process python | Stop-Process -Force

# Restart server
python server.py
```

### Can't Access Docs
- Check server is running: http://localhost:8000/
- Check firewall settings
- Try: http://127.0.0.1:8000/docs

### Video Upload Fails
- Check file size (< 100MB recommended)
- Check file format (MP4, AVI supported)
- Check file path is accessible

### Analysis Fails
- Check server console for error messages
- Verify video has person in frame
- Check YOLO model (yolo11n-pose.pt) exists

---

## 📊 Expected Test Results

### Test Summary:

```
┌──────────────────────────────────────────────────┐
│  TEST RESULTS                                     │
├──────────────────────────────────────────────────┤
│  ✅ Health Check                    200 OK        │
│  ✅ Create Session                  200 OK        │
│  ✅ Upload Video (13.38 MB)         200 OK        │
│  ✅ Start Analysis                  200 OK        │
│  ✅ Check Status (5 polls)          200 OK        │
│  ✅ Get Report                      200 OK        │
│  ✅ List Sessions (2 found)         200 OK        │
│  ✅ Delete Session                  200 OK        │
├──────────────────────────────────────────────────┤
│  Total: 8/8 Tests Passed            SUCCESS ✅    │
└──────────────────────────────────────────────────┘
```

---

## 🎬 Server Console Output

While testing in docs, watch the server console:

```
[63ee7b28] Starting BJUMP analysis...
[63ee7b28] Total frames: 236, FPS: 30
[63ee7b28] Progress: 10%
[63ee7b28] Progress: 20%
...
[63ee7b28] Progress: 100%
[63ee7b28] Analysis complete! Generating report...

==================================================
STANDING BROAD JUMP PERFORMANCE REPORT
==================================================
Duration: 6.4s
Total Jumps: 2 | Valid Jumps: 1
Accuracy: 50.0%
...
FINAL PERFORMANCE SCORE: 0.97 | KEEP PRACTICING
==================================================

[63ee7b28] Report generated successfully!
```

---

## 📝 Quick Reference

### Server URLs:
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **API**: http://localhost:8000

### Test Order:
1. Health Check → 2. Create → 3. Upload → 4. Analyze → 5. Status → 6. Report

### Exercise Codes:
`bjump` | `vjump` | `pushup` | `squat` | `situp` | `sitnreach` | `skipping` | `jumpingjacks`

---

**Ready to test! Open http://localhost:8000/docs and follow the steps above!** 🚀
