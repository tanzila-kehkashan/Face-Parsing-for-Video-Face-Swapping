# 🎭 FAKESYNCSTUDIO API Setup Guide

Professional AI Face Swapping API with optimal quality settings.

## 📋 Overview

The FAKESYNCSTUDIO API provides a RESTful interface to perform high-quality face swapping with:
- ✨ **CodeFormer Enhancement** (always enabled)
- 🎨 **BiSeNet Face Parsing** (always enabled) 
- 🌊 **Laplacian Blending** (optimal quality)
- ⚡ **GPU Acceleration** (CUDA support)
- 📊 **Real-time Progress Streaming**

## 🏗️ Project Structure

```
your-api-directory/
├── main.py                    # FastAPI server
├── processor.py               # Face swap processing logic
├── settings.py                # Configuration
├── api_utils.py              # Utility functions
├── start_api.py              # Startup script
├── requirements_api.txt      # API dependencies
├── uploads/                  # Temporary upload storage
├── outputs/                  # Processed results
└── logs/                     # API logs
```

## ⚙️ Prerequisites

### System Requirements
- **Python 3.9+**
- **8GB+ RAM** (16GB recommended)
- **NVIDIA GPU** (optional, for acceleration)
- **10GB+ free disk space**

### FAKESYNCSTUDIO Project
The API requires the main FAKESYNCSTUDIO project to be installed and working:
- Download/clone the updated FAKESYNCSTUDIO project
- Ensure all models are downloaded and placed correctly
- Verify the simplified version is working

## 📦 Installation

### Step 1: Install Dependencies

```bash
# Install API-specific dependencies
pip install -r requirements_api.txt

# OR install manually:
pip install fastapi uvicorn python-multipart aiofiles
```

### Step 2: Configure Settings

Edit `settings.py` and update the path to your FAKESYNCSTUDIO installation:

```python
# Update this path to your actual installation
SWAPFACE_DIR = Path("/path/to/your/FAKESYNCSTUDIO")
```

### Step 3: Verify Setup

```bash
# Run the startup script to check everything
python start_api.py
```

This will verify:
- ✅ Python version compatibility
- ✅ Required dependencies  
- ✅ FAKESYNCSTUDIO project access
- ✅ AI model files
- ✅ Directory structure

## 🚀 Starting the API

### Option 1: Using Startup Script (Recommended)
```bash
python start_api.py
```

### Option 2: Direct Launch
```bash
python main.py
```

### Option 3: Production Setup
```bash
uvicorn main:app --host 0.0.0.0 --port 9876 --workers 1
```

## 🌐 API Endpoints

### Core Endpoints

#### 1. **Upload Files for Processing**
```http
POST /upload/
Content-Type: multipart/form-data

Form Fields:
- source_image: Image file (JPG, PNG)
- target_video: Video file (MP4, AVI, MOV)  
- settings: Optional JSON settings
```

**Example Response:**
```json
{
  "job_id": "abc123-def456",
  "status": "processing", 
  "message": "Files uploaded successfully. Processing started with optimal AI settings.",
  "features": [
    "CodeFormer face enhancement",
    "BiSeNet face parsing",
    "Laplacian blending", 
    "Professional quality output"
  ]
}
```

#### 2. **Check Job Status**
```http
GET /jobs/{job_id}/status/
```

**Example Response:**
```json
{
  "status": "completed",
  "output_path": "/path/to/result.mp4",
  "file_size": 15728640,
  "created_at": "2024-01-15T10:30:00",
  "message": "Face swap completed successfully with professional quality"
}
```

#### 3. **Download Result**
```http
GET /jobs/{job_id}/result
```

Returns the processed video file.

#### 4. **Stream Processing Logs**
```http
GET /jobs/{job_id}/stream_logs
```

Returns real-time processing logs via Server-Sent Events.

#### 5. **List All Jobs**
```http
GET /jobs/
```

#### 6. **Health Check**
```http
GET /health
```

## 📱 Usage Examples

### Python Client Example

```python
import requests
import time

# Upload files
with open('source_face.jpg', 'rb') as source, open('target_video.mp4', 'rb') as target:
    response = requests.post('http://localhost:9876/upload/', files={
        'source_image': source,
        'target_video': target
    })

job_data = response.json()
job_id = job_data['job_id']
print(f"Job started: {job_id}")

# Check status
while True:
    status_response = requests.get(f'http://localhost:9876/jobs/{job_id}/status/')
    status = status_response.json()['status']
    
    if status == 'completed':
        # Download result
        result_response = requests.get(f'http://localhost:9876/jobs/{job_id}/result')
        with open(f'result_{job_id}.mp4', 'wb') as f:
            f.write(result_response.content)
        print("Processing completed!")
        break
    elif status == 'failed':
        print("Processing failed!")
        break
    else:
        print(f"Status: {status}")
        time.sleep(5)
```

### cURL Examples

```bash
# Upload files
curl -X POST "http://localhost:9876/upload/" \
  -F "source_image=@source_face.jpg" \
  -F "target_video=@target_video.mp4"

# Check status  
curl "http://localhost:9876/jobs/YOUR_JOB_ID/status/"

# Download result
curl -o result.mp4 "http://localhost:9876/jobs/YOUR_JOB_ID/result"

# Health check
curl "http://localhost:9876/health"
```

### JavaScript/Web Example

```javascript
// Upload files
const formData = new FormData();
formData.append('source_image', sourceImageFile);
formData.append('target_video', targetVideoFile);

const uploadResponse = await fetch('/upload/', {
    method: 'POST',
    body: formData
});

const jobData = await uploadResponse.json();
const jobId = jobData.job_id;

// Stream logs in real-time
const eventSource = new EventSource(`/jobs/${jobId}/stream_logs`);
eventSource.onmessage = function(event) {
    console.log('Log:', event.data);
};

// Check status periodically
const checkStatus = async () => {
    const response = await fetch(`/jobs/${jobId}/status/`);
    const status = await response.json();
    
    if (status.status === 'completed') {
        // Download result
        window.location.href = `/jobs/${jobId}/result`;
    }
};
```

## ⚙️ Configuration Options

### Available Settings (Limited)

The API uses optimal defaults but allows limited customization:

```json
{
  "swap_option": "All Face",        // "All Face", "Biggest", "All Male", "All Female"
  "age": 25,                        // 1-100 (for age-based filtering)
  "distance_slider": 0.6,           // 0.0-2.0 (for specific face matching)
  "face_scale": 1.0                 // 0.1-2.0 (face scaling)
}
```

**Example with custom settings:**
```bash
curl -X POST "http://localhost:9876/upload/" \
  -F "source_image=@source.jpg" \
  -F "target_video=@target.mp4" \
  -F 'settings={"swap_option": "Biggest", "face_scale": 1.2}'
```

### Fixed Optimal Settings

These settings are automatically applied and cannot be changed:
- **Face Enhancement**: CodeFormer (always enabled)
- **Face Parsing**: BiSeNet with optimal mask regions
- **Blending**: Laplacian pyramid with 4 levels
- **Mask Processing**: 10 soft iterations, optimal blur/erode
- **Quality**: Maximum settings for professional output

## 🔧 Troubleshooting

### Common Issues

1. **"SWAPFACE_DIR not found"**
   - Update the path in `settings.py`
   - Ensure FAKESYNCSTUDIO project is properly installed

2. **"Models missing"**
   - Download all required model files
   - Check file paths and permissions

3. **"CUDA not available"**
   - Install CUDA-compatible PyTorch
   - Verify GPU drivers

4. **"Processing failed"**
   - Check API logs and job stream logs
   - Verify input file formats (JPG/PNG for images, MP4/AVI/MOV for videos)
   - Ensure sufficient disk space

5. **"Job stuck in processing"**
   - Check server resources (RAM, disk space)
   - Restart API server if needed
   - Check if source video has detectable faces

### Monitoring

```bash
# Check server health
curl http://localhost:9876/health

# List all jobs  
curl http://localhost:9876/jobs/

# View real-time logs
curl http://localhost:9876/jobs/JOB_ID/stream_logs
```

## 🔒 Security Considerations

- **File Validation**: API validates file types and sizes
- **Temporary Storage**: Upload files are automatically cleaned up
- **Rate Limiting**: Consider adding rate limiting for production
- **Authentication**: Add authentication for production deployments
- **HTTPS**: Use HTTPS in production environments

## 📈 Performance Tips

### Optimization
- Use GPU acceleration when available
- Ensure sufficient RAM (16GB+ recommended)
- Use SSD storage for better I/O performance
- Process shorter videos for faster results

### Scaling
- Run multiple API instances behind a load balancer
- Use shared storage for outputs
- Implement job queuing for high load
- Monitor resource usage

## 🆘 Support

### API Documentation
- Interactive docs: `http://localhost:9876/docs`
- OpenAPI schema: `http://localhost:9876/openapi.json`

### Logs Location
- API logs: Console output
- Job logs: Available via `/jobs/{job_id}/stream_logs`
- Processing logs: Included in job streams

This API provides professional-quality face swapping with zero configuration complexity! 🎯