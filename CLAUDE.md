# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time person detection system using YOLOv11 (latest 2025 model) with modular architecture designed for extensibility.

## Key Commands

### Development Environment Setup
```bash
# Activate virtual environment (Python 3.12+)
source .venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Basic execution with default YOLOv11n model
python main.py

# High accuracy mode with YOLOv11m or YOLOv11x
python main.py --model yolo11m.pt --confidence 0.7

# Test without camera (system verification)
python debug/test_detector.py

# Debug mode with detailed logging
python main.py --debug
```

### Interactive Controls During Runtime
- `q`/`ESC`: Exit application
- `p`: Pause/resume detection
- `s`: Save screenshot
- `+`/`-`: Adjust confidence threshold
- `r`: Reset statistics
- `c`: Toggle center point display

## Architecture Overview

### Core Detection Pipeline
1. **CameraCapture** (`src/core/camera.py`): Handles video stream acquisition with automatic fallback to alternative camera indices
2. **PersonDetector** (`src/core/detector.py`): YOLOv11 wrapper focusing on person class (ID=0) detection
3. **Visualizer** (`src/ui/visualizer.py`): Draws bounding boxes, labels, and performance metrics
4. **PerformanceMonitor** (`src/utils/performance.py`): Tracks FPS, processing time, and detection statistics

### Model Selection Strategy
- **yolo11n.pt**: Development/testing (170+ FPS, 39.5% mAP)
- **yolo11s.pt**: Balanced performance (47.0% mAP)
- **yolo11m.pt**: Production use (51.5% mAP)
- **yolo11x.pt**: Maximum accuracy (54.7% mAP)

### Key Design Patterns
- **Modular architecture**: All components in `src/` with category-based folder structure
- **Context managers**: CameraCapture and Timer classes support `with` statements
- **Dynamic threshold adjustment**: Confidence threshold can be modified at runtime
- **Automatic model download**: YOLOv11 models download on first use

## Critical Implementation Details

### Person Detection Specifics
- COCO dataset class ID 0 = "person"
- Default confidence threshold: 0.5
- Processes at 1280x720 @ 30fps by default
- CPU-optimized PyTorch backend (no CUDA dependency)

### Performance Considerations
- Frame times tracked with 30-frame sliding window
- Statistics logged every 30 frames
- Supports both CPU and GPU inference (auto-detected)
- Buffer size set to 1 for minimal latency

## Extension Points

### Adding Face/Age/Gender Detection
Extend `PersonDetector.detect()` to process detected person bounding boxes:
```python
# In src/core/detector.py
for detection in person_detections:
    x1, y1, x2, y2 = detection['bbox']
    person_roi = frame[y1:y2, x1:x2]
    # Add face detection on person_roi
```

### Adding Tracking
Integrate ByteTrack or similar:
```python
# Create src/core/tracker.py
import supervision as sv
tracker = sv.ByteTrack()
tracked = tracker.update_with_detections(detections)
```

## Important Files

- `main.py`: Entry point with argument parsing and main loop
- `src/core/detector.py`: Core YOLOv11 detection logic
- `src/core/camera.py`: Camera management with fallback logic
- `src/ui/visualizer.py`: All visualization and overlay functions
- `src/utils/performance.py`: FPS and timing metrics
- `debug/test_detector.py`: Component verification script
- `docs/model_evaluation.md`: YOLOv11 performance analysis
- `docs/optimization_guide.md`: Enhancement recommendations

## Model Files
- Downloaded models stored in project root (e.g., `yolo11n.pt`)
- Ultralytics config at `~/.config/Ultralytics/settings.json`

## Testing Approach
- Unit test substitute: `debug/test_detector.py` verifies component initialization
- Integration testing: Run `python main.py` with camera connected
- Performance validation: Monitor FPS display during runtime

## Common Issues and Solutions

### Camera Not Found
System automatically tries indices 0-4. Manual override: `--camera N`

### Low FPS
Switch to lighter model: `--model yolo11n.pt` or reduce resolution: `--width 640 --height 480`

### Import Errors
Ensure virtual environment is activated and path includes project root