# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time person detection system with face detection and age/gender estimation capabilities, built with YOLOv11 and modern GUI framework.

## Commands

### Setup and Installation
```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/macOS

# Install dependencies
pip install -r requirements.txt

# Download required models (automatic on first GUI run)
python download_models.py
```

### Running the Application
```bash
# GUI Mode (Recommended)
python gui_main.py

# CLI Mode with options
python main.py --model yolo11n.pt --confidence 0.5 --camera 0

# Enhanced GUI (alternative interface)
python gui_main_enhanced.py
```

### Development Commands
- **No standard test framework** - Testing is manual through GUI/CLI
- **No linting configuration** - Use standard Python conventions
- **Version updates**: Edit `src/utils/version.py` and update CHANGELOG.md

## Architecture

### Core Components
- `src/core/` - Detection engines (YOLO, face detection, age/gender)
  - `detector.py` - Main YOLOPersonDetector class
  - `face_detector.py` - YuNet face detection
  - `age_gender_estimator.py` - Caffe model integration
  - `camera.py` - Camera handling and frame capture
  
- `src/gui/` - PySide6 GUI implementation
  - `main_window.py` - TabWidget-based main interface
  - `widgets/` - Stream and File mode widgets
  - `workers/` - Background processing threads
  - `dialogs/` - Export and settings dialogs

- `src/pipelines/` - Processing pipelines
  - `basic_pipeline.py` - Standard detection flow
  - `optimized_pipeline.py` - Performance-focused variant
  - `stable_pipeline.py` - Reliability-focused variant

### Key Design Patterns
1. **Worker Pattern**: All heavy processing runs in QThread workers to maintain UI responsiveness
2. **Signal/Slot Communication**: PySide6 signals for thread-safe updates
3. **Dual Mode Architecture**: Separate implementations for Stream (real-time) and File (batch) processing
4. **Plugin Architecture**: Interchangeable detection models and pipelines

### Model Management
- **YOLO Models**: Automatic download from Ultralytics (yolo11n.pt to yolo11x.pt)
- **Face Models**: YuNet (OpenCV) and InsightFace options
- **Age/Gender Models**: Caffe models downloaded via `download_models.py`
- **Model Location**: Root directory for YOLO, auto-cached for others

### Important Files
- `gui_main.py:45-150` - Main GUI initialization and tab setup
- `src/gui/widgets/stream_widget.py:200-400` - Real-time detection logic
- `src/gui/workers/detection_worker.py:50-150` - Core detection loop
- `src/core/detector.py:100-250` - YOLO integration and tracking
- `src/core/age_gender_estimator.py:30-120` - Age/gender model loading

## Critical Implementation Notes

### GUI Threading
- Always use QThread workers for detection operations
- Emit signals for UI updates, never directly modify UI from workers
- Use `QMetaObject.invokeMethod` for thread-safe calls when needed

### Performance Considerations
- Default to yolo11n.pt for real-time performance (170+ FPS)
- Frame skipping available in File mode for large videos
- GPU acceleration automatically detected (CUDA/MPS)

### Error Handling
- Models auto-download on first use with progress feedback
- Graceful fallback if face/age models unavailable
- Camera initialization retries with error dialogs

### Export Functionality
- JSON export uses custom encoder for int64 types (src/gui/dialogs/export_dialog.py:180)
- Video export maintains original FPS with annotation overlay
- CSV/XML formats for data analysis workflows

## Common Development Tasks

### Adding New Detection Features
1. Implement detector in `src/core/`
2. Create worker in `src/gui/workers/`
3. Add UI controls to relevant widget
4. Update export formats if needed

### Modifying GUI Layout
1. Edit widgets in `src/gui/widgets/`
2. Update main_window.py for tab changes
3. Maintain signal/slot connections

### Updating YOLO Models
1. Change default in `src/core/detector.py`
2. Update model list in GUI dropdowns
3. Test performance metrics

## Version Management
- Version defined in `src/utils/version.py`
- Update CHANGELOG.md for all releases
- Follow semantic versioning (MAJOR.MINOR.PATCH)