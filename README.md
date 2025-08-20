# YOLOv11 Person Detection System v2.2.0

Real-time person detection system with face detection and age/gender estimation capabilities. Powered by the latest YOLOv11 (2025) models and Caffe deep learning models.

## 🌟 Key Features

- **Dual Mode Operation**: Stream mode (webcam) and File mode (video processing)
- **Tab Interface**: Easy switching between Stream and File modes
- **Latest Technology**: YOLOv11 (2025's newest model) with up to 54.7% mAP
- **Face Detection**: Advanced face detection with YuNet model
- **Age/Gender Estimation**: Deep learning-based age and gender prediction using Caffe models
- **GUI & CUI Support**: PySide6 GUI with tab interface or command-line interface
- **High Performance**: Real-time detection at 30+ FPS
- **Export Formats**: JSON, CSV, XML for detection data
- **Drag & Drop**: Support for video file drag and drop
- **Resource Management**: Automatic pause/resume when switching tabs
- **Auto Model Download**: Automatic download of required models on first run

## 📋 Requirements

- Python 3.12+
- CUDA (optional, for GPU acceleration)
- Webcam (for Stream mode)
- 4GB+ RAM recommended

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/person-face-age-gender-detector.git
cd person-face-age-gender-detector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download age/gender models (automatic on first run, or manual)
python download_models.py
```

## 🤖 Age/Gender Models

The system uses Caffe models for age and gender estimation:
- Models will be automatically downloaded on first GUI launch (~90MB)
- Or manually download using: `python download_models.py`
- Models are saved to `models/age_gender_caffe/`

## 📖 Usage

### GUI Mode (Recommended) - v2.2.0

```bash
python gui_main.py
```

#### 🎥 Stream Tab (Webcam)
- Real-time person detection from webcam
- Face detection with tracking
- Age and gender estimation
- Live statistics display
- Adjustable confidence threshold
- Screenshot capture
- Model selection dropdown

#### 📁 File Tab (Video Processing)
- Process video files (MP4, AVI, MOV, etc.)
- Face detection and age/gender analysis
- Batch processing support
- Progress bar with ETA
- Export options:
  - Annotated video with bounding boxes
  - JSON detection data (including face/age/gender)
  - CSV detection data
  - Individual frames (optional)

#### 🎛️ Features Available in Both Tabs
- **Face Detection**: Toggle on/off
- **Age/Gender Estimation**: Deep learning-based prediction
- **Performance Monitoring**: FPS and processing time
- **Export Results**: Save detection data in multiple formats

### CUI Mode (Legacy)

```bash
# Basic usage
python main.py

# Advanced options
python main.py --model yolo11x.pt --confidence 0.7
python main.py --camera 0 --width 1920 --height 1080
python main.py --show-center --debug
```

## 🎮 Controls

### GUI Tab Interface
- **Stream Tab**: Click to switch to webcam mode
- **File Tab**: Click to switch to video processing mode
- **Auto Resource Management**: Stream pauses when switching to File tab

### Stream Tab Controls
- **Play/Pause**: Toggle detection
- **Screenshot**: Save current frame
- **Reset Stats**: Clear statistics
- **Confidence Slider**: Adjust detection threshold (0.1-0.95)
- **Model Selection**: Choose YOLOv11 variant

### File Tab Controls
- **Browse**: Select video file
- **Drag & Drop**: Drop video files directly
- **Process Settings**: 
  - Model selection
  - Confidence threshold
  - Frame skip rate
  - Output formats
- **Start/Stop**: Control processing

### CUI Keyboard Shortcuts
| Key | Function |
|-----|----------|
| `q` / `ESC` | Exit |
| `p` | Pause/Resume |
| `s` | Screenshot |
| `+` / `-` | Adjust confidence |
| `r` | Reset statistics |
| `c` | Toggle center display |

## 📂 Project Structure

```
person-face-age-gender-detector/
├── gui_main.py              # GUI application entry point
├── main.py                  # CUI application entry point
├── requirements.txt         # Python dependencies
├── CHANGELOG.md            # Version history
├── src/
│   ├── core/               # Core functionality
│   │   ├── detector.py     # YOLOv11 detection engine
│   │   ├── camera.py       # Camera capture
│   │   └── common.py       # Shared utilities
│   ├── gui/                # GUI components (PySide6)
│   │   ├── windows/        # Main window
│   │   │   └── main_window.py
│   │   ├── widgets/        # UI widgets
│   │   │   ├── video_display.py
│   │   │   ├── control_panel.py
│   │   │   └── file_processor.py
│   │   ├── workers/        # Background threads
│   │   │   ├── base_worker.py
│   │   │   ├── yolo_worker.py
│   │   │   └── file_worker.py
│   │   └── README.md       # GUI documentation
│   ├── ui/                 # Shared UI components
│   │   └── visualizer.py   # Detection visualization
│   └── utils/              # Utilities
│       ├── performance.py  # Performance monitoring
│       ├── version.py      # Version management
│       └── coordinate_exporter.py  # Data export
├── debug/                  # Testing & debugging
│   ├── test_detector.py    # Detection tests
│   ├── test_gui.py        # GUI tests
│   ├── test_gui_tabs.py   # Tab interface tests
│   ├── test_file_processing.py  # File mode tests
│   └── sample_video/       # Test videos
├── docs/                   # Documentation
│   ├── model_evaluation.md
│   ├── optimization_guide.md
│   └── code_review_report.md
└── .venv/                  # Python virtual environment
```

## 🤖 Model Performance

| Model | mAP (%) | Speed | GPU Memory | Use Case |
|-------|---------|-------|------------|----------|
| yolo11n.pt | 39.5 | 170+ FPS | 0.5 GB | Real-time, low-end hardware |
| yolo11s.pt | 47.0 | 140 FPS | 0.9 GB | Balanced performance |
| yolo11m.pt | 51.5 | 90 FPS | 1.8 GB | Production use |
| yolo11l.pt | 53.4 | 60 FPS | 2.5 GB | High accuracy |
| yolo11x.pt | 54.7 | 40 FPS | 3.5 GB | Maximum accuracy |

## 📊 Export Formats

### JSON Format
```json
{
  "frame": 1,
  "timestamp": 0.033,
  "detections": [
    {
      "bbox": [100, 200, 300, 400],
      "confidence": 0.95,
      "class": "person",
      "center": [200, 300]
    }
  ]
}
```

### CSV Format
```csv
Frame,Timestamp,Detection_ID,X1,Y1,X2,Y2,Confidence,Class
1,0.033,0,100,200,300,400,0.95,person
```

## ⚡ Performance Tips

### For Higher FPS
- Use lighter models (yolo11n.pt or yolo11s.pt)
- Reduce resolution: `--width 640 --height 480`
- Increase frame skip in File mode
- Use GPU if available

### For Better Accuracy
- Use heavier models (yolo11m.pt or yolo11x.pt)
- Increase confidence threshold
- Process every frame (no frame skip)
- Higher resolution input

## 🔧 Troubleshooting

### Camera Not Found
```bash
# Try different camera index
python main.py --camera 1
# Or check camera permissions
```

### Low FPS
```bash
# Use lighter model
python main.py --model yolo11n.pt
# Or reduce resolution
python main.py --width 640 --height 480
```

### GPU Not Detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

### File Processing Error
- Ensure video codec is supported (H.264 recommended)
- Check file path has no special characters
- Verify sufficient disk space for output

## 🆕 What's New in v2.1.0

- **Tab Interface**: Stream and File modes in separate tabs
- **File Processing Mode**: Process video files with export options
- **Auto Resource Management**: Smart pause/resume when switching tabs
- **Drag & Drop**: Direct video file dropping
- **Multiple Export Formats**: JSON, CSV, XML support
- **Base Worker Class**: Improved code reusability
- **Common Utilities**: Shared functions for detection processing
- **Bug Fixes**: Tuple conversion, text encoding issues

## 📝 Version History

See [CHANGELOG.md](CHANGELOG.md) for detailed version history.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) for the detection model
- [PySide6](https://www.qt.io/qt-for-python) for the GUI framework
- [OpenCV](https://opencv.org/) for image processing

## 📧 Contact

For issues and questions, please use the [GitHub Issues](https://github.com/yourusername/person-face-age-gender-detector/issues) page.

---
**Version**: 2.1.0 "Dual Vision"  
**Build**: 20250817.2  
**Last Updated**: August 17, 2025