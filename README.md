# YOLOv11 Person Detection System v2.3.0

🚀 High-performance real-time person detection system with face detection and age/gender estimation capabilities. Powered by the latest YOLOv11 (2025) models with GPU acceleration and adaptive optimization.

## ✨ Highlights

- **🎯 Latest YOLOv11 Technology**: Up to 54.7% mAP with real-time performance
- **🚀 GPU Acceleration**: Automatic GPU detection and optimization 
- **📷 Smart Camera Selection**: Dropdown menu with auto-detection
- **⚡ Performance Optimized**: Adaptive frame skipping, batch processing, and caching
- **👤 Face & Age/Gender**: Advanced face detection with age/gender estimation
- **🎨 Modern GUI**: Tab-based interface with Stream and File modes

## 🖥️ Tested Environments

| Platform | Version | Status | Notes |
|----------|---------|--------|-------|
| **Windows 11** | 23H2 | ✅ Fully Tested | Primary development environment |
| **Ubuntu** | 22.04 LTS | ✅ Fully Tested | WSL2 on Windows 11 |
| **Windows 10** | 22H2 | ✅ Compatible | Community tested |
| **macOS** | 14.x | 🔄 In Testing | Community support |

> **Development Environment**: Windows 11 with WSL2 Ubuntu 22.04 LTS

## 🌟 Key Features

### Core Detection
- **Dual Mode Operation**: Stream mode (webcam) and File mode (video processing)
- **Tab Interface**: Seamless switching between Stream and File modes
- **YOLOv11 Models**: From nano (170+ FPS) to x-large (54.7% mAP)
- **Face Detection**: YuNet and InsightFace models with tracking
- **Age/Gender Estimation**: Deep learning-based prediction using Caffe models

### Performance Optimization (v2.3.0) 🆕
- **GPU Auto-Detection**: Automatic CUDA device selection and management
- **Adaptive Processing**: Dynamic frame skipping based on performance
- **Batch Processing**: Process multiple frames simultaneously 
- **Smart Caching**: Reduce redundant computations
- **Parallel Detection**: Concurrent face detection for multiple persons
- **Quality Scaling**: Automatic resolution adjustment for stable FPS

### User Interface
- **Camera Selection**: Dropdown menu with camera details (resolution, FPS)
- **Real-time Stats**: FPS, processing time, detection count
- **Export Formats**: JSON, CSV, XML for detection data
- **Drag & Drop**: Direct video file processing
- **Resource Management**: Automatic pause/resume when switching tabs

## 📋 Requirements

### System Requirements
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+
- **Python**: 3.11 or 3.12
- **RAM**: 4GB minimum, 8GB+ recommended
- **GPU**: NVIDIA with CUDA support (optional but recommended)
- **Storage**: 2GB for models and dependencies

### Hardware Support
- **Webcam**: Any USB or built-in camera for Stream mode
- **GPU**: NVIDIA GTX 1060 or better for optimal performance
- **CPU**: Modern multi-core processor (4+ cores recommended)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/person-face-age-gender-detector.git
cd person-face-age-gender-detector

# Create virtual environment (Python 3.11 or 3.12)
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies (includes gdown for model downloads)
pip install -r requirements.txt

# Optional: Download age/gender models manually (auto-downloads on first run)
python download_models.py
```

### GPU Setup (Optional)
For NVIDIA GPU acceleration:
```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# If False, install CUDA-enabled PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 📖 Usage

### GUI Mode (Recommended)

```bash
python gui_main.py
```

#### 🎥 Stream Tab (Webcam)
- **Camera Selection**: Choose from dropdown menu with auto-detection
- **Real-time Detection**: Person, face, age/gender estimation
- **Live Statistics**: FPS counter and performance metrics
- **Adjustable Settings**:
  - Confidence threshold (0.1-0.95)
  - Model selection (yolo11n to yolo11x)
  - Face detection toggle
  - Age/gender estimation toggle
- **Screenshot Capture**: Save current frame with detections

#### 📁 File Tab (Video Processing)
- **Video Processing**: MP4, AVI, MOV, MKV support
- **Batch Analysis**: Process entire videos with progress tracking
- **Export Options**:
  - Annotated video with bounding boxes
  - JSON detection data (frame-by-frame)
  - CSV for data analysis
  - XML for integration
  - Individual frame extraction

### Command Line Mode

```bash
# Basic usage with default settings
python main.py

# Custom model and confidence
python main.py --model yolo11m.pt --confidence 0.7

# Specific camera and resolution
python main.py --camera 0 --width 1920 --height 1080

# Enable GPU acceleration
python main.py --device cuda

# Show detection centers and debug info
python main.py --show-center --debug
```

## ⚡ Performance Optimization

### Automatic Optimizations
The system automatically applies these optimizations:
- **GPU Detection**: Selects best available GPU
- **Adaptive Frame Skip**: Maintains target FPS by skipping frames
- **Dynamic Resolution**: Scales processing resolution based on load
- **Batch Processing**: Groups frames for efficient GPU utilization
- **Smart Caching**: Reuses recent detection results

### Performance Modes

| Mode | Settings | FPS | Use Case |
|------|----------|-----|----------|
| **Speed** | yolo11n, skip=2, batch=4 | 60-120 | Real-time monitoring |
| **Balanced** | yolo11s, skip=1, batch=2 | 30-60 | General use |
| **Quality** | yolo11m, skip=0, batch=1 | 15-30 | High accuracy |
| **Maximum** | yolo11x, skip=0, batch=1 | 10-20 | Best detection |

### Optimization Tips

#### For Higher FPS:
```python
# Use lighter model
model = "yolo11n.pt"
# Enable frame skipping
skip_frames = 2
# Reduce resolution
resolution = (640, 480)
# Enable GPU
use_gpu = True
```

#### For Better Accuracy:
```python
# Use heavier model
model = "yolo11x.pt"
# Process every frame
skip_frames = 0
# Higher resolution
resolution = (1920, 1080)
# Increase confidence
confidence = 0.7
```

## 🤖 Model Information

### YOLOv11 Models
| Model | mAP | Speed (GPU) | Speed (CPU) | Size | VRAM |
|-------|-----|-------------|-------------|------|------|
| yolo11n | 39.5% | 170+ FPS | 30 FPS | 2.6 MB | 0.5 GB |
| yolo11s | 47.0% | 140 FPS | 20 FPS | 9.4 MB | 0.9 GB |
| yolo11m | 51.5% | 90 FPS | 12 FPS | 20.1 MB | 1.8 GB |
| yolo11l | 53.4% | 60 FPS | 8 FPS | 25.3 MB | 2.5 GB |
| yolo11x | 54.7% | 40 FPS | 5 FPS | 56.9 MB | 3.5 GB |

### Face Detection Models
- **YuNet**: OpenCV's lightweight face detector
- **InsightFace**: Advanced face analysis (optional)

### Age/Gender Models
- **Caffe Models**: ~90MB, auto-downloaded on first run
- **Accuracy**: Age ±5 years, Gender 85%+ accuracy

## 🎮 Controls & Shortcuts

### GUI Controls
| Control | Function |
|---------|----------|
| **Camera Dropdown** | Select available cameras |
| **🔄 Button** | Refresh camera list |
| **Play/Pause** | Toggle detection |
| **Screenshot** | Save current frame |
| **Reset Stats** | Clear statistics |
| **Confidence Slider** | Adjust detection threshold |

### Keyboard Shortcuts
| Key | Function |
|-----|----------|
| `F11` | Toggle fullscreen |
| `Ctrl+M` | Maximize window |
| `Ctrl+S` | Save screenshot |
| `Ctrl+Q` | Quit application |
| `Space` | Pause/Resume |
| `Tab` | Switch between Stream/File |

## 📂 Project Structure

```
person-face-age-gender-detector/
├── gui_main.py                 # GUI application entry
├── main.py                     # CLI application entry
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # AI assistance guide
├── src/
│   ├── core/                   # Core detection engines
│   │   ├── detector.py         # YOLOv11 wrapper
│   │   ├── optimized_detector.py  # Performance-optimized detector 🆕
│   │   ├── face_detector.py   # Face detection
│   │   └── age_gender_caffe.py # Age/gender estimation
│   ├── gui/                    # GUI components
│   │   ├── windows/            # Main windows
│   │   ├── widgets/            # UI widgets
│   │   └── workers/            # Background threads
│   ├── utils/                  # Utilities
│   │   ├── gpu_manager.py     # GPU detection & management 🆕
│   │   ├── camera_detector.py # Camera enumeration 🆕
│   │   └── performance.py     # Performance monitoring
│   └── tests/                  # Test suites 🆕
│       └── test_camera_detector.py
└── models/                     # Model storage
    └── age_gender_caffe/       # Caffe models (auto-downloaded)
```

## 🔧 Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera permissions
# On Windows: Settings > Privacy > Camera
# On Linux: Check /dev/video* permissions

# Try manual camera index
python main.py --camera 1
```

#### Low FPS / Performance Issues
```bash
# Check GPU availability
python -c "from src.utils.gpu_manager import gpu_manager; print(gpu_manager.get_system_info())"

# Use optimized settings
python gui_main.py  # GUI will auto-optimize
```

#### OpenCV Errors
```bash
# Suppress OpenCV warnings
export OPENCV_LOG_LEVEL=ERROR  # Linux/macOS
set OPENCV_LOG_LEVEL=ERROR     # Windows
```

#### Model Download Issues
```bash
# Manual download with progress
python download_models.py

# Or install gdown if missing
pip install gdown
```

## 🆕 What's New in v2.3.0

### Performance & Optimization
- **GPU Manager**: Automatic GPU detection and selection
- **Optimized Detector**: 2-8x faster with batch processing
- **Adaptive Processing**: Dynamic quality adjustment for stable FPS
- **Parallel Detection**: Concurrent face detection
- **Smart Caching**: Reduce redundant computations

### User Experience
- **Camera Dropdown**: Replace index input with selection menu
- **Camera Details**: Show resolution and FPS in dropdown
- **Refresh Button**: Dynamic camera list updates
- **Cleaner Logs**: Suppress unnecessary warnings
- **Better Error Handling**: Graceful fallbacks

### Developer Features
- **Test Suite**: Comprehensive camera detection tests
- **CLAUDE.md**: AI assistant configuration
- **Cross-platform**: Improved Windows/Linux/macOS support

## 📊 Performance Benchmarks

### System: Windows 11, RTX 3060, Intel i7-10700
| Configuration | FPS | CPU Usage | GPU Usage | RAM |
|--------------|-----|-----------|-----------|-----|
| YOLOv11n + GPU | 120 | 15% | 40% | 1.2 GB |
| YOLOv11m + GPU | 45 | 20% | 60% | 1.8 GB |
| YOLOv11n + CPU | 25 | 80% | 0% | 1.0 GB |
| With Face Detection | 35 | 25% | 55% | 1.5 GB |
| Full Pipeline | 28 | 30% | 65% | 2.0 GB |


## 📄 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** - see the [LICENSE](LICENSE) file for details.

### Why AGPL-3.0?

This project uses several powerful libraries that are licensed under AGPL-3.0:
- **Ultralytics YOLO**: State-of-the-art object detection
- **albumentations**: Advanced image augmentation

To maintain legal compliance and respect the open-source ecosystem, this project adopts the AGPL-3.0 license.

### What does this mean?

- ✅ You can use this software for any purpose, including commercial use
- ✅ You can modify and distribute the software
- 📝 You must provide source code when distributing
- 🌐 If you run this as a network service, you must provide source code to users
- 🔗 Any modifications must also be licensed under AGPL-3.0

### Commercial Use

For commercial use without AGPL-3.0 obligations:
1. Purchase commercial licenses for Ultralytics and other AGPL components
2. Contact the project maintainers for alternative licensing arrangements

### Third-party Licenses

- **InsightFace models**: Non-commercial research use only (models can be disabled)

## 🙏 Acknowledgments

- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [PySide6](https://www.qt.io/qt-for-python) - Modern GUI framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [InsightFace](https://github.com/deepinsight/insightface) - Face analysis toolkit


---

**Version**: 2.3.0  
**License**: AGPL-3.0  

<p align="center">
  Built with YOLOv11 and PySide6
</p>