"""
Enhanced control panel with face detection and age/gender controls
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QComboBox,
    QCheckBox, QSpinBox, QGridLayout, QFrame
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)


class EnhancedControlPanel(QWidget):
    """Enhanced control panel with face and age/gender controls"""
    
    # Signals
    play_pause_clicked = Signal()
    screenshot_clicked = Signal()
    confidence_changed = Signal(float)
    model_changed = Signal(str)
    center_display_toggled = Signal(bool)
    reset_stats_clicked = Signal()
    camera_settings_changed = Signal(dict)
    
    # Face detection signals
    face_detection_toggled = Signal(bool)
    age_gender_toggled = Signal(bool)
    face_confidence_changed = Signal(float)
    detection_mode_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.is_playing = True
        self.face_detection_enabled = True
        self.age_gender_enabled = True
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Main controls
        layout.addWidget(self.create_main_controls())
        
        # Detection settings
        layout.addWidget(self.create_detection_settings())
        
        # Face detection settings
        layout.addWidget(self.create_face_detection_settings())
        
        # Camera settings
        layout.addWidget(self.create_camera_settings())
        
        # Statistics display
        layout.addWidget(self.create_statistics_display())
        
        # Enhanced statistics
        layout.addWidget(self.create_enhanced_statistics())
        
        # Add stretch
        layout.addStretch()
    
    def create_main_controls(self) -> QGroupBox:
        """Create main controls"""
        group = QGroupBox("Main Controls")
        layout = QVBoxLayout()
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("⏸ Pause")
        self.play_pause_btn.setMinimumHeight(40)
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.play_pause_btn.clicked.connect(self.on_play_pause_clicked)
        layout.addWidget(self.play_pause_btn)
        
        # Screenshot button
        self.screenshot_btn = QPushButton("📷 Screenshot")
        self.screenshot_btn.setMinimumHeight(35)
        self.screenshot_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
        layout.addWidget(self.screenshot_btn)
        
        # Reset stats button
        self.reset_stats_btn = QPushButton("🔄 Reset Stats")
        self.reset_stats_btn.setMinimumHeight(30)
        self.reset_stats_btn.clicked.connect(self.reset_stats_clicked.emit)
        layout.addWidget(self.reset_stats_btn)
        
        group.setLayout(layout)
        return group
    
    def create_detection_settings(self) -> QGroupBox:
        """Create detection settings"""
        group = QGroupBox("Person Detection")
        layout = QGridLayout()
        
        # Model selection
        layout.addWidget(QLabel("Model:"), 0, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt",
            "yolo11l.pt",
            "yolo11x.pt"
        ])
        self.model_combo.currentTextChanged.connect(self.model_changed.emit)
        layout.addWidget(self.model_combo, 0, 1)
        
        # Confidence threshold
        layout.addWidget(QLabel("Confidence:"), 1, 0)
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        layout.addWidget(self.confidence_slider, 1, 1)
        
        self.confidence_label = QLabel("0.50")
        layout.addWidget(self.confidence_label, 1, 2)
        
        # Center display checkbox
        self.center_checkbox = QCheckBox("Show Center Points")
        self.center_checkbox.toggled.connect(self.center_display_toggled.emit)
        layout.addWidget(self.center_checkbox, 2, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_face_detection_settings(self) -> QGroupBox:
        """Create face detection settings"""
        group = QGroupBox("Face & Age/Gender Detection")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #FF6B6B;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #FF6B6B;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Face detection toggle
        self.face_detection_checkbox = QCheckBox("Enable Face Detection")
        self.face_detection_checkbox.setChecked(True)
        self.face_detection_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                color: #333;
            }
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
            }
        """)
        self.face_detection_checkbox.toggled.connect(self.on_face_detection_toggled)
        layout.addWidget(self.face_detection_checkbox)
        
        # Age/Gender estimation toggle
        self.age_gender_checkbox = QCheckBox("Enable Age/Gender Estimation")
        self.age_gender_checkbox.setChecked(True)
        self.age_gender_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 13px;
                color: #333;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3;
            }
        """)
        self.age_gender_checkbox.toggled.connect(self.on_age_gender_toggled)
        layout.addWidget(self.age_gender_checkbox)
        
        # Detection mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        
        self.detection_mode_combo = QComboBox()
        self.detection_mode_combo.addItems([
            "Fast (Skip frames)",
            "Balanced",
            "Accurate (Every frame)"
        ])
        self.detection_mode_combo.setCurrentIndex(1)
        self.detection_mode_combo.currentTextChanged.connect(self.on_detection_mode_changed)
        mode_layout.addWidget(self.detection_mode_combo)
        
        layout.addLayout(mode_layout)
        
        # Face confidence threshold
        face_conf_layout = QHBoxLayout()
        face_conf_layout.addWidget(QLabel("Face Conf:"))
        
        self.face_confidence_slider = QSlider(Qt.Horizontal)
        self.face_confidence_slider.setRange(1, 100)
        self.face_confidence_slider.setValue(70)
        self.face_confidence_slider.valueChanged.connect(self.on_face_confidence_changed)
        face_conf_layout.addWidget(self.face_confidence_slider)
        
        self.face_confidence_label = QLabel("0.70")
        face_conf_layout.addWidget(self.face_confidence_label)
        
        layout.addLayout(face_conf_layout)
        
        group.setLayout(layout)
        return group
    
    def create_camera_settings(self) -> QGroupBox:
        """Create camera settings"""
        group = QGroupBox("Camera Settings")
        layout = QGridLayout()
        
        # Camera index
        layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 10)
        self.camera_spin.setValue(0)
        layout.addWidget(self.camera_spin, 0, 1)
        
        # Resolution
        layout.addWidget(QLabel("Resolution:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "640x480",
            "1280x720",
            "1920x1080"
        ])
        self.resolution_combo.setCurrentIndex(1)
        layout.addWidget(self.resolution_combo, 1, 1)
        
        # FPS
        layout.addWidget(QLabel("Target FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(30)
        layout.addWidget(self.fps_spin, 2, 1)
        
        # Apply button
        self.apply_camera_btn = QPushButton("Apply")
        self.apply_camera_btn.clicked.connect(self.on_camera_settings_apply)
        layout.addWidget(self.apply_camera_btn, 3, 0, 1, 2)
        
        group.setLayout(layout)
        return group
    
    def create_statistics_display(self) -> QGroupBox:
        """Create statistics display"""
        group = QGroupBox("Statistics")
        layout = QGridLayout()
        
        # FPS
        layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setStyleSheet("font-weight: bold; color: #2196F3;")
        layout.addWidget(self.fps_label, 0, 1)
        
        # Person count
        layout.addWidget(QLabel("Persons:"), 1, 0)
        self.person_count_label = QLabel("0")
        self.person_count_label.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout.addWidget(self.person_count_label, 1, 1)
        
        # Total detections
        layout.addWidget(QLabel("Total:"), 2, 0)
        self.total_detections_label = QLabel("0")
        layout.addWidget(self.total_detections_label, 2, 1)
        
        # Average confidence
        layout.addWidget(QLabel("Avg Conf:"), 3, 0)
        self.avg_confidence_label = QLabel("0.00")
        layout.addWidget(self.avg_confidence_label, 3, 1)
        
        group.setLayout(layout)
        return group
    
    def create_enhanced_statistics(self) -> QGroupBox:
        """Create enhanced statistics for face and age/gender"""
        group = QGroupBox("Face & Age/Gender Stats")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #9C27B0;
                border-radius: 5px;
                margin-top: 10px;
            }
            QGroupBox::title {
                color: #9C27B0;
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        layout = QGridLayout()
        
        # Face count
        layout.addWidget(QLabel("Faces:"), 0, 0)
        self.face_count_label = QLabel("0")
        self.face_count_label.setStyleSheet("font-weight: bold; color: #FF6B6B;")
        layout.addWidget(self.face_count_label, 0, 1)
        
        # Gender distribution
        layout.addWidget(QLabel("Gender:"), 1, 0)
        self.gender_label = QLabel("M:0 F:0")
        self.gender_label.setStyleSheet("color: #673AB7;")
        layout.addWidget(self.gender_label, 1, 1)
        
        # Age distribution
        layout.addWidget(QLabel("Ages:"), 2, 0)
        self.age_label = QLabel("N/A")
        self.age_label.setStyleSheet("color: #3F51B5;")
        layout.addWidget(self.age_label, 2, 1)
        
        # Processing time
        layout.addWidget(QLabel("Process:"), 3, 0)
        self.process_time_label = QLabel("0ms")
        self.process_time_label.setStyleSheet("color: #607D8B;")
        layout.addWidget(self.process_time_label, 3, 1)
        
        group.setLayout(layout)
        return group
    
    # Slot methods
    def on_play_pause_clicked(self):
        """Handle play/pause button click"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸ Pause")
        else:
            self.play_pause_btn.setText("▶ Play")
        self.play_pause_clicked.emit()
    
    def on_confidence_changed(self, value):
        """Handle confidence slider change"""
        confidence = value / 100.0
        self.confidence_label.setText(f"{confidence:.2f}")
        self.confidence_changed.emit(confidence)
    
    def on_face_confidence_changed(self, value):
        """Handle face confidence slider change"""
        confidence = value / 100.0
        self.face_confidence_label.setText(f"{confidence:.2f}")
        self.face_confidence_changed.emit(confidence)
    
    def on_face_detection_toggled(self, checked):
        """Handle face detection toggle"""
        self.face_detection_enabled = checked
        self.age_gender_checkbox.setEnabled(checked)
        if not checked:
            self.age_gender_checkbox.setChecked(False)
        self.face_detection_toggled.emit(checked)
    
    def on_age_gender_toggled(self, checked):
        """Handle age/gender toggle"""
        self.age_gender_enabled = checked
        self.age_gender_toggled.emit(checked)
    
    def on_detection_mode_changed(self, mode):
        """Handle detection mode change"""
        self.detection_mode_changed.emit(mode)
    
    def on_camera_settings_apply(self):
        """Apply camera settings"""
        settings = {
            'camera_index': self.camera_spin.value(),
            'resolution': self.resolution_combo.currentText(),
            'fps': self.fps_spin.value()
        }
        self.camera_settings_changed.emit(settings)
    
    def update_statistics(self, stats):
        """Update statistics display"""
        # Basic stats
        self.fps_label.setText(f"{stats.get('fps', 0):.1f}")
        self.person_count_label.setText(str(stats.get('person_count', 0)))
        self.total_detections_label.setText(str(stats.get('frame_count', 0)))
        
        avg_conf = stats.get('avg_person_confidence', 0)
        self.avg_confidence_label.setText(f"{avg_conf:.2f}")
        
        # Face stats
        self.face_count_label.setText(str(stats.get('face_count', 0)))
        
        # Gender distribution
        gender_dist = stats.get('gender_distribution', {})
        male = gender_dist.get('Male', 0)
        female = gender_dist.get('Female', 0)
        self.gender_label.setText(f"M:{male} F:{female}")
        
        # Age distribution
        age_dist = stats.get('age_distribution', {})
        if age_dist:
            # Show top age ranges
            age_text = ", ".join([f"{k}:{v}" for k, v in list(age_dist.items())[:3]])
            self.age_label.setText(age_text[:20])  # Limit length
        else:
            self.age_label.setText("N/A")
        
        # Processing time
        proc_time = stats.get('processing_time', {})
        total_time = proc_time.get('total', 0) * 1000  # Convert to ms
        self.process_time_label.setText(f"{total_time:.1f}ms")