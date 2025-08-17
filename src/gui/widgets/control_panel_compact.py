"""
Compact control panel widget with collapsible sections
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QSlider, QLabel, QComboBox,
    QCheckBox, QSpinBox, QGridLayout, QToolBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
import logging

logger = logging.getLogger(__name__)

class CompactControlPanel(QWidget):
    """Compact control panel with collapsible sections"""
    
    # Signal definitions (same as original)
    play_pause_clicked = Signal()
    screenshot_clicked = Signal()
    confidence_changed = Signal(float)
    model_changed = Signal(str)
    center_display_toggled = Signal(bool)
    reset_stats_clicked = Signal()
    camera_settings_changed = Signal(dict)
    face_detection_toggled = Signal(bool)
    age_gender_toggled = Signal(bool)
    face_confidence_changed = Signal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.is_playing = True
        self.setup_ui()
    
    def setup_ui(self):
        """Setup UI with collapsible sections"""
        layout = QVBoxLayout(self)
        layout.setSpacing(5)  # Reduce spacing
        
        # Main controls (always visible)
        layout.addWidget(self.create_main_controls())
        
        # Collapsible sections using QToolBox
        self.toolbox = QToolBox()
        self.toolbox.setStyleSheet("""
            QToolBox::tab {
                background: #454545;
                color: white;
                padding: 5px;
                border-radius: 3px;
                margin-bottom: 2px;
            }
            QToolBox::tab:selected {
                background: #2196F3;
                font-weight: bold;
            }
        """)
        
        # Add collapsible sections
        self.toolbox.addItem(self.create_detection_settings(), "Detection Settings")
        self.toolbox.addItem(self.create_face_detection_settings(), "Face & Age/Gender")
        self.toolbox.addItem(self.create_camera_settings(), "Camera Settings")
        
        layout.addWidget(self.toolbox)
        
        # Statistics (always visible but compact)
        layout.addWidget(self.create_compact_statistics())
        
        # Add stretch
        layout.addStretch()
    
    def create_main_controls(self) -> QGroupBox:
        """Create main controls (compact version)"""
        group = QGroupBox("Controls")
        layout = QHBoxLayout()  # Horizontal layout for compactness
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("⏸")
        self.play_pause_btn.setMaximumWidth(40)
        self.play_pause_btn.setMinimumHeight(35)
        self.play_pause_btn.setToolTip("Play/Pause Detection")
        self.play_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.play_pause_btn.clicked.connect(self.on_play_pause_clicked)
        layout.addWidget(self.play_pause_btn)
        
        # Screenshot button
        self.screenshot_btn = QPushButton("📷")
        self.screenshot_btn.setMaximumWidth(40)
        self.screenshot_btn.setMinimumHeight(35)
        self.screenshot_btn.setToolTip("Take Screenshot")
        self.screenshot_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.screenshot_btn.clicked.connect(self.screenshot_clicked.emit)
        layout.addWidget(self.screenshot_btn)
        
        # Reset button
        self.reset_stats_btn = QPushButton("🔄")
        self.reset_stats_btn.setMaximumWidth(40)
        self.reset_stats_btn.setMinimumHeight(35)
        self.reset_stats_btn.setToolTip("Reset Statistics")
        self.reset_stats_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        self.reset_stats_btn.clicked.connect(self.reset_stats_clicked.emit)
        layout.addWidget(self.reset_stats_btn)
        
        layout.addStretch()
        group.setLayout(layout)
        return group
    
    def create_detection_settings(self) -> QWidget:
        """Create detection settings widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n.pt",
            "yolo11s.pt",
            "yolo11m.pt"
        ])
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Confidence slider (compact)
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Conf:"))
        self.confidence_slider = QSlider(Qt.Horizontal)
        self.confidence_slider.setMinimum(10)
        self.confidence_slider.setMaximum(95)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.on_confidence_changed)
        conf_layout.addWidget(self.confidence_slider)
        self.confidence_value_label = QLabel("0.50")
        self.confidence_value_label.setMinimumWidth(35)
        conf_layout.addWidget(self.confidence_value_label)
        layout.addLayout(conf_layout)
        
        # Center display checkbox
        self.center_display_check = QCheckBox("Show center points")
        self.center_display_check.toggled.connect(self.center_display_toggled.emit)
        layout.addWidget(self.center_display_check)
        
        widget.setLayout(layout)
        return widget
    
    def create_face_detection_settings(self) -> QWidget:
        """Create face detection settings widget"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Face detection checkbox
        self.face_detection_check = QCheckBox("Enable Face Detection")
        self.face_detection_check.toggled.connect(self.on_face_detection_toggled)
        layout.addWidget(self.face_detection_check)
        
        # Age/gender checkbox
        self.age_gender_check = QCheckBox("Enable Age/Gender")
        self.age_gender_check.setEnabled(False)
        self.age_gender_check.toggled.connect(self.age_gender_toggled.emit)
        layout.addWidget(self.age_gender_check)
        
        # Face confidence slider (compact)
        face_conf_layout = QHBoxLayout()
        face_conf_layout.addWidget(QLabel("Face:"))
        self.face_confidence_slider = QSlider(Qt.Horizontal)
        self.face_confidence_slider.setMinimum(50)
        self.face_confidence_slider.setMaximum(95)
        self.face_confidence_slider.setValue(80)
        self.face_confidence_slider.setEnabled(False)
        self.face_confidence_slider.valueChanged.connect(self.on_face_confidence_changed)
        face_conf_layout.addWidget(self.face_confidence_slider)
        self.face_confidence_value_label = QLabel("0.80")
        self.face_confidence_value_label.setMinimumWidth(35)
        face_conf_layout.addWidget(self.face_confidence_value_label)
        layout.addLayout(face_conf_layout)
        
        # Face stats (compact)
        stats_layout = QGridLayout()
        stats_layout.addWidget(QLabel("Faces:"), 0, 0)
        self.face_count_label = QLabel("0")
        stats_layout.addWidget(self.face_count_label, 0, 1)
        stats_layout.addWidget(QLabel("Gender:"), 1, 0)
        self.gender_label = QLabel("M:0 F:0")
        stats_layout.addWidget(self.gender_label, 1, 1)
        layout.addLayout(stats_layout)
        
        widget.setLayout(layout)
        return widget
    
    def create_camera_settings(self) -> QWidget:
        """Create camera settings widget"""
        widget = QWidget()
        layout = QGridLayout()
        layout.setSpacing(5)
        
        # Camera index
        layout.addWidget(QLabel("Camera:"), 0, 0)
        self.camera_spin = QSpinBox()
        self.camera_spin.setMinimum(0)
        self.camera_spin.setMaximum(10)
        self.camera_spin.setValue(0)
        layout.addWidget(self.camera_spin, 0, 1)
        
        # Resolution
        layout.addWidget(QLabel("Res:"), 1, 0)
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "1280x720",
            "640x480"
        ])
        layout.addWidget(self.resolution_combo, 1, 1)
        
        # FPS
        layout.addWidget(QLabel("FPS:"), 2, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setMinimum(10)
        self.fps_spin.setMaximum(60)
        self.fps_spin.setValue(30)
        layout.addWidget(self.fps_spin, 2, 1)
        
        # Apply button
        self.apply_camera_btn = QPushButton("Apply")
        self.apply_camera_btn.clicked.connect(self.on_camera_settings_apply)
        layout.addWidget(self.apply_camera_btn, 3, 0, 1, 2)
        
        widget.setLayout(layout)
        return widget
    
    def create_compact_statistics(self) -> QGroupBox:
        """Create compact statistics display"""
        group = QGroupBox("Statistics")
        layout = QGridLayout()
        layout.setSpacing(2)
        
        # Compact font
        stats_font = QFont()
        stats_font.setPointSize(9)
        
        # FPS
        layout.addWidget(QLabel("FPS:"), 0, 0)
        self.fps_label = QLabel("0.0")
        self.fps_label.setFont(stats_font)
        self.fps_label.setStyleSheet("color: #00ff00;")
        layout.addWidget(self.fps_label, 0, 1)
        
        # Processing time
        layout.addWidget(QLabel("Time:"), 0, 2)
        self.processing_time_label = QLabel("0ms")
        self.processing_time_label.setFont(stats_font)
        self.processing_time_label.setStyleSheet("color: #00ffff;")
        layout.addWidget(self.processing_time_label, 0, 3)
        
        # Detection count
        layout.addWidget(QLabel("Persons:"), 1, 0)
        self.detection_count_label = QLabel("0")
        self.detection_count_label.setFont(stats_font)
        self.detection_count_label.setStyleSheet("color: #ffff00;")
        layout.addWidget(self.detection_count_label, 1, 1)
        
        # Total frames
        layout.addWidget(QLabel("Frames:"), 1, 2)
        self.total_frames_label = QLabel("0")
        self.total_frames_label.setFont(stats_font)
        layout.addWidget(self.total_frames_label, 1, 3)
        
        # Total detections (hidden for compactness)
        self.total_detections_label = QLabel("0")
        self.total_detections_label.setVisible(False)
        
        group.setLayout(layout)
        return group
    
    # Event handlers (same as original)
    def on_play_pause_clicked(self):
        """Play/pause button click handler"""
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸")
        else:
            self.play_pause_btn.setText("▶")
        self.play_pause_clicked.emit()
    
    def on_confidence_changed(self, value):
        """Confidence slider change handler"""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")
        self.confidence_changed.emit(confidence)
    
    def on_model_changed(self, text):
        """Model selection change handler"""
        model_name = text.split()[0]
        self.model_changed.emit(model_name)
    
    def on_camera_settings_apply(self):
        """Camera settings apply handler"""
        resolution = self.resolution_combo.currentText().split('x')
        settings = {
            'camera_index': self.camera_spin.value(),
            'width': int(resolution[0]),
            'height': int(resolution[1]),
            'fps': self.fps_spin.value()
        }
        self.camera_settings_changed.emit(settings)
    
    def on_face_detection_toggled(self, checked):
        """Face detection toggle handler"""
        self.age_gender_check.setEnabled(checked)
        self.face_confidence_slider.setEnabled(checked)
        if not checked:
            self.age_gender_check.setChecked(False)
        self.face_detection_toggled.emit(checked)
    
    def on_face_confidence_changed(self, value):
        """Face confidence slider change handler"""
        confidence = value / 100.0
        self.face_confidence_value_label.setText(f"{confidence:.2f}")
        self.face_confidence_changed.emit(confidence)
    
    def update_statistics(self, stats: dict):
        """Update statistics display"""
        self.fps_label.setText(f"{stats.get('fps', 0):.1f}")
        self.processing_time_label.setText(
            f"{stats.get('processing_time', 0) * 1000:.0f}ms"
        )
        self.detection_count_label.setText(
            str(stats.get('person_count', 0))
        )
        self.total_frames_label.setText(
            str(stats.get('frame_count', 0))
        )
        
        # Update face statistics
        if 'face_count' in stats:
            self.face_count_label.setText(str(stats['face_count']))
        
        if 'gender_distribution' in stats:
            gender_dist = stats['gender_distribution']
            male = gender_dist.get('Male', 0)
            female = gender_dist.get('Female', 0)
            self.gender_label.setText(f"M:{male} F:{female}")
    
    def set_play_state(self, is_playing: bool):
        """Set play/pause state programmatically"""
        self.is_playing = is_playing
        if self.is_playing:
            self.play_pause_btn.setText("⏸")
        else:
            self.play_pause_btn.setText("▶")