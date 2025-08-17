#!/usr/bin/env python3
"""
Enhanced GUI application with face detection and age/gender estimation
"""

import sys
import os
import cv2
import logging
from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QStatusBar, QMessageBox, QTabWidget, QCheckBox,
    QLabel, QPushButton, QStyle
)
from PySide6.QtCore import Qt, QTimer, Slot, QSettings
from PySide6.QtGui import QAction, QIcon, QKeySequence

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui.widgets.video_display import VideoWidget as VideoDisplay
from src.gui.widgets.enhanced_control_panel import EnhancedControlPanel
from src.gui.widgets.file_processor import FileProcessor
from src.gui.workers.enhanced_worker import EnhancedDetectionWorker
from src.gui.workers.file_worker import FileProcessingWorker
from src.utils.version import VersionInfo
from src.gui.windows.window_fix import WaylandWindowMixin

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedMainWindow(WaylandWindowMixin, QMainWindow):
    """Enhanced main window with face detection capabilities"""
    
    def __init__(self):
        super().__init__()
        
        # Settings
        self.settings = QSettings('PersonDetector', 'EnhancedGUI')
        
        # Workers
        self.detection_worker = None
        self.file_worker = None
        
        # State
        self.stream_was_running = False
        self.face_detection_enabled = True
        self.age_gender_enabled = True
        
        # Initialize UI
        self.setup_ui()
        self.setup_menu()
        self.setup_statusbar()
        self.setup_connections()
        
        # Load settings
        self.load_settings()
        
        # Apply Wayland fixes if needed
        self.apply_wayland_fixes()
        
        logger.info("Enhanced GUI initialized with face detection support")
    
    def setup_ui(self):
        """Setup main UI"""
        self.setWindowTitle(f"Person & Face Detection - {VersionInfo.get_version_string()}")
        self.setMinimumSize(1200, 800)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Tab widget for Stream/File modes
        self.tab_widget = QTabWidget()
        self.tab_widget.currentChanged.connect(self.on_tab_changed)
        
        # Stream tab
        self.stream_tab = self.create_stream_tab()
        self.tab_widget.addTab(self.stream_tab, "🎥 Stream Mode")
        
        # File tab
        self.file_tab = self.create_file_tab()
        self.tab_widget.addTab(self.file_tab, "📁 File Mode")
        
        main_layout.addWidget(self.tab_widget)
        
        # Feature toggle bar
        self.feature_bar = self.create_feature_bar()
        main_layout.addWidget(self.feature_bar)
    
    def create_stream_tab(self) -> QWidget:
        """Create stream tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Video display
        self.video_display = VideoDisplay()
        splitter.addWidget(self.video_display)
        
        # Right: Control panel
        self.control_panel = EnhancedControlPanel()
        splitter.addWidget(self.control_panel)
        
        # Set splitter sizes (70% video, 30% controls)
        splitter.setSizes([700, 300])
        
        layout.addWidget(splitter)
        return tab
    
    def create_file_tab(self) -> QWidget:
        """Create file processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        self.file_processor = FileProcessor()
        layout.addWidget(self.file_processor)
        
        return tab
    
    def create_feature_bar(self) -> QWidget:
        """Create feature toggle bar"""
        bar = QWidget()
        bar.setMaximumHeight(50)
        bar.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                border-top: 1px solid #ccc;
            }
        """)
        
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Feature toggles
        self.face_toggle = QCheckBox("👤 Face Detection")
        self.face_toggle.setChecked(True)
        self.face_toggle.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.face_toggle.toggled.connect(self.on_face_detection_toggled)
        layout.addWidget(self.face_toggle)
        
        self.age_toggle = QCheckBox("🎂 Age/Gender")
        self.age_toggle.setChecked(True)
        self.age_toggle.setStyleSheet("font-size: 13px; font-weight: bold;")
        self.age_toggle.toggled.connect(self.on_age_gender_toggled)
        layout.addWidget(self.age_toggle)
        
        layout.addStretch()
        
        # Status indicator
        self.status_indicator = QLabel("⚪ Ready")
        self.status_indicator.setStyleSheet("font-size: 13px;")
        layout.addWidget(self.status_indicator)
        
        return bar
    
    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        screenshot_action = QAction("Screenshot", self)
        screenshot_action.setShortcut(QKeySequence("Ctrl+S"))
        screenshot_action.triggered.connect(self.capture_screenshot)
        file_menu.addAction(screenshot_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        self.fullscreen_action = QAction("Fullscreen", self)
        self.fullscreen_action.setShortcut(QKeySequence("F11"))
        self.fullscreen_action.setCheckable(True)
        self.fullscreen_action.triggered.connect(self.toggle_fullscreen)
        view_menu.addAction(self.fullscreen_action)
        
        # Detection menu
        detection_menu = menubar.addMenu("Detection")
        
        face_action = QAction("Toggle Face Detection", self)
        face_action.setShortcut(QKeySequence("Ctrl+F"))
        face_action.triggered.connect(lambda: self.face_toggle.toggle())
        detection_menu.addAction(face_action)
        
        age_action = QAction("Toggle Age/Gender", self)
        age_action.setShortcut(QKeySequence("Ctrl+A"))
        age_action.triggered.connect(lambda: self.age_toggle.toggle())
        detection_menu.addAction(age_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_statusbar(self):
        """Setup status bar"""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Mode label
        self.mode_label = QLabel("Stream Mode")
        self.statusbar.addWidget(self.mode_label)
        
        # Separator
        self.statusbar.addWidget(QLabel(" | "))
        
        # Status label
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label)
        
        # Detection info
        self.detection_label = QLabel("")
        self.statusbar.addPermanentWidget(self.detection_label)
    
    def setup_connections(self):
        """Setup signal connections"""
        # Control panel signals
        self.control_panel.play_pause_clicked.connect(self.toggle_detection)
        self.control_panel.screenshot_clicked.connect(self.capture_screenshot)
        self.control_panel.confidence_changed.connect(self.update_confidence)
        self.control_panel.model_changed.connect(self.update_model)
        self.control_panel.center_display_toggled.connect(self.toggle_center_display)
        self.control_panel.reset_stats_clicked.connect(self.reset_statistics)
        
        # Face detection controls
        self.control_panel.face_detection_toggled.connect(self.on_face_detection_toggled)
        self.control_panel.age_gender_toggled.connect(self.on_age_gender_toggled)
        self.control_panel.detection_mode_changed.connect(self.on_detection_mode_changed)
        
        # File processor signals
        self.file_processor.processing_requested.connect(self.start_file_processing)
        self.file_processor.stop_requested.connect(self.stop_file_processing)
    
    def start_detection(self):
        """Start detection"""
        try:
            # Create and configure worker
            self.detection_worker = EnhancedDetectionWorker()
            
            # Configure face detection
            self.detection_worker.config['enable_face_detection'] = self.face_detection_enabled
            self.detection_worker.config['enable_age_gender'] = self.age_gender_enabled
            
            # Connect signals
            self.detection_worker.frame_ready.connect(self.video_display.update_frame)
            self.detection_worker.stats_updated.connect(self.update_statistics)
            self.detection_worker.error_occurred.connect(self.handle_error)
            
            # Start worker
            self.detection_worker.start()
            
            # Update UI
            self.status_label.setText("Detecting...")
            self.status_indicator.setText("🟢 Running")
            self.control_panel.play_pause_btn.setText("⏸ Pause")
            
            logger.info("Detection started with face detection")
            
        except Exception as e:
            logger.error(f"Failed to start detection: {e}")
            self.handle_error(str(e))
    
    def stop_detection(self):
        """Stop detection"""
        if self.detection_worker:
            self.detection_worker.stop()
            self.detection_worker = None
            
            # Update UI
            self.status_label.setText("Stopped")
            self.status_indicator.setText("🔴 Stopped")
            self.control_panel.play_pause_btn.setText("▶ Play")
            
            logger.info("Detection stopped")
    
    def toggle_detection(self):
        """Toggle detection on/off"""
        if self.detection_worker and self.detection_worker.is_running:
            if self.detection_worker.is_paused:
                self.detection_worker.resume()
                self.status_label.setText("Detecting...")
                self.status_indicator.setText("🟢 Running")
            else:
                self.detection_worker.pause()
                self.status_label.setText("Paused")
                self.status_indicator.setText("🟡 Paused")
        else:
            self.start_detection()
    
    @Slot(bool)
    def on_face_detection_toggled(self, enabled):
        """Handle face detection toggle"""
        self.face_detection_enabled = enabled
        
        if self.detection_worker:
            self.detection_worker.toggle_face_detection(enabled)
        
        # Update feature bar
        self.face_toggle.setChecked(enabled)
        
        # Disable age/gender if face detection is off
        if not enabled:
            self.age_toggle.setChecked(False)
            self.age_toggle.setEnabled(False)
        else:
            self.age_toggle.setEnabled(True)
        
        logger.info(f"Face detection {'enabled' if enabled else 'disabled'}")
    
    @Slot(bool)
    def on_age_gender_toggled(self, enabled):
        """Handle age/gender toggle"""
        self.age_gender_enabled = enabled
        
        if self.detection_worker:
            self.detection_worker.toggle_age_gender(enabled)
        
        # Update feature bar
        self.age_toggle.setChecked(enabled)
        
        logger.info(f"Age/gender estimation {'enabled' if enabled else 'disabled'}")
    
    @Slot(str)
    def on_detection_mode_changed(self, mode):
        """Handle detection mode change"""
        if self.detection_worker:
            if "Fast" in mode:
                skip_frames = 3
            elif "Accurate" in mode:
                skip_frames = 1
            else:  # Balanced
                skip_frames = 2
            
            self.detection_worker.config['face_skip_frames'] = skip_frames
            self.detection_worker.config['age_skip_frames'] = skip_frames + 1
    
    @Slot(dict)
    def update_statistics(self, stats):
        """Update statistics display"""
        self.control_panel.update_statistics(stats)
        
        # Update status bar
        persons = stats.get('person_count', 0)
        faces = stats.get('face_count', 0)
        fps = stats.get('fps', 0)
        
        self.detection_label.setText(
            f"FPS: {fps:.1f} | Persons: {persons} | Faces: {faces}"
        )
    
    @Slot(int)
    def on_tab_changed(self, index):
        """Handle tab change"""
        if index == 0:  # Stream tab
            self.mode_label.setText("Stream Mode")
            
            # Resume stream if it was running
            if self.stream_was_running and self.detection_worker:
                self.detection_worker.resume()
                
        elif index == 1:  # File tab
            self.mode_label.setText("File Mode")
            
            # Pause stream detection
            if self.detection_worker and self.detection_worker.is_running:
                self.stream_was_running = not self.detection_worker.is_paused
                if self.stream_was_running:
                    self.detection_worker.pause()
            else:
                self.stream_was_running = False
    
    def capture_screenshot(self):
        """Capture screenshot"""
        if self.detection_worker:
            frame = self.detection_worker.capture_screenshot()
            if frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                self.statusbar.showMessage(f"Screenshot saved: {filename}", 3000)
                logger.info(f"Screenshot saved: {filename}")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        if self.isFullScreen():
            self.showNormal()
            self.fullscreen_action.setChecked(False)
        else:
            self.showFullScreen()
            self.fullscreen_action.setChecked(True)
    
    def toggle_center_display(self, enabled):
        """Toggle center point display"""
        if self.detection_worker:
            self.detection_worker.toggle_center_display()
    
    def update_confidence(self, value):
        """Update confidence threshold"""
        if self.detection_worker:
            self.detection_worker.update_confidence_threshold(value)
    
    def update_model(self, model_name):
        """Update detection model"""
        if self.detection_worker:
            self.detection_worker.update_model(model_name)
    
    def reset_statistics(self):
        """Reset statistics"""
        if self.detection_worker:
            self.detection_worker.reset_stats()
    
    def start_file_processing(self, params):
        """Start file processing"""
        try:
            self.file_worker = FileProcessingWorker()
            
            # Add face detection parameters
            params['enable_face_detection'] = self.face_detection_enabled
            params['enable_age_gender'] = self.age_gender_enabled
            
            self.file_worker.set_parameters(params)
            
            # Connect signals
            self.file_worker.progress_updated.connect(self.file_processor.update_progress)
            self.file_worker.processing_completed.connect(self.file_processor.on_processing_completed)
            self.file_worker.error_occurred.connect(self.file_processor.on_processing_error)
            self.file_worker.log_message.connect(self.file_processor.append_log)
            
            # Start processing
            self.file_worker.start()
            
        except Exception as e:
            logger.error(f"Failed to start file processing: {e}")
            self.handle_error(str(e))
    
    def stop_file_processing(self):
        """Stop file processing"""
        if self.file_worker:
            self.file_worker.stop()
            self.file_worker = None
    
    def handle_error(self, error_msg):
        """Handle errors"""
        logger.error(f"Error: {error_msg}")
        QMessageBox.critical(self, "Error", error_msg)
        self.status_label.setText("Error")
        self.status_indicator.setText("🔴 Error")
    
    def show_about(self):
        """Show about dialog"""
        about_text = f"""
        <h2>Person & Face Detection System</h2>
        <p>Version: {VersionInfo.get_version_string()}</p>
        <p><b>Features:</b></p>
        <ul>
        <li>Real-time person detection (YOLOv11)</li>
        <li>Face detection (SCRFD/OpenCV)</li>
        <li>Age and gender estimation</li>
        <li>Video file processing</li>
        </ul>
        <p><b>Technologies:</b></p>
        <ul>
        <li>YOLOv11 for person detection</li>
        <li>InsightFace/OpenCV for face detection</li>
        <li>Deep learning for age/gender estimation</li>
        <li>PySide6 for GUI</li>
        </ul>
        <p>© 2025 Enhanced Detection System</p>
        """
        
        QMessageBox.about(self, "About", about_text)
    
    def load_settings(self):
        """Load application settings"""
        # Window geometry
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # Feature states
        self.face_detection_enabled = self.settings.value("face_detection", True, type=bool)
        self.age_gender_enabled = self.settings.value("age_gender", True, type=bool)
        
        self.face_toggle.setChecked(self.face_detection_enabled)
        self.age_toggle.setChecked(self.age_gender_enabled)
    
    def save_settings(self):
        """Save application settings"""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("face_detection", self.face_detection_enabled)
        self.settings.setValue("age_gender", self.age_gender_enabled)
    
    def closeEvent(self, event):
        """Handle close event"""
        # Stop workers
        if self.detection_worker:
            self.detection_worker.stop()
        if self.file_worker:
            self.file_worker.stop()
        
        # Save settings
        self.save_settings()
        
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("Enhanced Person & Face Detector")
    app.setOrganizationName("PersonDetector")
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    window = EnhancedMainWindow()
    window.show()
    
    # Start detection automatically
    QTimer.singleShot(500, window.start_detection)
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()