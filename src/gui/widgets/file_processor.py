"""
File processor widget for video file detection
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QProgressBar,
    QComboBox, QCheckBox, QTextEdit, QFileDialog,
    QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QDragEnterEvent, QDropEvent
import os
import logging
from pathlib import Path
from typing import Optional, List

from src.utils.version import SUPPORTED_VIDEO_FORMATS, EXPORT_FORMATS

logger = logging.getLogger(__name__)

class FileProcessorWidget(QWidget):
    """Widget for processing video files"""
    
    # Signals
    process_started = Signal(dict)  # Processing parameters
    process_stopped = Signal()
    file_selected = Signal(str)  # File path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_file = None
        self.processing = False
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        
        # Create splitter for main layout
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # File selection
        left_layout.addWidget(self.create_file_selection())
        
        # Processing options
        left_layout.addWidget(self.create_processing_options())
        
        # Output settings
        left_layout.addWidget(self.create_output_settings())
        
        # Control buttons
        left_layout.addWidget(self.create_control_buttons())
        
        # Progress section
        left_layout.addWidget(self.create_progress_section())
        
        left_layout.addStretch()
        
        # Right panel - Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Results display
        right_layout.addWidget(self.create_results_display())
        
        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 600])
        
        layout.addWidget(splitter)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
        
    def create_file_selection(self) -> QGroupBox:
        """Create file selection section"""
        group = QGroupBox("Input File")
        layout = QVBoxLayout()
        
        # File path input
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Select a video file or drag & drop here...")
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)
        
        # Browse button
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(self.browse_btn)
        
        layout.addLayout(file_layout)
        
        # File info label
        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(self.file_info_label)
        
        # Drag & drop hint
        drop_hint = QLabel("💡 Tip: You can drag and drop video files here")
        drop_hint.setStyleSheet("color: #2196F3; font-size: 11px;")
        layout.addWidget(drop_hint)
        
        group.setLayout(layout)
        return group
        
    def create_processing_options(self) -> QGroupBox:
        """Create processing options section"""
        group = QGroupBox("Processing Options")
        layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "yolo11n.pt (Fastest)",
            "yolo11s.pt (Balanced)",
            "yolo11m.pt (Accurate)",
            "yolo11l.pt (More Accurate)",
            "yolo11x.pt (Most Accurate)"
        ])
        self.model_combo.setCurrentIndex(1)  # Default to balanced
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence:"))
        self.confidence_spin = QComboBox()
        self.confidence_spin.addItems([
            "0.3 (Low)",
            "0.5 (Medium)",
            "0.7 (High)",
            "0.9 (Very High)"
        ])
        self.confidence_spin.setCurrentIndex(1)  # Default to 0.5
        conf_layout.addWidget(self.confidence_spin)
        layout.addLayout(conf_layout)
        
        # Frame skip
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Process every:"))
        self.frame_skip_combo = QComboBox()
        self.frame_skip_combo.addItems([
            "Every frame",
            "Every 2 frames",
            "Every 5 frames",
            "Every 10 frames"
        ])
        skip_layout.addWidget(self.frame_skip_combo)
        layout.addLayout(skip_layout)
        
        # Options checkboxes
        self.draw_boxes_check = QCheckBox("Draw bounding boxes")
        self.draw_boxes_check.setChecked(True)
        layout.addWidget(self.draw_boxes_check)
        
        self.draw_labels_check = QCheckBox("Draw labels")
        self.draw_labels_check.setChecked(True)
        layout.addWidget(self.draw_labels_check)
        
        self.draw_confidence_check = QCheckBox("Show confidence scores")
        self.draw_confidence_check.setChecked(True)
        layout.addWidget(self.draw_confidence_check)
        
        group.setLayout(layout)
        return group
        
    def create_output_settings(self) -> QGroupBox:
        """Create output settings section"""
        group = QGroupBox("Output Settings")
        layout = QVBoxLayout()
        
        # Output directory
        dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Same as input file")
        dir_layout.addWidget(self.output_dir_edit)
        
        self.output_browse_btn = QPushButton("Browse...")
        self.output_browse_btn.clicked.connect(self.browse_output_dir)
        dir_layout.addWidget(self.output_browse_btn)
        layout.addLayout(dir_layout)
        
        # Output format
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Video format:"))
        self.video_format_combo = QComboBox()
        self.video_format_combo.addItems([".mp4", ".avi"])
        format_layout.addWidget(self.video_format_combo)
        layout.addLayout(format_layout)
        
        # Data export format
        data_layout = QHBoxLayout()
        data_layout.addWidget(QLabel("Data format:"))
        self.data_format_combo = QComboBox()
        self.data_format_combo.addItems([".json", ".csv", ".xml"])
        data_layout.addWidget(self.data_format_combo)
        layout.addLayout(data_layout)
        
        # Export options
        self.export_video_check = QCheckBox("Export processed video")
        self.export_video_check.setChecked(True)
        layout.addWidget(self.export_video_check)
        
        self.export_data_check = QCheckBox("Export detection data")
        self.export_data_check.setChecked(True)
        layout.addWidget(self.export_data_check)
        
        self.export_frames_check = QCheckBox("Export individual frames")
        self.export_frames_check.setChecked(False)
        layout.addWidget(self.export_frames_check)
        
        group.setLayout(layout)
        return group
        
    def create_control_buttons(self) -> QGroupBox:
        """Create control buttons"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        
        # Start/Stop button
        self.start_stop_btn = QPushButton("🚀 Start Processing")
        self.start_stop_btn.setMinimumHeight(40)
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.start_stop_btn.clicked.connect(self.toggle_processing)
        self.start_stop_btn.setEnabled(False)
        layout.addWidget(self.start_stop_btn)
        
        # Clear button
        self.clear_btn = QPushButton("🗑️ Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)
        layout.addWidget(self.clear_btn)
        
        group.setLayout(layout)
        return group
        
    def create_progress_section(self) -> QGroupBox:
        """Create progress section"""
        group = QGroupBox("Progress")
        layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Statistics
        stats_layout = QHBoxLayout()
        
        self.frames_label = QLabel("Frames: 0/0")
        stats_layout.addWidget(self.frames_label)
        
        self.time_label = QLabel("Time: 0.0s")
        stats_layout.addWidget(self.time_label)
        
        self.fps_label = QLabel("FPS: 0.0")
        stats_layout.addWidget(self.fps_label)
        
        layout.addLayout(stats_layout)
        
        group.setLayout(layout)
        return group
        
    def create_results_display(self) -> QGroupBox:
        """Create results display section"""
        group = QGroupBox("Results")
        layout = QVBoxLayout()
        
        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "Frame", "Time (s)", "Detections", "Confidence", "Objects"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.results_table)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(150)
        self.log_output.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ff00;
                font-family: Consolas, monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.log_output)
        
        group.setLayout(layout)
        return group
        
    def browse_file(self):
        """Browse for input file"""
        formats = " ".join([f"*{fmt}" for fmt in SUPPORTED_VIDEO_FORMATS])
        file_filter = f"Video Files ({formats});;All Files (*.*)"
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            file_filter
        )
        
        if file_path:
            self.set_input_file(file_path)
            
    def browse_output_dir(self):
        """Browse for output directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            ""
        )
        
        if dir_path:
            self.output_dir_edit.setText(dir_path)
            
    def set_input_file(self, file_path: str):
        """Set the input file"""
        self.current_file = file_path
        self.file_path_edit.setText(file_path)
        
        # Update file info
        file_info = Path(file_path)
        size_mb = file_info.stat().st_size / (1024 * 1024)
        self.file_info_label.setText(
            f"File: {file_info.name} | Size: {size_mb:.1f} MB | "
            f"Format: {file_info.suffix}"
        )
        self.file_info_label.setStyleSheet("color: green;")
        
        # Enable start button
        self.start_stop_btn.setEnabled(True)
        
        # Emit signal
        self.file_selected.emit(file_path)
        
        # Log
        self.log_message(f"File loaded: {file_info.name}")
        
    def toggle_processing(self):
        """Toggle processing state"""
        if not self.processing:
            self.start_processing()
        else:
            self.stop_processing()
            
    def start_processing(self):
        """Start processing"""
        if not self.current_file:
            QMessageBox.warning(self, "No File", "Please select a video file first.")
            return
            
        self.processing = True
        self.start_stop_btn.setText("⏹ Stop Processing")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        # Disable controls
        self.browse_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.confidence_spin.setEnabled(False)
        
        # Get processing parameters
        params = self.get_processing_params()
        
        # Reset progress
        self.progress_bar.setValue(0)
        self.status_label.setText("Processing...")
        
        # Clear previous results
        self.results_table.setRowCount(0)
        
        # Emit signal
        self.process_started.emit(params)
        
        # Log
        self.log_message("Processing started...")
        
    def stop_processing(self):
        """Stop processing"""
        self.processing = False
        self.start_stop_btn.setText("🚀 Start Processing")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        # Enable controls
        self.browse_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.confidence_spin.setEnabled(True)
        
        self.status_label.setText("Processing stopped")
        
        # Emit signal
        self.process_stopped.emit()
        
        # Log
        self.log_message("Processing stopped by user")
        
    def get_processing_params(self) -> dict:
        """Get processing parameters"""
        # Parse confidence value
        conf_text = self.confidence_spin.currentText()
        confidence = float(conf_text.split()[0])
        
        # Parse frame skip
        skip_text = self.frame_skip_combo.currentText()
        if "Every frame" in skip_text:
            frame_skip = 1
        else:
            frame_skip = int(skip_text.split()[1])
            
        # Get output directory
        output_dir = self.output_dir_edit.text()
        if not output_dir:
            output_dir = str(Path(self.current_file).parent)
            
        return {
            'input_file': self.current_file,
            'model': self.model_combo.currentText().split()[0],
            'confidence': confidence,
            'frame_skip': frame_skip,
            'draw_boxes': self.draw_boxes_check.isChecked(),
            'draw_labels': self.draw_labels_check.isChecked(),
            'draw_confidence': self.draw_confidence_check.isChecked(),
            'output_dir': output_dir,
            'video_format': self.video_format_combo.currentText(),
            'data_format': self.data_format_combo.currentText(),
            'export_video': self.export_video_check.isChecked(),
            'export_data': self.export_data_check.isChecked(),
            'export_frames': self.export_frames_check.isChecked()
        }
        
    def update_progress(self, current: int, total: int, fps: float, elapsed: float):
        """Update progress display"""
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            
        self.frames_label.setText(f"Frames: {current}/{total}")
        self.time_label.setText(f"Time: {elapsed:.1f}s")
        self.fps_label.setText(f"FPS: {fps:.1f}")
        
    def add_result_row(self, frame_num: int, timestamp: float, 
                      detections: int, avg_confidence: float, objects: str):
        """Add a row to results table"""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        self.results_table.setItem(row, 0, QTableWidgetItem(str(frame_num)))
        self.results_table.setItem(row, 1, QTableWidgetItem(f"{timestamp:.2f}"))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(detections)))
        self.results_table.setItem(row, 3, QTableWidgetItem(f"{avg_confidence:.2f}"))
        self.results_table.setItem(row, 4, QTableWidgetItem(objects))
        
    def log_message(self, message: str):
        """Add message to log"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.append(f"[{timestamp}] {message}")
        
    def clear_results(self):
        """Clear all results"""
        self.results_table.setRowCount(0)
        self.log_output.clear()
        self.progress_bar.setValue(0)
        self.frames_label.setText("Frames: 0/0")
        self.time_label.setText("Time: 0.0s")
        self.fps_label.setText("FPS: 0.0")
        self.status_label.setText("Ready")
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile():
                file_path = urls[0].toLocalFile()
                if any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_VIDEO_FORMATS):
                    event.acceptProposedAction()
                    
    def dropEvent(self, event: QDropEvent):
        """Handle drop event"""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile():
                file_path = urls[0].toLocalFile()
                if any(file_path.lower().endswith(fmt) for fmt in SUPPORTED_VIDEO_FORMATS):
                    self.set_input_file(file_path)
                    event.acceptProposedAction()
                    
    def processing_completed(self, output_files: dict):
        """Handle processing completion"""
        self.processing = False
        self.start_stop_btn.setText("🚀 Start Processing")
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        
        # Enable controls
        self.browse_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.confidence_spin.setEnabled(True)
        
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing completed!")
        
        # Log output files
        self.log_message("Processing completed successfully!")
        if 'video' in output_files:
            self.log_message(f"Output video: {output_files['video']}")
        if 'data' in output_files:
            self.log_message(f"Detection data: {output_files['data']}")
            
        # Show completion message
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Video processing completed successfully!\n\n"
            f"Output files:\n{chr(10).join(output_files.values())}"
        )