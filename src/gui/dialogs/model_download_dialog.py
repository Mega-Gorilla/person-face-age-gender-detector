"""
Model download dialog for GUI
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QProgressBar,
    QTextEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QFont
import logging
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.core.age_gender_caffe import CaffeAgeGenderEstimator, check_gdown_installed

logger = logging.getLogger(__name__)


class ModelDownloadThread(QThread):
    """Thread for downloading models"""
    
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(bool)
    error = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.estimator = None
        
    def run(self):
        """Download models"""
        try:
            self.status.emit("Initializing model downloader...")
            self.progress.emit(10)
            
            # Check if gdown is installed
            if not check_gdown_installed():
                self.error.emit("gdown package is not installed.\nPlease install it with: pip install gdown")
                self.finished.emit(False)
                return
            
            self.status.emit("Checking for Caffe models...")
            self.progress.emit(20)
            
            # Initialize estimator (this will trigger download if needed)
            self.estimator = CaffeAgeGenderEstimator(use_gpu=False)
            
            if self.estimator.method == 'caffe':
                self.status.emit("✓ Models loaded successfully!")
                self.progress.emit(100)
                self.finished.emit(True)
            else:
                self.status.emit("⚠ Models could not be loaded")
                self.progress.emit(100)
                self.finished.emit(False)
                
        except Exception as e:
            self.error.emit(f"Error during model initialization: {str(e)}")
            self.finished.emit(False)


class ModelDownloadDialog(QDialog):
    """Dialog for model download with progress"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.download_thread = None
        self.download_success = False
        
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Age/Gender Model Setup")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("Caffe Model Setup for Age/Gender Estimation")
        title_font = QFont()
        title_font.setPointSize(12)
        title_font.setBold(True)
        title.setFont(title_font)
        layout.addWidget(title)
        
        # Info text
        info = QLabel(
            "The application needs to download Caffe models for age and gender estimation.\n"
            "This is a one-time download of approximately 90MB."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Status label
        self.status_label = QLabel("Ready to check models...")
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Log output
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(200)
        layout.addWidget(self.log_output)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.download_btn = QPushButton("Check && Download Models")
        self.download_btn.clicked.connect(self.start_download)
        button_layout.addWidget(self.download_btn)
        
        self.skip_btn = QPushButton("Skip (No Age/Gender)")
        self.skip_btn.clicked.connect(self.skip_download)
        button_layout.addWidget(self.skip_btn)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.accept)
        self.close_btn.setEnabled(False)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
        
    def start_download(self):
        """Start the model download process"""
        self.download_btn.setEnabled(False)
        self.skip_btn.setEnabled(False)
        
        self.log_output.append("Starting model check and download...")
        
        # Create and start download thread
        self.download_thread = ModelDownloadThread()
        self.download_thread.progress.connect(self.update_progress)
        self.download_thread.status.connect(self.update_status)
        self.download_thread.error.connect(self.show_error)
        self.download_thread.finished.connect(self.download_finished)
        
        self.download_thread.start()
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_bar.setValue(value)
        
    def update_status(self, message):
        """Update status message"""
        self.status_label.setText(message)
        self.log_output.append(message)
        
    def show_error(self, error_msg):
        """Show error message"""
        self.log_output.append(f"ERROR: {error_msg}")
        self.status_label.setText("Error occurred during download")
        
    def download_finished(self, success):
        """Handle download completion"""
        self.download_success = success
        
        if success:
            self.log_output.append("\n✓ Models are ready for use!")
            self.status_label.setText("Models loaded successfully!")
            self.close_btn.setText("Continue")
        else:
            self.log_output.append("\n⚠ Models could not be loaded.")
            self.log_output.append("Age/Gender estimation will not be available.")
            self.close_btn.setText("Continue without models")
            
        self.close_btn.setEnabled(True)
        self.download_btn.setEnabled(True)
        self.download_btn.setText("Retry")
        
    def skip_download(self):
        """Skip the download"""
        self.log_output.append("\nSkipping model download.")
        self.log_output.append("Age/Gender estimation will not be available.")
        self.download_success = False
        self.accept()
        
    def get_download_status(self):
        """Get the download status"""
        return self.download_success


def check_and_download_models(parent=None):
    """Check if models exist and offer to download if not"""
    from src.core.age_gender_caffe import CaffeAgeGenderEstimator
    
    # Check if models already exist
    model_dir = Path("models/age_gender_caffe")
    age_model = model_dir / "age_net.caffemodel"
    gender_model = model_dir / "gender_net.caffemodel"
    
    if age_model.exists() and gender_model.exists():
        # Models already exist
        logger.info("Caffe models already downloaded")
        return True
        
    # Show download dialog
    dialog = ModelDownloadDialog(parent)
    dialog.exec()
    
    return dialog.get_download_status()