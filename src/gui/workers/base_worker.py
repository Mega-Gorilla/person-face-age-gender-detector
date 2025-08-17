"""
Base worker class for detection processing
"""

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class BaseDetectionWorker(QThread):
    """Base class for detection workers"""
    
    # Common signals
    error_occurred = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_running = False
        
    def convert_cv_to_qimage(self, cv_img: np.ndarray) -> QImage:
        """
        Convert OpenCV image to QImage
        
        Args:
            cv_img: OpenCV image (BGR format)
            
        Returns:
            QImage in RGB format
        """
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        
        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        
        # Create QImage
        qimage = QImage(
            rgb_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888
        )
        
        return qimage.copy()
    
    def format_detection_data(self, detection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format detection data for export
        
        Args:
            detection: Raw detection dictionary
            
        Returns:
            Formatted detection dictionary
        """
        # Convert bbox to list format if needed
        bbox = detection.get('bbox', (0, 0, 0, 0))
        if isinstance(bbox, tuple):
            bbox = list(bbox)
        elif hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
            
        return {
            'bbox': bbox,
            'confidence': float(detection.get('confidence', 0.0)),
            'class': detection.get('class_name', 'person'),
            'center': list(detection.get('center', (0, 0))),
            'area': detection.get('area', 0),
            'width': detection.get('width', 0),
            'height': detection.get('height', 0)
        }
    
    def handle_error(self, error_message: str, exception: Optional[Exception] = None):
        """
        Handle and log errors
        
        Args:
            error_message: Error message to log and emit
            exception: Optional exception object
        """
        if exception:
            logger.error(f"{error_message}: {exception}")
            full_message = f"{error_message}: {str(exception)}"
        else:
            logger.error(error_message)
            full_message = error_message
            
        if self.is_running:
            self.error_occurred.emit(full_message)
    
    def stop(self):
        """Stop the worker thread"""
        self.is_running = False
        
        # Wait for thread to finish
        if not self.wait(2000):
            logger.warning(f"{self.__class__.__name__} did not stop gracefully")
            self.terminate()
            self.wait(500)
    
    def cleanup(self):
        """Cleanup resources - override in subclasses"""
        pass