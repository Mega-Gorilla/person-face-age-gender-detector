#!/usr/bin/env python3
"""
Quick GUI test to verify checkbox functionality
"""

import sys
import time
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

sys.path.insert(0, str(Path(__file__).parent))

from src.gui.windows.main_window import MainWindow

def test_gui():
    """Quick GUI test"""
    app = QApplication(sys.argv)
    
    # Create main window
    window = MainWindow()
    
    # Check initial states
    print("\n" + "="*60)
    print("GUI Checkbox State Test")
    print("="*60)
    
    print("\n1. Initial checkbox states:")
    face_checkbox = window.control_panel.face_detection_checkbox if hasattr(window.control_panel, 'face_detection_checkbox') else None
    age_checkbox = window.control_panel.age_gender_checkbox if hasattr(window.control_panel, 'age_gender_checkbox') else None
    
    if face_checkbox:
        print(f"   Face detection checkbox: {face_checkbox.isChecked()}")
    if age_checkbox:
        print(f"   Age/gender checkbox: {age_checkbox.isChecked()}")
    
    print("\n2. Worker states:")
    print(f"   Worker face detection: {window.detection_worker.enable_face_detection}")
    print(f"   Worker age/gender: {window.detection_worker.enable_age_gender}")
    
    print("\n3. Pipeline existence:")
    print(f"   Pipeline exists: {window.detection_worker.pipeline is not None}")
    
    if window.detection_worker.pipeline:
        print(f"   Pipeline face detector: {window.detection_worker.pipeline.face_detector is not None}")
        print(f"   Pipeline age estimator: {window.detection_worker.pipeline.age_gender_estimator is not None}")
    
    # Simulate toggling face detection
    def toggle_test():
        print("\n4. Simulating checkbox toggle...")
        if face_checkbox:
            face_checkbox.setChecked(True)
            print(f"   Face checkbox now: {face_checkbox.isChecked()}")
            
        # Give it a moment to process
        QTimer.singleShot(500, check_after_toggle)
    
    def check_after_toggle():
        print("\n5. After toggle:")
        print(f"   Worker face detection: {window.detection_worker.enable_face_detection}")
        if window.detection_worker.pipeline:
            print(f"   Pipeline face detection: {window.detection_worker.pipeline.enable_face_detection}")
        
        print("\n" + "="*60)
        print("Test completed! Closing in 2 seconds...")
        print("="*60)
        
        # Close after delay
        QTimer.singleShot(2000, app.quit)
    
    # Start test after window is shown
    window.show()
    QTimer.singleShot(1000, toggle_test)
    
    return app.exec()

if __name__ == "__main__":
    test_gui()