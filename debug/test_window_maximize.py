#!/usr/bin/env python3
"""
Test script for window maximize functionality on Ubuntu/Wayland
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLabel
from PySide6.QtCore import Qt
from src.gui.windows.window_fix import WaylandWindowMixin, apply_platform_specific_fixes, get_window_manager_info
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestWindow(WaylandWindowMixin, QMainWindow):
    """Test window with maximize fixes"""
    
    def __init__(self):
        super().__init__()
        
        # Apply platform fixes
        apply_platform_specific_fixes(self)
        
        # Setup UI
        self.setWindowTitle("Window Maximize Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Window info
        wm_info = get_window_manager_info()
        info_text = f"""
        <h2>Window Manager Information</h2>
        <p><b>Session Type:</b> {wm_info['session_type']}</p>
        <p><b>Desktop Session:</b> {wm_info['desktop_session']}</p>
        <p><b>Current Desktop:</b> {wm_info['xdg_current_desktop']}</p>
        <p><b>Is Wayland:</b> {wm_info['is_wayland']}</p>
        <p><b>Is X11:</b> {wm_info['is_x11']}</p>
        
        <h3>Keyboard Shortcuts:</h3>
        <ul>
        <li><b>F11 / Alt+Enter:</b> Toggle Fullscreen</li>
        <li><b>Ctrl+M:</b> Toggle Maximize</li>
        <li><b>Ctrl+Shift+M:</b> Cycle Window States</li>
        </ul>
        """
        
        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Test buttons
        btn_maximize = QPushButton("Toggle Maximize (Ctrl+M)")
        btn_maximize.clicked.connect(self.toggle_maximize)
        layout.addWidget(btn_maximize)
        
        btn_fullscreen = QPushButton("Toggle Fullscreen (F11)")
        btn_fullscreen.clicked.connect(self.toggle_fullscreen_safe)
        layout.addWidget(btn_fullscreen)
        
        btn_smart = QPushButton("Smart Maximize (Fill Available Space)")
        btn_smart.clicked.connect(self.smart_maximize)
        layout.addWidget(btn_smart)
        
        btn_center = QPushButton("Center Window")
        btn_center.clicked.connect(self.center_window)
        layout.addWidget(btn_center)
        
        btn_cycle = QPushButton("Cycle States (Ctrl+Shift+M)")
        btn_cycle.clicked.connect(self.cycle_window_state)
        layout.addWidget(btn_cycle)
        
        btn_normal = QPushButton("Show Normal")
        btn_normal.clicked.connect(self.showNormal)
        layout.addWidget(btn_normal)
        
        # State label
        self.state_label = QLabel("Current State: Normal")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet("""
            QLabel {
                background-color: #2196F3;
                color: white;
                padding: 10px;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.state_label)
        
        # Setup window fixes
        self.setup_window_fixes()
        
        # Update state label periodically
        from PySide6.QtCore import QTimer
        self.state_timer = QTimer()
        self.state_timer.timeout.connect(self.update_state_label)
        self.state_timer.start(500)
        
        logger.info("Test window initialized")
        
    def update_state_label(self):
        """Update the state label with current window state"""
        states = []
        if self.isFullScreen():
            states.append("Fullscreen")
        elif self.isMaximized():
            states.append("Maximized")
        else:
            states.append("Normal")
            
        if self.isMinimized():
            states.append("(Minimized)")
            
        self.state_label.setText(f"Current State: {' '.join(states)}")
        
        # Update button colors based on state
        if self.isFullScreen():
            self.state_label.setStyleSheet("""
                QLabel {
                    background-color: #f44336;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        elif self.isMaximized():
            self.state_label.setStyleSheet("""
                QLabel {
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)
        else:
            self.state_label.setStyleSheet("""
                QLabel {
                    background-color: #2196F3;
                    color: white;
                    padding: 10px;
                    border-radius: 5px;
                    font-size: 14px;
                    font-weight: bold;
                }
            """)

def main():
    """Main test function"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show test window
    window = TestWindow()
    window.show()
    
    # Log initial state
    logger.info("Window shown. Testing maximize functionality...")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())