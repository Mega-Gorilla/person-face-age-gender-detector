#!/usr/bin/env python3
"""
Test script for GUI tab functionality
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from src.gui.windows.main_window import MainWindow
from src.utils.version import get_full_version, APP_NAME

def test_gui_tabs():
    """Test GUI with tabs"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {APP_NAME}")
    logger.info(f"Version: {get_full_version()}")
    
    # Create application
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    try:
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Check tabs
        logger.info(f"Number of tabs: {window.tab_widget.count()}")
        logger.info(f"Tab 0: {window.tab_widget.tabText(0)}")
        logger.info(f"Tab 1: {window.tab_widget.tabText(1)}")
        
        # Verify components
        assert hasattr(window, 'stream_tab'), "Stream tab not found"
        assert hasattr(window, 'file_tab'), "File tab not found"
        assert hasattr(window, 'video_widget'), "Video widget not found"
        assert hasattr(window, 'control_panel'), "Control panel not found"
        
        logger.info("✓ All tabs and components initialized successfully")
        
        # Start application
        return app.exec()
        
    except Exception as e:
        logger.error(f"GUI test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_gui_tabs())